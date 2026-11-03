"""Sparse MLA 的统一入口：校验形状、分配工作区、发射主核和可选 merge。"""
import torch
import torch_npu  # 注册 PyTorch 的 NPU 设备与 runtime。

from .config import CASE_SHAPES, SELECTED, Tuning
from .merge import merge_partials
from .ragged import ragged_attention
from .workspace import workspace_attention

# 与归档中已测实现一致；没有借重构增加近似或改变编译优化策略。
LAUNCH_OPTIONS = {
    "sanitize_overflow": False,
    "enable_auto_bind_sub_block": False,
    "enable_preload": False,
}


def _check_inputs(q, kv, q_rope, k_rope, indices):
    # Tensor 元数据仍逐次校验，但每个属性只读取所需次数。
    q_shape, kv_shape, ids_shape = q.shape, kv.shape, indices.shape
    if len(q_shape) != 4 or len(kv_shape) != 4 or len(ids_shape) != 4:
        raise ValueError("Q、KV 和 indices 必须是四维张量")
    batch, queries, heads, width = q_shape
    length, topk = kv_shape[1], ids_shape[-1]
    if (width != 512 or kv_shape != (batch, length, 1, 512) or q_rope.shape != (batch, queries, heads, 64)
            or k_rope.shape != (batch, length, 1, 64) or ids_shape != (batch, queries, 1, topk)
            or min(batch, queries, heads, length, topk) <= 0):
        raise ValueError("需要 Q[B,SQ,H,512]、KV[B,L,1,512]、RoPE64、indices[B,SQ,1,K]")
    device = q.device
    if (device.type != "npu" or kv.device != device or q_rope.device != device or k_rope.device != device
            or indices.device != device or not q.is_contiguous() or not kv.is_contiguous()
            or not q_rope.is_contiguous() or not k_rope.is_contiguous() or not indices.is_contiguous()):
        raise ValueError("所有输入必须位于同一 NPU，且连续存放")
    bf16 = torch.bfloat16
    if (q.dtype != bf16 or kv.dtype != bf16 or q_rope.dtype != bf16 or k_rope.dtype != bf16
            or indices.dtype != torch.int32):
        raise ValueError("Q/KV/RoPE 必须为 BF16；indices 必须为 int32")
    return (batch, queries, heads, length, topk), device


def _check_tuning(config, shape, cores):
    if not isinstance(config, Tuning):
        raise TypeError("config 必须是 dsa.Tuning")
    _, _, heads, _, topk = shape
    if type(cores) is not int or cores <= 0:
        raise ValueError("cores 必须是正整数")
    if any(type(v) is not int or v <= 0 for v in (config.bm, config.bn, config.split, config.pv_cols)):
        raise ValueError("BM/BN/split/PV 列数必须是正整数")
    if config.bm not in (16, 32, 64, 128) or heads % config.bm:
        raise ValueError("BM 必须为 16/32/64/128，且整除 head 数")
    if config.direct_nz:
        if config.bn != 64 or topk % 64 or config.split > topk // 64:
            raise ValueError("直接 NZ 路径需要 BN64、完整 tile 和非空 split")
    else:
        if config.bn not in (128, 256) or topk % (2 * config.bn * config.split):
            raise ValueError("GM 工作区路径需要 BN128/256，每个 split 含偶数个 tile")
        if config.pv_cols not in (128, 256, 512):
            raise ValueError("PV 列数必须为 128/256/512")
        if config.share_score and config.pv_cols < config.bn:
            raise ValueError("共享 UB 必须能装下 score tile")


def _validated_defaults():
    """固定形状与不可变 Tuning 只需校验一次，不缓存任何张量或输入值。"""
    defaults = {}
    for name, shape in CASE_SHAPES.items():
        config = SELECTED[name]
        _check_tuning(config, shape, 28)
        defaults[shape] = config
    return defaults


_DEFAULT_CONFIGS = _validated_defaults()


def forward(q, kv, q_rope, k_rope, indices, *, config=None, cores=28, return_kernels=False):
    """计算给定 indices 的稀疏注意力，返回 output、softmax_max、softmax_sum。

    output 为 BF16 [B,SQ,H,512]；两个统计量为 FP32 [B,1,SQ,H]。
    缩放系数固定为 1/sqrt(512+64)。重复索引保留重复权重，不会去重。
    每个 query 至少需要一个有效索引；无效项按 [0,L) 掩码处理。
    这不是 indexer/TopK，也不含 paged/causal 模式；不在这里扫描输入数据。

    return_kernels=True 时返回 (outputs, compiled_kernels)，便于核对 IR、
    二进制及完整的主核+merge 计时。默认调用方式只返回三个结果张量。
    """
    shape, device = _check_inputs(q, kv, q_rope, k_rope, indices)
    if config is None:
        try:
            config = _DEFAULT_CONFIGS[shape]
        except KeyError as error:
            raise ValueError(f"没有验证过的 DSA 配置: {shape}") from error
        # 配置已在导入时检查；调用者可变的 core 数仍须逐次检查。
        if type(cores) is not int or cores <= 0:
            raise ValueError("cores 必须是正整数")
    else:
        _check_tuning(config, shape, cores)
    batch, queries, heads, length, topk = shape
    split = config.split

    # 所有 gather 和分片合并均属于本次调用，没有跨调用缓存输入或结果。
    workspace = None
    if not config.direct_nz:
        workspace = torch.empty((cores, 2, config.bn, 576), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(q)
    maximum = torch.empty((batch, 1, queries, heads), device=device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    if split == 1:
        numerator, partial_max, partial_sum = output, maximum, denominator
    else:
        numerator = torch.empty((batch, queries, split, heads, 512), device=device, dtype=torch.float32)
        partial_max = torch.empty((batch, queries, split, heads), device=device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)

    if config.direct_nz:
        main = ragged_attention[(cores, )](q, kv, q_rope, k_rope, indices, numerator, partial_max,
                                           partial_sum, batch, queries, length, heads, topk, config.bm, cores,
                                           config.bn, split, **LAUNCH_OPTIONS)
    else:
        main = workspace_attention[(cores, )](q, kv, q_rope, k_rope, indices, workspace, numerator,
                                              partial_max, partial_sum, batch, queries, length, heads, topk,
                                              config.bm, cores, config.bn, split, config.pv_cols,
                                              config.share_score, **LAUNCH_OPTIONS)
    if split > 1:
        # 这一步是运算的一部分；benchmark 必须包含，不能只量主核。
        merged = merge_partials[(cores * 2, )](numerator, partial_max, partial_sum, output, maximum,
                                               denominator, batch * queries, heads, split, cores * 2,
                                               **LAUNCH_OPTIONS)
    result = (output, maximum, denominator)
    if return_kernels:
        return result, (main, merged) if split > 1 else (main, )
    return result
