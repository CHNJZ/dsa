# DSA / Sparse MLA：审计、重构与复现

这里重构的是附件 `issue105-dsa-tla-20261102.zip` 中已经选中的高性能实现，
不是重新寻找一组更有利的 benchmark 参数。主核保留原来的运算顺序、tiling、
同步协议和编译选项；按形状分派，不根据输入数值、随机种子或参考答案分派。

## 建议阅读顺序

1. [从原理到实现](DSA从原理到实现.md)：从 attention 和 MLA 开始，逐步读懂真实 kernel。
2. [公平性审计](FAIRNESS_AUDIT.md)：数学合同、精度、官方 AscendC 源码、计时边界。
3. [重构说明](REFACTOR_NOTES.md)：文件结构、保持运算不变的方法及等价证据。
4. [编译器接口分析](COMPILER_ANALYSIS.md)：旧 wheel、当前 main-dev、新增公共接口和后续方向。
5. [最终性能复测](PERFORMANCE_FINAL.md)：全部 16 规格，六种调用顺序平衡，保留所有样本。
6. [交付结论](DELIVERY.md) 与 `validation/`：完整结论、原始证据和验证边界。

社区 PR：[triton-ascend #2940](https://github.com/triton-lang/triton-ascend/pull/2940)。
本地可直接应用的补丁及验证说明在 [compiler/](compiler/README.md)。

“未发现 reward hack”不等于支持整个原生 API，也不等于整个 DSA 模型加速。
这个实现只计算已给定稀疏索引的 attention 阶段，不包含 indexer、TopK 或 KV cache 更新。
详细判断和验证边界见公平性审计。

## 文件组织

```text
dsa/
  api.py          输入检查、workspace 分配、主核及 merge 的统一入口
  config.py       16 个规格的固定调度参数
  workspace.py    双缓冲 GM gather 路径；BN128/256、PV256/512
  ragged.py       小 decode 的 BN64、不等长 split、直接 NZ 路径
  merge.py        FP32 split 归并
  compat.py       明确分开的 legacy / public 接口模式
reference/        附件选中实现的原始副本，只为可重复的前后比较
tools/            精度、计时、环境检查、AST 等价验证
validation/       原始证据与日志，不放编译缓存和大型 profiler trace
```

## 运行环境与接口模式

需要 Ascend950PR、兼容的 CANN / torch_npu 和 Triton-Ascend。
普通电脑可以运行 AST 等价检查，无法验证设备精度和性能。
先激活自己的 Python 环境，再加载 CANN：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /path/to/x
export ASCEND_RT_VISIBLE_DEVICES=1
export TRITON_CACHE_DIR="$PWD/.cache-legacy"
export DSA_API_MODE=legacy
```

`ASCEND_RT_VISIBLE_DEVICES=1` 是本次复测使用的物理卡映射，运行时使用逻辑 `npu:0`。
在其他机器上请按实际设备调整。不会自动独占设备，也不会停止其他用户的任务。

- **legacy**：适配旧 wheel 中已有的 AVE/HIVM 操作，会在进程内扩展 CodeGenerator；
  不改安装文件。适用于原版和重构版的同编译器比较。
- **public**：要求本次 PR 的三个接口，以及当前 main-dev 已有的 MMAD / SIMD 接口；
  不安装 monkeypatch、不解析临时 MLIR。必须用对应构建验证，不能沿用 legacy 的性能结论。
- **auto**：按接口可用性选择；审计和正式 benchmark 应显式指定前两种模式之一。

本次检查的旧 wheel 为 `3.6.0.dev20261031+git8ca956c1`；选定安装文件与 wheel RECORD
一致，证据见 [环境记录](validation/environment-installed.json)。RECORD 检查是本机安装
一致性检查，不是独立签名认证，也不保证任意历史或未来 wheel 均兼容。

## 最小调用

以下张量均须连续，且位于同一 NPU：

```python
from dsa import forward

# q:       BF16 [B, SQ, H, 512]
# kv:      BF16 [B, L, 1, 512]，同时作为内容 key 和 value
# q_rope:  BF16 [B, SQ, H, 64]
# k_rope:  BF16 [B, L, 1, 64]
# indices: int32 [B, SQ, 1, TOPK]
output, softmax_max, softmax_sum = forward(q, kv, q_rope, k_rope, indices)
```

输出分别是 BF16 `[B,SQ,H,512]` 和两个 FP32 `[B,1,SQ,H]` 张量。
缩放系数固定为本 benchmark 的 `1/24`；真实模型接入必须保留模型原有 scale，
不能把这里的系数直接套到所有 MLA 模型。每个 query 至少要有一个有效索引。
重复索引保留其重复权重；目前不提供 paged、causal 或全无效行的通用实现。
默认只接受配置表中的 16 个规格；额外规格必须显式传入 `Tuning` 并重新验证。

## 可重复验证

工具使用 PyTorch / torch_npu / Triton 和 `pyelftools`；推荐在独立虚拟环境中安装缺失工具。
不要为复现而修改共享安装包。以下结果目录必须尚不存在，避免覆盖旧证据。

```bash
# 不需要 NPU：核对原函数和重构函数的专门化 AST。
python tools/check_refactor_ast.py

# 不需要 NPU：用包内证据重新生成最终性能表。
python tools/summarize_results.py --timing-dir validation/balanced-all16 \
  --markdown-out PERFORMANCE_FINAL.md --json-out validation/summary-final.json \
  --require-complete

# 全部 16 个规格：同一输入、三个输出、完整 FP32 参考、逐位重构比较。
python tools/run_matrix.py --mode accuracy --out validation/my-accuracy

# 3 轮，每轮每种实现 30 个样本；全部六种顺序等频，平衡前驱缓存影响。
# --interleaved 时 samples 应为 6 的倍数。
python tools/run_matrix.py --mode timing --interleaved --samples 30 --rounds 3 \
  --out validation/my-timing

# 用整数视图比较原始位，包括正负零；不以 torch.equal 的浮点数值相等代替。
python tools/check_bitwise.py --out validation/my-bitwise.json

# 更换分布与随机种子，覆盖 ragged、共享 UB、wide、BN256 四条实际路径。
python tools/run_matrix.py --mode accuracy \
  --cases decode_b1 decode_b8 decode_tp8 prefill_tp8 \
  --families uniform peaked duplicates invalid --seeds 20261103 \
  --allow-invalid-native-failure \
  --out validation/my-adversarial

# 安装/构建本次 PR 后，在全新进程里只运行公共接口，不导入原版适配层。
DSA_API_MODE=public TRITON_CACHE_DIR="$PWD/.cache-public" \
  python tools/public_smoke.py --case decode_b1 --out validation/my-public
```

参考计算在 NPU 上使用 FP32，并显式关闭 HF32；这不是 CPU reference。
全张量归一化误差沿用附件的研究阈值，同时记录逐行误差和 RMS，不冒称官方 CANN 验收标准。
原版与重构版还要求 output/max/sum 逐位相同。`invalid` 是边界行为检查，不用于宣称相对
AscendC 的加速；原生版本对该输入的适用合同及结果必须单独解释。
普通精度检查要求三方都通过；只有显式指定的 invalid 边界探索才允许原生失败继续记录，
其 `comparison_passed` 仍为 false，绝不据此计算公平性能结论。

性能工具同时记录设备任务耗时之和、设备任务起止跨度和包含 Python 调度/分配/同步的
eager 调用延迟。它们回答不同的问题，不能混成一个加速比。split merge 始终包含在内。
共享设备上的波动应与原始样本、轮次范围一起看；不要把单个最小值当作无损证明。

## 后续优化的约束

先保留数学、精度和计时合同，再定位实际瓶颈。小 decode 重点是调度与 merge 开销；
大 prefill 重点是 gather 搬运、双缓冲重叠、score/PV 的 UB 占用与矩阵利用率。
接口补齐首先提升实现的可靠性与可维护性，不自动等于性能提升。
更多可执行方向见中文教程及编译器分析。
