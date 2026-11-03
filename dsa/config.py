"""经过原附件选择的 16 个规格与调度参数。

这里只按张量形状分派，不读取输入值、随机种子或参考结果。
BM = 一次处理的 head 数；BN = 一次处理的被选 KV 行数；
split = 一个 query/head tile 在 KV 方向上的分片数。
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Tuning:
    """只描述编译期调度，不改变算子的数学含义。"""

    bm: int
    bn: int
    split: int
    direct_nz: bool = False
    pv_cols: int = 512
    share_score: bool = False


# (batch, query length, heads, KV length, number of selected indices)
CASE_SHAPES = {
    "decode_b1": (1, 1, 128, 8192, 2048),
    "decode_b8": (8, 1, 128, 32768, 2048),
    "decode_b32": (32, 1, 128, 32768, 2048),
    "decode_tp8": (8, 1, 16, 32768, 2048),
    "prefill_q16": (1, 16, 128, 8192, 2048),
    "prefill_q64": (1, 64, 128, 32768, 2048),
    "prefill_tp8": (1, 128, 16, 32768, 2048),
    "prefill_k512": (1, 128, 64, 32768, 512),
    "decode_b4_h128": (4, 1, 128, 8192, 2048),
    "prefill_b4_h128": (4, 8192, 128, 8192, 2048),
    "decode_b4_h64": (4, 1, 64, 8192, 2048),
    "prefill_b4_h64": (4, 8192, 64, 8192, 2048),
    "decode_b4_h32": (4, 1, 32, 8192, 2048),
    "prefill_b4_h32": (4, 8192, 32, 8192, 2048),
    "decode_b4_h16": (4, 1, 16, 8192, 2048),
    "prefill_b4_h16": (4, 8192, 16, 8192, 2048),
}

SELECTED = {
    "decode_b1": Tuning(64, 64, 14, direct_nz=True, pv_cols=256),
    "decode_b8": Tuning(128, 128, 2, pv_cols=256, share_score=True),
    "decode_b32": Tuning(128, 128, 4, pv_cols=256, share_score=True),
    "decode_tp8": Tuning(16, 128, 8),
    "prefill_q16": Tuning(128, 128, 4, pv_cols=256, share_score=True),
    "prefill_q64": Tuning(128, 128, 1, pv_cols=256, share_score=True),
    "prefill_tp8": Tuning(16, 256, 1),
    "prefill_k512": Tuning(64, 128, 1),
    "decode_b4_h128": Tuning(128, 128, 8, pv_cols=256, share_score=True),
    "prefill_b4_h128": Tuning(128, 128, 1, pv_cols=256, share_score=True),
    "decode_b4_h64": Tuning(64, 128, 8),
    "prefill_b4_h64": Tuning(64, 128, 1),
    "decode_b4_h32": Tuning(32, 128, 8),
    "prefill_b4_h32": Tuning(32, 128, 1),
    "decode_b4_h16": Tuning(16, 128, 8),
    "prefill_b4_h16": Tuning(16, 256, 1),
}

_BY_SHAPE = {CASE_SHAPES[name]: tuning for name, tuning in SELECTED.items()}


def select(shape):
    """只接受已有验证配置；额外规格需要调用者显式给出 Tuning。"""
    try:
        return _BY_SHAPE[tuple(shape)]
    except KeyError as error:
        raise ValueError(f"没有验证过的 DSA 配置: {tuple(shape)}") from error
