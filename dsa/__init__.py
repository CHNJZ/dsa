"""Ascend950 上的 BF16 Sparse MLA；公开入口是 forward。"""
from .api import forward
from .config import CASE_SHAPES, SELECTED, Tuning

__all__ = ["forward", "Tuning", "CASE_SHAPES", "SELECTED"]
