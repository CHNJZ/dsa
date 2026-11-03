"""Typed bindings to existing memref/HIVM operations absent from public TLA.

No compiler flags, assembly rewriting, or numerical approximations. Helpers
are inlined by the normal pipeline. Keep installed site-packages unchanged.
"""

import triton.language as tl
from triton.language.core import builtin, _unwrap_if_constexpr
from tla_native_ext import _call


@builtin
def scalar_load(buf, index, _semantic=None):
    if buf.space != "ub" or buf.dtype != tl.int32 or len(buf.shape) != 1:
        raise ValueError("scalar_load requires a rank-one int32 UB buffer")
    i = _semantic.to_tensor(index)
    bt = str(buf.handle.get_type())
    it = str(i.handle.get_type())
    body = f"""%i = arith.index_cast %a1 : {it} to index
%value = memref.load %a0[%i] : {bt}
tt.return %value : i32"""
    values = _call(_semantic, [buf.handle, i.handle], body, ("i32",))
    return tl.tensor(values.get_result(0), tl.int32)


@builtin
def load_pair(base, offset, stride, dst, _semantic=None):
    """Two contiguous rows from GM with a runtime row stride (in elements).

    Caller ensures both rows are in bounds and stride is positive and fits DMA
    hardware limits. base is the complete, zero-offset rank-one GM buffer.
    """
    if (
        base.space != "gm"
        or dst.space != "ub"
        or base.dtype != dst.dtype
        or len(base.shape) != 1
        or dst.shape[0] != 2
    ):
        raise ValueError("load_pair requires GM base and a two-row UB destination")
    off = _semantic.to_tensor(offset)
    st = _semantic.to_tensor(stride)
    bt, dt = str(base.handle.get_type()), str(dst.handle.get_type())
    ot, stt = str(off.handle.get_type()), str(st.handle.get_type())
    width = dst.shape[1]
    elem = "bf16"
    if dst.dtype != tl.bfloat16:
        raise ValueError("BF16 only")
    view = (
        f"memref<2x{width}x{elem}, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>"
    )
    body = f"""%offset = arith.index_cast %a1 : {ot} to index
%stride = arith.index_cast %a2 : {stt} to index
%source = memref.reinterpret_cast %a0 to offset: [%offset], sizes: [2, {width}], strides: [%stride, 1] : {bt} to {view}
hivm.hir.load ins(%source : {view}) outs(%a3 : {dt})
tt.return"""
    _call(_semantic, [base.handle, off.handle, st.handle, dst.handle], body)


@builtin
def byte_view(base, shape, dtype, _semantic=None):
    """Zero-offset typed view into a byte allocation; aliases are explicit IR.

    The caller must respect lifetimes. Compiler alias and synchronization analysis
    see the memref.view operation; this is not a no-alias claim.
    """
    from triton.language.extra.tla.core import buffer, buffer_type
    import math

    shape = tuple(_unwrap_if_constexpr(x) for x in _unwrap_if_constexpr(shape))
    dtype = _unwrap_if_constexpr(dtype)
    if base.space != "ub" or base.dtype != tl.int8 or len(base.shape) != 1:
        raise ValueError("byte_view requires a rank-one UB byte allocation")
    bits = {"bf16": 16, "fp32": 32}[str(dtype)]
    if math.prod(shape) * bits // 8 > base.shape[0]:
        raise ValueError("view exceeds byte allocation")
    ty = buffer_type(shape, dtype, "ub")
    rt = str(ty.to_ir(_semantic.builder))
    bt = str(base.handle.get_type())
    body = f"""%zero = arith.constant 0 : index
%view = memref.view %a0[%zero][] : {bt} to {rt}
tt.return %view : {rt}"""
    result = _call(_semantic, [base.handle], body, (rt,))
    return buffer(result.get_result(0), ty)


@builtin
def load_rows(base, offset, stride, count, dst, _semantic=None):
    """One or two BF16 rows; positive stride, runtime DMA burst count.

    One-row copies allow repeated indices without an out-of-bounds second read.
    The caller fills the duplicate row from row zero during the layout conversion.
    """
    if (
        base.space != "gm"
        or dst.space != "ub"
        or dst.shape[0] != 2
        or base.dtype != tl.bfloat16
    ):
        raise ValueError("load_rows expects BF16 GM and a two-row UB tile")
    off, st, n = (_semantic.to_tensor(x) for x in (offset, stride, count))
    bt, dt = str(base.handle.get_type()), str(dst.handle.get_type())
    width = dst.shape[1]
    src = f"memref<?x{width}xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>"
    full = (
        f"memref<2x{width}xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>"
    )
    dst_offset = "?" if dst.offset is None else str(dst.offset)
    dest = f"memref<?x{width}xbf16, strided<[{dst.strides[0]}, 1], offset: {dst_offset}>, #hivm.address_space<ub>>"
    body = f"""%offset = arith.index_cast %a1 : {off.handle.get_type()} to index
%stride = arith.index_cast %a2 : {st.handle.get_type()} to index
%count = arith.index_cast %a4 : {n.handle.get_type()} to index
%full = memref.reinterpret_cast %a0 to offset: [%offset], sizes: [2, {width}], strides: [%stride, 1] : {bt} to {full}
%src = memref.subview %full[0, 0] [%count, {width}] [1, 1] : {full} to {src}
%dst = memref.subview %a3[0, 0] [%count, {width}] [1, 1] : {dt} to {dest}
hivm.hir.load ins(%src : {src}) outs(%dst : {dest})
tt.return"""
    _call(_semantic, [base.handle, off.handle, st.handle, dst.handle, n.handle], body)
