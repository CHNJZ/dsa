"""旧 wheel 的兼容层：只构造已有的 AVE/HIVM/memref 操作。

性能主核只依赖本文件提供的七个操作。这里没有修改已安装的文件，
也没有降低精度。legacy 模式保留 typed IR 桥接供原版等价复验；
public 模式只调用正式 API，不必修改注意力算法或流水调度。
"""
import hashlib
import os
import triton.language.extra.tla as tla
from pathlib import Path
import tempfile
import triton.language as tl
from triton.language.core import builtin, _unwrap_if_constexpr
from triton.language.extra.tla.core import vector
from triton.language.extra.tla.cube import _logical_shape
from triton.language.extra.tla.simd import _all_mask, _scalar_handle

# 在 import 前明确选择。默认 legacy 用于与附件逐指令复验；public 严格要求
# 上游 API 完整；auto 仅供交付使用，测试报告必须记录 SELECTED_API_MODE。
REQUESTED_API_MODE = os.environ.get("DSA_API_MODE", "legacy")
if REQUESTED_API_MODE not in ("legacy", "public", "auto"):
    raise ValueError("DSA_API_MODE must be legacy, public or auto")
_REQUIRED = {
    "scalar_load": hasattr(tla, "scalar_load"),
    "byte_view": hasattr(tla, "byte_view"),
    "load_gm_to_ub_strided": hasattr(tla, "load_gm_to_ub_strided"),
    "cube.mmad": hasattr(tla.cube, "mmad"),
    "simd.vmuls": hasattr(tla.simd, "vmuls"),
    "simd.vexpdif": hasattr(tla.simd, "vexpdif"),
}
MISSING_PUBLIC_APIS = tuple(name for name, available in _REQUIRED.items() if not available)
if REQUESTED_API_MODE == "public" and MISSING_PUBLIC_APIS:
    raise RuntimeError("Missing public TLA APIs: " + ", ".join(MISSING_PUBLIC_APIS))
_USE_PUBLIC = REQUESTED_API_MODE == "public" or (REQUESTED_API_MODE == "auto" and not MISSING_PUBLIC_APIS)
SELECTED_API_MODE = "public" if _USE_PUBLIC else "legacy"


def install():
    if _USE_PUBLIC:
        return
    from triton.compiler.code_generator import CodeGenerator

    if getattr(CodeGenerator, "_issue104_native_ext", False):
        return
    original = CodeGenerator.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.builder._tla_extension_module = self.module

    CodeGenerator.__init__ = init
    CodeGenerator._issue104_native_ext = True


def _call(semantic, arguments, body, returns=()):
    from triton._C.libtriton import ir

    builder = semantic.builder
    module = builder._tla_extension_module
    types = [str(a.get_type()) for a in arguments]
    signature = ", ".join(f"%a{i}: {ty}" for i, ty in enumerate(types))
    result = " -> (" + ", ".join(returns) + ")" if returns else ""
    key = hashlib.sha256((signature + result + body).encode()).hexdigest()[:20]
    name = "__tla_native_" + key
    if not module.has_function(name):
        text = (f"module {{ tt.func private @{name}({signature}){result} {{\n{body}\n}} }}")
        with tempfile.TemporaryDirectory(prefix="tla-native-op-") as directory:
            path = Path(directory) / "op.mlir"
            path.write_text(text)
            helper = ir.parse_mlir_module(str(path), arguments[0].get_context())
        # ModuleOp.push_back requires an already-detached operation. Splice
        # the helper module's zero-argument body instead, preserving ownership.
        source_block = helper.get_function(name).get_operation().get_block()
        destination_block = (module.get_function(module.get_entry_func_name()).get_operation().get_block())
        source_block.merge_block_before(destination_block)
    return builder.call(module.get_function(name), arguments)


@builtin
def vmuls(src, scalar, _semantic=None):
    """Native FP32 vector-scalar multiply through the existing TLA binding."""
    if src.dtype != tl.float32 or src.shape != (64, ):
        raise ValueError("vmuls currently requires vector<64xf32>")
    handle = _scalar_handle(src, scalar, _semantic)
    mask = _all_mask(64, _semantic)
    result = _semantic.builder.create_simd_vec_scalar("muls", src.handle, handle, mask.handle)
    return vector(result, src.type)


@builtin
def vexpdif(lhs, rhs, _semantic=None):
    if lhs.type != rhs.type or lhs.dtype != tl.float32 or lhs.shape != (64, ):
        raise ValueError("vexpdif currently requires two vector<64xf32> operands")
    mask = _all_mask(64, _semantic)
    ty, mt = str(lhs.handle.get_type()), str(mask.handle.get_type())
    # The installed AVE dialect has no inliner interface, unlike HIVM.
    # Splice a typed native-op region at this insertion point. The constant
    # condition is removed by the normal TTIR canonicalizer; final IR must
    # contain vexpdif directly, with no helper call or conditional remaining.
    from triton._C.libtriton import ir

    builder = _semantic.builder
    text = f"""module {{ tt.func private @primitive(%a: {ty}, %b: {ty}, %p: {mt}) -> {ty} {{
      %r = scf.execute_region -> {ty} {{
        %e = ave.hir.vexpdif %a, %b, %p : {ty}, {mt}
        scf.yield %e : {ty}
      }}
      tt.return %r : {ty}
    }} }}"""
    with tempfile.TemporaryDirectory(prefix="tla-native-exp-") as directory:
        path = Path(directory) / "op.mlir"
        path.write_text(text)
        helper = ir.parse_mlir_module(str(path), lhs.handle.get_context())
    fn = helper.get_function("primitive")
    for i, handle in enumerate((lhs.handle, rhs.handle, mask.handle)):
        fn.args(i).replace_all_uses_with(handle)
    native = []
    helper.walk(lambda op: native.append(op) if op.get_name() == "ave.hir.vexpdif" else None)
    condition = builder.get_int1(True)
    branch = builder.create_if_op([lhs.handle.get_type()], condition, True)
    native[0].get_block().merge_block_before(branch.get_then_block())
    point = builder.get_insertion_point()
    builder.set_insertion_point_to_end(branch.get_else_block())
    builder.create_yield_op([lhs.handle])
    builder.restore_insertion_point(point)
    return vector(branch.get_result(0), lhs.type)


@builtin
def matmul(a, b, acc, transpose_b=False, clear_accum=False, transpose_a=False, _semantic=None):
    ta, tb, clear = (bool(_unwrap_if_constexpr(x)) for x in (transpose_a, transpose_b, clear_accum))
    m, k = _logical_shape(a.shape)
    k2, n = _logical_shape(b.shape)
    if ta:
        m, k = k, m
    if tb:
        k2, n = n, k2
    if (a.space, b.space, acc.space) != ("l1", "l1", "l0c"):
        raise ValueError("MMAD requires L1/L1/L0C")
    if a.dtype != b.dtype or k != k2 or _logical_shape(acc.shape) != (m, n):
        raise ValueError("MMAD logical shapes or dtypes do not agree")
    attrs = ", ".join(x for x, enabled in [("a_transpose", ta), ("b_transpose", tb)] if enabled)
    attrs = " {" + attrs + "}" if attrs else ""
    at, bt, ct = (str(x.handle.get_type()) for x in (a, b, acc))
    body = f"""%init = arith.constant {str(clear).lower()}
%m = arith.constant {m} : index
%k = arith.constant {k} : index
%n = arith.constant {n} : index
hivm.hir.mmadL1{attrs} ins(%a0, %a1, %init, %m, %k, %n : {at}, {bt}, i1, index, index, index) outs(%a2 : {ct})
tt.return"""
    _call(_semantic, [a.handle, b.handle, acc.handle], body)


install()


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
    values = _call(_semantic, [buf.handle, i.handle], body, ("i32", ))
    return tl.tensor(values.get_result(0), tl.int32)


@builtin
def load_pair(base, offset, stride, dst, _semantic=None):
    """Two contiguous rows from GM with a runtime row stride (in elements).

    Caller ensures both rows are in bounds and stride is positive and fits DMA
    hardware limits. base is the complete, zero-offset rank-one GM buffer.
    """
    if (base.space != "gm" or dst.space != "ub" or base.dtype != dst.dtype or len(base.shape) != 1
            or dst.shape[0] != 2):
        raise ValueError("load_pair requires GM base and a two-row UB destination")
    off = _semantic.to_tensor(offset)
    st = _semantic.to_tensor(stride)
    bt, dt = str(base.handle.get_type()), str(dst.handle.get_type())
    ot, stt = str(off.handle.get_type()), str(st.handle.get_type())
    width = dst.shape[1]
    elem = "bf16"
    if dst.dtype != tl.bfloat16:
        raise ValueError("BF16 only")
    view = (f"memref<2x{width}x{elem}, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>")
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
    result = _call(_semantic, [base.handle], body, (rt, ))
    return buffer(result.get_result(0), ty)


@builtin
def load_rows(base, offset, stride, count, dst, _semantic=None):
    """One or two BF16 rows; positive stride, runtime DMA burst count.

    One-row copies allow repeated indices without an out-of-bounds second read.
    The caller fills the duplicate row from row zero during the layout conversion.
    """
    if (base.space != "gm" or dst.space != "ub" or dst.shape[0] != 2 or base.dtype != tl.bfloat16):
        raise ValueError("load_rows expects BF16 GM and a two-row UB tile")
    off, st, n = (_semantic.to_tensor(x) for x in (offset, stride, count))
    bt, dt = str(base.handle.get_type()), str(dst.handle.get_type())
    width = dst.shape[1]
    src = f"memref<?x{width}xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>"
    full = (f"memref<2x{width}xbf16, strided<[?, 1], offset: ?>, #hivm.address_space<gm>>")
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


if _USE_PUBLIC:
    # 主核仍使用同样的数学与调度；这里只替换前端构造 IR 的入口。
    # public 模式不会 monkeypatch CodeGenerator，也不解析/拼接 MLIR 文本。
    scalar_load = tla.scalar_load
    byte_view = tla.byte_view
    matmul = tla.cube.mmad
    vmuls = tla.simd.vmuls
    vexpdif = tla.simd.vexpdif

    @builtin
    def load_pair(base, offset, stride, dst, _semantic=None):
        return tla.load_gm_to_ub_strided(base, dst, offset, stride, _semantic=_semantic)

    @builtin
    def load_rows(base, offset, stride, count, dst, _semantic=None):
        return tla.load_gm_to_ub_strided(base, dst, offset, stride, rows=count, _semantic=_semantic)
