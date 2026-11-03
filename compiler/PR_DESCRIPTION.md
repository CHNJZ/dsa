## Problem and DSA analysis

A high-performance sparse-attention implementation needs three memory operations
that were not exposed through public TLA APIs on `main-dev@0988a7ed2`:

1. Read an int32 index from UB into scalar control flow to calculate a selected
   KV row address.
2. Reuse one UB byte allocation as differently typed temporary tiles after the
   previous tile's last use has completed.
3. Copy a small number of contiguous KV rows whose starting offset, row pitch,
   and row count are known at runtime.

The implementation previously expressed these operations by parsing temporary
MLIR helpers and patching the Python code generator. It could run with unchanged
installed package files, but it still depended on private runtime compiler
adaptations. This PR replaces those adaptations with native builder-backed,
public APIs. It does not introduce a DSA-specific operation or a general dynamic
shape extension to the TLA buffer type.

Current main-dev already provides the other required capabilities:
`cube.mmad` transpose/actual extents, offset-preserving `subview` and `reshape`,
and SIMD `vmuls`/`vexpdif`. This PR does not duplicate those changes.

## Public interfaces

```python
# A scalar read for address calculations outside scope.simd().
index = tla.scalar_load(indices, position)
index = tla.scalar_load(indices_2d[row, column])

# Typed aliases of one UB allocation; no data copy or conversion.
storage = tla.alloc((1024,), tl.int8, space="ub")
fp32_tile = tla.byte_view(storage, (128,), tl.float32, byte_offset=128)
bf16_tile = tla.byte_view(storage, (128,), tl.bfloat16, byte_offset=128)

# Rows are contiguous individually; their GM pitch can be dynamic.
tla.load_gm_to_ub_strided(base, tile, offset, row_stride, rows=count)
```

- `scalar_load` accepts int32 UB buffers and one index per dimension. It preserves
  source offsets/strides and emits `memref.load`.
- `byte_view` accepts a zero-offset, contiguous, rank-one int8 UB allocation and
  compile-time shape, dtype, and byte offset. It checks capacity and element
  alignment and emits `memref.view`.
- `load_gm_to_ub_strided` accepts a rank-one, unit-stride GM source and a rank-two
  UB destination with contiguous columns. Source/destination dtypes must match.
  It preserves destination pitch/offset and untouched rows, and emits native
  memref views followed by one `hivm.hir.load`. Omitting `rows` copies all
  destination rows.

Runtime indices, offsets, pitches, and row counts accept signed int32/int64
scalar tensors. Python integers must fit signed int64. Unsigned, narrow, and
boolean tensors are rejected explicitly because `arith.index_cast` sign-extends;
for example, directly widening uint8(200) would produce a negative index.
Static scalar-load indices are first converted correctly to int64, including
Python constants that Triton would otherwise infer as uint32.

## Addressing, aliases, and numerical behavior

`memref.reinterpret_cast` replaces the source descriptor offset; it does not
add an offset relative to the current view. The strided DMA builder extracts
that existing offset and adds the requested relative offset. Destination
subviews retain their original offset and row pitch. Tests compose pointer,
entry-view, source-subview, and transfer offsets to catch double counting or
lost offsets.

The source reinterpret-cast descriptor keeps a static row capacity, and a
source subview carries the runtime row count. Zero-offset GM entries can
canonicalize directly onto `hivm.hir.pointer_cast`, whose existing conversion
requires statically shaped entry descriptors. A dynamic-shaped reinterpret cast
would assert in that path. This scoped construction avoids changing the general
conversion pass. The single DMA accesses only the active subview, so an unused
second row does not need to be valid memory.

Byte views share storage and remain visible as native aliases to memory planning
and synchronization. They make no no-alias assertion. The caller must complete
previous reads before reusing overlapping storage. Scalar reads are ordinary
memory reads, not barriers; automatically synchronized code retains its memory
dependencies, while manually scheduled pipelines remain responsible for theirs.

The frontend rejects invalid static shapes, layouts, types, capacity, and index
bounds, including a statically out-of-bounds first row when other DMA dimensions
are dynamic. Runtime bounds and target-specific DMA restrictions remain the
caller's/backend's responsibility; no runtime bounds checks are inserted.

These are addressing and memory interfaces. They do not change arithmetic,
softmax, accumulation precision, masking, rounding, or repeated-index semantics.
The DMA API neither sorts nor deduplicates indices. The PR makes no claim that a
compiler-version change alone preserves or improves end-to-end DSA performance.

## Validation

Built the native C++ extension against `main-dev@0988a7ed2` and tested on the
Ascend950 environment with Python 3.11. LLVM and NPU-IR dependency revisions are
`f6ded0be897e2878612dd903f7e8bb85448269e5` and
`4eb12142e16fc8c7ed33c3b568658b773ae0d1c9`, respectively.

- **121 tests passed:** 79 host contracts plus 14 configurations × IR/compile/run.
- Every run configuration checks two seeds with exact equality: **28 input sets**.
- Scalar tests cover direct DMA-to-scalar and SIMD-to-scalar dependencies,
  nonzero-offset/strided aliases, and multidimensional indices.
- Byte-view tests check overlapping FP32/BF16 reuse, typed outputs, full backing
  bytes, and untouched guards at byte offsets 0 and 128.
- Strided DMA tests cover FP16/BF16, static/runtime row counts and pitches,
  composed source offsets, pitched destination guards, and wider rows.
  Zero-offset entry cases include a one-row transfer at the very end of the
  source allocation, where a second row would be out of bounds. `rows` is marked
  `do_not_specialize` and the test checks its compiled scalar signature, so
  `rows=1` also exercises the runtime-count path.
- Four complete public-API DSA paths (`decode_b1`, `decode_b8`, `decode_tp8`,
  `prefill_tp8`), including the applicable merge kernels, passed on-device
  numerical checks against the benchmark's FP32 reference. These cover direct
  NZ/ragged split, different PV widths and UB reuse, and BN256. They use the
  existing research validation gates, not an official CANN accuracy certification.
- These complete paths also passed unmodified AST-to-ttadapter lowering. The
  public-path check verifies no DSA-local CodeGenerator patch or legacy helper
  module is used; it permits the stock Ascend backend's normal lazy target-
  attribute initialization.
- Full configured pre-commit checks and `git diff --check` passed; all changed
  source files match the tested SHA-256 manifest. C++ was explicitly formatted
  with `clang-format -i -style=file`. Hooks whose existing filters do not select
  these files report their normal skip.

Reproduce after rebuilding the native extension:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
TLA_TEST_STAGE=run python -m pytest -q -s \
  third_party/ascend/unittest/tla_ut/vector/scalar_load.py \
  third_party/ascend/unittest/tla_ut/vector/byte_view.py \
  third_party/ascend/unittest/tla_ut/dma/strided_load.py \
  third_party/ascend/unittest/tla_ut/memory_frontend_contract.py
```

The complete DSA tests validate integration and numerical behavior. They do not
establish an end-to-end performance improvement from this compiler PR; there is
no such performance claim here.

## Scope of follow-up work

The DSA review also identified useful future work: diagnostics for overlapping
buffer lifetimes and C/V ready/reuse/drain dependencies, constrained coalescing
of compatible row transfers, reusable verified layout transformations, and
compiler receipts that identify the actual native library and effective
options. Those require separate analysis and tests. This PR only adds the three
missing memory interfaces with explicit contracts and native construction.
