"""BF16 indexed sparse MLA for Ascend950 (D=512, RoPE=64).

Heads form Cube M; selected KV rows form N. Two AIVs gather sorted index
pairs, pack NZ tiles, and run FP32 online softmax and output updates. L1 KV
slots alternate while gather overlaps QK/PV; partial FP32 results are merged
when split > 1. Pair sorting preserves the selected multiset, including repeats.

All data-dependent fast paths are checked on device. No skipped rescaling,
quantized KV, pre-gathered inputs, or ignored merge work. See README.md for
input restrictions, numerical contract, synchronization, and reproduction.
"""

import torch
import torch_npu
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction
import tla_native_ext as ext
import dsa_ext

ext.install()


@TLAJITFunction
def sparse_mla_kernel(
    Q,
    KV,
    QR,
    KR,
    IDS,
    O,
    MAX,
    SUM,
    B: tl.constexpr,
    SQ: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BM: tl.constexpr,
    CORES: tl.constexpr,
    BN: tl.constexpr,
    SPLIT: tl.constexpr,
):
    D: tl.constexpr = 512
    DR: tl.constexpr = 64
    VM: tl.constexpr = BM // 2
    tl.static_assert(BN == 64)
    VN: tl.constexpr = BN // 2
    SCALE: tl.constexpr = 576**-0.5
    for job in tl.range(tl.program_id(0), B * SQ * (H // BM) * SPLIT, CORES):
        token = job // ((H // BM) * SPLIT)
        split = job % SPLIT
        output_token = token * SPLIT + split
        head = ((job // SPLIT) % (H // BM)) * BM
        batch = token // SQ
        q_l1 = tla.alloc((D // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        qr_l1 = tla.alloc((DR // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        kv_l1 = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        kr_l1 = tla.alloc((DR // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        kv_next = tla.alloc((D // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        kr_next = tla.alloc((DR // 16, BN // 16, 16, 16), tl.bfloat16, space="l1")
        kflat_next = tla.reshape(kv_next, (D // 16, BN * 16))
        rflat_next = tla.reshape(kr_next, (DR // 16, BN * 16))
        p_l1 = tla.alloc((BN // 16, BM // 16, 16, 16), tl.bfloat16, space="l1")
        kflat = tla.reshape(kv_l1, (D // 16, BN * 16))
        rflat = tla.reshape(kr_l1, (DR // 16, BN * 16))
        pflat = tla.reshape(p_l1, (BN // 16, BM * 16))
        shared = tla.alloc((VM * 256 * 4,), tl.int8, space="ub")
        kpairs = (
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
            tla.alloc((2, D), tl.bfloat16, space="ub"),
        )
        rpairs = (
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
            tla.alloc((2, 128), tl.bfloat16, space="ub"),
        )
        kpack = tla.alloc((D // 16, (VN + 1) * 16), tl.bfloat16, space="ub")
        pack_shared = tla.alloc(
            (max((DR // 16) * (VN + 1) * 16, (BN // 16) * (VM + 1) * 16) * 2,),
            tl.int8,
            space="ub",
        )
        rpack = dsa_ext.byte_view(pack_shared, (DR // 16, (VN + 1) * 16), tl.bfloat16)
        lows = tla.alloc((64,), tl.int32, space="ub")
        strides = tla.alloc((64,), tl.int32, space="ub")
        counts = tla.alloc((64,), tl.int32, space="ub")
        any_duplicate = tla.alloc((8,), tl.int32, space="ub")
        duplicate_flags = tla.alloc((64,), tl.int32, space="ub")
        ids0 = tla.alloc((BN,), tl.int32, space="ub")
        ids1 = tla.alloc((BN,), tl.int32, space="ub")
        score = tla.alloc((VM, BN), tl.float32, space="ub")
        pub = dsa_ext.byte_view(pack_shared, (BN // 16, (VM + 1) * 16), tl.bfloat16)
        pv = dsa_ext.byte_view(shared, (VM, 256), tl.float32)
        acc = tla.alloc((VM, D), tl.float32, space="ub")
        acc_alt = tla.alloc((VM, D), tl.float32, space="ub")
        out = dsa_ext.byte_view(shared, (VM, 256), tl.bfloat16)
        m = tla.alloc((64,), tl.float32, space="ub")
        l = tla.alloc((64,), tl.float32, space="ub")
        m_alt = tla.alloc((64,), tl.float32, space="ub")
        l_alt = tla.alloc((64,), tl.float32, space="ub")
        alpha = tla.alloc((64,), tl.float32, space="ub")
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + (token * H + head) * D, (BM, D)), q_l1)
            tla.load_gm_to_l1(tla.view(QR + (token * H + head) * DR, (BM, DR)), qr_l1)
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                neginf = tla.simd.vfull(float("-inf"), tl.float32, 64)
                if SPLIT > 1:
                    neginf = tla.simd.vfull(-3.4028234663852886e38, tl.float32, 64)
                tla.simd.vsts(m[0], neginf)
                tla.simd.vsts(l[0], zero)
                for row in tl.range(0, VM):
                    for col in tl.static_range(0, D // 64):
                        tla.simd.vsts(acc[row, col * 64], zero)
        pc = tla.alloc((D // 16, BM // 16, 16, 16), tl.float32, space="l0c")
        # Seed the first ready slot and one free PV-fragment credit.
        first_start = ((split * (TOPK // BN)) // SPLIT) * BN
        split_end = (((split + 1) * (TOPK // BN)) // SPLIT) * BN
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.load_gm_to_ub(tla.view(IDS + token * TOPK + first_start, (BN,)), ids0)
            with tla.scope.simd():
                ii = tla.simd.vlds(ids0[0])
                even, odd = tla.simd.vdintlv(ii, ii)
                low = tla.simd.vmin(even, odd)
                high = tla.simd.vmax(even, odd)
                sorted_ids, unused = tla.simd.vintlv(low, high)
                tla.simd.vsts(ids0[0], sorted_ids)
                zero = tla.simd.vfull(0, tl.int32, 64)
                limit = tla.simd.vfull(L - 1, tl.int32, 64)
                low = tla.simd.vmax(zero, tla.simd.vmin(limit, low))
                high = tla.simd.vmax(zero, tla.simd.vmin(limit, high))
                equal = tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ)
                flags = tla.simd.vsel(equal, tla.simd.vfull(65537, tl.int32, 64), zero)
                tla.simd.vsts(duplicate_flags[0], flags)
                tla.simd.vsts(
                    any_duplicate[0], tla.simd.vreduce_max(flags), dist="ONEPT_B32"
                )
                one = tla.simd.vfull(1, tl.int32, 64)
                two = tla.simd.vfull(2, tl.int32, 64)
                tla.simd.vsts(lows[0], low)
                tla.simd.vsts(strides[0], tla.simd.vmax(one, tla.simd.vsub(high, low)))
                tla.simd.vsts(counts[0], tla.simd.vsel(equal, one, two))
            if dsa_ext.scalar_load(any_duplicate, 0) == 0:
                for pair in tl.static_range(0, VN // 2):
                    row = pair * 2
                    s0 = dsa_ext.scalar_load(lows, sub * (VN // 2) + pair)
                    stride = dsa_ext.scalar_load(strides, sub * (VN // 2) + pair)
                    dsa_ext.load_pair(
                        tla.view(KV, (B * L * D,)),
                        (batch * L + s0) * D,
                        stride * D,
                        kpairs[pair],
                    )
                    dsa_ext.load_pair(
                        tla.view(KR, (B * L * DR,)),
                        (batch * L + s0) * DR,
                        stride * DR,
                        tla.view(rpairs[pair], (2, DR)),
                    )
            else:
                for pair in tl.static_range(0, VN // 2):
                    row = pair * 2
                    s0 = dsa_ext.scalar_load(lows, sub * (VN // 2) + pair)
                    stride = dsa_ext.scalar_load(strides, sub * (VN // 2) + pair)
                    count = dsa_ext.scalar_load(counts, sub * (VN // 2) + pair)
                    dsa_ext.load_rows(
                        tla.view(KV, (B * L * D,)),
                        (batch * L + s0) * D,
                        stride * D,
                        count,
                        kpairs[pair],
                    )
                    dsa_ext.load_rows(
                        tla.view(KR, (B * L * DR,)),
                        (batch * L + s0) * DR,
                        stride * DR,
                        count,
                        tla.view(rpairs[pair], (2, DR)),
                    )
            if dsa_ext.scalar_load(any_duplicate, 0) == 0:
                with tla.scope.simd():
                    for pair in tl.static_range(16):
                        for localrow in tl.static_range(2):
                            row = pair * 2 + localrow
                            for chunk in tl.static_range(D // 128):
                                values = tla.simd.vlds(
                                    kpairs[pair][localrow, chunk * 128]
                                )
                                tla.simd.vsstb(
                                    values, kpack[chunk * 8, row * 16], (VN + 1) * 16
                                )
                            rv = tla.simd.vlds(rpairs[pair][localrow, 0])
                            tla.simd.vsstb(
                                rv,
                                rpack[0, row * 16],
                                (VN + 1) * 16,
                                mask=tla.simd.pge(64, 128),
                            )
            else:
                with tla.scope.simd():
                    for pair in tl.static_range(16):
                        flag = tla.simd.vbitcast(
                            tla.simd.vlds(
                                duplicate_flags[sub * (VN // 2) + pair], dist="BRC_B32"
                            ),
                            "i16",
                        )
                        duplicate = tla.simd.vcmp(
                            flag, tla.simd.vfull(0, tl.int16, 128), op=tla.simd.CMP.NE
                        )
                        for localrow in tl.static_range(2):
                            row = pair * 2 + localrow
                            for chunk in tl.static_range(D // 128):
                                values = tla.simd.vlds(
                                    kpairs[pair][localrow, chunk * 128]
                                )
                                if localrow == 1:
                                    values = tla.simd.vsel(
                                        duplicate,
                                        tla.simd.vlds(kpairs[pair][0, chunk * 128]),
                                        values,
                                    )
                                tla.simd.vsstb(
                                    values, kpack[chunk * 8, row * 16], (VN + 1) * 16
                                )
                            rv = tla.simd.vlds(rpairs[pair][localrow, 0])
                            if localrow == 1:
                                rv = tla.simd.vsel(
                                    duplicate, tla.simd.vlds(rpairs[pair][0, 0]), rv
                                )
                            tla.simd.vsstb(
                                rv,
                                rpack[0, row * 16],
                                (VN + 1) * 16,
                                mask=tla.simd.pge(64, 128),
                            )
            tla.copy_ub_to_l1(
                tla.view(kpack, (D // 16, VN * 16)),
                tla.view(kflat, (D // 16, VN * 16), offsets=(0, sub * VN * 16)),
            )
            tla.copy_ub_to_l1(
                tla.view(rpack, (DR // 16, VN * 16)),
                tla.view(rflat, (DR // 16, VN * 16), offsets=(0, sub * VN * 16)),
            )
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
        with tla.scope.vector():
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        # Flags: 0/5=KV slot ready, 1=score ready, 2=P ready,
        # 3=PV fragment ready, 4=PV fragment consumed/free.
        for pair_start in tl.range(first_start, split_end, 2 * BN):
            for phase in tl.static_range(2):
                start = pair_start + phase * BN
                if start < split_end:
                    if phase == 0:
                        acc_old = acc
                        acc_new = acc_alt
                        m_old = m
                        m_new = m_alt
                        l_old = l
                        l_new = l_alt
                    else:
                        acc_old = acc_alt
                        acc_new = acc
                        m_old = m_alt
                        m_new = m
                        l_old = l_alt
                        l_new = l
                    if phase == 0:
                        # Distinct old/new states avoid in-place vector hazards.
                        key = kv_l1
                        rope = kr_l1
                        ids_cur = ids0
                        ids_prefetch = ids1
                        pack_key = kflat_next
                        pack_rope = rflat_next
                    else:
                        key = kv_next
                        rope = kr_next
                        ids_cur = ids1
                        ids_prefetch = ids0
                        pack_key = kflat
                        pack_rope = rflat
                    with tla.scope.cube():
                        tla.sync.block_wait_flag(
                            tla.sync.MODE.BLOCK, tla.sync.PIPE.S, phase * 5
                        )
                        sc = tla.alloc(
                            (BN // 16, BM // 16, 16, 16), tl.float32, space="l0c"
                        )
                        ext.matmul(q_l1, key, sc, transpose_b=True, clear_accum=True)
                        ext.matmul(qr_l1, rope, sc, transpose_b=True, clear_accum=False)
                        tla.copy_l0c_to_ub(
                            sc,
                            score,
                            tla.FixpipeL0C2UBParams(
                                dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                sub_block_idx=None,
                            ),
                        )
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 1)
                    if start + BN < split_end:
                        with tla.scope.vector():
                            sub = tla.sub_block_id()
                            tla.load_gm_to_ub(
                                tla.view(IDS + token * TOPK + (start + BN), (BN,)),
                                ids_prefetch,
                            )
                            with tla.scope.simd():
                                ii = tla.simd.vlds(ids_prefetch[0])
                                even, odd = tla.simd.vdintlv(ii, ii)
                                low = tla.simd.vmin(even, odd)
                                high = tla.simd.vmax(even, odd)
                                sorted_ids, unused = tla.simd.vintlv(low, high)
                                tla.simd.vsts(ids_prefetch[0], sorted_ids)
                                zero = tla.simd.vfull(0, tl.int32, 64)
                                limit = tla.simd.vfull(L - 1, tl.int32, 64)
                                low = tla.simd.vmax(zero, tla.simd.vmin(limit, low))
                                high = tla.simd.vmax(zero, tla.simd.vmin(limit, high))
                                equal = tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ)
                                flags = tla.simd.vsel(
                                    equal, tla.simd.vfull(65537, tl.int32, 64), zero
                                )
                                tla.simd.vsts(duplicate_flags[0], flags)
                                tla.simd.vsts(
                                    any_duplicate[0],
                                    tla.simd.vreduce_max(flags),
                                    dist="ONEPT_B32",
                                )
                                one = tla.simd.vfull(1, tl.int32, 64)
                                two = tla.simd.vfull(2, tl.int32, 64)
                                tla.simd.vsts(lows[0], low)
                                tla.simd.vsts(
                                    strides[0],
                                    tla.simd.vmax(one, tla.simd.vsub(high, low)),
                                )
                                tla.simd.vsts(counts[0], tla.simd.vsel(equal, one, two))
                            if dsa_ext.scalar_load(any_duplicate, 0) == 0:
                                for pair in tl.static_range(0, VN // 2):
                                    row = pair * 2
                                    s0 = dsa_ext.scalar_load(
                                        lows, sub * (VN // 2) + pair
                                    )
                                    stride = dsa_ext.scalar_load(
                                        strides, sub * (VN // 2) + pair
                                    )
                                    dsa_ext.load_pair(
                                        tla.view(KV, (B * L * D,)),
                                        (batch * L + s0) * D,
                                        stride * D,
                                        kpairs[pair],
                                    )
                                    dsa_ext.load_pair(
                                        tla.view(KR, (B * L * DR,)),
                                        (batch * L + s0) * DR,
                                        stride * DR,
                                        tla.view(rpairs[pair], (2, DR)),
                                    )
                            else:
                                for pair in tl.static_range(0, VN // 2):
                                    row = pair * 2
                                    s0 = dsa_ext.scalar_load(
                                        lows, sub * (VN // 2) + pair
                                    )
                                    stride = dsa_ext.scalar_load(
                                        strides, sub * (VN // 2) + pair
                                    )
                                    count = dsa_ext.scalar_load(
                                        counts, sub * (VN // 2) + pair
                                    )
                                    dsa_ext.load_rows(
                                        tla.view(KV, (B * L * D,)),
                                        (batch * L + s0) * D,
                                        stride * D,
                                        count,
                                        kpairs[pair],
                                    )
                                    dsa_ext.load_rows(
                                        tla.view(KR, (B * L * DR,)),
                                        (batch * L + s0) * DR,
                                        stride * DR,
                                        count,
                                        tla.view(rpairs[pair], (2, DR)),
                                    )
                    with tla.scope.vector():
                        sub = tla.sub_block_id()
                        tla.sync.block_wait_flag(
                            tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1
                        )
                        with tla.scope.simd():
                            neginf = tla.simd.vfull(float("-inf"), tl.float32, 64)
                            izero = tla.simd.vfull(0, tl.int32, 64)
                            ilimit = tla.simd.vfull(L, tl.int32, 64)
                            i0 = tla.simd.vlds(ids_cur[0])
                            i1 = i0
                            if BN == 128:
                                i1 = tla.simd.vlds(ids_cur[64])
                            for row in tl.range(0, VM):
                                x0 = ext.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                                x1 = neginf
                                if BN == 128:
                                    x1 = ext.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
                                x0 = tla.simd.vsel(
                                    tla.simd.vcmp(i0, izero, op=tla.simd.CMP.GE),
                                    x0,
                                    neginf,
                                )
                                x1 = tla.simd.vsel(
                                    tla.simd.vcmp(i1, izero, op=tla.simd.CMP.GE),
                                    x1,
                                    neginf,
                                )
                                x0 = tla.simd.vsel(
                                    tla.simd.vcmp(i0, ilimit, op=tla.simd.CMP.LT),
                                    x0,
                                    neginf,
                                )
                                x1 = tla.simd.vsel(
                                    tla.simd.vcmp(i1, ilimit, op=tla.simd.CMP.LT),
                                    x1,
                                    neginf,
                                )
                                new_m = tla.simd.vreduce_max(tla.simd.vmax(x0, x1))
                                a = tla.simd.vfull(1.0, tl.float32, 64)
                                old_m = tla.simd.vlds(m_old[row], dist="BRC_B32")
                                new_m = tla.simd.vmax(new_m, old_m)
                                a = ext.vexpdif(old_m, new_m)
                                tla.simd.vsts(alpha[row], a, dist="ONEPT_B32")
                                p0 = ext.vexpdif(x0, new_m)
                                p1 = ext.vexpdif(x1, new_m)
                                new_l = tla.simd.vbroadcast(
                                    tla.simd.vreduce_sum(tla.simd.vadd(p0, p1))
                                )
                                old_l = tla.simd.vlds(l_old[row], dist="BRC_B32")
                                new_l = tla.simd.vadd(new_l, tla.simd.vmul(old_l, a))
                                tla.simd.vsts(m_new[row], new_m, dist="ONEPT_B32")
                                tla.simd.vsts(l_new[row], new_l, dist="ONEPT_B32")
                                packed, unused = tla.simd.vdintlv(
                                    tla.simd.vtruncf(p0, tl.bfloat16),
                                    tla.simd.vtruncf(p1, tl.bfloat16),
                                )
                                if BN == 128:
                                    tla.simd.vsstb(
                                        packed, pub[0, row * 16], (VM + 1) * 16
                                    )
                                else:
                                    tla.simd.vsstb(
                                        packed,
                                        pub[0, row * 16],
                                        (VM + 1) * 16,
                                        mask=tla.simd.pge(64, 128),
                                    )
                        tla.copy_ub_to_l1(
                            tla.view(pub, (BN // 16, VM * 16)),
                            tla.view(
                                pflat, (BN // 16, VM * 16), offsets=(0, sub * VM * 16)
                            ),
                        )
                        tla.sync.block_set_flag(
                            tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 2
                        )
                    with tla.scope.cube():
                        tla.sync.block_wait_flag(
                            tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2
                        )
                        ext.matmul(p_l1, key, pc, clear_accum=True)
                    if start + BN < split_end:
                        with tla.scope.vector():
                            sub = tla.sub_block_id()
                            if dsa_ext.scalar_load(any_duplicate, 0) == 0:
                                with tla.scope.simd():
                                    for pair in tl.static_range(16):
                                        for localrow in tl.static_range(2):
                                            row = pair * 2 + localrow
                                            for chunk in tl.static_range(D // 128):
                                                values = tla.simd.vlds(
                                                    kpairs[pair][localrow, chunk * 128]
                                                )
                                                tla.simd.vsstb(
                                                    values,
                                                    kpack[chunk * 8, row * 16],
                                                    (VN + 1) * 16,
                                                )
                                            rv = tla.simd.vlds(
                                                rpairs[pair][localrow, 0]
                                            )
                                            tla.simd.vsstb(
                                                rv,
                                                rpack[0, row * 16],
                                                (VN + 1) * 16,
                                                mask=tla.simd.pge(64, 128),
                                            )
                            else:
                                with tla.scope.simd():
                                    for pair in tl.static_range(16):
                                        flag = tla.simd.vbitcast(
                                            tla.simd.vlds(
                                                duplicate_flags[sub * (VN // 2) + pair],
                                                dist="BRC_B32",
                                            ),
                                            "i16",
                                        )
                                        duplicate = tla.simd.vcmp(
                                            flag,
                                            tla.simd.vfull(0, tl.int16, 128),
                                            op=tla.simd.CMP.NE,
                                        )
                                        for localrow in tl.static_range(2):
                                            row = pair * 2 + localrow
                                            for chunk in tl.static_range(D // 128):
                                                values = tla.simd.vlds(
                                                    kpairs[pair][localrow, chunk * 128]
                                                )
                                                if localrow == 1:
                                                    values = tla.simd.vsel(
                                                        duplicate,
                                                        tla.simd.vlds(
                                                            kpairs[pair][0, chunk * 128]
                                                        ),
                                                        values,
                                                    )
                                                tla.simd.vsstb(
                                                    values,
                                                    kpack[chunk * 8, row * 16],
                                                    (VN + 1) * 16,
                                                )
                                            rv = tla.simd.vlds(
                                                rpairs[pair][localrow, 0]
                                            )
                                            if localrow == 1:
                                                rv = tla.simd.vsel(
                                                    duplicate,
                                                    tla.simd.vlds(rpairs[pair][0, 0]),
                                                    rv,
                                                )
                                            tla.simd.vsstb(
                                                rv,
                                                rpack[0, row * 16],
                                                (VN + 1) * 16,
                                                mask=tla.simd.pge(64, 128),
                                            )
                            tla.copy_ub_to_l1(
                                tla.view(kpack, (D // 16, VN * 16)),
                                tla.view(
                                    pack_key,
                                    (D // 16, VN * 16),
                                    offsets=(0, sub * VN * 16),
                                ),
                            )
                            tla.copy_ub_to_l1(
                                tla.view(rpack, (DR // 16, VN * 16)),
                                tla.view(
                                    pack_rope,
                                    (DR // 16, VN * 16),
                                    offsets=(0, sub * VN * 16),
                                ),
                            )
                            tla.sync.block_set_flag(
                                tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, (1 - phase) * 5
                            )
                    for fragment in tl.range(2):
                        with tla.scope.cube():
                            tla.sync.block_wait_flag(
                                tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4
                            )
                            tla.copy_l0c_to_ub(
                                tla.view(
                                    pc,
                                    (16, BM // 16, 16, 16),
                                    offsets=(fragment * 16, 0, 0, 0),
                                ),
                                pv,
                                tla.FixpipeL0C2UBParams(
                                    dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                    sub_block_idx=None,
                                ),
                            )
                            tla.sync.block_set_flag(
                                tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 3
                            )
                        with tla.scope.vector():
                            tla.sync.block_wait_flag(
                                tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 3
                            )
                            with tla.scope.simd():
                                for row in tl.range(VM):
                                    a = tla.simd.vlds(alpha[row], dist="BRC_B32")
                                    for chunk in tl.static_range(2):
                                        y0 = tla.simd.vlds(pv[row, chunk * 128])
                                        y1 = tla.simd.vlds(pv[row, chunk * 128 + 64])
                                        y0 = tla.simd.vadd(
                                            y0,
                                            tla.simd.vmul(
                                                tla.simd.vlds(
                                                    acc_old[
                                                        row,
                                                        fragment * 256 + chunk * 128,
                                                    ]
                                                ),
                                                a,
                                            ),
                                        )
                                        y1 = tla.simd.vadd(
                                            y1,
                                            tla.simd.vmul(
                                                tla.simd.vlds(
                                                    acc_old[
                                                        row,
                                                        fragment * 256
                                                        + chunk * 128
                                                        + 64,
                                                    ]
                                                ),
                                                a,
                                            ),
                                        )
                                        tla.simd.vsts(
                                            acc_new[row, fragment * 256 + chunk * 128],
                                            y0,
                                        )
                                        tla.simd.vsts(
                                            acc_new[
                                                row, fragment * 256 + chunk * 128 + 64
                                            ],
                                            y1,
                                        )
                            tla.sync.block_set_flag(
                                tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4
                            )
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        if ((split_end - first_start) // BN) % 2 == 0:
            with tla.scope.vector():
                sub = tla.sub_block_id()
                if SPLIT == 1:
                    for fragment in tl.range(2):
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                den = tla.simd.vlds(l[row], dist="BRC_B32")
                                for chunk in tl.static_range(2):
                                    y0 = tla.simd.vlds(
                                        acc[row, fragment * 256 + chunk * 128]
                                    )
                                    y1 = tla.simd.vlds(
                                        acc[row, fragment * 256 + chunk * 128 + 64]
                                    )
                                    b0 = tla.simd.vtruncf(
                                        tla.simd.vdiv(y0, den), tl.bfloat16
                                    )
                                    b1 = tla.simd.vtruncf(
                                        tla.simd.vdiv(y1, den), tl.bfloat16
                                    )
                                    packed, unused = tla.simd.vdintlv(b0, b1)
                                    tla.simd.vsts(out[row, chunk * 128], packed)
                        tla.store_ub_to_gm(
                            out,
                            tla.view(
                                O
                                + (output_token * H + head + sub * VM) * D
                                + fragment * 256,
                                (VM, 256),
                                strides=(D, 1),
                            ),
                        )
                else:
                    tla.store_ub_to_gm(
                        acc,
                        tla.view(O + (output_token * H + head + sub * VM) * D, (VM, D)),
                    )
                tla.store_ub_to_gm(
                    tla.view(m, (VM,), offsets=(0,)),
                    tla.view(MAX + output_token * H + head + sub * VM, (VM,)),
                )
                tla.store_ub_to_gm(
                    tla.view(l, (VM,), offsets=(0,)),
                    tla.view(SUM + output_token * H + head + sub * VM, (VM,)),
                )
        else:
            with tla.scope.vector():
                sub = tla.sub_block_id()
                if SPLIT == 1:
                    for fragment in tl.range(2):
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                den = tla.simd.vlds(l_alt[row], dist="BRC_B32")
                                for chunk in tl.static_range(2):
                                    y0 = tla.simd.vlds(
                                        acc_alt[row, fragment * 256 + chunk * 128]
                                    )
                                    y1 = tla.simd.vlds(
                                        acc_alt[row, fragment * 256 + chunk * 128 + 64]
                                    )
                                    b0 = tla.simd.vtruncf(
                                        tla.simd.vdiv(y0, den), tl.bfloat16
                                    )
                                    b1 = tla.simd.vtruncf(
                                        tla.simd.vdiv(y1, den), tl.bfloat16
                                    )
                                    packed, unused = tla.simd.vdintlv(b0, b1)
                                    tla.simd.vsts(out[row, chunk * 128], packed)
                        tla.store_ub_to_gm(
                            out,
                            tla.view(
                                O
                                + (output_token * H + head + sub * VM) * D
                                + fragment * 256,
                                (VM, 256),
                                strides=(D, 1),
                            ),
                        )
                else:
                    tla.store_ub_to_gm(
                        acc_alt,
                        tla.view(O + (output_token * H + head + sub * VM) * D, (VM, D)),
                    )
                tla.store_ub_to_gm(
                    tla.view(m_alt, (VM,), offsets=(0,)),
                    tla.view(MAX + output_token * H + head + sub * VM, (VM,)),
                )
                tla.store_ub_to_gm(
                    tla.view(l_alt, (VM,), offsets=(0,)),
                    tla.view(SUM + output_token * H + head + sub * VM, (VM,)),
                )


def forward(q, kv, qr, kr, indices, bm=64, cores=28, bn=64, split=1):
    from merge import merge_partials

    b, sq, h, d = q.shape
    topk = indices.shape[-1]
    assert (
        d == 512
        and qr.shape[-1] == 64
        and h % bm == 0
        and topk % bn == 0
        and 1 <= split <= topk // bn
    )
    assert all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in (q, kv, qr, kr))
    assert indices.dtype == torch.int32 and indices.is_contiguous()
    options = dict(
        sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False
    )
    o = torch.empty_like(q)
    mx = torch.empty((b, 1, sq, h), device=q.device, dtype=torch.float32)
    su = torch.empty_like(mx)
    if split == 1:
        partial, pm, pl = o, mx, su
    else:
        partial = torch.empty(
            (b, sq, split, h, d), device=q.device, dtype=torch.float32
        )
        pm = torch.empty((b, sq, split, h), device=q.device, dtype=torch.float32)
        pl = torch.empty_like(pm)
    main_kernel = sparse_mla_kernel[(cores,)](
        q,
        kv,
        qr,
        kr,
        indices,
        partial,
        pm,
        pl,
        b,
        sq,
        kv.shape[1],
        h,
        topk,
        bm,
        cores,
        bn,
        split,
        **options
    )
    global LAST_KERNELS
    LAST_KERNELS = [main_kernel]
    if split > 1:
        merge_kernel = merge_partials[(cores * 2,)](
            partial, pm, pl, o, mx, su, b * sq, h, split, cores * 2, **options
        )
        LAST_KERNELS.append(merge_kernel)
    return o, mx, su
