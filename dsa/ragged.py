"""小 decode 的直接 NZ 路径：BN64 和不等长 split。

本路径把 gather 结果在 Vector 中用 vsstb 打成 NZ，直接送入 L1，
省去 GM 工作区。decode_b1 只有一个 query；split14 用更多 Cube 并行
处理 32 个 KV tile。状态、分片边界与每一条同步保持归档实现不变。

这份内核单独保留，是因为其数据流、QK 的两次 MMAD 以及 ping-pong
状态与 workspace 路径不同，强行统一会掩盖正确性所依赖的时序。
"""
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction

from . import compat as ops


@TLAJITFunction
def ragged_attention(
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
    # 把 TOPK/64 个 tile 均分到 SPLIT 份，允许分片含不同数量的 tile。
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
        shared = tla.alloc((VM * 256 * 4, ), tl.int8, space="ub")
        # 两行 DMA 暂存；按原顺序独立分配，避免改变 UB 布局与生命周期。
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
            (max((DR // 16) * (VN + 1) * 16, (BN // 16) * (VM + 1) * 16) * 2, ),
            tl.int8,
            space="ub",
        )
        rpack = ops.byte_view(pack_shared, (DR // 16, (VN + 1) * 16), tl.bfloat16)
        lows = tla.alloc((64, ), tl.int32, space="ub")
        strides = tla.alloc((64, ), tl.int32, space="ub")
        counts = tla.alloc((64, ), tl.int32, space="ub")
        any_duplicate = tla.alloc((8, ), tl.int32, space="ub")
        duplicate_flags = tla.alloc((64, ), tl.int32, space="ub")
        ids0 = tla.alloc((BN, ), tl.int32, space="ub")
        ids1 = tla.alloc((BN, ), tl.int32, space="ub")
        score = tla.alloc((VM, BN), tl.float32, space="ub")
        pub = ops.byte_view(pack_shared, (BN // 16, (VM + 1) * 16), tl.bfloat16)
        pv = ops.byte_view(shared, (VM, 256), tl.float32)
        acc = tla.alloc((VM, D), tl.float32, space="ub")
        acc_alt = tla.alloc((VM, D), tl.float32, space="ub")
        out = ops.byte_view(shared, (VM, 256), tl.bfloat16)
        m = tla.alloc((64, ), tl.float32, space="ub")
        l = tla.alloc((64, ), tl.float32, space="ub")
        m_alt = tla.alloc((64, ), tl.float32, space="ub")
        l_alt = tla.alloc((64, ), tl.float32, space="ub")
        alpha = tla.alloc((64, ), tl.float32, space="ub")
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
        # 按 tile 数切分，而非要求 TOPK 可整除 SPLIT；split14 覆盖全部32个tile。
        first_start = ((split * (TOPK // BN)) // SPLIT) * BN
        split_end = (((split + 1) * (TOPK // BN)) // SPLIT) * BN
        with tla.scope.vector():
            sub = tla.sub_block_id()
            tla.load_gm_to_ub(tla.view(IDS + token * TOPK + first_start, (BN, )), ids0)
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
                tla.simd.vsts(any_duplicate[0], tla.simd.vreduce_max(flags), dist="ONEPT_B32")
                one = tla.simd.vfull(1, tl.int32, 64)
                two = tla.simd.vfull(2, tl.int32, 64)
                tla.simd.vsts(lows[0], low)
                tla.simd.vsts(strides[0], tla.simd.vmax(one, tla.simd.vsub(high, low)))
                tla.simd.vsts(counts[0], tla.simd.vsel(equal, one, two))
            if ops.scalar_load(any_duplicate, 0) == 0:
                for pair in tl.static_range(0, VN // 2):
                    row = pair * 2
                    s0 = ops.scalar_load(lows, sub * (VN // 2) + pair)
                    stride = ops.scalar_load(strides, sub * (VN // 2) + pair)
                    ops.load_pair(
                        tla.view(KV, (B * L * D, )),
                        (batch * L + s0) * D,
                        stride * D,
                        kpairs[pair],
                    )
                    ops.load_pair(
                        tla.view(KR, (B * L * DR, )),
                        (batch * L + s0) * DR,
                        stride * DR,
                        tla.view(rpairs[pair], (2, DR)),
                    )
            else:
                for pair in tl.static_range(0, VN // 2):
                    row = pair * 2
                    s0 = ops.scalar_load(lows, sub * (VN // 2) + pair)
                    stride = ops.scalar_load(strides, sub * (VN // 2) + pair)
                    count = ops.scalar_load(counts, sub * (VN // 2) + pair)
                    ops.load_rows(
                        tla.view(KV, (B * L * D, )),
                        (batch * L + s0) * D,
                        stride * D,
                        count,
                        kpairs[pair],
                    )
                    ops.load_rows(
                        tla.view(KR, (B * L * DR, )),
                        (batch * L + s0) * DR,
                        stride * DR,
                        count,
                        tla.view(rpairs[pair], (2, DR)),
                    )
            if ops.scalar_load(any_duplicate, 0) == 0:
                with tla.scope.simd():
                    for pair in tl.static_range(16):
                        for localrow in tl.static_range(2):
                            row = pair * 2 + localrow
                            for chunk in tl.static_range(D // 128):
                                values = tla.simd.vlds(kpairs[pair][localrow, chunk * 128])
                                tla.simd.vsstb(values, kpack[chunk * 8, row * 16], (VN + 1) * 16)
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
                            tla.simd.vlds(duplicate_flags[sub * (VN // 2) + pair], dist="BRC_B32"),
                            "i16",
                        )
                        duplicate = tla.simd.vcmp(flag, tla.simd.vfull(0, tl.int16, 128), op=tla.simd.CMP.NE)
                        for localrow in tl.static_range(2):
                            row = pair * 2 + localrow
                            for chunk in tl.static_range(D // 128):
                                values = tla.simd.vlds(kpairs[pair][localrow, chunk * 128])
                                if localrow == 1:
                                    values = tla.simd.vsel(
                                        duplicate,
                                        tla.simd.vlds(kpairs[pair][0, chunk * 128]),
                                        values,
                                    )
                                tla.simd.vsstb(values, kpack[chunk * 8, row * 16], (VN + 1) * 16)
                            rv = tla.simd.vlds(rpairs[pair][localrow, 0])
                            if localrow == 1:
                                rv = tla.simd.vsel(duplicate, tla.simd.vlds(rpairs[pair][0, 0]), rv)
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
        # 先发下一 tile 的 gather，等当前 score 后做 softmax，再在 PV 期间打包 NZ。
        # old/new 双份状态避免原地更新的 Vector hazard。
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
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, phase * 5)
                        sc = tla.alloc((BN // 16, BM // 16, 16, 16), tl.float32, space="l0c")
                        ops.matmul(q_l1, key, sc, transpose_b=True, clear_accum=True)
                        ops.matmul(qr_l1, rope, sc, transpose_b=True, clear_accum=False)
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
                                tla.view(IDS + token * TOPK + (start + BN), (BN, )),
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
                                flags = tla.simd.vsel(equal, tla.simd.vfull(65537, tl.int32, 64), zero)
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
                            if ops.scalar_load(any_duplicate, 0) == 0:
                                for pair in tl.static_range(0, VN // 2):
                                    row = pair * 2
                                    s0 = ops.scalar_load(lows, sub * (VN // 2) + pair)
                                    stride = ops.scalar_load(strides, sub * (VN // 2) + pair)
                                    ops.load_pair(
                                        tla.view(KV, (B * L * D, )),
                                        (batch * L + s0) * D,
                                        stride * D,
                                        kpairs[pair],
                                    )
                                    ops.load_pair(
                                        tla.view(KR, (B * L * DR, )),
                                        (batch * L + s0) * DR,
                                        stride * DR,
                                        tla.view(rpairs[pair], (2, DR)),
                                    )
                            else:
                                for pair in tl.static_range(0, VN // 2):
                                    row = pair * 2
                                    s0 = ops.scalar_load(lows, sub * (VN // 2) + pair)
                                    stride = ops.scalar_load(strides, sub * (VN // 2) + pair)
                                    count = ops.scalar_load(counts, sub * (VN // 2) + pair)
                                    ops.load_rows(
                                        tla.view(KV, (B * L * D, )),
                                        (batch * L + s0) * D,
                                        stride * D,
                                        count,
                                        kpairs[pair],
                                    )
                                    ops.load_rows(
                                        tla.view(KR, (B * L * DR, )),
                                        (batch * L + s0) * DR,
                                        stride * DR,
                                        count,
                                        tla.view(rpairs[pair], (2, DR)),
                                    )
                    with tla.scope.vector():
                        sub = tla.sub_block_id()
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
                        with tla.scope.simd():
                            neginf = tla.simd.vfull(float("-inf"), tl.float32, 64)
                            izero = tla.simd.vfull(0, tl.int32, 64)
                            ilimit = tla.simd.vfull(L, tl.int32, 64)
                            i0 = tla.simd.vlds(ids_cur[0])
                            i1 = i0
                            if BN == 128:
                                i1 = tla.simd.vlds(ids_cur[64])
                            for row in tl.range(0, VM):
                                x0 = ops.vmuls(tla.simd.vlds(score[row, 0]), SCALE)
                                x1 = neginf
                                if BN == 128:
                                    x1 = ops.vmuls(tla.simd.vlds(score[row, 64]), SCALE)
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
                                a = ops.vexpdif(old_m, new_m)
                                tla.simd.vsts(alpha[row], a, dist="ONEPT_B32")
                                p0 = ops.vexpdif(x0, new_m)
                                p1 = ops.vexpdif(x1, new_m)
                                new_l = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(p0, p1)))
                                old_l = tla.simd.vlds(l_old[row], dist="BRC_B32")
                                new_l = tla.simd.vadd(new_l, tla.simd.vmul(old_l, a))
                                tla.simd.vsts(m_new[row], new_m, dist="ONEPT_B32")
                                tla.simd.vsts(l_new[row], new_l, dist="ONEPT_B32")
                                packed, unused = tla.simd.vdintlv(
                                    tla.simd.vtruncf(p0, tl.bfloat16),
                                    tla.simd.vtruncf(p1, tl.bfloat16),
                                )
                                if BN == 128:
                                    tla.simd.vsstb(packed, pub[0, row * 16], (VM + 1) * 16)
                                else:
                                    tla.simd.vsstb(
                                        packed,
                                        pub[0, row * 16],
                                        (VM + 1) * 16,
                                        mask=tla.simd.pge(64, 128),
                                    )
                        tla.copy_ub_to_l1(
                            tla.view(pub, (BN // 16, VM * 16)),
                            tla.view(pflat, (BN // 16, VM * 16), offsets=(0, sub * VM * 16)),
                        )
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 2)
                    with tla.scope.cube():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                        ops.matmul(p_l1, key, pc, clear_accum=True)
                    if start + BN < split_end:
                        with tla.scope.vector():
                            sub = tla.sub_block_id()
                            if ops.scalar_load(any_duplicate, 0) == 0:
                                with tla.scope.simd():
                                    for pair in tl.static_range(16):
                                        for localrow in tl.static_range(2):
                                            row = pair * 2 + localrow
                                            for chunk in tl.static_range(D // 128):
                                                values = tla.simd.vlds(kpairs[pair][localrow, chunk * 128])
                                                tla.simd.vsstb(
                                                    values,
                                                    kpack[chunk * 8, row * 16],
                                                    (VN + 1) * 16,
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
                                                values = tla.simd.vlds(kpairs[pair][localrow, chunk * 128])
                                                if localrow == 1:
                                                    values = tla.simd.vsel(
                                                        duplicate,
                                                        tla.simd.vlds(kpairs[pair][0, chunk * 128]),
                                                        values,
                                                    )
                                                tla.simd.vsstb(
                                                    values,
                                                    kpack[chunk * 8, row * 16],
                                                    (VN + 1) * 16,
                                                )
                                            rv = tla.simd.vlds(rpairs[pair][localrow, 0])
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
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, (1 - phase) * 5)
                    for fragment in tl.range(2):
                        with tla.scope.cube():
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
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
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 3)
                        with tla.scope.vector():
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 3)
                            with tla.scope.simd():
                                for row in tl.range(VM):
                                    a = tla.simd.vlds(alpha[row], dist="BRC_B32")
                                    for chunk in tl.static_range(2):
                                        y0 = tla.simd.vlds(pv[row, chunk * 128])
                                        y1 = tla.simd.vlds(pv[row, chunk * 128 + 64])
                                        y0 = tla.simd.vadd(
                                            y0,
                                            tla.simd.vmul(
                                                tla.simd.vlds(acc_old[
                                                    row,
                                                    fragment * 256 + chunk * 128,
                                                ]),
                                                a,
                                            ),
                                        )
                                        y1 = tla.simd.vadd(
                                            y1,
                                            tla.simd.vmul(
                                                tla.simd.vlds(acc_old[
                                                    row,
                                                    fragment * 256 + chunk * 128 + 64,
                                                ]),
                                                a,
                                            ),
                                        )
                                        tla.simd.vsts(
                                            acc_new[row, fragment * 256 + chunk * 128],
                                            y0,
                                        )
                                        tla.simd.vsts(
                                            acc_new[row, fragment * 256 + chunk * 128 + 64],
                                            y1,
                                        )
                            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        # 分片长度可能是奇数：最终状态位于不同的 ping-pong 缓冲区。
        if ((split_end - first_start) // BN) % 2 == 0:
            with tla.scope.vector():
                sub = tla.sub_block_id()
                if SPLIT == 1:
                    for fragment in tl.range(2):
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                den = tla.simd.vlds(l[row], dist="BRC_B32")
                                for chunk in tl.static_range(2):
                                    y0 = tla.simd.vlds(acc[row, fragment * 256 + chunk * 128])
                                    y1 = tla.simd.vlds(acc[row, fragment * 256 + chunk * 128 + 64])
                                    b0 = tla.simd.vtruncf(tla.simd.vdiv(y0, den), tl.bfloat16)
                                    b1 = tla.simd.vtruncf(tla.simd.vdiv(y1, den), tl.bfloat16)
                                    packed, unused = tla.simd.vdintlv(b0, b1)
                                    tla.simd.vsts(out[row, chunk * 128], packed)
                        tla.store_ub_to_gm(
                            out,
                            tla.view(
                                O + (output_token * H + head + sub * VM) * D + fragment * 256,
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
                    tla.view(m, (VM, ), offsets=(0, )),
                    tla.view(MAX + output_token * H + head + sub * VM, (VM, )),
                )
                tla.store_ub_to_gm(
                    tla.view(l, (VM, ), offsets=(0, )),
                    tla.view(SUM + output_token * H + head + sub * VM, (VM, )),
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
                                    y0 = tla.simd.vlds(acc_alt[row, fragment * 256 + chunk * 128])
                                    y1 = tla.simd.vlds(acc_alt[row, fragment * 256 + chunk * 128 + 64])
                                    b0 = tla.simd.vtruncf(tla.simd.vdiv(y0, den), tl.bfloat16)
                                    b1 = tla.simd.vtruncf(tla.simd.vdiv(y1, den), tl.bfloat16)
                                    packed, unused = tla.simd.vdintlv(b0, b1)
                                    tla.simd.vsts(out[row, chunk * 128], packed)
                        tla.store_ub_to_gm(
                            out,
                            tla.view(
                                O + (output_token * H + head + sub * VM) * D + fragment * 256,
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
                    tla.view(m_alt, (VM, ), offsets=(0, )),
                    tla.view(MAX + output_token * H + head + sub * VM, (VM, )),
                )
                tla.store_ub_to_gm(
                    tla.view(l_alt, (VM, ), offsets=(0, )),
                    tla.view(SUM + output_token * H + head + sub * VM, (VM, )),
                )
