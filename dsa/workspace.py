"""GM 双槽流水：BN128/BN256 共用一份算法，保留各自的调度。

Vector 将选中的 KV/位置编码合并为 576 列的连续 GM 工作区；Cube 用
ND2NZ 读入 L1，以一次 QK 矩阵乘覆盖 512+64 维。BM128 复用 score/PV
的 UB 空间并使用 256 列 PV 分块；较小 BM 使用独立 score 和 512 列 PV。
BN 是编译期常量，两个 softmax 分支保持原实现各自的浮点求和次序。
"""
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction

from . import compat as ops


@TLAJITFunction
def workspace_attention(Q, KV, QR, KR, IDS, WORK, O, MAX, SUM, B: tl.constexpr, SQ: tl.constexpr,
                        L: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr, BM: tl.constexpr,
                        CORES: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr, PV_COLS: tl.constexpr,
                        SHARE_SCORE: tl.constexpr):
    tl.static_assert(BN == 128 or BN == 256)
    VM: tl.constexpr = BM // 2
    VP: tl.constexpr = PV_COLS
    HALF_N: tl.constexpr = BN // 2
    # 一个 job = 一个 query × 一个 head tile × 一个 KV 分片。
    for job in tl.range(tl.program_id(0), B * SQ * (H // BM) * SPLIT, CORES):
        token = job // (H // BM * SPLIT)
        split = job % SPLIT
        output_token = token * SPLIT + split
        head = job // SPLIT % (H // BM) * BM
        batch = token // SQ
        # L1/NZ 供 Cube 使用；UB 保存每个 Vector 子核负责的半个 head tile。
        query_l1 = tla.alloc((36, BM // 16, 16, 16), tl.bfloat16, space='l1')
        keys_l1 = tla.alloc((36, BN // 16, 16, 16), tl.bfloat16, space='l1')
        prob_l1 = tla.alloc((BN // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
        prob_flat_l1 = tla.reshape(prob_l1, (BN // 16, BM * 16))
        ids0 = tla.alloc((BN, ), tl.int32, space='ub')
        ids1 = tla.alloc((BN, ), tl.int32, space='ub')
        if not SHARE_SCORE:
            scores_ub = tla.alloc((VM, BN), tl.float32, space='ub')
        prob_padded_ub = tla.alloc((BN // 16, (VM + 1) * 16), tl.bfloat16, space='ub')
        shared_bytes = tla.alloc((VM * VP * 4, ), tl.int8, space='ub')
        pv_ub = ops.byte_view(shared_bytes, (VM, VP), tl.float32)
        if SHARE_SCORE:
            scores_ub = ops.byte_view(shared_bytes, (VM, BN), tl.float32)
        numerator_ub = tla.alloc((VM, 512), tl.float32, space='ub')
        output_bf16 = ops.byte_view(shared_bytes, (VM, VP), tl.bfloat16)
        need_safe_gather = tla.alloc((8, ), tl.int32, space='ub')
        gather_key_even = tla.alloc((16, 512), tl.bfloat16, space='ub')
        gather_key_odd = tla.alloc((16, 512), tl.bfloat16, space='ub')
        gather_rope_even = tla.alloc((16, 64), tl.bfloat16, space='ub')
        gather_rope_odd = tla.alloc((16, 64), tl.bfloat16, space='ub')
        running_max = tla.alloc((64, ), tl.float32, space='ub')
        running_sum = tla.alloc((64, ), tl.float32, space='ub')
        rescale = tla.alloc((64, ), tl.float32, space='ub')
        score_l0c = tla.alloc((BN // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        pv_l0c = tla.alloc((VP // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        # 此路径等长切分；每个分片含偶数个 BN tile。
        begin = split * (TOPK // SPLIT)
        end = (split + 1) * (TOPK // SPLIT)
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + (token * H + head) * 512, (BM, 512)),
                              tla.view(query_l1, (32, BM // 16, 16, 16)))
            tla.load_gm_to_l1(tla.view(QR + (token * H + head) * 64, (BM, 64)),
                              tla.view(query_l1, (4, BM // 16, 16, 16), offsets=(32, 0, 0, 0)))
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                negative = tla.simd.vfull(-3.4028234663852886e+38, tl.float32, 64)
                tla.simd.vsts(running_max[0], negative)
                tla.simd.vsts(running_sum[0], zero)
                for row in tl.range(VM):
                    for col in tl.static_range(8):
                        tla.simd.vsts(numerator_ub[row, col * 64], zero)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        # 预取第一槽。后面的重复 gather 块刻意放在 QK 后，以覆盖 Cube 工作。
        with tla.scope.vector():
            gather_ids = ids0
            gather_start = begin
            gather_slot = 0
            sub = tla.sub_block_id()
            tla.load_gm_to_ub(tla.view(IDS + token * TOPK + gather_start, (BN, )), gather_ids)
            with tla.scope.simd():
                combined_flags = tla.simd.vfull(0, tl.int32, 64)
                for index_half in tl.static_range(BN // 64):
                    ii = tla.simd.vlds(gather_ids[index_half * 64])
                    even, odd = tla.simd.vdintlv(ii, ii)
                    low = tla.simd.vmin(even, odd)
                    high = tla.simd.vmax(even, odd)
                    ordered, unused = tla.simd.vintlv(low, high)
                    tla.simd.vsts(gather_ids[index_half * 64], ordered)
                    zero = tla.simd.vfull(0, tl.int32, 64)
                    limit = tla.simd.vfull(L - 1, tl.int32, 64)
                    invalid_low = tla.simd.vsel(tla.simd.vcmp(low, zero, op=tla.simd.CMP.LT),
                                                tla.simd.vfull(1, tl.int32, 64), zero)
                    invalid_high = tla.simd.vsel(tla.simd.vcmp(high, limit, op=tla.simd.CMP.GT),
                                                 tla.simd.vfull(1, tl.int32, 64), zero)
                    flags = tla.simd.vsel(tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ),
                                          tla.simd.vfull(1, tl.int32, 64), zero)
                    combined_flags = tla.simd.vmax(
                        combined_flags, tla.simd.vmax(flags, tla.simd.vmax(invalid_low, invalid_high)))
                tla.simd.vsts(need_safe_gather[0], tla.simd.vreduce_max(combined_flags), dist='ONEPT_B32')
            for pair_chunk in tl.range(BN // 64):
                for gather_phase in tl.static_range(2):
                    chunk = (pair_chunk * 2 + gather_phase) * 16
                    if gather_phase == 0:
                        keys = gather_key_even
                        ropes = gather_rope_even
                    else:
                        keys = gather_key_odd
                        ropes = gather_rope_odd
                    if ops.scalar_load(need_safe_gather, 0) == 0:
                        for pair in tl.static_range(8):
                            i0 = ops.scalar_load(gather_ids, sub * HALF_N + chunk + pair * 2)
                            i1 = ops.scalar_load(gather_ids, sub * HALF_N + chunk + pair * 2 + 1)
                            s0 = i0
                            s1 = i1
                            ops.load_pair(tla.view(KV, (B * L * 512, )), (batch * L + s0) * 512,
                                          (s1 - s0) * 512, tla.view(keys, (2, 512), offsets=(pair * 2, 0)))
                            ops.load_pair(tla.view(KR, (B * L * 64, )), (batch * L + s0) * 64, (s1 - s0) * 64,
                                          tla.view(ropes, (2, 64), offsets=(pair * 2, 0)))
                    else:
                        for row in tl.static_range(16):
                            index = ops.scalar_load(gather_ids, sub * HALF_N + chunk + row)
                            safe = tl.maximum(0, tl.minimum(index, L - 1))
                            tla.load_gm_to_ub(tla.view(KV + (batch * L + safe) * 512, (1, 512)),
                                              tla.view(keys, (1, 512), offsets=(row, 0)))
                            tla.load_gm_to_ub(tla.view(KR + (batch * L + safe) * 64, (1, 64)),
                                              tla.view(ropes, (1, 64), offsets=(row, 0)))
                    tla.store_ub_to_gm(
                        keys,
                        tla.view(WORK +
                                 ((tl.program_id(0) * 2 + gather_slot) * BN + sub * HALF_N + chunk) * 576,
                                 (16, 512),
                                 strides=(576, 1)))
                    tla.store_ub_to_gm(
                        ropes,
                        tla.view(WORK +
                                 ((tl.program_id(0) * 2 + gather_slot) * BN + sub * HALF_N + chunk) * 576 +
                                 512, (16, 64),
                                 strides=(576, 1)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
        # 稳态：QK → 下一个 tile 的 gather → softmax → PV → FP32 累加。
        # flag 0/5: GM 槽就绪；1: score 就绪；2: P 就绪；3/4: PV 就绪/已消费。
        for pair_start in tl.range(begin, end, 2 * BN):
            for phase in tl.static_range(2):
                start = pair_start + phase * BN
                if phase == 0:
                    current = keys_l1
                    indices = ids0
                    next_indices = ids1
                else:
                    current = keys_l1
                    indices = ids1
                    next_indices = ids0
                with tla.scope.cube():
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, phase * 5)
                    tla.load_gm_to_l1(tla.view(WORK + (tl.program_id(0) * 2 + phase) * BN * 576, (BN, 576)),
                                      current)
                    ops.matmul(query_l1, current, score_l0c, transpose_b=True, clear_accum=True)
                    if SHARE_SCORE:
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                    tla.copy_l0c_to_ub(
                        score_l0c, scores_ub,
                        tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                                sub_block_idx=None))
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 1)
                with tla.scope.vector():
                    if start + BN < end:
                        gather_ids = next_indices
                        gather_start = start + BN
                        gather_slot = 1 - phase
                        sub = tla.sub_block_id()
                        tla.load_gm_to_ub(tla.view(IDS + token * TOPK + gather_start, (BN, )), gather_ids)
                        with tla.scope.simd():
                            combined_flags = tla.simd.vfull(0, tl.int32, 64)
                            for index_half in tl.static_range(BN // 64):
                                ii = tla.simd.vlds(gather_ids[index_half * 64])
                                even, odd = tla.simd.vdintlv(ii, ii)
                                low = tla.simd.vmin(even, odd)
                                high = tla.simd.vmax(even, odd)
                                ordered, unused = tla.simd.vintlv(low, high)
                                tla.simd.vsts(gather_ids[index_half * 64], ordered)
                                zero = tla.simd.vfull(0, tl.int32, 64)
                                limit = tla.simd.vfull(L - 1, tl.int32, 64)
                                invalid_low = tla.simd.vsel(tla.simd.vcmp(low, zero, op=tla.simd.CMP.LT),
                                                            tla.simd.vfull(1, tl.int32, 64), zero)
                                invalid_high = tla.simd.vsel(tla.simd.vcmp(high, limit, op=tla.simd.CMP.GT),
                                                             tla.simd.vfull(1, tl.int32, 64), zero)
                                flags = tla.simd.vsel(tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ),
                                                      tla.simd.vfull(1, tl.int32, 64), zero)
                                combined_flags = tla.simd.vmax(
                                    combined_flags,
                                    tla.simd.vmax(flags, tla.simd.vmax(invalid_low, invalid_high)))
                            tla.simd.vsts(need_safe_gather[0],
                                          tla.simd.vreduce_max(combined_flags),
                                          dist='ONEPT_B32')
                        for pair_chunk in tl.range(BN // 64):
                            for gather_phase in tl.static_range(2):
                                chunk = (pair_chunk * 2 + gather_phase) * 16
                                if gather_phase == 0:
                                    keys = gather_key_even
                                    ropes = gather_rope_even
                                else:
                                    keys = gather_key_odd
                                    ropes = gather_rope_odd
                                if ops.scalar_load(need_safe_gather, 0) == 0:
                                    for pair in tl.static_range(8):
                                        i0 = ops.scalar_load(gather_ids, sub * HALF_N + chunk + pair * 2)
                                        i1 = ops.scalar_load(gather_ids, sub * HALF_N + chunk + pair * 2 + 1)
                                        s0 = i0
                                        s1 = i1
                                        ops.load_pair(tla.view(KV, (B * L * 512, )), (batch * L + s0) * 512,
                                                      (s1 - s0) * 512,
                                                      tla.view(keys, (2, 512), offsets=(pair * 2, 0)))
                                        ops.load_pair(tla.view(KR, (B * L * 64, )), (batch * L + s0) * 64,
                                                      (s1 - s0) * 64,
                                                      tla.view(ropes, (2, 64), offsets=(pair * 2, 0)))
                                else:
                                    for row in tl.static_range(16):
                                        index = ops.scalar_load(gather_ids, sub * HALF_N + chunk + row)
                                        safe = tl.maximum(0, tl.minimum(index, L - 1))
                                        tla.load_gm_to_ub(tla.view(KV + (batch * L + safe) * 512, (1, 512)),
                                                          tla.view(keys, (1, 512), offsets=(row, 0)))
                                        tla.load_gm_to_ub(tla.view(KR + (batch * L + safe) * 64, (1, 64)),
                                                          tla.view(ropes, (1, 64), offsets=(row, 0)))
                                tla.store_ub_to_gm(
                                    keys,
                                    tla.view(
                                        WORK +
                                        ((tl.program_id(0) * 2 + gather_slot) * BN + sub * HALF_N + chunk) *
                                        576, (16, 512),
                                        strides=(576, 1)))
                                tla.store_ub_to_gm(
                                    ropes,
                                    tla.view(
                                        WORK +
                                        ((tl.program_id(0) * 2 + gather_slot) * BN + sub * HALF_N + chunk) *
                                        576 + 512, (16, 64),
                                        strides=(576, 1)))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, (1 - phase) * 5)
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
                    sub = tla.sub_block_id()
                    if BN == 128:
                        with tla.scope.simd():
                            negative = tla.simd.vfull(float('-inf'), tl.float32, 64)
                            zero = tla.simd.vfull(0, tl.int32, 64)
                            limit = tla.simd.vfull(L, tl.int32, 64)
                            i0 = tla.simd.vlds(indices[0])
                            i1 = tla.simd.vlds(indices[64])
                            for row in tl.range(VM):
                                x0 = ops.vmuls(tla.simd.vlds(scores_ub[row, 0]), 576**(-0.5))
                                x1 = ops.vmuls(tla.simd.vlds(scores_ub[row, 64]), 576**(-0.5))
                                x0 = tla.simd.vsel(tla.simd.vcmp(i0, zero, op=tla.simd.CMP.GE), x0, negative)
                                x1 = tla.simd.vsel(tla.simd.vcmp(i1, zero, op=tla.simd.CMP.GE), x1, negative)
                                x0 = tla.simd.vsel(tla.simd.vcmp(i0, limit, op=tla.simd.CMP.LT), x0, negative)
                                x1 = tla.simd.vsel(tla.simd.vcmp(i1, limit, op=tla.simd.CMP.LT), x1, negative)
                                old_m = tla.simd.vlds(running_max[row], dist='BRC_B32')
                                maximum = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(x0, x1)), old_m)
                                scale_old = ops.vexpdif(old_m, maximum)
                                p0 = ops.vexpdif(x0, maximum)
                                p1 = ops.vexpdif(x1, maximum)
                                den = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(p0, p1)))
                                den = tla.simd.vadd(
                                    den,
                                    tla.simd.vmul(tla.simd.vlds(running_sum[row], dist='BRC_B32'), scale_old))
                                tla.simd.vsts(running_max[row], maximum, dist='ONEPT_B32')
                                tla.simd.vsts(running_sum[row], den, dist='ONEPT_B32')
                                tla.simd.vsts(rescale[row], scale_old, dist='ONEPT_B32')
                                packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p0, tl.bfloat16),
                                                                  tla.simd.vtruncf(p1, tl.bfloat16))
                                tla.simd.vsstb(packed, prob_padded_ub[0, row * 16], (VM + 1) * 16)
                    else:
                        with tla.scope.simd():
                            negative = tla.simd.vfull(float('-inf'), tl.float32, 64)
                            zero = tla.simd.vfull(0, tl.int32, 64)
                            limit = tla.simd.vfull(L, tl.int32, 64)
                            i0 = tla.simd.vlds(indices[0])
                            i1 = tla.simd.vlds(indices[64])
                            i2 = tla.simd.vlds(indices[128])
                            i3 = tla.simd.vlds(indices[192])
                            for row in tl.range(VM):
                                x0 = ops.vmuls(tla.simd.vlds(scores_ub[row, 0]), 576**(-0.5))
                                x0 = tla.simd.vsel(tla.simd.vcmp(i0, zero, op=tla.simd.CMP.GE), x0, negative)
                                x0 = tla.simd.vsel(tla.simd.vcmp(i0, limit, op=tla.simd.CMP.LT), x0, negative)
                                x1 = ops.vmuls(tla.simd.vlds(scores_ub[row, 64]), 576**(-0.5))
                                x1 = tla.simd.vsel(tla.simd.vcmp(i1, zero, op=tla.simd.CMP.GE), x1, negative)
                                x1 = tla.simd.vsel(tla.simd.vcmp(i1, limit, op=tla.simd.CMP.LT), x1, negative)
                                x2 = ops.vmuls(tla.simd.vlds(scores_ub[row, 128]), 576**(-0.5))
                                x2 = tla.simd.vsel(tla.simd.vcmp(i2, zero, op=tla.simd.CMP.GE), x2, negative)
                                x2 = tla.simd.vsel(tla.simd.vcmp(i2, limit, op=tla.simd.CMP.LT), x2, negative)
                                x3 = ops.vmuls(tla.simd.vlds(scores_ub[row, 192]), 576**(-0.5))
                                x3 = tla.simd.vsel(tla.simd.vcmp(i3, zero, op=tla.simd.CMP.GE), x3, negative)
                                x3 = tla.simd.vsel(tla.simd.vcmp(i3, limit, op=tla.simd.CMP.LT), x3, negative)
                                old_m = tla.simd.vlds(running_max[row], dist='BRC_B32')
                                maximum = tla.simd.vmax(
                                    tla.simd.vreduce_max(
                                        tla.simd.vmax(tla.simd.vmax(x0, x1), tla.simd.vmax(x2, x3))), old_m)
                                scale_old = ops.vexpdif(old_m, maximum)
                                p0 = ops.vexpdif(x0, maximum)
                                p1 = ops.vexpdif(x1, maximum)
                                p2 = ops.vexpdif(x2, maximum)
                                p3 = ops.vexpdif(x3, maximum)
                                den = tla.simd.vbroadcast(
                                    tla.simd.vreduce_sum(
                                        tla.simd.vadd(tla.simd.vadd(p0, p1), tla.simd.vadd(p2, p3))))
                                den = tla.simd.vadd(
                                    den,
                                    tla.simd.vmul(tla.simd.vlds(running_sum[row], dist='BRC_B32'), scale_old))
                                tla.simd.vsts(running_max[row], maximum, dist='ONEPT_B32')
                                tla.simd.vsts(running_sum[row], den, dist='ONEPT_B32')
                                tla.simd.vsts(rescale[row], scale_old, dist='ONEPT_B32')
                                packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p0, tl.bfloat16),
                                                                  tla.simd.vtruncf(p1, tl.bfloat16))
                                tla.simd.vsstb(packed, prob_padded_ub[0, row * 16], (VM + 1) * 16)
                                packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p2, tl.bfloat16),
                                                                  tla.simd.vtruncf(p3, tl.bfloat16))
                                tla.simd.vsstb(packed, prob_padded_ub[8, row * 16], (VM + 1) * 16)
                    tla.copy_ub_to_l1(tla.view(prob_padded_ub, (BN // 16, VM * 16)),
                                      tla.view(prob_flat_l1, (BN // 16, VM * 16), offsets=(0, sub * VM * 16)))
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 2)
                with tla.scope.cube():
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                for fragment in tl.range(512 // VP):
                    with tla.scope.cube():
                        ops.matmul(prob_l1,
                                   tla.view(current, (VP // 16, BN // 16, 16, 16),
                                            offsets=(fragment * (VP // 16), 0, 0, 0)),
                                   pv_l0c,
                                   clear_accum=True)
                        if SHARE_SCORE:
                            if fragment > 0:
                                tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                        else:
                            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                        tla.copy_l0c_to_ub(
                            pv_l0c, pv_ub,
                            tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                                    dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT,
                                                    sub_block_idx=None))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 3)
                    with tla.scope.vector():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 3)
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                scale_old = tla.simd.vlds(rescale[row], dist='BRC_B32')
                                for col in tl.static_range(VP // 64):
                                    value = tla.simd.vadd(
                                        tla.simd.vlds(pv_ub[row, col * 64]),
                                        tla.simd.vmul(
                                            tla.simd.vlds(numerator_ub[row, fragment * VP + col * 64]),
                                            scale_old))
                                    tla.simd.vsts(numerator_ub[row, fragment * VP + col * 64], value)
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        # 回收最后一个 PV credit，再输出；split>1 保留未归一化 FP32 分子。
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        with tla.scope.vector():
            sub = tla.sub_block_id()
            if SPLIT == 1:
                for fragment in tl.range(512 // VP):
                    with tla.scope.simd():
                        for row in tl.range(VM):
                            den = tla.simd.vlds(running_sum[row], dist='BRC_B32')
                            for col in tl.static_range(VP // 128):
                                b0 = tla.simd.vtruncf(
                                    tla.simd.vdiv(tla.simd.vlds(numerator_ub[row, fragment * VP + col * 128]),
                                                  den), tl.bfloat16)
                                b1 = tla.simd.vtruncf(
                                    tla.simd.vdiv(
                                        tla.simd.vlds(numerator_ub[row, fragment * VP + col * 128 + 64]),
                                        den), tl.bfloat16)
                                packed, unused = tla.simd.vdintlv(b0, b1)
                                tla.simd.vsts(output_bf16[row, col * 128], packed)
                    tla.store_ub_to_gm(
                        output_bf16,
                        tla.view(O + (output_token * H + head + sub * VM) * 512 + fragment * VP, (VM, VP),
                                 strides=(512, 1)))
            else:
                tla.store_ub_to_gm(numerator_ub,
                                   tla.view(O + (output_token * H + head + sub * VM) * 512, (VM, 512)))
            tla.store_ub_to_gm(tla.view(running_max, (VM, )),
                               tla.view(MAX + output_token * H + head + sub * VM, (VM, )))
            tla.store_ub_to_gm(tla.view(running_sum, (VM, )),
                               tla.view(SUM + output_token * H + head + sub * VM, (VM, )))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 7)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 8)
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 7)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 8)
