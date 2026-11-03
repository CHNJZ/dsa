"""Fused indexed sparse MLA with separate score/PV UB and wide PV transfers.

Selected for BM16/BM64: their smaller FP32 output state permits a full D512
PV transfer. Otherwise the dataflow and arithmetic match dsa_kernel.py.
All gather and optional merge work is included. See README.md and configs.json.
"""
import torch, torch_npu
import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction
import tla_native_ext as ext
import dsa_ext
ext.install()

@TLAJITFunction
def sparse_mla_kernel(Q, KV, QR, KR, IDS, WORK, O, MAX, SUM, B: tl.constexpr, SQ: tl.constexpr, L: tl.constexpr, H: tl.constexpr, TOPK: tl.constexpr, BM: tl.constexpr, CORES: tl.constexpr, BN: tl.constexpr, SPLIT: tl.constexpr):
    tl.static_assert(BN == 256)
    VM: tl.constexpr = BM // 2
    VP: tl.constexpr = 512 if BM <= 64 else 128
    for job in tl.range(tl.program_id(0), B * SQ * (H // BM) * SPLIT, CORES):
        token = job // (H // BM * SPLIT)
        split = job % SPLIT
        output_token = token * SPLIT + split
        head = job // SPLIT % (H // BM) * BM
        batch = token // SQ
        q = tla.alloc((36, BM // 16, 16, 16), tl.bfloat16, space='l1')
        k0 = tla.alloc((36, BN // 16, 16, 16), tl.bfloat16, space='l1')
        p = tla.alloc((BN // 16, BM // 16, 16, 16), tl.bfloat16, space='l1')
        pflat = tla.reshape(p, (BN // 16, BM * 16))
        ids0 = tla.alloc((BN,), tl.int32, space='ub')
        ids1 = tla.alloc((BN,), tl.int32, space='ub')
        score = tla.alloc((VM, BN), tl.float32, space='ub')
        pub = tla.alloc((BN // 16, (VM + 1) * 16), tl.bfloat16, space='ub')
        pv_storage = tla.alloc((VM * VP * 4,), tl.int8, space='ub')
        pv = dsa_ext.byte_view(pv_storage, (VM, VP), tl.float32)
        acc = tla.alloc((VM, 512), tl.float32, space='ub')
        out = dsa_ext.byte_view(pv_storage, (VM, VP), tl.bfloat16)
        any_duplicate = tla.alloc((8,), tl.int32, space='ub')
        gk0 = tla.alloc((16, 512), tl.bfloat16, space='ub')
        gk1 = tla.alloc((16, 512), tl.bfloat16, space='ub')
        gr0 = tla.alloc((16, 64), tl.bfloat16, space='ub')
        gr1 = tla.alloc((16, 64), tl.bfloat16, space='ub')
        m = tla.alloc((64,), tl.float32, space='ub')
        l = tla.alloc((64,), tl.float32, space='ub')
        alpha = tla.alloc((64,), tl.float32, space='ub')
        sc = tla.alloc((BN // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        pc = tla.alloc((VP // 16, BM // 16, 16, 16), tl.float32, space='l0c')
        begin = split * (TOPK // SPLIT)
        end = (split + 1) * (TOPK // SPLIT)
        with tla.scope.cube():
            tla.load_gm_to_l1(tla.view(Q + (token * H + head) * 512, (BM, 512)), tla.view(q, (32, BM // 16, 16, 16)))
            tla.load_gm_to_l1(tla.view(QR + (token * H + head) * 64, (BM, 64)), tla.view(q, (4, BM // 16, 16, 16), offsets=(32, 0, 0, 0)))
        with tla.scope.vector():
            with tla.scope.simd():
                zero = tla.simd.vfull(0.0, tl.float32, 64)
                negative = tla.simd.vfull(-3.4028234663852886e+38, tl.float32, 64)
                tla.simd.vsts(m[0], negative)
                tla.simd.vsts(l[0], zero)
                for row in tl.range(VM):
                    for col in tl.static_range(8):
                        tla.simd.vsts(acc[row, col * 64], zero)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        with tla.scope.vector():
            gather_ids = ids0
            gather_start = begin
            gather_slot = 0
            sub = tla.sub_block_id()
            tla.load_gm_to_ub(tla.view(IDS + token * TOPK + gather_start, (BN,)), gather_ids)
            with tla.scope.simd():
                combined_flags = tla.simd.vfull(0, tl.int32, 64)
                for index_half in tl.static_range(4):
                    ii = tla.simd.vlds(gather_ids[index_half * 64])
                    even, odd = tla.simd.vdintlv(ii, ii)
                    low = tla.simd.vmin(even, odd)
                    high = tla.simd.vmax(even, odd)
                    ordered, unused = tla.simd.vintlv(low, high)
                    tla.simd.vsts(gather_ids[index_half * 64], ordered)
                    zero = tla.simd.vfull(0, tl.int32, 64)
                    limit = tla.simd.vfull(L - 1, tl.int32, 64)
                    invalid_low = tla.simd.vsel(tla.simd.vcmp(low, zero, op=tla.simd.CMP.LT), tla.simd.vfull(1, tl.int32, 64), zero)
                    invalid_high = tla.simd.vsel(tla.simd.vcmp(high, limit, op=tla.simd.CMP.GT), tla.simd.vfull(1, tl.int32, 64), zero)
                    flags = tla.simd.vsel(tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ), tla.simd.vfull(1, tl.int32, 64), zero)
                    combined_flags = tla.simd.vmax(combined_flags, tla.simd.vmax(flags, tla.simd.vmax(invalid_low, invalid_high)))
                tla.simd.vsts(any_duplicate[0], tla.simd.vreduce_max(combined_flags), dist='ONEPT_B32')
            for pair_chunk in tl.range(4):
                for gather_phase in tl.static_range(2):
                    chunk = (pair_chunk * 2 + gather_phase) * 16
                    if gather_phase == 0:
                        keys = gk0
                        ropes = gr0
                    else:
                        keys = gk1
                        ropes = gr1
                    if dsa_ext.scalar_load(any_duplicate, 0) == 0:
                        for pair in tl.static_range(8):
                            i0 = dsa_ext.scalar_load(gather_ids, sub * 128 + chunk + pair * 2)
                            i1 = dsa_ext.scalar_load(gather_ids, sub * 128 + chunk + pair * 2 + 1)
                            s0 = i0
                            s1 = i1
                            dsa_ext.load_pair(tla.view(KV, (B * L * 512,)), (batch * L + s0) * 512, (s1 - s0) * 512, tla.view(keys, (2, 512), offsets=(pair * 2, 0)))
                            dsa_ext.load_pair(tla.view(KR, (B * L * 64,)), (batch * L + s0) * 64, (s1 - s0) * 64, tla.view(ropes, (2, 64), offsets=(pair * 2, 0)))
                    else:
                        for row in tl.static_range(16):
                            index = dsa_ext.scalar_load(gather_ids, sub * 128 + chunk + row)
                            safe = tl.maximum(0, tl.minimum(index, L - 1))
                            tla.load_gm_to_ub(tla.view(KV + (batch * L + safe) * 512, (1, 512)), tla.view(keys, (1, 512), offsets=(row, 0)))
                            tla.load_gm_to_ub(tla.view(KR + (batch * L + safe) * 64, (1, 64)), tla.view(ropes, (1, 64), offsets=(row, 0)))
                    tla.store_ub_to_gm(keys, tla.view(WORK + ((tl.program_id(0) * 2 + gather_slot) * BN + sub * 128 + chunk) * 576, (16, 512), strides=(576, 1)))
                    tla.store_ub_to_gm(ropes, tla.view(WORK + ((tl.program_id(0) * 2 + gather_slot) * BN + sub * 128 + chunk) * 576 + 512, (16, 64), strides=(576, 1)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)
        for pair_start in tl.range(begin, end, 2 * BN):
            for phase in tl.static_range(2):
                start = pair_start + phase * BN
                if phase == 0:
                    current = k0
                    indices = ids0
                    next_indices = ids1
                else:
                    current = k0
                    indices = ids1
                    next_indices = ids0
                with tla.scope.cube():
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, phase * 5)
                    tla.load_gm_to_l1(tla.view(WORK + (tl.program_id(0) * 2 + phase) * BN * 576, (BN, 576)), current)
                    ext.matmul(q, current, sc, transpose_b=True, clear_accum=True)
                    tla.copy_l0c_to_ub(sc, score, tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 1)
                with tla.scope.vector():
                    if start + BN < end:
                        gather_ids = next_indices
                        gather_start = start + BN
                        gather_slot = 1 - phase
                        sub = tla.sub_block_id()
                        tla.load_gm_to_ub(tla.view(IDS + token * TOPK + gather_start, (BN,)), gather_ids)
                        with tla.scope.simd():
                            combined_flags = tla.simd.vfull(0, tl.int32, 64)
                            for index_half in tl.static_range(4):
                                ii = tla.simd.vlds(gather_ids[index_half * 64])
                                even, odd = tla.simd.vdintlv(ii, ii)
                                low = tla.simd.vmin(even, odd)
                                high = tla.simd.vmax(even, odd)
                                ordered, unused = tla.simd.vintlv(low, high)
                                tla.simd.vsts(gather_ids[index_half * 64], ordered)
                                zero = tla.simd.vfull(0, tl.int32, 64)
                                limit = tla.simd.vfull(L - 1, tl.int32, 64)
                                invalid_low = tla.simd.vsel(tla.simd.vcmp(low, zero, op=tla.simd.CMP.LT), tla.simd.vfull(1, tl.int32, 64), zero)
                                invalid_high = tla.simd.vsel(tla.simd.vcmp(high, limit, op=tla.simd.CMP.GT), tla.simd.vfull(1, tl.int32, 64), zero)
                                flags = tla.simd.vsel(tla.simd.vcmp(low, high, op=tla.simd.CMP.EQ), tla.simd.vfull(1, tl.int32, 64), zero)
                                combined_flags = tla.simd.vmax(combined_flags, tla.simd.vmax(flags, tla.simd.vmax(invalid_low, invalid_high)))
                            tla.simd.vsts(any_duplicate[0], tla.simd.vreduce_max(combined_flags), dist='ONEPT_B32')
                        for pair_chunk in tl.range(4):
                            for gather_phase in tl.static_range(2):
                                chunk = (pair_chunk * 2 + gather_phase) * 16
                                if gather_phase == 0:
                                    keys = gk0
                                    ropes = gr0
                                else:
                                    keys = gk1
                                    ropes = gr1
                                if dsa_ext.scalar_load(any_duplicate, 0) == 0:
                                    for pair in tl.static_range(8):
                                        i0 = dsa_ext.scalar_load(gather_ids, sub * 128 + chunk + pair * 2)
                                        i1 = dsa_ext.scalar_load(gather_ids, sub * 128 + chunk + pair * 2 + 1)
                                        s0 = i0
                                        s1 = i1
                                        dsa_ext.load_pair(tla.view(KV, (B * L * 512,)), (batch * L + s0) * 512, (s1 - s0) * 512, tla.view(keys, (2, 512), offsets=(pair * 2, 0)))
                                        dsa_ext.load_pair(tla.view(KR, (B * L * 64,)), (batch * L + s0) * 64, (s1 - s0) * 64, tla.view(ropes, (2, 64), offsets=(pair * 2, 0)))
                                else:
                                    for row in tl.static_range(16):
                                        index = dsa_ext.scalar_load(gather_ids, sub * 128 + chunk + row)
                                        safe = tl.maximum(0, tl.minimum(index, L - 1))
                                        tla.load_gm_to_ub(tla.view(KV + (batch * L + safe) * 512, (1, 512)), tla.view(keys, (1, 512), offsets=(row, 0)))
                                        tla.load_gm_to_ub(tla.view(KR + (batch * L + safe) * 64, (1, 64)), tla.view(ropes, (1, 64), offsets=(row, 0)))
                                tla.store_ub_to_gm(keys, tla.view(WORK + ((tl.program_id(0) * 2 + gather_slot) * BN + sub * 128 + chunk) * 576, (16, 512), strides=(576, 1)))
                                tla.store_ub_to_gm(ropes, tla.view(WORK + ((tl.program_id(0) * 2 + gather_slot) * BN + sub * 128 + chunk) * 576 + 512, (16, 64), strides=(576, 1)))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, (1 - phase) * 5)
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 1)
                    sub = tla.sub_block_id()
                    with tla.scope.simd():
                        negative = tla.simd.vfull(float('-inf'), tl.float32, 64)
                        zero = tla.simd.vfull(0, tl.int32, 64)
                        limit = tla.simd.vfull(L, tl.int32, 64)
                        i0 = tla.simd.vlds(indices[0])
                        i1 = tla.simd.vlds(indices[64])
                        i2 = tla.simd.vlds(indices[128])
                        i3 = tla.simd.vlds(indices[192])
                        for row in tl.range(VM):
                            x0 = ext.vmuls(tla.simd.vlds(score[row, 0]), 576 ** (-0.5))
                            x0 = tla.simd.vsel(tla.simd.vcmp(i0, zero, op=tla.simd.CMP.GE), x0, negative)
                            x0 = tla.simd.vsel(tla.simd.vcmp(i0, limit, op=tla.simd.CMP.LT), x0, negative)
                            x1 = ext.vmuls(tla.simd.vlds(score[row, 64]), 576 ** (-0.5))
                            x1 = tla.simd.vsel(tla.simd.vcmp(i1, zero, op=tla.simd.CMP.GE), x1, negative)
                            x1 = tla.simd.vsel(tla.simd.vcmp(i1, limit, op=tla.simd.CMP.LT), x1, negative)
                            x2 = ext.vmuls(tla.simd.vlds(score[row, 128]), 576 ** (-0.5))
                            x2 = tla.simd.vsel(tla.simd.vcmp(i2, zero, op=tla.simd.CMP.GE), x2, negative)
                            x2 = tla.simd.vsel(tla.simd.vcmp(i2, limit, op=tla.simd.CMP.LT), x2, negative)
                            x3 = ext.vmuls(tla.simd.vlds(score[row, 192]), 576 ** (-0.5))
                            x3 = tla.simd.vsel(tla.simd.vcmp(i3, zero, op=tla.simd.CMP.GE), x3, negative)
                            x3 = tla.simd.vsel(tla.simd.vcmp(i3, limit, op=tla.simd.CMP.LT), x3, negative)
                            old_m = tla.simd.vlds(m[row], dist='BRC_B32')
                            maximum = tla.simd.vmax(tla.simd.vreduce_max(tla.simd.vmax(tla.simd.vmax(x0, x1), tla.simd.vmax(x2, x3))), old_m)
                            a = ext.vexpdif(old_m, maximum)
                            p0 = ext.vexpdif(x0, maximum)
                            p1 = ext.vexpdif(x1, maximum)
                            p2 = ext.vexpdif(x2, maximum)
                            p3 = ext.vexpdif(x3, maximum)
                            den = tla.simd.vbroadcast(tla.simd.vreduce_sum(tla.simd.vadd(tla.simd.vadd(p0, p1), tla.simd.vadd(p2, p3))))
                            den = tla.simd.vadd(den, tla.simd.vmul(tla.simd.vlds(l[row], dist='BRC_B32'), a))
                            tla.simd.vsts(m[row], maximum, dist='ONEPT_B32')
                            tla.simd.vsts(l[row], den, dist='ONEPT_B32')
                            tla.simd.vsts(alpha[row], a, dist='ONEPT_B32')
                            packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p0, tl.bfloat16), tla.simd.vtruncf(p1, tl.bfloat16))
                            tla.simd.vsstb(packed, pub[0, row * 16], (VM + 1) * 16)
                            packed, unused = tla.simd.vdintlv(tla.simd.vtruncf(p2, tl.bfloat16), tla.simd.vtruncf(p3, tl.bfloat16))
                            tla.simd.vsstb(packed, pub[8, row * 16], (VM + 1) * 16)
                    tla.copy_ub_to_l1(tla.view(pub, (BN // 16, VM * 16)), tla.view(pflat, (BN // 16, VM * 16), offsets=(0, sub * VM * 16)))
                    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 2)
                with tla.scope.cube():
                    tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 2)
                for fragment in tl.range(512 // VP):
                    with tla.scope.cube():
                        ext.matmul(p, tla.view(current, (VP // 16, BN // 16, 16, 16), offsets=(fragment * (VP // 16), 0, 0, 0)), pc, clear_accum=True)
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
                        tla.copy_l0c_to_ub(pc, pv, tla.FixpipeL0C2UBParams(dma_mode=tla.FixpipeDMAMode.NZ2ND, dual_dst_mode=tla.FixpipeDualDstMode.ROW_SPLIT, sub_block_idx=None))
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.F, 3)
                    with tla.scope.vector():
                        tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 3)
                        with tla.scope.simd():
                            for row in tl.range(VM):
                                a = tla.simd.vlds(alpha[row], dist='BRC_B32')
                                for col in tl.static_range(VP // 64):
                                    value = tla.simd.vadd(tla.simd.vlds(pv[row, col * 64]), tla.simd.vmul(tla.simd.vlds(acc[row, fragment * VP + col * 64]), a))
                                    tla.simd.vsts(acc[row, fragment * VP + col * 64], value)
                        tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.V, 4)
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 4)
        with tla.scope.vector():
            sub = tla.sub_block_id()
            if SPLIT == 1:
                for fragment in tl.range(512 // VP):
                    with tla.scope.simd():
                        for row in tl.range(VM):
                            den = tla.simd.vlds(l[row], dist='BRC_B32')
                            for col in tl.static_range(VP // 128):
                                b0 = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(acc[row, fragment * VP + col * 128]), den), tl.bfloat16)
                                b1 = tla.simd.vtruncf(tla.simd.vdiv(tla.simd.vlds(acc[row, fragment * VP + col * 128 + 64]), den), tl.bfloat16)
                                packed, unused = tla.simd.vdintlv(b0, b1)
                                tla.simd.vsts(out[row, col * 128], packed)
                    tla.store_ub_to_gm(out, tla.view(O + (output_token * H + head + sub * VM) * 512 + fragment * VP, (VM, VP), strides=(512, 1)))
            else:
                tla.store_ub_to_gm(acc, tla.view(O + (output_token * H + head + sub * VM) * 512, (VM, 512)))
            tla.store_ub_to_gm(tla.view(m, (VM,)), tla.view(MAX + output_token * H + head + sub * VM, (VM,)))
            tla.store_ub_to_gm(tla.view(l, (VM,)), tla.view(SUM + output_token * H + head + sub * VM, (VM,)))
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 7)
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 8)
        with tla.scope.cube():
            tla.sync.block_wait_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 7)
            tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.S, 8)

def forward(q, kv, qr, kr, indices, bm=128, cores=28, bn=256, split=1):
    from merge import merge_partials
    b, sq, h, d = q.shape
    topk = indices.shape[-1]
    assert d == 512 and qr.shape[-1] == 64 and (h % bm == 0) and (bn == 256) and (topk % (2 * bn * split) == 0)
    assert all((t.dtype == torch.bfloat16 and t.is_contiguous() for t in (q, kv, qr, kr)))
    assert indices.dtype == torch.int32 and indices.is_contiguous()
    options = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
    wk = torch.empty((cores, 2, bn, 576), dtype=q.dtype, device=q.device)
    o = torch.empty_like(q)
    mx = torch.empty((b, 1, sq, h), device=q.device, dtype=torch.float32)
    su = torch.empty_like(mx)
    if split == 1:
        partial, pm, pl = (o, mx, su)
    else:
        partial = torch.empty((b, sq, split, h, d), device=q.device, dtype=torch.float32)
        pm = torch.empty((b, sq, split, h), device=q.device, dtype=torch.float32)
        pl = torch.empty_like(pm)
    k = sparse_mla_kernel[cores,](q, kv, qr, kr, indices, wk, partial, pm, pl, b, sq, kv.shape[1], h, topk, bm, cores, bn, split, **options)
    global LAST_KERNELS
    LAST_KERNELS = [k]
    if split > 1:
        m = merge_partials[cores * 2,](partial, pm, pl, o, mx, su, b * sq, h, split, cores * 2, **options)
        LAST_KERNELS.append(m)
    return (o, mx, su)
