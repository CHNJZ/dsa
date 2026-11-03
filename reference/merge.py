"""FP32 split-MLA merge with grouped GM transfers and register-held weights."""

import triton.language as tl
import triton.language.extra.tla as tla
from triton.language.extra.tla.jit import TLAJITFunction
import tla_native_ext as ext


@TLAJITFunction
def merge_partials(
    P,
    PM,
    PL,
    O,
    OM,
    OL,
    T: tl.constexpr,
    H: tl.constexpr,
    SPLIT: tl.constexpr,
    CORES: tl.constexpr,
):
    GH: tl.constexpr = 4 if T * H >= 512 else 1
    with tla.scope.vector():
        for job in tl.range(tl.program_id(0), T * (H // GH), CORES):
            token = job // (H // GH)
            head = (job % (H // GH)) * GH
            values = tla.alloc((SPLIT, GH * 512), tl.float32, space="ub")
            vals = tla.reshape(values, (SPLIT, GH, 512))
            maxes = tla.alloc((SPLIT, 8), tl.float32, space="ub")
            sums = tla.alloc((SPLIT, 8), tl.float32, space="ub")
            mout = tla.alloc((64,), tl.float32, space="ub")
            lout = tla.alloc((64,), tl.float32, space="ub")
            out = tla.alloc((GH, 512), tl.bfloat16, space="ub")
            tla.load_gm_to_ub(
                tla.view(
                    P + (token * SPLIT * H + head) * 512,
                    (SPLIT, GH * 512),
                    strides=(H * 512, 1),
                ),
                values,
            )
            tla.load_gm_to_ub(
                tla.view(PM + token * SPLIT * H + head, (SPLIT, GH), strides=(H, 1)),
                tla.view(maxes, (SPLIT, GH)),
            )
            tla.load_gm_to_ub(
                tla.view(PL + token * SPLIT * H + head, (SPLIT, GH), strides=(H, 1)),
                tla.view(sums, (SPLIT, GH)),
            )
            with tla.scope.simd():
                for h in tl.range(GH):
                    maximum = tla.simd.vfull(float("-inf"), tl.float32, 64)
                    for i in tl.static_range(SPLIT):
                        maximum = tla.simd.vmax(
                            maximum, tla.simd.vlds(maxes[i, h], dist="BRC_B32")
                        )
                    den = tla.simd.vfull(0.0, tl.float32, 64)
                    weights = ()
                    for i in tl.static_range(SPLIT):
                        weight = ext.vexpdif(
                            tla.simd.vlds(maxes[i, h], dist="BRC_B32"), maximum
                        )
                        den = tla.simd.vadd(
                            den,
                            tla.simd.vmul(
                                weight, tla.simd.vlds(sums[i, h], dist="BRC_B32")
                            ),
                        )
                        weights = weights + (weight,)
                    tla.simd.vsts(mout[h], maximum, dist="ONEPT_B32")
                    tla.simd.vsts(lout[h], den, dist="ONEPT_B32")
                    for chunk in tl.range(4):
                        y0 = tla.simd.vfull(0.0, tl.float32, 64)
                        y1 = tla.simd.vfull(0.0, tl.float32, 64)
                        for i in tl.static_range(SPLIT):
                            y0 = tla.simd.vadd(
                                y0,
                                tla.simd.vmul(
                                    weights[i], tla.simd.vlds(vals[i, h, chunk * 128])
                                ),
                            )
                            y1 = tla.simd.vadd(
                                y1,
                                tla.simd.vmul(
                                    weights[i],
                                    tla.simd.vlds(vals[i, h, chunk * 128 + 64]),
                                ),
                            )
                        b0 = tla.simd.vtruncf(tla.simd.vdiv(y0, den), tl.bfloat16)
                        b1 = tla.simd.vtruncf(tla.simd.vdiv(y1, den), tl.bfloat16)
                        packed, unused = tla.simd.vdintlv(b0, b1)
                        tla.simd.vsts(out[h, chunk * 128], packed)
            tla.store_ub_to_gm(out, tla.view(O + (token * H + head) * 512, (GH, 512)))
            tla.store_ub_to_gm(
                tla.view(mout, (GH,)), tla.view(OM + token * H + head, (GH,))
            )
            tla.store_ub_to_gm(
                tla.view(lout, (GH,)), tla.view(OL + token * H + head, (GH,))
            )
