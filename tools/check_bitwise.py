"""Recheck all 16 cases using integer bit views, including signed zero.

The original full FP32-reference run used torch.equal on floating tensors.
This additional check strengthens that evidence without claiming that floating
torch.equal distinguishes signed zeros. It does not time kernels.
"""
import argparse
import gc
import json
from pathlib import Path

from review import CASES, calls_for, make_inputs, receipts, torch
from dsa import forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.npu.set_device(0)
    records = []
    for case, shape in CASES.items():
        data = make_inputs(shape, args.seed, "normal")
        calls, original = calls_for(case, data)
        before = calls["original"]()
        after, kernels = forward(*data, return_kernels=True)
        torch.npu.synchronize()
        same = []
        for lhs, rhs in zip(before, after):
            assert lhs.shape == rhs.shape and lhs.dtype == rhs.dtype
            dtype = torch.int16 if lhs.dtype == torch.bfloat16 else torch.int32
            same.append(torch.equal(lhs.contiguous().view(dtype), rhs.contiguous().view(dtype)))
        record = dict(case=case, seed=args.seed, shape=shape, bitwise_equal=dict(zip(("output", "maximum", "denominator"), same)),
                      equality_method="dtype-and-shape-check-then-integer-bit-view",
                      original_kernels=receipts(original.LAST_KERNELS), refactored_kernels=receipts(kernels), passed=all(same))
        records.append(record)
        args.out.write_text(json.dumps(records, indent=2))
        print(case, record["bitwise_equal"], flush=True)
        if not record["passed"]:
            raise SystemExit(1)
        del before, after, lhs, rhs, calls, data, kernels
        gc.collect()


if __name__ == "__main__":
    main()
