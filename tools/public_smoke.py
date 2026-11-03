"""Validate the public-API kernel without importing the legacy extension.

Use a Triton-Ascend build containing the accompanying PR and set
DSA_API_MODE=public before Python starts. The reference uses full FP32
operations on NPU with HF32 disabled, as described in review.py.
"""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys

if os.environ.get("DSA_API_MODE") != "public":
    raise RuntimeError("Set DSA_API_MODE=public explicitly")

from review import CASES, accuracy, make_inputs, receipts, torch, triton
from triton.compiler.code_generator import CodeGenerator

original_init = CodeGenerator.__init__
assert not getattr(CodeGenerator, "_issue104_native_ext", False)
from dsa import forward
from dsa.compat import SELECTED_API_MODE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--family", default="normal", choices=("normal", "uniform", "peaked", "duplicates", "invalid"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    data = make_inputs(CASES[args.case], args.seed, args.family)
    values, kernels = forward(*data, return_kernels=True)
    torch.npu.synchronize()
    check = accuracy({"public": values}, data)
    # There is no legacy implementation in this process: do not report the
    # helper's vacuously true original/refactor comparison as an actual test.
    check.pop("original_refactored_bitwise_equal")
    initializer = CodeGenerator.__init__
    # The stock Ascend backend lazily wraps CodeGenerator to attach hacc.target.
    # Permit that package-owned wrapper, but never a DSA-local adaptation.
    initializer_file = Path(initializer.__code__.co_filename).resolve()
    stock_backend = Path(triton.__file__).resolve().parent / "backends/ascend/__init__.py"
    assert initializer is original_init or initializer_file == stock_backend
    assert not getattr(CodeGenerator, "_issue104_native_ext", False)
    assert "dsa_ext" not in sys.modules and "tla_native_ext" not in sys.modules
    record = dict(case=args.case, shape=CASES[args.case], seed=args.seed,
                  family=args.family, api_mode=SELECTED_API_MODE,
                  triton_path=triton.__file__, no_dsa_codegen_monkeypatch=True,
                  code_generator_origin=str(initializer_file),
                  kernels=receipts(kernels), accuracy=check,
                  passed=all(row["attachment_gate_passed"] for row in check["metrics"]["public"].values()))
    (args.out / "result.json").write_text(json.dumps(record, indent=2))
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps(record), flush=True)
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
