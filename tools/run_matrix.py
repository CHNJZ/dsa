"""Run each DSA case in a fresh process and retain successes and failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
spec = json.loads((ROOT / "validation/attachment-specifications.json").read_text())
CASES = {**spec["cases"]["issue"], **spec["cases"]["comment"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("accuracy", "timing"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--families", nargs="+", default=["normal"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261102])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--allow-invalid-native-failure", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for case in args.cases:
        for family in args.families:
            for seed in args.seeds:
                label = f"{case}-{family}-{seed}"
                command = [sys.executable, str(ROOT / "tools/review.py"), "--case", case,
                           "--family", family, "--seed", str(seed), "--mode", args.mode,
                           "--samples", str(args.samples), "--rounds", str(args.rounds),
                           "--out", str(args.out / label)]
                if args.interleaved:
                    command.append("--interleaved")
                if args.allow_invalid_native_failure and family == "invalid":
                    command.append("--allow-native-failure")
                snapshot = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
                (args.out / f"{label}-device.txt").write_text(snapshot.stdout + snapshot.stderr)
                start = time.time()
                with (args.out / f"{label}.log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                record = dict(case=case, family=family, seed=seed, exit_code=completed.returncode,
                              elapsed_seconds=time.time() - start, command=command,
                              api_mode=os.environ.get("DSA_API_MODE", "legacy"))
                records.append(record)
                (args.out / "matrix.json").write_text(json.dumps(records, indent=2))
                print(json.dumps(record), flush=True)
                if completed.returncode:
                    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
