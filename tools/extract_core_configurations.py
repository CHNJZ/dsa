"""Join a matrix run to CANN op_summary.csv by profiler process id.

Chrome traces omit Block Num in this environment. Preserve the original CSV
values and file hashes instead of inferring physical core counts from task names.
Run this on the machine that retains export_only_prof_dir.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--profiler-dir", type=Path, default=Path("export_only_prof_dir"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fields = ("Op Name", "OP Type", "Task Type", "Block Num", "Mix Block Num", "HF32 Eligible")
    records = []
    for entry in json.loads((args.matrix_dir / "matrix.json").read_text()):
        label = f"{entry['case']}-{entry['family']}-{entry['seed']}"
        log = (args.matrix_dir / f"{label}.log").read_text()
        pids = sorted(set(re.findall(r"\[WARNING\]\s+\[(\d+)\]\s+profiler\.py", log)))
        result_file = args.matrix_dir / label / "result.json"
        if result_file.exists():
            pid = json.loads(result_file.read_text()).get("process_id")
            if pid:
                pids = sorted(set(pids + [str(pid)]))
        evidence, configurations = [], set()
        for pid in pids:
            for directory in sorted(args.profiler_dir.glob(f"*_{pid}_*_ascend_pt")):
                for path in sorted(directory.rglob("op_summary*.csv")):
                    matched = 0
                    for row in csv.DictReader(path.open()):
                        if row.get("Op Name") not in ("sparse_mla_kernel", "workspace_attention", "ragged_attention",
                                                     "merge_partials", "SparseFlashAttention"):
                            continue
                        configurations.add(tuple(row.get(key, "") for key in fields))
                        matched += 1
                    evidence.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                         matching_rows=matched))
        records.append(dict(case=entry["case"], family=entry["family"], seed=entry["seed"], process_ids=pids,
                            configurations=[dict(zip(fields, row)) for row in sorted(configurations)],
                            csv_evidence=evidence))
    args.out.write_text(json.dumps({"matrix": str(args.matrix_dir), "records": records,
                                    "interpretation": "CANN launch configuration, not measured active-core occupancy"}, indent=2))


if __name__ == "__main__":
    main()
