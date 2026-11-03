"""One-case, reproducible correctness and timing audit (run from x/).

Original, refactored and AscendC implementations receive the same tensors.
The profiler reports both summed task durations and the span of all tasks.
An independent eager measurement includes Python dispatch and synchronization.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference"))

import torch
import torch_npu
import triton

SPEC = json.loads((ROOT / "validation/attachment-specifications.json").read_text())
CASES = {**SPEC["cases"]["issue"], **SPEC["cases"]["comment"]}
CONFIGS = json.loads((ROOT / "reference/configs-selected.json").read_text())


def make_inputs(shape, seed, family):
    """Generate inputs before timing; keep index order and multiplicity."""
    batch, queries, heads, length, selected = shape
    generator = torch.Generator().manual_seed(seed)

    def normal(*dims):
        return torch.randn(dims, generator=generator).bfloat16().npu()

    q = normal(batch, queries, heads, 512)
    kv = normal(batch, length, 1, 512)
    qr = normal(batch, queries, heads, 64)
    kr = normal(batch, length, 1, 64)
    ids = torch.stack([
        torch.randperm(length, generator=generator)[:selected]
        for _ in range(batch * queries)
    ]).reshape(batch, queries, 1, selected).int().npu()
    if family == "uniform":
        q.zero_()
        qr.zero_()
    elif family == "peaked":
        q.mul_(4)
        qr.mul_(4)
    elif family == "duplicates":
        ids[..., 1::2] = ids[..., 0::2]
    elif family == "invalid":
        ids[..., selected // 2:] = -1
    return q, kv, qr, kr, ids


def native(data):
    q, kv, qr, kr, ids = data
    return torch_npu.npu_sparse_flash_attention(
        q, kv, kv, ids, 576 ** -0.5, query_rope=qr, key_rope=kr,
        attention_mode=2, sparse_mode=0, return_softmax_lse=True,
    )


def calls_for(case, data, include_refactor=True):
    cfg = CONFIGS[case]
    original = importlib.import_module(cfg["module"])
    calls = {
        "original": lambda: original.forward(
            *data, bm=cfg["bm"], bn=cfg["bn"], split=cfg["split"], cores=28),
        "ascendc": lambda: native(data),
    }
    if include_refactor:
        from dsa import forward
        calls["refactored"] = lambda: forward(*data)
    return calls, original


def receipts(kernels):
    """Record full binary plus executable sections, separating debug names."""
    from elftools.elf.elffile import ELFFile

    result = []
    for kernel in kernels:
        binary = kernel.asm["npubin"]
        adapter = kernel.asm.get("ttadapter", "")
        normalized = re.sub(r"loc\([^\n]*?\)", "", adapter)
        normalized = re.sub(r"@[A-Za-z0-9_.$]+", "@symbol", normalized)
        normalized = "\n".join(line.rstrip() for line in normalized.splitlines()
                               if not line.startswith("#loc"))
        sections = {}
        try:
            elf = ELFFile(io.BytesIO(binary))
            sections = {section.name: hashlib.sha256(section.data()).hexdigest()
                        for section in elf.iter_sections() if section["sh_flags"] & 4}
        except Exception as exc:
            sections = {"unavailable": str(exc)}
        result.append({
            "name": kernel.name,
            "binary_sha256": hashlib.sha256(binary).hexdigest(),
            "executable_sections": sections,
            "normalized_adapter_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "metadata": str(kernel.metadata),
        })
    return result


def accuracy(outputs, data, chunk=16):
    """Full-output FP32 reference, chunked only to bound scratch memory.

    CPU-generated BF16 inputs are converted to FP32 on NPU for the reference;
    HF32 is explicitly disabled. This is not a CPU reference or an official
    CANN acceptance rule. Max-row error supplements the attachment's global
    normalization so one erroneous row cannot hide behind another row's scale.
    """
    q, kv, qr, kr, ids = data
    batch, queries, heads, _ = q.shape
    metrics = {model: {key: dict(max_abs=0., reference_max=0., error_sq=0., ref_sq=0.,
                                max_row_normalized=0., finite=True, elements=0)
                       for key in ("output", "maximum", "denominator")}
               for model in outputs}
    exact = {key: True for key in ("output", "maximum", "denominator")}
    for bi in range(batch):
        for start in range(0, queries, chunk):
            end = min(queries, start + chunk)
            index = ids[bi, start:end, 0].long()
            valid = (index >= 0) & (index < kv.shape[1])
            index = index.clamp(0, kv.shape[1] - 1)
            selected = kv[bi, index, 0].float()
            rope = kr[bi, index, 0].float()
            score = (torch.bmm(q[bi, start:end].float(), selected.transpose(1, 2))
                     + torch.bmm(qr[bi, start:end].float(), rope.transpose(1, 2))) / 24.
            score.masked_fill_(~valid[:, None, :], -float("inf"))
            maximum = score.amax(-1)
            exponent = (score - maximum[..., None]).exp()
            denominator = exponent.sum(-1)
            reference = (torch.bmm(exponent / denominator[..., None], selected),
                         maximum, denominator)
            pieces = {model: (value[0][bi, start:end], value[1][bi, 0, start:end],
                               value[2][bi, 0, start:end]) for model, value in outputs.items()}
            for model, values in pieces.items():
                for key, actual, expected in zip(metrics[model], values, reference):
                    if actual.shape != expected.shape:
                        raise AssertionError((model, key, actual.shape, expected.shape))
                    delta = actual.float() - expected
                    row = metrics[model][key]
                    row["max_abs"] = max(row["max_abs"], float(delta.abs().max()))
                    row["reference_max"] = max(row["reference_max"], float(expected.abs().max()))
                    row["error_sq"] += float(delta.square().sum())
                    row["ref_sq"] += float(expected.square().sum())
                    row["finite"] &= bool(actual.isfinite().all())
                    row["elements"] += expected.numel()
                    row_error = delta.abs().amax(-1) / expected.abs().amax(-1).clamp_min(1e-30)
                    row["max_row_normalized"] = max(row["max_row_normalized"], float(row_error.max()))
            if "refactored" in pieces:
                for i, key in enumerate(exact):
                    lhs, rhs = pieces["original"][i], pieces["refactored"][i]
                    if lhs.dtype != rhs.dtype or lhs.shape != rhs.shape:
                        exact[key] = False
                    else:
                        bits = torch.int16 if lhs.dtype == torch.bfloat16 else torch.int32
                        exact[key] &= torch.equal(lhs.contiguous().view(bits), rhs.contiguous().view(bits))
    for model, values in metrics.items():
        for row, tolerance in zip(values.values(), (.005, .0001, .0001)):
            row["max_normalized"] = row["max_abs"] / max(row["reference_max"], 1e-30)
            row["relative_rms"] = (row["error_sq"] / max(row["ref_sq"], 1e-30)) ** .5
            row["attachment_gate_passed"] = row["finite"] and row["max_normalized"] <= tolerance
            row["attachment_tolerance"] = tolerance
    return {"metrics": metrics, "original_refactored_bitwise_equal": exact,
            "equality_method": "dtype-and-shape-check-then-integer-bit-view",
            "row_metric_definition": "output: per head over 512 columns; maximum/denominator: per query over H heads; diagnostic only"}


def profile_tasks(call, path, samples, expected_names):
    for _ in range(3):
        call()
    torch.npu.synchronize()
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=config,
    ) as profiler:
        for _ in range(samples):
            call()
        torch.npu.synchronize()
    profiler.export_chrome_trace(str(path))
    raw = json.loads(path.read_text())
    events = raw.get("traceEvents", []) if isinstance(raw, dict) else raw
    tasks = sorted((event for event in events if event.get("ph") == "X"
                    and str((event.get("args") or {}).get("Task Type", "")).startswith(
                        ("AI_CORE", "AI_VECTOR", "MIX_", "KERNEL"))), key=lambda event: float(event["ts"]))
    if not tasks or len(tasks) % samples:
        raise RuntimeError(f"incomplete profiler record: {len(tasks)} tasks / {samples} calls")
    count = len(tasks) // samples
    names = [event["name"] for event in tasks[:count]]
    if names != expected_names:
        raise RuntimeError(f"Expected all main/merge tasks {expected_names}, got {names}")
    if len({(event.get("pid"), event.get("args", {}).get("Physic Stream Id")) for event in tasks}) != 1:
        raise RuntimeError("Expected a single device and stream for serial task grouping")
    summed, spans = [], []
    for start in range(0, len(tasks), count):
        group = tasks[start:start + count]
        if [event["name"] for event in group] != names:
            raise RuntimeError("device task sequence changed within one profile")
        summed.append(sum(float(event["dur"]) for event in group))
        spans.append(max(float(event["ts"]) + float(event["dur"]) for event in group)
                     - float(group[0]["ts"]))
    return {"task_sum_us": summed, "task_span_us": spans, "tasks_per_call": count,
            "tasks": [{"name": event["name"], "args": event.get("args", {})} for event in tasks[:count]]}


def balanced_order(names, sample, round_id):
    """Balance all six orders and predecessor pairs for three implementations."""
    if len(names) == 3:
        orders = ((0, 1, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1))
        return [names[index] for index in orders[(sample + round_id) % 6]]
    rotate = (sample + round_id) % len(names)
    return names[rotate:] + names[:rotate]


def eager_samples(calls, samples, round_id):
    """Rotate order every sample; include wrapper allocation and all launches."""
    result = {name: [] for name in calls}
    names = list(calls)
    for sample in range(samples):
        for name in balanced_order(names, sample, round_id):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            output = calls[name]()
            torch.npu.synchronize()
            result[name].append((time.perf_counter_ns() - start) / 1000)
            del output
    return result


def profile_interleaved(calls, task_names, path, samples, round_id):
    """Keep competing implementations close in time within one profiler run.

    Each sample runs all implementations, rotating their order. Samples are
    retained regardless of external interference; this is not a best-of run.
    """
    names = list(calls)
    for _ in range(3):
        for call in calls.values():
            call()
    torch.npu.synchronize()
    schedule = []
    config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
        experimental_config=config,
    ) as profiler:
        for sample in range(samples):
            for name in balanced_order(names, sample, round_id):
                calls[name]()
                schedule.append(name)
        torch.npu.synchronize()
    profiler.export_chrome_trace(str(path))
    raw = json.loads(path.read_text())
    events = raw.get("traceEvents", []) if isinstance(raw, dict) else raw
    tasks = sorted((event for event in events if event.get("ph") == "X"
                    and str((event.get("args") or {}).get("Task Type", "")).startswith(
                        ("AI_CORE", "AI_VECTOR", "MIX_", "KERNEL"))), key=lambda event: float(event["ts"]))
    expected_count = sum(len(task_names[name]) for name in schedule)
    if len(tasks) != expected_count:
        raise RuntimeError(f"Incomplete interleaved trace: {len(tasks)} != {expected_count}")
    if len({(event.get("pid"), event.get("args", {}).get("Physic Stream Id")) for event in tasks}) != 1:
        raise RuntimeError("Expected a single device and stream for serial task grouping")
    result = {name: dict(task_sum_us=[], task_span_us=[], tasks_per_call=len(task_names[name]), tasks=[])
              for name in names}
    cursor = 0
    for name in schedule:
        group = tasks[cursor:cursor + len(task_names[name])]
        cursor += len(group)
        if [event["name"] for event in group] != task_names[name]:
            raise RuntimeError(f"Unexpected task order for {name}: {[event['name'] for event in group]}")
        row = result[name]
        row["task_sum_us"].append(sum(float(event["dur"]) for event in group))
        row["task_span_us"].append(max(float(event["ts"]) + float(event["dur"]) for event in group)
                                    - float(group[0]["ts"]))
        if not row["tasks"]:
            row["tasks"] = [{"name": event["name"], "args": event.get("args", {})} for event in group]
    return result


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(180, repeat=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("accuracy", "timing", "smoke"), default="accuracy")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--family", choices=("normal", "uniform", "peaked", "duplicates", "invalid"), default="normal")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--original-only", action="store_true")
    parser.add_argument("--interleaved", action="store_true",
                        help="Rotate all implementations per call within each device trace")
    parser.add_argument("--allow-native-failure", action="store_true",
                        help="Only for invalid-index accuracy exploration; never qualifies a performance comparison")
    args = parser.parse_args()
    period = 2 if args.original_only else 6
    if args.interleaved and args.samples % period:
        parser.error(f"Balanced interleaving requires --samples to be a multiple of {period}")
    if args.allow_native_failure and (args.family != "invalid" or args.mode == "timing"):
        parser.error("--allow-native-failure is restricted to invalid-index accuracy checks")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.npu.set_device(0)
    torch.npu.matmul.allow_hf32 = False
    print("PHASE: create inputs", flush=True)
    data = make_inputs(CASES[args.case], args.seed, args.family)
    calls, original = calls_for(args.case, data, not args.original_only)
    record = dict(case=args.case, shape=CASES[args.case], seed=args.seed, family=args.family,
                  process_id=os.getpid(),
                  mode=args.mode, api_mode=os.environ.get("DSA_API_MODE", "legacy"),
                  physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                  torch=torch.__version__, torch_npu=torch_npu.__version__, triton=triton.__version__,
                  triton_path=triton.__file__, config=CONFIGS[args.case])
    record["sampling_method"] = "per-call-six-order-balanced" if args.interleaved else "per-implementation-blocks"
    outputs = {}
    for name, call in calls.items():
        print(f"PHASE: first launch {name}", flush=True)
        outputs[name] = call()
        torch.npu.synchronize()
        print(f"PHASE: first launch completed {name}", flush=True)
    record["original_kernels"] = receipts(original.LAST_KERNELS)
    if "refactored" in calls:
        from dsa import forward
        _, kernels = forward(*data, return_kernels=True)
        torch.npu.synchronize()
        record["refactored_kernels"] = receipts(kernels)
    if args.mode in ("accuracy", "smoke"):
        print("PHASE: full FP32 reference", flush=True)
        record["accuracy"] = accuracy(outputs, data)
        tla_passed = all(row["attachment_gate_passed"]
                         for model in ("original", "refactored") if model in outputs
                         for row in record["accuracy"]["metrics"][model].values())
        if "refactored" in outputs:
            tla_passed &= all(record["accuracy"]["original_refactored_bitwise_equal"].values())
        native_passed = all(row["attachment_gate_passed"]
                            for row in record["accuracy"]["metrics"]["ascendc"].values())
        record["tla_passed"] = tla_passed
        record["native_passed"] = native_passed
        record["comparison_passed"] = tla_passed and native_passed
        record["native_failure_allowed"] = args.allow_native_failure
        record["passed"] = record["comparison_passed"] or (args.allow_native_failure and tla_passed)
    else:
        outputs.clear()
        record["rounds"] = []
        names = list(calls)
        task_names = {"original": [item["name"] for item in record["original_kernels"]],
                      "ascendc": ["SparseFlashAttention"]}
        if "refactored_kernels" in record:
            task_names["refactored"] = [item["name"] for item in record["refactored_kernels"]]
        for round_id in range(args.rounds):
            print(f"PHASE: timing round {round_id}", flush=True)
            row = {"id": round_id, "profiles": {}}
            if args.interleaved:
                row["profiles"] = profile_interleaved(calls, task_names, args.out / f"trace-{round_id}-interleaved.json",
                                                      args.samples, round_id)
            else:
                shift = round_id % len(names)
                for name in names[shift:] + names[:shift]:
                    row["profiles"][name] = profile_tasks(calls[name], args.out / f"trace-{round_id}-{name}.json",
                                                         args.samples, task_names[name])
            row["eager_us"] = eager_samples(calls, args.samples, round_id)
            record["rounds"].append(row)
            (args.out / "result.json").write_text(json.dumps(record, indent=2, default=str))
        record["passed"] = True
    (args.out / "result.json").write_text(json.dumps(record, indent=2, default=str))
    faulthandler.cancel_dump_traceback_later()
    print(json.dumps({k: v for k, v in record.items() if k not in ("rounds", "original_kernels", "refactored_kernels")}, default=str), flush=True)
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
