"""Summarize saved DSA results without importing or launching any NPU code.

Run from any directory: python x/tools/summarize_results.py
Missing/incomplete cases remain explicit. --require-complete returns failure
until all 16 cases and their accuracy/binary/timing evidence are complete.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics


MODELS = ("original", "refactored", "ascendc")
OUTPUTS = ("output", "maximum", "denominator")
TIMERS = ("task_sum_us", "task_span_us", "eager_us")
LABELS = {"original": "原始 TLA", "refactored": "重构 TLA", "ascendc": "AscendC"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CORE_FIELDS = {"blocknum", "blockdim", "mixblocknum", "mixblockdim", "aicnum", "aivnum",
               "aicorenum", "aivectornum", "coretype", "tasktype", "tilingkey"}
CORE_COUNT_FIELDS = CORE_FIELDS - {"coretype", "tasktype", "tilingkey"}
IDENTITY_FIELDS = ("shape", "seed", "family", "api_mode", "physical_device", "torch",
                   "torch_npu", "triton", "triton_path", "config")


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def relative_path(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_records(directory, root, cases, family, seed):
    """Select an explicit input family/seed; never silently choose duplicate runs."""
    selected = defaultdict(list)
    ignored, errors = [], []
    for path in sorted(directory.glob("*/result.json")):
        try:
            record = read_json(path)
            if not isinstance(record, dict):
                raise ValueError("result must be an object")
        except (OSError, ValueError) as error:
            errors.append({"path": relative_path(path, root), "error": str(error)})
            continue
        if record.get("case") not in cases or record.get("family") != family or record.get("seed") != seed:
            ignored.append(relative_path(path, root))
            continue
        selected[record["case"]].append((record, path))
    return selected, ignored, errors


def one_record(records, case, root):
    matches = records.get(case, [])
    if len(matches) != 1:
        return None, {"status": "missing" if not matches else "duplicate",
                      "paths": [relative_path(path, root) for _, path in matches]}
    record, path = matches[0]
    return record, {"status": "present", "path": relative_path(path, root)}


def valid_sections(sections):
    return (isinstance(sections, dict) and bool(sections) and "unavailable" not in sections
            and all(isinstance(name, str) and isinstance(value, str) and SHA256.fullmatch(value)
                    for name, value in sections.items()))


def compare_code(record, expected_kernels):
    if record is None:
        return {"verified": False, "reason": "result missing or ambiguous", "kernels": []}
    original, refactored = (record.get(model + "_kernels", []) for model in MODELS[:2])
    if not isinstance(original, list) or not isinstance(refactored, list):
        return {"verified": False, "reason": "invalid receipt list", "kernels": []}
    pairs = []
    for index in range(max(expected_kernels, len(original), len(refactored))):
        left = original[index] if index < len(original) and isinstance(original[index], dict) else {}
        right = refactored[index] if index < len(refactored) and isinstance(refactored[index], dict) else {}
        a, b = left.get("executable_sections"), right.get("executable_sections")
        valid = valid_sections(a) and valid_sections(b)
        pairs.append({"role": "main" if index == 0 else "merge" if index == 1 else f"extra_{index}",
                      "original_name": left.get("name"), "refactored_name": right.get("name"),
                      "valid_executable_sections": bool(valid), "equal": bool(valid and a == b),
                      "original_sections": a, "refactored_sections": b})
    count_ok = len(original) == len(refactored) == expected_kernels
    return {"verified": count_ok and all(row["equal"] for row in pairs),
            "expected_kernels": expected_kernels, "kernel_count_matches": count_ok, "kernels": pairs}


def correlate_binaries(accuracy, timing, expected_kernels):
    rows = []
    for model in MODELS[:2]:
        checked = accuracy.get(model + "_kernels", []) if accuracy else []
        measured = timing.get(model + "_kernels", []) if timing else []
        for index in range(expected_kernels):
            left = checked[index].get("binary_sha256") if index < len(checked) else None
            right = measured[index].get("binary_sha256") if index < len(measured) else None
            valid = isinstance(left, str) and SHA256.fullmatch(left) and isinstance(right, str) and SHA256.fullmatch(right)
            rows.append({"model": model, "role": "main" if index == 0 else "merge",
                         "accuracy_sha256": left, "timing_sha256": right,
                         "verified": bool(valid and left == right)})
    return {"verified": all(row["verified"] for row in rows), "kernels": rows}


def summarize_accuracy(record):
    if record is None:
        return {"complete": False, "metrics": {}, "bitwise_equal": {}}
    data = record.get("accuracy", {})
    metrics = data.get("metrics", {})
    complete = True
    gates = {}
    for model in MODELS:
        values = metrics.get(model, {})
        gates[model] = True
        for output in OUTPUTS:
            row = values.get(output, {})
            valid = all(finite_number(row.get(key)) for key in
                        ("max_normalized", "max_row_normalized", "relative_rms", "max_abs", "error_sq", "ref_sq"))
            complete &= valid and row.get("finite") is True
            tolerance = .005 if output == "output" else .0001
            gate = bool(valid and row.get("finite") is True and 0 <= row["max_normalized"] <= tolerance)
            complete &= row.get("attachment_tolerance") == tolerance and row.get("attachment_gate_passed") == gate
            gates[model] &= gate
    comparison = data.get("original_refactored_bitwise_equal", {})
    numeric_equal = all(comparison.get(key) is True for key in OUTPUTS)
    strict = data.get("equality_method") == "dtype-and-shape-check-then-integer-bit-view"
    return {"complete": bool(complete), "record_passed": record.get("passed") is True,
            "attachment_gates": gates, "original_comparison_flags": comparison,
            "all_outputs_numerically_equal": numeric_equal,
            "all_outputs_bitwise_equal": strict and numeric_equal, "metrics": metrics,
            "equality_method": data.get("equality_method", "legacy torch.equal numeric comparison")}


def strict_bitwise_evidence(record, accuracy, timing, expected_kernels):
    """Bind the separate integer-view check to the binaries actually measured."""
    if record is None:
        return {"verified": False, "reason": "strict integer-view record missing or duplicated"}
    method = record.get("equality_method")
    equal = all(record.get("bitwise_equal", {}).get(key) is True for key in OUTPUTS)
    inputs_match = bool(accuracy and timing and all(record.get(key) == accuracy.get(key) == timing.get(key)
                                                  for key in ("case", "shape", "seed")))
    links = {"accuracy": correlate_binaries(record, accuracy, expected_kernels),
             "timing": correlate_binaries(record, timing, expected_kernels)}
    return {"verified": method == "dtype-and-shape-check-then-integer-bit-view" and equal and inputs_match
                         and all(link["verified"] for link in links.values()),
            "equality_method": method, "bitwise_equal": record.get("bitwise_equal", {}),
            "case_shape_seed_match": inputs_match,
            "binary_links": links}


def statistics_from_rounds(samples):
    medians = [statistics.median(values) for values in samples]
    return {"median_us": statistics.median(medians), "round_min_us": min(medians),
            "round_max_us": max(medians), "round_medians_us": medians,
            "round_spread_ratio": max(medians) / min(medians),
            "samples_per_round": [len(values) for values in samples]}


def summarize_timing(record, expected_rounds, expected_kernels):
    if record is None:
        return {"complete": False, "statistics": {}, "issues": ["result missing or ambiguous"], "task_records": []}
    rounds = record.get("rounds", [])
    issues, task_records = [], []
    if len(rounds) != expected_rounds or [row.get("id") for row in rounds] != list(range(expected_rounds)):
        issues.append(f"expected round IDs 0..{expected_rounds - 1}; got {[row.get('id') for row in rounds]}")
    if record.get("passed") is not True:
        issues.append("result has no final passed=true marker")
    values = {model: {timer: [] for timer in TIMERS} for model in MODELS}
    for index, row in enumerate(rounds):
        for model in MODELS:
            profile = row.get("profiles", {}).get(model, {})
            sizes = []
            for timer in TIMERS:
                samples = row.get("eager_us", {}).get(model) if timer == "eager_us" else profile.get(timer)
                if not isinstance(samples, list) or not samples or not all(finite_number(v) and v > 0 for v in samples):
                    issues.append(f"round {index} {model} {timer}: missing/nonpositive/nonfinite samples")
                    continue
                values[model][timer].append(samples)
                sizes.append(len(samples))
            if len(set(sizes)) > 1:
                issues.append(f"round {index} {model}: sample counts disagree")
            tasks = profile.get("tasks", [])
            if not tasks or profile.get("tasks_per_call") != len(tasks):
                issues.append(f"round {index} {model}: missing/inconsistent task list")
            if model != "ascendc" and len(tasks) != expected_kernels:
                issues.append(f"round {index} {model}: expected {expected_kernels} main/merge tasks; got {len(tasks)}")
            for task_index, task in enumerate(tasks):
                args = task.get("args", {})
                core = {key: value for key, value in args.items()
                        if re.sub(r"[^a-z0-9]", "", key.lower()) in CORE_FIELDS}
                task_records.append({"round": row.get("id"), "model": model, "task_index": task_index,
                                     "name": task.get("name"), "reported_fields": core, "args": args,
                                     "core_count_fields_present": any(re.sub(r"[^a-z0-9]", "", key.lower())
                                                                      in CORE_COUNT_FIELDS for key in core)})
    stats = {}
    for model in MODELS:
        stats[model] = {timer: statistics_from_rounds(values[model][timer])
                        for timer in TIMERS if values[model][timer]}
    ratios = {}
    for timer in TIMERS:
        if all(timer in stats.get(model, {}) for model in MODELS):
            ref = stats["refactored"][timer]["median_us"]
            ratios[timer] = {"original_over_refactored": stats["original"][timer]["median_us"] / ref,
                             "ascendc_over_refactored": stats["ascendc"][timer]["median_us"] / ref}
    return {"complete": not issues, "issues": issues, "statistics": stats,
            "ratios": ratios, "task_records": task_records,
            "sampling_method": record.get("sampling_method", "not recorded; see the saved collector source")}


def aggregate_accuracy(cases):
    aggregate = {}
    for model in MODELS:
        aggregate[model] = {}
        for output in OUTPUTS:
            rows = [(case, data["accuracy"].get("metrics", {}).get(model, {}).get(output, {}))
                    for case, data in cases.items()]
            rows = [(case, row) for case, row in rows if all(finite_number(row.get(key)) for key in
                    ("max_normalized", "max_row_normalized", "relative_rms", "error_sq", "ref_sq"))]
            result = {"cases": len(rows), "expected_cases": len(cases)}
            for key in ("max_normalized", "max_row_normalized", "relative_rms"):
                if rows:
                    case, row = max(rows, key=lambda item: item[1][key])
                    result["worst_" + key] = {"value": row[key], "case": case}
            if rows:
                result["pooled_relative_rms"] = math.sqrt(sum(row["error_sq"] for _, row in rows)
                                                           / max(sum(row["ref_sq"] for _, row in rows), 1e-30))
            aggregate[model][output] = result
    return aggregate


def build_summary(root, accuracy_dir, timing_dir, family, seed, expected_rounds):
    spec = read_json(root / "validation/attachment-specifications.json")
    expected = {**spec["cases"]["issue"], **spec["cases"]["comment"]}
    if len(expected) != 16:
        raise ValueError(f"this review requires exactly 16 distinct specifications; found {len(expected)}")
    configs = read_json(root / "reference/configs-selected.json")
    strict_path = root / "validation/bitwise-final-all16.json"
    strict_records = defaultdict(list)
    if strict_path.is_file():
        for record in read_json(strict_path):
            if record.get("seed") == seed and family == "normal":
                strict_records[record.get("case")].append(record)
    acc, acc_ignored, acc_errors = load_records(accuracy_dir, root, expected, family, seed)
    timing, timing_ignored, timing_errors = load_records(timing_dir, root, expected, family, seed)
    cases = {}
    for case, shape in expected.items():
        a, a_source = one_record(acc, case, root)
        t, t_source = one_record(timing, case, root)
        count = 2 if configs[case]["split"] > 1 else 1
        strict_matches = strict_records.get(case, [])
        strict_record = strict_matches[0] if len(strict_matches) == 1 else None
        item = {"shape": shape, "accuracy_source": a_source, "timing_source": t_source,
                "accuracy": summarize_accuracy(a), "timing": summarize_timing(t, expected_rounds, count),
                "strict_bitwise": strict_bitwise_evidence(strict_record, a, t, count),
                "accuracy_code": compare_code(a, count), "timing_code": compare_code(t, count),
                "timing_binaries_checked_for_accuracy": correlate_binaries(a, t, count),
                "accuracy_environment": {key: a.get(key) for key in IDENTITY_FIELDS} if a else {},
                "timing_environment": {key: t.get(key) for key in IDENTITY_FIELDS} if t else {},
                "input_contract_issues": [f"{phase}: shape/config differs from selected specification"
                                          for phase, record in (("accuracy", a), ("timing", t))
                                          if record and (record.get("shape") != shape or record.get("config") != configs[case])],
                "accuracy_timing_differences": {key: {"accuracy": a.get(key), "timing": t.get(key)}
                                                for key in IDENTITY_FIELDS if a and t and a.get(key) != t.get(key)}}
        item["complete"] = (item["accuracy"]["complete"] and item["accuracy"].get("record_passed", False)
                            and (item["accuracy"].get("all_outputs_bitwise_equal", False)
                                 or item["strict_bitwise"]["verified"])
                            and all(item["accuracy"].get("attachment_gates", {}).values())
                            and item["timing"]["complete"] and item["accuracy_code"]["verified"]
                            and item["timing_code"]["verified"]
                            and item["timing_binaries_checked_for_accuracy"]["verified"]
                            and not item["input_contract_issues"])
        cases[case] = item
    historical = read_json(root / "validation/attachment-performance.json")
    core_suffix = {"paired-all16": "-paired", "balanced-all16": "-balanced"}.get(Path(timing_dir).name, "")
    core_relative = f"validation/core-configurations{core_suffix}.json"
    core_path = root / core_relative
    core_evidence = {"source": core_relative, "present": core_path.is_file()}
    if core_path.is_file():
        try:
            core_evidence["data"] = read_json(core_path)
        except (OSError, ValueError) as error:
            core_evidence["read_error"] = str(error)
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_directories": {"accuracy": relative_path(accuracy_dir, root), "timing": relative_path(timing_dir, root)},
            "input_family": family, "seed": seed, "expected_rounds": expected_rounds,
            "expected_cases": list(expected), "cases": cases,
            "complete": all(row["complete"] for row in cases.values()) and not (acc_errors or timing_errors),
            "missing_or_unverified_cases": [case for case, row in cases.items() if not row["complete"]],
            "accuracy_aggregate": aggregate_accuracy(cases),
            "core_csv_evidence": core_evidence,
            "read_errors": acc_errors + timing_errors, "ignored_records": acc_ignored + timing_ignored,
            "historical_attachment": {"source": "validation/attachment-performance.json", "rows": historical},
            "gate_note": "attachment research tolerances, not official CANN acceptance criteria",
            "statistics_note": "median of per-round medians; min/max are extrema of round medians, not confidence intervals"}


def number(value, digits=6):
    return f"{value:.{digits}g}" if finite_number(value) else "—"


def flag(value):
    return "是" if value else "未确认"


def table(lines, columns, rows):
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    lines.append("")


def render(summary):
    cases = summary["cases"]
    total = len(cases)
    counts = {key: sum(row[key]["complete"] for row in cases.values()) for key in ("accuracy", "timing")}
    code_counts = {phase: sum(row[phase + "_code"]["verified"] for row in cases.values())
                   for phase in ("accuracy", "timing")}
    exact_count = sum(row["accuracy"].get("all_outputs_bitwise_equal", False) or row["strict_bitwise"]["verified"]
                      for row in cases.values())
    linked = sum(row["timing_binaries_checked_for_accuracy"]["verified"] for row in cases.values())
    lines = ["# DSA 重构验证与性能比较", "", f"生成时间（UTC）：`{summary['generated_at_utc']}`。",
             "本文件由 `python tools/summarize_results.py` 从保存的 JSON 重算；不启动 NPU，不执行附件代码。", "",
             f"本轮来源：精度 `{summary['source_directories']['accuracy']}`；计时 `{summary['source_directories']['timing']}`。", "",
             "## 本轮证据是否完整", "",
             f"预期 **{total} 个规格**；完整精度数据 **{counts['accuracy']}/{total}**，完整计时数据 **{counts['timing']}/{total}**。",
             f"原始/重构三输出逐位相同 **{exact_count}/{total}**；全部 main/merge 可执行代码段相同：精度记录 **{code_counts['accuracy']}/{total}**，计时记录 **{code_counts['timing']}/{total}**；计时二进制关联到精度二进制 **{linked}/{total}**。",
             f"本轮输入为 `{summary['input_family']}`，seed=`{summary['seed']}`；预期每种实现测量 {summary['expected_rounds']} 轮。", ""]
    if summary["complete"]:
        lines.append("全部规格的完整数据和关联检查已具备。相同可执行代码段与逐位相同输出支持本次重构保持了已测路径的实际计算；共享机器上的时间差仍需按测量波动解释。")
    else:
        lines.append("**当前证据不完整，不能宣称已完成全部 16 规格的性能无损验证。** 待补齐或核查：" + "、".join(summary["missing_or_unverified_cases"]) + "。")
    lines += ["", "检查不会把空 executable_sections、`unavailable` 或缺失 receipt 当成代码相同。完整二进制 SHA 可以因调试/符号信息不同而变化；代码相同的判据是所有非空 executable section 的 SHA 映射相等。AscendC 未记录可执行代码段 receipt，此项比较仅针对原始和重构 TLA。", ""]
    lines += ["早期 accuracy JSON 中名字含 `bitwise` 的字段实际由 `torch.equal` 计算，只说明数值相同。这里的位级结论要求明确记录 integer-bit-view 方法，或采用 [严格位级复验](validation/bitwise-final-all16.json)，并将其中每个 main/merge 的完整 binary SHA 关联到当前精度与计时记录；不会仅凭旧字段名认定位级相同。", ""]
    table(lines, ["规格", "精度数据", "三输出逐位相同", "精度代码段相同", "计时代码段相同", "计时数据", "计时 SHA 关联精度"],
          [[case, flag(row["accuracy"]["complete"]), flag(row["accuracy"].get("all_outputs_bitwise_equal") or row["strict_bitwise"]["verified"]),
            flag(row["accuracy_code"]["verified"]), flag(row["timing_code"]["verified"]),
            flag(row["timing"]["complete"]), flag(row["timing_binaries_checked_for_accuracy"]["verified"])]
           for case, row in cases.items()])
    lines += ["## 本轮性能：三种计时边界分别报告", "",
              "每个数字为 **每轮样本中位数的中位数**；方括号为这些轮中位数的 `[最小值, 最大值]`。它们不是样本极值或置信区间。单位均为微秒（µs）。`原始/重构` 和 `AscendC/重构` 大于 1 表示重构版对应指标较小。缺失或未完成全部轮次的规格显示“未完整”，不拿部分轮次作最终比较。", "",
              "- `task_sum_us`：一次调用全部设备 kernel 任务时长相加，包含 TLA 的 merge。",
              "- `task_span_us`：同一调用从首个设备任务开始到最后任务结束的跨度，包含任务间空隙；不包含首任务之前或末任务之后的完整主机开销。",
              "- `eager_us`：同步设备后开始主机计时，调用完整 wrapper 后再次同步；包含 Python、分配、分发和同步成本。它与 profiler 指标分开采样，不能直接相减解释为某一项硬件开销。", ""]
    lines += ["采集器按时间排序后的重复任务序列分组，并检查每次任务名称/数量一致；本汇总检查 TLA 的 main/merge 任务数，但未独立重放完整 trace 的 correlation 关系。因此任务跨度是当前采集器定义的设备区间。", ""]
    for timer in TIMERS:
        lines += [f"### {timer}", ""]
        rows = []
        for case, data in cases.items():
            timing = data["timing"]
            if not timing["complete"]:
                rows.append([case, "未完整", "未完整", "未完整", "—", "—"])
                continue
            values = []
            for model in MODELS:
                stat = timing["statistics"][model][timer]
                values.append(f"{number(stat['median_us'])} [{number(stat['round_min_us'])}, {number(stat['round_max_us'])}]")
            ratios = timing["ratios"][timer]
            rows.append([case, *values, number(ratios["original_over_refactored"], 5), number(ratios["ascendc_over_refactored"], 5)])
        table(lines, ["规格", "原始 TLA", "重构 TLA", "AscendC", "原始/重构", "AscendC/重构"], rows)
    lines += ["共享服务器的负载、频率和缓存状态会影响结果。这里报告观测比值和轮间范围，不对差异作没有统计证据的显著性判断；可执行代码相同也不保证每次计时相同。请同时看代码证据、三个计时边界与原始各轮数据。", "",
              "## 精度：相同参考与公开研究阈值", "",
              "参考在 NPU 上以 FP32 计算，并关闭 HF32；不是 CPU/FP64 参考。附件研究判据为 output 的全张量最大归一化误差 ≤0.005，maximum/denominator ≤0.0001，并要求有限值；**这不是官方 CANN 验收规则**。逐行误差和 RMS 在本轮补充报告，不能将没有定义的官方 gate 标为通过。", "",
              "以下 `N / R / RMS` 依次为全张量最大归一化误差、最大行归一化误差、相对 RMS。output 按每个 query/head 的 512 维行归一化；maximum/denominator 的记录按每个 query 的 head 轴归一化。接近零的参考行可能令 R 放大，应结合绝对误差（保存在 JSON）解释。", ""]
    lines += ["研究阈值通过规格数：" + "；".join(
        f"{LABELS[model]} {sum(row['accuracy'].get('attachment_gates', {}).get(model, False) for row in cases.values())}/{total}"
        for model in MODELS) + "。", ""]
    for output in OUTPUTS:
        lines += [f"### {output}", ""]
        rows = []
        for case, data in cases.items():
            values = []
            for model in MODELS:
                row = data["accuracy"].get("metrics", {}).get(model, {}).get(output, {})
                values.append(" / ".join(number(row.get(key), 5) for key in
                                         ("max_normalized", "max_row_normalized", "relative_rms")))
            rows.append([case, *values])
        table(lines, ["规格", "原始 TLA：N / R / RMS", "重构 TLA：N / R / RMS", "AscendC：N / R / RMS"], rows)
    rows = []
    for model in MODELS:
        for output in OUTPUTS:
            row = summary["accuracy_aggregate"][model][output]
            rows.append([LABELS[model], output, f"{row['cases']}/{total}",
                         *[number(row.get("worst_" + key, {}).get("value"), 6) for key in
                           ("max_normalized", "max_row_normalized", "relative_rms")],
                         number(row.get("pooled_relative_rms"), 6)])
    lines += ["### 全规格精度汇总", "", "最大值对应的具体规格、累计平方误差和各模型研究 gate 状态保存在 `validation/summary.json`。汇总 RMS 由累计平方误差/累计参考平方和重算，不取 RMS 的算术平均。", ""]
    table(lines, ["实现", "输出", "覆盖", "最大 N", "最大 R", "最坏规格 RMS", "合并 RMS"], rows)
    lines += ["## trace 中报告的核心配置字段", "",
              "下表只展示首轮每个任务的原始字段；全部轮次、所有实现与原始 args 保存在 summary JSON。`Block Num`、`Mix Block Num`、`Block Dim` 等是 profiler 报告的任务配置，不能直接等同于某一时刻实际活动的物理 AIC/AIV 数。缺字段时明确标为未记录，不按 grid 或任务名称补猜。", ""]
    if summary["core_csv_evidence"]["present"]:
        lines += ["另有从 CANN `op_summary*.csv` 提取并关联规格的独立证据：[core-configurations.json](validation/core-configurations.json)。trace 缺少 Block 字段不等于 CSV 没有这项证据；请结合该文件中的覆盖范围、原始字段和来源判定。这里保留其完整内容到 summary JSON，不根据 trace 的 Task Type 反推核心数量。", ""]
        if "read_error" in summary["core_csv_evidence"]:
            lines += ["**CSV 衍生证据 JSON 当前无法读取：** " + summary["core_csv_evidence"]["read_error"], ""]
    else:
        lines += ["当前尚未取回 `validation/core-configurations.json`。后续若有 CANN op_summary CSV 的规格关联记录，应与本表一并审查；本表的“未记录”只指 trace args。", ""]
    rows = []
    for case, data in cases.items():
        values = []
        for model in MODELS:
            records = [task for task in data["timing"]["task_records"] if task["model"] == model and task["round"] == 0]
            values.append("; ".join(f"{task['name']}: " + (", ".join(f"{key}={value}" for key, value in task['reported_fields'].items()) or "未记录相关字段")
                                     + ("；未记录核数字段" if not task["core_count_fields_present"] else "")
                                     for task in records) or "未记录")
        rows.append([case, *values])
    table(lines, ["规格", "原始 TLA", "重构 TLA", "AscendC"], rows)
    lines += ["## 环境与未解决项", ""]
    methods = sorted({str(data["timing"].get("sampling_method", "未记录")) for data in cases.values()})
    lines += ["采样方法字段：" + "；".join(f"`{method}`" for method in methods) + "。不同采样方法应分别报告，不能合并样本或删除不利轮次。", ""]
    environments = {}
    for case, data in cases.items():
        for phase in ("accuracy", "timing"):
            env = data[phase + "_environment"]
            if env:
                fields = {key: env.get(key) for key in ("api_mode", "physical_device", "torch", "torch_npu", "triton", "triton_path")}
                key = json.dumps(fields, sort_keys=True)
                environments.setdefault(key, {"fields": fields, "runs": []})["runs"].append(f"{case}/{phase}")
    for env in environments.values():
        lines.append(f"- {len(env['runs'])} 条记录：`{json.dumps(env['fields'], ensure_ascii=False)}`。")
    if not environments:
        lines.append("- 尚无本轮环境记录；不使用旧附件环境代填。")
    for case, data in cases.items():
        issues = []
        for phase in ("accuracy", "timing"):
            if data[phase + "_source"]["status"] != "present":
                issues.append(phase + ": " + data[phase + "_source"]["status"])
        issues += data["timing"]["issues"]
        issues += data["input_contract_issues"]
        spreads = []
        for model in MODELS:
            for timer, stat in data["timing"]["statistics"].get(model, {}).items():
                if stat["round_spread_ratio"] >= 2:
                    spreads.append(f"{model}/{timer}={number(stat['round_spread_ratio'], 4)}×")
        if spreads:
            issues.append("轮中位数最大/最小至少 2×：" + ", ".join(spreads)
                          + "；这是描述性波动提示，不是显著性检验，不能据三轮比值认定算法退化")
        if data["accuracy_timing_differences"]:
            issues.append("accuracy/timing 字段不同：" + json.dumps(data["accuracy_timing_differences"], ensure_ascii=False))
        if data["accuracy"].get("metrics"):
            for model, passed in data["accuracy"].get("attachment_gates", {}).items():
                if not passed:
                    issues.append(f"{model} 未通过已记录研究 gate 或指标不完整")
        if not data["accuracy_code"]["verified"]:
            issues.append("accuracy 的 main/merge 代码段同一性未确认")
        if not data["timing_code"]["verified"]:
            issues.append("timing 的 main/merge 代码段同一性未确认")
        if not data["timing_binaries_checked_for_accuracy"]["verified"]:
            issues.append("timing binary SHA 与 accuracy 关联未确认")
        if issues:
            lines.append(f"- `{case}`：" + "；".join(issues) + "。")
    for error in summary["read_errors"]:
        lines.append(f"- JSON 读取失败 `{error['path']}`：{error['error']}。")
    if summary["ignored_records"]:
        lines.append(f"- {len(summary['ignored_records'])} 条其他规格/输入族/seed 的记录未混入本表；路径保存在 summary JSON。")
    lines += ["", "## 历史附件数据（独立口径，不与本轮混列）", "",
              "以下仅转录 `validation/attachment-performance.json` 的旧环境结果。它使用设备任务耗时之和；不是本轮重构测量，也不能与上表相减作为优化收益。历史 16 规格中六组 decode 达到或超过 AscendC，其中约 1.010× 的一组接近持平，八组 prefill 均未超过 AscendC。", ""]
    table(lines, ["规格", "历史 AscendC µs", "历史最优 TLA µs", "历史 AscendC/TLA"],
          [[row["case"], number(row.get("ascendc_us")), number(row.get("optimized_us")),
            number(row.get("ascendc_over_tla"), 5)] for row in summary["historical_attachment"]["rows"]])
    lines += ["原始数据路径、每轮中位数/样本数、代码段和 binary 关联、任务 args、精度指标及其覆盖范围均可从 `validation/summary.json` 追溯。公平性背景见 [FAIRNESS_AUDIT.md](FAIRNESS_AUDIT.md)。", ""]
    return "\n".join(lines).replace("validation/core-configurations.json", summary["core_csv_evidence"]["source"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--accuracy-dir", type=Path)
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--family", default="normal")
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--markdown-out", type=Path, help="default: ROOT/PERFORMANCE.md")
    parser.add_argument("--json-out", type=Path, help="default: ROOT/validation/summary.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    root = args.root.resolve()
    summary = build_summary(root, args.accuracy_dir or root / "validation/accuracy-all16",
                            args.timing_dir or root / "validation/timing-all16", args.family, args.seed, args.rounds)
    json_out = args.json_out or root / "validation/summary.json"
    markdown_out = args.markdown_out or root / "PERFORMANCE.md"
    json_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    markdown_out.write_text(render(summary).replace("validation/summary.json", relative_path(json_out, root)), encoding="utf-8")
    print(json.dumps({"complete": summary["complete"], "expected_cases": len(summary["cases"]),
                      "missing_or_unverified_cases": summary["missing_or_unverified_cases"]}, ensure_ascii=False))
    if args.require_complete and not summary["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
