import argparse
import math
import os
import sys
import textwrap
from typing import Any

import pandas as pd


# Hack to have something similar to DISABLED_TEST. These models are flaky.

flaky_models = {
    "yolov3",
    "detectron2_maskrcnn_r_101_c4",
    "XGLMForCausalLM",  # discovered in https://github.com/pytorch/pytorch/pull/128148
    "detectron2_fcos_r_50_fpn",
}

# BenchmarkRunner.check_accuracy normalizes eager_two_runs_differ to pass, so
# a pass can legitimately have no Dynamo counters. Keep in sync with the
# non_deterministic list in torchbench.yaml.
PASS_WITHOUT_CAPTURE_MODELS = {"mobilenet_v3_large"}

CAPTURE_OPTIONAL_STATUSES = {
    "OOM",
    "fail_to_run",
    "model_fail_to_load",
    "pass_due_to_skip",
    "timeout",
}


def get_field(csv: pd.DataFrame, model_name: str, field: str) -> Any | None:
    try:
        return csv.loc[csv["name"] == model_name][field].item()
    except Exception:
        return None


def _parse_counter(value: Any) -> int | None:
    if pd.api.types.is_bool(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0 or not value.is_integer():
        return None
    return int(value)


def _capture_required(model: str, actual_status: str, expected_status: str) -> bool:
    if actual_status == "pass" and model in PASS_WITHOUT_CAPTURE_MODELS:
        return False
    # check_accuracy owns status changes. A matching non-pass result is a
    # reviewed expected failure, while eager/skip/run failures may occur before
    # Dynamo has a chance to capture a graph.
    if actual_status == expected_status and actual_status != "pass":
        return False
    return not (
        actual_status in CAPTURE_OPTIONAL_STATUSES or actual_status.startswith("eager_")
    )


# Dynamo metrics tracked against the expected-accuracy baselines, and how each is
# compared. A regression is a move in the "bad" direction:
#   "exact_lower" -- lower-is-better, EXACT match: any change flags (regression
#     when actual > expected, improvement when actual < expected). For small,
#     stable signals (graph breaks, eager fallbacks).
#   "drop_threshold" -- higher-is-better coverage count (ops captured). Only a
#     DROP beyond COVERAGE_DROP_TOL is a regression; equal, small drops, and any
#     increase are PASS. calls_captured is a large, fine-grained count that
#     drifts on routine decomp/lowering changes, so exact-matching it would
#     churn the baselines constantly (and even bump on benign increases) -- the
#     threshold keeps it as a capture-collapse guard without the noise.
# `unique_graphs` is intentionally NOT tracked: a change is ambiguous in both
# directions (fewer graphs can mean graphs were merged / fewer breaks -- an
# improvement -- not less capture; more can mean fragmentation), and it is
# already covered indirectly by graph_breaks (fragmentation) and calls_captured
# (how much is captured). It must be positive when capture is required, unless
# the baseline explicitly expects zero captured ops.
# A metric is only compared when the expected row supplies a value, preserving
# baselines that predate a metric or leave it unavailable for a particular model.
# Keep in sync with METRIC_COLUMNS in ci_expected_accuracy/update_expected.py
# (the baseline writer).
COVERAGE_DROP_TOL = 0.05  # a >5% drop in captured ops is a regression
TRACKED_METRICS = {
    "graph_breaks": "exact_lower",
    "calls_captured": "drop_threshold",  # ops captured
    "fallbacks_to_eager": "exact_lower",
}


def _classify(actual: Any, expected: Any, mode: str) -> str:
    """Return "PASS" / "FAIL" / "IMPROVED" for one metric per its `mode`."""
    if mode == "exact_lower":
        if actual == expected:
            return "PASS"
        return "FAIL" if actual > expected else "IMPROVED"
    # "drop_threshold": higher is better; only a drop past the tolerance is a
    # regression. Increases and small drops are absorbed as PASS so benign
    # drift does not force baseline bumps.
    if actual >= expected:
        return "PASS"
    return "FAIL" if actual < expected * (1 - COVERAGE_DROP_TOL) else "PASS"


def check_graph_breaks(
    actual_csv: pd.DataFrame, expected_csv: pd.DataFrame, expected_filename: str
) -> tuple[list[str], str]:
    failed: list[str] = []
    improved: list[str] = []
    invalid: list[str] = []

    for label, csv in (("actual", actual_csv), ("expected", expected_csv)):
        if csv.empty:
            return [f"{label} CSV"], f"Error: {label} CSV contains no model results."
        required = {"name", "accuracy"}
        if label == "expected":
            required.add("graph_breaks")
        missing = required - set(csv.columns)
        if missing:
            columns = ", ".join(sorted(missing))
            return [f"{label} CSV"], (
                f"Error: {label} CSV is missing required columns: {columns}."
            )

        names = csv["name"]
        for row_number, model in enumerate(names, start=2):
            if not isinstance(model, str) or not model.strip():
                return [f"{label} CSV"], (
                    f"Error: {label} CSV row {row_number} has a missing or invalid "
                    f"model name: {model!r}."
                )
        duplicates = names[names.duplicated(keep=False)].drop_duplicates()
        if not duplicates.empty:
            models = " ".join(duplicates)
            return list(duplicates), (
                f"Error: {label} CSV contains multiple results for: {models}."
            )

    expected_models = set(expected_csv["name"])

    if "rocm" in expected_filename:
        flaky_models.update(
            {
                "alexnet",
                "demucs",
                "densenet121",
                "detectron2_fcos_r_50_fpn",
                "doctr_det_predictor",
                "doctr_reco_predictor",
                "levit_128",
                "llava",
                "microbench_unbacked_tolist_sum",
                "resnet50",
                "resnet152",
                "sam",
                "sam_fast",
                "timm_efficientdet",
                "torchrec_dlrm",
                "vgg16",
                # LLM
                "meta-llama/Llama-3.2-1B",
                "google/gemma-2-2b",
                "google/gemma-3-4b-it",
                "openai/whisper-tiny",
                "Qwen/Qwen3-0.6B",
                "mistralai/Mistral-7B-Instruct-v0.3",
                "openai/gpt-oss-20b",
                # Discovered after gfx950 CI enablement and rocm 7.2
                "mobilenet_v3_large",
                "mnasnet1_0",
            }
        )

    for model in actual_csv["name"]:
        flaky = model in flaky_models

        if model not in expected_models:
            actual_graph_breaks = get_field(actual_csv, model, "graph_breaks")
            print(
                f"{model:34}  {'MISSING:':19} "
                f"graph_breaks={actual_graph_breaks}, expected=None"
            )
            improved.append(model)
            continue

        actual_status = get_field(actual_csv, model, "accuracy")
        expected_status = get_field(expected_csv, model, "accuracy")
        if not isinstance(actual_status, str) or not actual_status.strip():
            print(
                f"{model:34}  {'INVALID:':19} "
                f"accuracy={actual_status!r}, expected a result status"
            )
            invalid.append(model)
            continue
        if not isinstance(expected_status, str) or not expected_status.strip():
            print(
                f"{model:34}  {'INVALID:':19} "
                f"expected accuracy={expected_status!r}, expected a result status"
            )
            invalid.append(model)
            continue

        num_graphs_raw = get_field(actual_csv, model, "unique_graphs")
        num_graphs = _parse_counter(num_graphs_raw)
        capture_required = _capture_required(model, actual_status, expected_status)
        capture_unavailable = num_graphs_raw is None or pd.isna(num_graphs_raw)
        if num_graphs is None and (capture_required or not capture_unavailable):
            print(
                f"{model:34}  {'INVALID:':19} "
                f"unique_graphs={num_graphs_raw!r}, expected a finite nonnegative integer"
            )
            invalid.append(model)
            continue
        expected_ops = _parse_counter(get_field(expected_csv, model, "calls_captured"))
        if capture_required and num_graphs == 0 and expected_ops != 0:
            print(
                f"{model:34}  {'INVALID:':19} "
                "unique_graphs=0, expected Dynamo graph capture"
            )
            invalid.append(model)
            continue
        check_metrics = capture_required or (num_graphs is not None and num_graphs > 0)

        model_failed = False
        model_improved = False
        model_invalid = False
        printed_detail = False
        for field, mode in TRACKED_METRICS.items():
            expected_raw = get_field(expected_csv, model, field)
            expected_missing = expected_raw is None or pd.isna(expected_raw)
            if field != "graph_breaks" and expected_missing:
                continue
            actual_raw = get_field(actual_csv, model, field)
            expected = _parse_counter(expected_raw)
            actual = _parse_counter(actual_raw)
            if expected is None:
                print(
                    f"{model:34}  {'INVALID:':19} "
                    f"expected {field}={expected_raw!r}, expected a finite nonnegative integer"
                )
                model_invalid = True
                continue
            if actual is None:
                if not check_metrics and (actual_raw is None or pd.isna(actual_raw)):
                    continue
                print(
                    f"{model:34}  {'INVALID:':19} "
                    f"{field}={actual_raw!r}, expected a finite nonnegative integer"
                )
                model_invalid = True
                continue
            if not check_metrics:
                continue
            result = _classify(actual, expected, mode)
            if result == "PASS":
                continue
            if flaky:
                status = f"{result}_BUT_FLAKY:"
            else:
                status = f"{result}:"
                if result == "FAIL":
                    model_failed = True
                else:
                    model_improved = True
            print(f"{model:34}  {status:19} {field}={actual}, expected={expected}")
            printed_detail = True

        if model_invalid:
            invalid.append(model)
            continue
        if not check_metrics:
            print(f"{model:34}  EAGER_FAILED")
            continue
        if not printed_detail:
            status = "PASS_BUT_FLAKY" if flaky else "PASS"
            print(f"{model:34}  {status}")
        if model_failed:
            failed.append(model)
        elif model_improved:
            improved.append(model)

    msg = ""
    if invalid or failed or improved:
        if invalid:
            msg += textwrap.dedent(
                f"""
            Error: {len(invalid)} models have missing or invalid required Dynamo metrics:
                {" ".join(invalid)}

            """
            )
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have regressed dynamo metrics (more graph
            breaks / eager fallbacks, or fewer ops / graphs captured):
                {" ".join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have improved dynamo metrics:
                {" ".join(improved)}

            """
            )
        if failed or improved:
            sha = os.getenv("SHA1", "{your CI commit sha}")
            msg += textwrap.dedent(
                f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        from pytorch/pytorch root, run
        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`
        and then `git add` the resulting local changes to expected CSVs to your commit.
        """
            )
    return invalid or failed or improved, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    args = parser.parse_args()

    try:
        actual = pd.read_csv(args.actual, dtype={"name": str})
        expected = pd.read_csv(args.expected, dtype={"name": str})
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        parser.error(str(e))

    failed, msg = check_graph_breaks(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
