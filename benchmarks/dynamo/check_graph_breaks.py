import argparse
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


def get_field(csv: pd.DataFrame, model_name: str, field: str) -> Any | None:
    try:
        return csv.loc[csv["name"] == model_name][field].item()
    except Exception:
        return None


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
# (how much is captured). It is still read from the actual CSV below for the
# dynamo_called check, just not gated.
# A metric is only checked for models whose expected baseline has that column,
# so baselines predating a metric are silently skipped until regenerated.
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
                "stable_diffusion_text_encoder",
                "stable_diffusion_unet",
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
        num_graphs = get_field(actual_csv, model, "unique_graphs")
        dynamo_called = num_graphs is not None and int(num_graphs) != 0

        expected_graph_breaks = get_field(expected_csv, model, "graph_breaks")
        flaky = model in flaky_models

        # graph_breaks is the primary metric: if the model has no baseline
        # graph_breaks entry it is missing from the baseline entirely.
        if expected_graph_breaks is None:
            actual_graph_breaks = get_field(actual_csv, model, "graph_breaks")
            print(
                f"{model:34}  {'MISSING:':19} "
                f"graph_breaks={actual_graph_breaks}, expected=None"
            )
            improved.append(model)
            continue
        if not dynamo_called:
            print(f"{model:34}  EAGER_FAILED")
            continue

        model_failed = False
        model_improved = False
        printed_detail = False
        for field, mode in TRACKED_METRICS.items():
            expected = get_field(expected_csv, model, field)
            actual = get_field(actual_csv, model, field)
            # Skip a metric absent from this baseline or the output, or whose
            # cell is unreadable (NaN) -- get_field returns NaN for a
            # present-but-empty cell, which must not be treated as a value.
            if (
                expected is None
                or actual is None
                or pd.isna(expected)
                or pd.isna(actual)
            ):
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

        if not printed_detail:
            status = "PASS_BUT_FLAKY" if flaky else "PASS"
            print(f"{model:34}  {status}")
        if model_failed:
            failed.append(model)
        elif model_improved:
            improved.append(model)

    msg = ""
    if failed or improved:
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
        sha = os.getenv("SHA1", "{your CI commit sha}")
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        from pytorch/pytorch root, run
        `python benchmarks/dynamo/ci_expected_accuracy/update_expected.py {sha}`
        and then `git add` the resulting local changes to expected CSVs to your commit.
        """
        )
    return failed or improved, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    args = parser.parse_args()

    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)

    failed, msg = check_graph_breaks(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
