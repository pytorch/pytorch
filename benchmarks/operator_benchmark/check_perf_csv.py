import argparse
import sys
import textwrap

import pandas as pd


SKIP_TEST_LISTS = [
    # https://github.com/pytorch/pytorch/issues/143852
    "channel_shuffle_batch_size4_channels_per_group64_height64_width64_groups4_channel_lastTrue",
    "batchnorm_N3136_C256_cpu_trainingTrue_cudnnFalse",
    "index_add__M256_N512_K1_dim1_cpu_dtypetorch.float32",
    "interpolate_input_size(1,3,600,400)_output_size(240,240)_channels_lastTrue_modelinear",
    "original_kernel_tensor_N1_C3_H512_W512_zero_point_dtypetorch.int32_nbits4_cpu",
    "original_kernel_tensor_N1_C3_H512_W512_zero_point_dtypetorch.int32_nbits8_cpu",
]


def get_field(csv, case: str, field: str):
    try:
        return csv.loc[csv["Case Name"] == case][field].item()
    except Exception:
        return None


def check_perf(actual_csv, expected_csv, expected_filename, threshold):
    failed = []
    improved = []
    baseline_not_found = []

    actual_csv = actual_csv[~actual_csv["Case Name"].isin(set(SKIP_TEST_LISTS))]

    for case in actual_csv["Case Name"]:
        perf = get_field(actual_csv, case, "Execution Time")
        expected_perf = get_field(expected_csv, case, "Execution Time")

        if expected_perf is None:
            status = "Baseline Not Found"
            print(f"{case:34}  {status}")
            baseline_not_found.append(case)
            continue

        speed_up = expected_perf / perf

        if (1 - threshold) <= speed_up < (1 + threshold):
            status = "PASS"
            print(f"{case:34}  {status}")
            continue
        elif speed_up >= 1 + threshold:
            status = "IMPROVED:"
            improved.append(case)
        else:
            status = "FAILED:"
            failed.append(case)
        print(f"{case:34}  {status:9} perf={perf}, expected={expected_perf}")

    msg = ""
    if failed or improved or baseline_not_found:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have performance status regressed:
                {" ".join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have performance status improved:
                {" ".join(improved)}

            """
            )

        if baseline_not_found:
            msg += textwrap.dedent(
                f"""
            Baseline Not Found: {len(baseline_not_found)} models don't have the baseline data:
                {" ".join(baseline_not_found)}

            """
            )

        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        """
        )
    return failed or improved or baseline_not_found, msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="threshold to define regression/improvement",
    )
    args = parser.parse_args()

    actual = pd.read_csv(args.actual)
    actual.drop_duplicates(subset=["Case Name"], keep="first", inplace=True)
    expected = pd.read_csv(args.expected)

    failed, msg = check_perf(actual, expected, args.expected, args.threshold)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
