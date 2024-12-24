import argparse
import os
import sys
import textwrap

import pandas as pd


def get_field(csv, case: str, field: str):
    try:
        return csv.loc[csv["Case Name"] == case][field].item()
    except Exception as e:
        return None


def check_perf(actual_csv, expected_csv, expected_filename, threshold):
    failed = []
    improved = []

    for case in actual_csv["Case Name"]:
        perf = get_field(actual_csv, case, "Execution Time")
        expected_perf = get_field(expected_csv, case, "Execution Time")

        if expected_perf is None:
            status = "FAIL"
            print(
                f"\n{case:34}  {status:9} Not Found. if it is expected, \
                you can update in {expected_filename} to reflect the new module. "
            )
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
    if failed or improved:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have performance status regressed:
                {' '.join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have performance status improved:
                {' '.join(improved)}

            """
            )
        sha = os.getenv("SHA1", "{your CI commit sha}")
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{expected_filename}` to reflect the new baseline.
        """
        )
    return failed or improved, msg


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
