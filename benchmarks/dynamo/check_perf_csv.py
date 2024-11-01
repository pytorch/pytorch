import argparse
import sys
import textwrap

import pandas as pd


def check_perf_csv(filename, threshold, threshold_scale):
    """
    Basic performance checking.
    """

    df = pd.read_csv(filename)

    failed = []
    for _, row in df.iterrows():
        model_name = row["name"]
        speedup = row["speedup"]
        if speedup < threshold * threshold_scale:
            failed.append(model_name)

        print(f"{model_name:34} {speedup}")

    if failed:
        print(
            textwrap.dedent(
                f"""
                Error {len(failed)} models performance regressed
                    {' '.join(failed)}
                """
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", type=str, help="csv file name")
    parser.add_argument(
        "--threshold", "-t", type=float, help="threshold speedup value to check against"
    )
    parser.add_argument(
        "--threshold-scale",
        "-s",
        type=float,
        default=1.0,
        help="multiple threshold by this value to relax the check",
    )
    args = parser.parse_args()
    check_perf_csv(args.file, args.threshold, args.threshold_scale)
