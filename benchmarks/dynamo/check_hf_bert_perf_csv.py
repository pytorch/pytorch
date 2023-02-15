import argparse
import sys
import textwrap

import pandas as pd


def check_hf_bert_perf_csv(filename):
    """
    Basic performance checking.
    """

    df = pd.read_csv(filename)

    failed = []
    for _, row in df.iterrows():
        model_name = row["name"]
        speedup = row["speedup"]
        # Reduced from 1.19 to 1.17, see https://github.com/pytorch/pytorch/issues/94687
        # Reduce further to 1.165 due to runner and run to run variances
        if speedup < 1.165:
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
    args = parser.parse_args()
    check_hf_bert_perf_csv(args.file)
