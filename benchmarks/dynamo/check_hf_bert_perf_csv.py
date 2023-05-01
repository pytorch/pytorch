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
        # Reduce from 1.165 to 1.160, see https://github.com/pytorch/pytorch/issues/96530
        # Reduce from 1.160 to 1.140 after a transformer version upgrade, see https://github.com/pytorch/benchmark/pull/1406
        # The speedup is not backed to 1.16 after the extra graph break issue is fixed in transformer upstream
        if speedup < 1.150:
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
