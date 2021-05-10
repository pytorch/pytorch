import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_a",
        type=str,
        default="input_a.csv",
        help="path to input_a csv"
    )

    parser.add_argument(
        "--input_b",
        type=str,
        default="input_b.csv",
        help="path to input_b csv"
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="diff.csv",
        help="name of the diff csv"
    )

    args = parser.parse_args()

    df_a = pd.read_csv(args.input_a)
    df_b = pd.read_csv(args.input_b)

    cols = df_a.columns

    for row_a, row_b in zip(df_a.iterrows(), df_b.iterrows()):
        i = row_a[0]
        row_a = row_a[1]
        row_b = row_b[1]
        for col in cols[1:]:
            df_a.at[i, col] = df_a.at[i, col] - df_b.at[i, col]

    print(df_a)
    df_a.to_csv(args.output_name, encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
