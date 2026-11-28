import argparse
import math
import sys
import textwrap

import pandas as pd


def _read_results(filename):
    try:
        return pd.read_csv(filename, dtype={"name": str})
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: {filename} contains no benchmark results")
        sys.exit(1)


def _valid_compression_ratio(value):
    if pd.api.types.is_bool(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def main(args):
    actual = _read_results(args.actual)
    expected = _read_results(args.expected)

    for filename, results in ((args.actual, actual), (args.expected, expected)):
        if results.empty:
            print(f"Error: {filename} contains no benchmark results")
            sys.exit(1)
        missing_columns = {"name", "compression_ratio"} - set(results.columns)
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            print(f"Error: {filename} is missing required column(s): {columns}")
            sys.exit(1)

    failed = []
    invalid = []

    for row_number, (_, row) in enumerate(actual.iterrows(), start=2):
        name = row["name"]
        if pd.isna(name) or not name.strip():
            invalid.append(
                f"row {row_number} in {args.actual} has a missing model name"
            )
            continue

        actual_value = row["compression_ratio"]
        actual_memory_compression = _valid_compression_ratio(actual_value)
        if actual_memory_compression is None:
            invalid.append(
                f"{name} has invalid compression_ratio={actual_value!r} "
                f"in {args.actual}; expected a finite value greater than zero"
            )
            continue

        # Extra baseline models are allowed; each actual row needs one baseline.
        expected_rows = expected.loc[expected["name"] == name, "compression_ratio"]
        if len(expected_rows) != 1:
            if expected_rows.empty:
                invalid.append(f"{name} is missing from {args.expected}")
            else:
                invalid.append(f"{name} has multiple entries in {args.expected}")
            continue

        expected_value = expected_rows.iloc[0]
        expected_memory_compression = _valid_compression_ratio(expected_value)
        if expected_memory_compression is None:
            invalid.append(
                f"{name} has invalid compression_ratio={expected_value!r} "
                f"in {args.expected}; expected a finite value greater than zero"
            )
            continue

        if actual_memory_compression >= expected_memory_compression * 0.95:
            status = "PASS"
        else:
            status = "FAIL"
            failed.append(name)
        print(
            f"""
            {name:34}:
                actual_memory_compression={actual_memory_compression:.2f},
                expected_memory_compression={expected_memory_compression:.2f},
                {status}
            """
        )

    if invalid:
        print("Error: invalid memory compression result(s):")
        for message in invalid:
            print(f"  - {message}")
        sys.exit(1)

    if failed:
        print(
            textwrap.dedent(
                f"""
                Error: {len(failed)} models below expected memory compression ratio:
                    {" ".join(failed)}
                If this drop is expected, you can update `{args.expected}`.
                """
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True)
    parser.add_argument("--expected", type=str, required=True)
    args = parser.parse_args()
    main(args)
