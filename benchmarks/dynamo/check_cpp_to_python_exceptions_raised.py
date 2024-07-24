import argparse
import os
import sys
import textwrap

import pandas as pd


def get_field(csv, model_name: str, field: str):
    try:
        return csv.loc[csv["name"] == model_name][field].item()
    except Exception as e:
        return None


def check_cpp_to_python_exceptions_raised(actual_csv, expected_csv, expected_filename):
    failed = []
    improved = []

    for model in actual_csv["name"]:
        cpp_to_python_exceptions_raised = get_field(
            actual_csv, model, "cpp_to_python_exceptions_raised"
        )
        expected_cpp_to_python_exceptions_raised = get_field(
            expected_csv, model, "cpp_to_python_exceptions_raised"
        )

        if cpp_to_python_exceptions_raised == expected_cpp_to_python_exceptions_raised:
            status = "PASS"
            print(f"{model:34}  {status}")
            continue

        elif cpp_to_python_exceptions_raised > expected_cpp_to_python_exceptions_raised:
            status = "FAIL:"
            failed.append(model)
        elif cpp_to_python_exceptions_raised < expected_cpp_to_python_exceptions_raised:
            status = "IMPROVED:"
            improved.append(model)
        exp_str = expected_cpp_to_python_exceptions_raised
        act_str = cpp_to_python_exceptions_raised
        print(
            f"{model:34}  {status:9} cpp_to_python_exceptions_raised={act_str}, expected={exp_str}"
        )

    msg = ""
    if failed or improved:
        if failed:
            msg += textwrap.dedent(
                f"""
            Error: {len(failed)} models have new dynamo graph breaks:
                {' '.join(failed)}

            """
            )
        if improved:
            msg += textwrap.dedent(
                f"""
            Improvement: {len(improved)} models have fixed dynamo graph breaks:
                {' '.join(improved)}

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

    failed, msg = check_cpp_to_python_exceptions_raised(actual, expected, args.expected)
    if failed:
        print(msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
