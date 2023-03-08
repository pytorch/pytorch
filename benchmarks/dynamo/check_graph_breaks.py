import argparse
import sys
import textwrap

import pandas as pd


def get_field(csv, model_name: str, field: str, typ=float):
    return typ(csv.loc[csv["name"] == model_name][field])


def main(args):
    actual = pd.read_csv(args.actual)
    expected = pd.read_csv(args.expected)
    failed = []
    improved = []

    for model in actual["name"]:

        graph_breaks = get_field(actual, model, "graph_breaks", typ=int)
        expected_graph_breaks = get_field(expected, model, "graph_breaks", typ=int)

        if graph_breaks == expected_graph_breaks:
            status = "PASS"
        elif graph_breaks > expected_graph_breaks:
            status = "FAIL"
            failed.append(model)
        elif graph_breaks < expected_graph_breaks:
            status = "IMPROVED"
            improved.append(model)
        print(
            f"""
            {model:34}:
                graph_breaks={graph_breaks},
                expected_graph_breaks={expected_graph_breaks},
                {status}
            """
        )

    if failed or improved:
        msg = ""
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
        msg += textwrap.dedent(
            f"""
        If this change is expected, you can update `{args.expected}` to reflect the new baseline.
        This can either be done manually, or by downloading artifacts from your PR CI job."
        """
        )

        print(msg)
        sys.exit(1)


parser = argparse.ArgumentParser()
parser.add_argument("--actual", type=str, required=True)
parser.add_argument("--expected", type=str, required=True)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)
