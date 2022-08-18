#!/usr/bin/env python3

from typing import Any


def parse_args() -> Any:
    from argparse import ArgumentParser
    parser = ArgumentParser("Filter all test configurations and keep only requested ones")
    parser.add_argument("--test-matrix", type=str, required=True, help="the original test matrix")
    parser.add_argument("--labels", type=str, help="the list of labels from the PR")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The original test matrix set by the workflow
    test_matrix = args.test_matrix
    labels = args.labels

    print(test_matrix)
    print("====")
    print(labels)

    # DEBUG
    print(f"::set-output name=test-matrix::{test_matrix}")


if __name__ == "__main__":
    main()
