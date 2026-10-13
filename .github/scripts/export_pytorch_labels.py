#!/usr/bin/env python3
"""
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

As a part of enforcing test ownership, we want to maintain a list of existing PyTorch labels
to verify the owners' existence. This script outputs a file containing a list of existing
pytorch/pytorch labels so that the file could be uploaded to S3, or written locally with
--output-file to refresh the snapshot used by the TESTOWNERS linter.

Uploading to S3 assumes the correct env vars are set for AWS permissions.

"""

import json
from typing import Any

from label_utils import gh_get_labels


def parse_args() -> Any:
    from argparse import ArgumentParser

    parser = ArgumentParser("Export PR labels")
    parser.add_argument("org", type=str)
    parser.add_argument("repo", type=str)
    parser.add_argument(
        "--output-file",
        type=str,
        help="write the labels here instead of uploading them to S3",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Exporting labels for {args.org}/{args.repo}")
    labels = json.dumps(sorted(gh_get_labels(args.org, args.repo)), indent=2) + "\n"
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(labels)
        return

    import boto3  # type: ignore[import]

    obj = boto3.resource("s3").Object("ossci-metrics", "pytorch_labels.json")
    obj.put(Body=labels.encode())


if __name__ == "__main__":
    main()
