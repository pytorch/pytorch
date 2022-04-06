#!/usr/bin/env python3
'''
Test ownership was introduced in https://github.com/pytorch/pytorch/issues/66232.

This lint verifies that every Python test file (file that matches test_*.py or *_test.py in the test folder)
has valid ownership information in a comment header. Valid means:
  - The format of the header follows the pattern "# Owner(s): ["list", "of owner", "labels"]
  - Each owner label actually exists in PyTorch
  - Each owner label starts with "module: " or "oncall: " or is in ACCEPTABLE_OWNER_LABELS

This file is expected to run in the root directory of pytorch/pytorch.
'''
import boto3  # type: ignore[import]
import botocore  # type: ignore[import]
import fnmatch
import json
import sys
from pathlib import Path
from typing import List, Any


# Team/owner labels usually start with "module: " or "oncall: ", but the following are acceptable exceptions
ACCEPTABLE_OWNER_LABELS = ["NNC", "high priority"]
GLOB_EXCEPTIONS = [
    "**/test/run_test.py"
]

PYTORCH_ROOT = Path(__file__).resolve().parent.parent.parent
TEST_DIR = PYTORCH_ROOT / "test"
CURRENT_FILE_NAME = Path(__file__).resolve().relative_to(PYTORCH_ROOT)

S3_RESOURCE_READ_ONLY = boto3.resource("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))


def get_all_test_files() -> List[Path]:
    test_files = list(TEST_DIR.glob("**/test_*.py"))
    test_files.extend(list(TEST_DIR.glob("**/*_test.py")))
    return [f for f in test_files if not any([fnmatch.fnmatch(str(f), g) for g in GLOB_EXCEPTIONS])]


def get_pytorch_labels() -> Any:
    bucket = S3_RESOURCE_READ_ONLY.Bucket("ossci-metrics")
    summaries = bucket.objects.filter(Prefix="pytorch_labels.json")
    for summary in summaries:
        labels = summary.get()["Body"].read()
    return json.loads(labels)


# Returns a string denoting the error invalidating the label OR an empty string if nothing is wrong
def validate_label(label: str, pytorch_labels: List[str]) -> str:
    if label not in pytorch_labels:
        return f"{label} is not a PyTorch label (please choose from https://github.com/pytorch/pytorch/labels)"
    if label.startswith("module:") or label.startswith("oncall:") or label in ACCEPTABLE_OWNER_LABELS:
        return ""
    return f"{label} is not an acceptable owner (please update to another label or edit ACCEPTABLE_OWNERS_LABELS " \
        "in {CURRENT_FILE_NAME}"


# Returns a string denoting the error invalidating the file OR an empty string if nothing is wrong
def validate_file(filename: Path, pytorch_labels: List[str]) -> str:
    prefix = "# Owner(s): "
    relative_name = Path(filename).relative_to(PYTORCH_ROOT)
    with open(filename) as f:
        for line in f.readlines():
            if line.startswith(prefix):
                labels = json.loads(line[len(prefix):])
                labels_msgs = [validate_label(label, pytorch_labels) for label in labels]
                file_msg = ", ".join([x for x in labels_msgs if x != ""])
                return f"{relative_name}: {file_msg}" if file_msg != "" else ""
    return f"{relative_name}: missing a comment header with ownership information."


def main() -> None:
    test_file_paths = get_all_test_files()
    pytorch_labels = get_pytorch_labels()

    file_msgs = [validate_file(f, pytorch_labels) for f in test_file_paths]
    err_msg = "\n".join([x for x in file_msgs if x != ""])
    if err_msg != "":
        err_msg = err_msg + "\n\nIf you see files with missing ownership information above, " \
            "please add the following line\n\n# Owner(s): [\"<owner: label>\"]\n\nto the top of each test file. " \
            "The owner should be an existing pytorch/pytorch label."
        print(err_msg)
        sys.exit(1)


if __name__ == '__main__':
    main()
