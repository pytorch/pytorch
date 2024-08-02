#!/usr/bin/env python3

"""Updates the default value of opset_version.

The current policy is that the default should be set to the
latest released version as of 18 months ago.

Usage:
Run with no arguments.
"""

import argparse
import datetime
import os
import re
import subprocess
import sys
from pathlib import Path
from subprocess import DEVNULL
from typing import Any


def read_sub_write(path: str, prefix_pat: str, new_default: int) -> None:
    with open(path, encoding="utf-8") as f:
        content_str = f.read()
    content_str = re.sub(prefix_pat, rf"\g<1>{new_default}", content_str)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content_str)
    print("modified", path)


def main(args: Any) -> None:
    pytorch_dir = Path(__file__).parent.parent.parent.resolve()
    onnx_dir = pytorch_dir / "third_party" / "onnx"
    os.chdir(onnx_dir)

    date = datetime.datetime.now() - datetime.timedelta(days=18 * 30)
    onnx_commit = subprocess.check_output(
        ("git", "log", f"--until={date}", "--max-count=1", "--format=%H"),
        encoding="utf-8",
    ).strip()
    onnx_tags = subprocess.check_output(
        ("git", "tag", "--list", f"--contains={onnx_commit}"), encoding="utf-8"
    )
    tag_tups = []
    semver_pat = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
    for tag in onnx_tags.splitlines():
        match = semver_pat.match(tag)
        if match:
            tag_tups.append(tuple(int(x) for x in match.groups()))

    # Take the release 18 months ago
    version_str = "{}.{}.{}".format(*min(tag_tups))

    print("Using ONNX release", version_str)

    head_commit = subprocess.check_output(
        ("git", "log", "--max-count=1", "--format=%H", "HEAD"), encoding="utf-8"
    ).strip()

    new_default = None

    subprocess.check_call(
        ("git", "checkout", f"v{version_str}"), stdout=DEVNULL, stderr=DEVNULL
    )
    try:
        from onnx import helper  # type: ignore[import]

        for version in helper.VERSION_TABLE:
            if version[0] == version_str:
                new_default = version[2]
                print("found new default opset_version", new_default)
                break
        if not new_default:
            sys.exit(
                f"failed to find version {version_str} in onnx.helper.VERSION_TABLE at commit {onnx_commit}"
            )
    finally:
        subprocess.check_call(
            ("git", "checkout", head_commit), stdout=DEVNULL, stderr=DEVNULL
        )

    os.chdir(pytorch_dir)

    read_sub_write(
        os.path.join("torch", "onnx", "_constants.py"),
        r"(ONNX_DEFAULT_OPSET = )\d+",
        new_default,
    )
    read_sub_write(
        os.path.join("torch", "onnx", "utils.py"),
        r"(opset_version \(int, default )\d+",
        new_default,
    )

    if not args.skip_build:
        print("Building PyTorch...")
        subprocess.check_call(
            ("python", "setup.py", "develop"),
        )
    print("Updating operator .expect files")
    subprocess.check_call(
        ("python", os.path.join("test", "onnx", "test_operators.py"), "--accept"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-build",
        "--skip_build",
        action="store_true",
        help="Skip building pytorch",
    )
    main(parser.parse_args())
