#!/usr/bin/env python3

"""Updates the default value of opset_version.

The current policy is that the default should be set to the
latest released version as of 18 months ago.

Usage:
Run with no arguments.
"""

import datetime
import os
import pathlib
import re
import sys
import subprocess


pytorch_dir = pathlib.Path(__file__).parent.parent.parent.resolve()
onnx_dir = pytorch_dir / "third_party" / "onnx"
os.chdir(onnx_dir)

date = datetime.datetime.now() - datetime.timedelta(days=18 * 30)
onnx_commit = subprocess.check_output(("git", "log", f"--until={date}", "--max-count=1", "--format=%H"), encoding="utf-8").strip()
onnx_tags = subprocess.check_output(("git", "tag", "--list", f"--contains={onnx_commit}"), encoding="utf-8")
tag_tups = []
semver_pat = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
for tag in onnx_tags.splitlines():
    match = semver_pat.match(tag)
    if match:
        tag_tups.append(tuple(int(x) for x in match.groups()))

min_tup = sorted(tag_tups)[0]
version_str = "{}.{}.{}".format(*min_tup)

print("Using ONNX release", version_str)

head_commit = subprocess.check_output(("git", "log", "--max-count=1", "--format=%H", "HEAD"), encoding="utf-8").strip()

new_default = None

subprocess.check_call(("git", "checkout", f"v{version_str}"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    from onnx import helper  # type: ignore
    for version in helper.VERSION_TABLE:
        if version[0] == version_str:
            new_default = version[2]
            print("found new default opset_version", new_default)
            break
    if not new_default:
        sys.exit(f"failed to find version {version_str} in onnx.helper.VERSION_TABLE at commit {onnx_commit}")
finally:
    subprocess.check_call(("git", "checkout", head_commit), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

os.chdir(pytorch_dir)


def read_sub_write(path: str, prefix_pat: str) -> None:
    with open(path, encoding="utf-8") as f:
        content_str = f.read()
    content_str = re.sub(prefix_pat, r"\g<1>{}".format(new_default), content_str)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content_str)
    print("modified", path)

read_sub_write("torch/onnx/symbolic_helper.py", r"(_default_onnx_opset_version = )\d+")
read_sub_write("torch/onnx/__init__.py", r"(opset_version \(int, default )\d+")
