# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

from . import BENCHMARK_DATA_DIR


def decompress_s3_data(s3_tarball_path: Path):
    assert str(s3_tarball_path.absolute()).endswith(".tar.gz"), (
        f"Expected .tar.gz file path but got {s3_tarball_path}."
    )
    import tarfile

    # Hide decompressed file in .data directory so that they won't be checked in
    decompress_dir = os.path.join(
        BENCHMARK_DATA_DIR, s3_tarball_path.name.removesuffix(".tar.gz")
    )

    os.makedirs(decompress_dir, exist_ok=True)
    print(f"Decompressing input tarball: {s3_tarball_path}...", end="", flush=True)
    tar = tarfile.open(s3_tarball_path)
    tar.extractall(path=decompress_dir)
    tar.close()
    print("OK")


def checkout_s3_data(name: str, decompress: bool = True):
    S3_URL_BASE = "https://ossci-datasets.s3.amazonaws.com/torchbench"
    download_dir = Path(BENCHMARK_DATA_DIR)
    download_dir.mkdir(parents=True, exist_ok=True)
    import requests

    full_path = download_dir.joinpath(name)
    s3_url = f"{S3_URL_BASE}/traces/{name}"
    r = requests.get(s3_url, allow_redirects=True)
    with open(str(full_path.absolute()), "wb") as output:
        print(f"Checking out {s3_url} to {full_path}")
        output.write(r.content)
    if decompress:
        decompress_s3_data(full_path)
