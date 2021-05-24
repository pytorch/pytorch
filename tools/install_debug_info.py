#!/usr/bin/env python3

import subprocess
import zipfile
import tempfile
import io
import os
import urllib
from pathlib import Path
from urllib import parse
from importlib import metadata
import argparse


import torch


def chunk_size() -> int:
    try:
        with open("/proc/sys/net/core/rmem_default") as f:
            return int(f.read())
    except Exception:
        return 1024 * 1024


def download_url(url: str, progress: bool = True) -> io.BytesIO:
    print(f"Downloading {url}")
    buf = io.BytesIO()
    with urllib.request.urlopen(url) as Response:  # type: ignore[attr-defined]
        content_length = Response.getheader("content-length")
        block_size = chunk_size()

        if content_length:
            content_length = int(content_length)
            block_size = max(block_size, content_length // 5)

        size = 0
        while True:
            new_chunk = Response.read(block_size)
            if not new_chunk:
                break
            buf.write(new_chunk)
            size += len(new_chunk)
            if progress and content_length:
                Percent = int((size / content_length) * 100)
                print(f"\tdownloaded: {Percent}%")

    return buf


def get_crc32(file: Path) -> str:
    # TODO: Implement this without dropping to Linux-specific shell commands
    with tempfile.NamedTemporaryFile() as f:
        subprocess.run(
            f"objcopy --dump-section .gnu_debuglink={f.name} {file}",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        )
        with open(f.name, "rb") as f:
            debuglink_bytes = f.read()
        if len(debuglink_bytes) <= 4:
            raise RuntimeError(f"Could not objcopy out .gnu_debuglink section from {file}")

        return bytes(reversed(debuglink_bytes[-4:])).hex()


def get_link(crc32: str) -> str:
    pip_version = metadata.version("torch")  # type: ignore[attr-defined]
    if ".dev" in pip_version:
        # nightly release
        encoded = parse.quote_plus(pip_version)
        base = "https://download.pytorch.org/libtorch/nightly/cpu/debug-libtorch-cxx11-abi-shared-with-deps-"
        url = f"{base}{encoded}-{crc32}.zip"
        return url
    else:
        raise RuntimeError(
            f"Did not detect {pip_version} as a nightly release, and only nightlies are supported right now"
        )


def install_libtorch_debuginfo() -> None:
    lib_path = Path(torch.__path__[0]) / "lib"  # type: ignore[attr-defined]
    libtorch_path = lib_path / "libtorch_cpu.so"

    if torch.version.debug:
        raise RuntimeError(
            "No need to get debug info, 'torch' is already built in debug mode"
        )

    if not libtorch_path.exists():
        raise RuntimeError(f"Could not find libtorch_cpu.so at {libtorch_path}")

    crc32 = get_crc32(libtorch_path)
    if crc32 == "":
        raise RuntimeError(f"Could not get CRC32 from {libtorch_path}")

    # link = get_link(crc32)
    # zip_bytes = download_url(link)
    zip_bytes = io.BytesIO(open("temp.bin", "rb").read())

    print("Unzipping debug info")
    zip_file = zipfile.ZipFile(zip_bytes)
    debug_so = zip_file.open("tmp/debug/libtorch_cpu.so.dbg").read()

    # Create destination folder (gdb will automatically look here for
    # libtorch_cpu.so.dbg
    debug_path = lib_path / ".debug"
    if not debug_path.exists():
        os.mkdir(debug_path)

    debug_so_path = debug_path / "libtorch_cpu.so.dbg"
    print("Installing debug info")
    with open(debug_so_path, "wb") as f:
        f.write(debug_so)

    print(f"Successfully installed debug info to '{debug_so_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch libtorch_cpu debug information and install it alongside existing torch libs. Once installed,"
                    " gdb will automatically detect the debuginfo and associate the symbols and line information when"
                    " debugging a program that uses libtorch_cpu.so"
    )
    args = parser.parse_args()

    install_libtorch_debuginfo()
