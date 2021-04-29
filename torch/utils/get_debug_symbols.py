import subprocess
import zipfile
import io
import os
import sys
import urllib
from urllib import parse
from importlib import metadata
import argparse


import torch


if sys.version_info < (3, 8):
    raise RuntimeError("Python 3.8+ required")


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


def get_crc32(file: str) -> str:
    # TODO: Implement this without dropping to Linux-specific shell commands
    proc = subprocess.run(
        f"objcopy --dump-section .gnu_debuglink=>(tail -c4 | od -t x4 -An | xargs echo) {file}",
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
    )
    return proc.stdout.decode().strip()


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


def install_libtorch() -> None:
    lib_path = os.path.join(torch.__path__[0], "lib")  # type: ignore[attr-defined]
    libtorch_path = os.path.join(lib_path, "libtorch_cpu.so")

    if torch.version.debug:
        raise RuntimeError(
            "No need to get debug info, 'torch' is already built in debug mode"
        )

    if not os.path.exists(libtorch_path):
        raise RuntimeError(f"Could not find libtorch_cpu.so at {libtorch_path}")

    crc32 = get_crc32(libtorch_path)
    # TODO: Remove this once nightly wheels actually have the .gnu_debuglink section
    crc32 = "3b1333e1"
    if crc32 == "":
        raise RuntimeError(f"Could not get CRC32 from {libtorch_path}")

    link = get_link(crc32)
    zip_bytes = download_url(link)

    print("Unzipping debug file")
    zip_file = zipfile.ZipFile(zip_bytes)
    debug_so = zip_file.open("debug/libtorch_cpu.so.dbg").read()

    # Create destination folder (gdb will automatically look here for
    # libtorch_cpu.so.dbg
    debug_path = os.path.join(lib_path, ".debug")
    if not os.path.exists(debug_path):
        os.mkdir(debug_path)

    debug_so_path = os.path.join(debug_path, "libtorch_cpu.so.dbg")
    print(f"Installing to {debug_so_path}")
    with open(debug_so_path, "wb") as debug_so_file:
        debug_so_file.write(debug_so)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch libtorch_cpu debug information and install it alongside existing torch libs."
    )
    args = parser.parse_args()

    install_libtorch()
