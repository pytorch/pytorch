#!/usr/bin/env python3

import os
import shutil
import sys
from subprocess import check_call
from tempfile import TemporaryDirectory

from auditwheel.elfutils import elf_file_filter
from auditwheel.lddtree import lddtree
from auditwheel.patcher import Patchelf
from auditwheel.repair import copylib
from auditwheel.wheeltools import InWheelCtx


def replace_tag(filename):
    with open(filename) as f:
        lines = f.read().split("\\n")
    for i, line in enumerate(lines):
        if not line.startswith("Tag: "):
            continue
        lines[i] = line.replace("-linux_", "-manylinux2014_")
        print(f"Updated tag from {line} to {lines[i]}")

    with open(filename, "w") as f:
        f.write("\\n".join(lines))


class AlignedPatchelf(Patchelf):
    def set_soname(self, file_name: str, new_soname: str) -> None:
        check_call(
            ["patchelf", "--page-size", "65536", "--set-soname", new_soname, file_name]
        )

    def replace_needed(self, file_name: str, soname: str, new_soname: str) -> None:
        check_call(
            [
                "patchelf",
                "--page-size",
                "65536",
                "--replace-needed",
                soname,
                new_soname,
                file_name,
            ]
        )


def embed_library(whl_path, lib_soname, update_tag=False):
    patcher = AlignedPatchelf()
    out_dir = TemporaryDirectory()
    whl_name = os.path.basename(whl_path)
    tmp_whl_name = os.path.join(out_dir.name, whl_name)
    with InWheelCtx(whl_path) as ctx:
        torchlib_path = os.path.join(ctx._tmpdir.name, "torch", "lib")
        ctx.out_wheel = tmp_whl_name
        new_lib_path, new_lib_soname = None, None
        for filename, _ in elf_file_filter(ctx.iter_files()):
            if not filename.startswith("torch/lib"):
                continue
            libtree = lddtree(filename)
            if lib_soname not in libtree["needed"]:
                continue
            lib_path = libtree["libs"][lib_soname]["path"]
            if lib_path is None:
                print(f"Can't embed {lib_soname} as it could not be found")
                break
            if lib_path.startswith(torchlib_path):
                continue

            if new_lib_path is None:
                new_lib_soname, new_lib_path = copylib(lib_path, torchlib_path, patcher)
            patcher.replace_needed(filename, lib_soname, new_lib_soname)
            print(f"Replacing {lib_soname} with {new_lib_soname} for {filename}")
        if update_tag:
            # Add manylinux2014 tag
            for filename in ctx.iter_files():
                if os.path.basename(filename) != "WHEEL":
                    continue
                replace_tag(filename)
    shutil.move(tmp_whl_name, whl_path)


if __name__ == "__main__":
    embed_library(
        sys.argv[1], "libgomp.so.1", len(sys.argv) > 2 and sys.argv[2] == "--update-tag"
    )
