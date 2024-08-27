import glob
import json
import os
import shlex
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch._inductor
import torch.utils._pytree as pytree
from torch._inductor import config, exc
from torch._inductor.cpp_builder import BuildOptionsBase, CppBuilder
from torch.export._tree_utils import reorder_kwargs

from .build_package import build_package_contents
from .pt2_archive_constants import AOTINDUCTOR_DIR, ARCHIVE_VERSION


class PT2ArchiveWriter:
    def __init__(self, archive_path: str) -> None:
        self.archive_path: str = archive_path
        self.archive_file: Optional[zipfile.ZipFile] = None

    def __enter__(self) -> "PT2ArchiveWriter":
        assert self.archive_file is None
        self.archive_file = zipfile.ZipFile(
            self.archive_path, "w", compression=zipfile.ZIP_STORED
        )
        self.writestr("version", str(ARCHIVE_VERSION))
        self.writestr("archive_format", "pt2")
        return self

    def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
        assert self.archive_file is not None
        self.archive_file.close()
        self.archive_file = None
        return None

    def writestr(self, name: str, data: Union[bytes, str]) -> None:
        assert self.archive_file is not None
        self.archive_file.writestr(name, data)

    def write_file(self, name: str, file_path: str) -> None:
        """
        Copy a file into the archive.
        name: The destination file inside the archive.
        file_path: The source file on disk.
        """
        assert Path(file_path).is_file(), f"{file_path} is not a valid file path"
        assert self.archive_file is not None
        self.archive_file.write(file_path, arcname=name)


class PT2ArchiveReader:
    def __init__(self, archive_path: str) -> None:
        self.archive_path: str = archive_path
        self.archive_file: Optional[zipfile.ZipFile] = None

    def __enter__(self) -> "PT2ArchiveReader":
        self.archive_file = zipfile.ZipFile(
            self.archive_path, "r", compression=zipfile.ZIP_STORED
        )
        return self

    def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
        if self.archive_file is not None:
            self.archive_file.close()
        return None

    def read(self, name: str) -> bytes:
        assert self.archive_file is not None
        return self.archive_file.read(name)

    def extract_to_path(self, member: str, path: str) -> str:
        assert self.archive_file is not None
        return self.archive_file.extract(member, path)

    def extractall(self, path: str) -> None:
        assert self.archive_file is not None
        self.archive_file.extractall(path)

    def get_file_names(self) -> List[str]:
        assert self.archive_file is not None
        return self.archive_file.namelist()


def _run_command_and_check(cmd: str) -> None:
    cmd = shlex.split(cmd)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise exc.CppCompileError(cmd, e.output) from e


def compile_so(aoti_dir: str, aoti_files: List[str], so_path: str) -> str:
    def get_aoti_file_with_suffix(suffix: str) -> str:
        for file in aoti_files:
            if file.endswith(suffix):
                return file
        raise RuntimeError(f"Unable to find file with suffix {suffix}")

    # Compile all the files into a .so
    cpp_file = os.path.join(aoti_dir, get_aoti_file_with_suffix(".cpp"))
    consts_o = os.path.join(aoti_dir, get_aoti_file_with_suffix(".o"))

    file_name = os.path.splitext(cpp_file)[0]

    # Parse compile flags and build the .o file
    with open(file_name + "_compile_flags.json") as f:
        compile_flags = json.load(f)

    compile_options = BuildOptionsBase(**compile_flags)
    object_builder = CppBuilder(
        name=file_name,
        sources=cpp_file,
        BuildOption=compile_options,
    )
    compile_cmd = object_builder.get_command_line()
    output_o = object_builder.get_target_file_path()

    _run_command_and_check(compile_cmd)

    # Parse linker flags and build the .so file
    with open(file_name + "_linker_flags.json") as f:
        linker_flags = json.load(f)

    linker_options = BuildOptionsBase(**linker_flags)
    so_builder = CppBuilder(
        name=os.path.split(so_path)[-1],
        sources=[output_o, consts_o],
        BuildOption=linker_options,
        output_dir=so_path,
    )
    link_cmd = so_builder.get_command_line()
    output_so = so_builder.get_target_file_path()

    _run_command_and_check(link_cmd)

    # mmapped weights
    serialized_weights_filename = file_name + "_serialized_weights.bin"
    if serialized_weights_filename in aoti_files:
        with open(serialized_weights_filename, "rb") as f_weights:
            serialized_weights = f_weights.read()

        with open(output_so, "a+b") as f_so:
            so_size = f_so.tell()
            # Page align the weights
            f_so.write(b" " * (16384 - so_size % 16384))
            f_so.write(serialized_weights)

    return output_so


def package_aoti(aoti_output_dir: str) -> str:
    """
    Saves the AOTInductor generated files to the PT2Archive format.
    """

    # Add a makefile and python script
    build_package_filename = "build_package.py"
    with open(os.path.join(aoti_output_dir, build_package_filename), "w") as f:
        f.write(build_package_contents)

    with open(os.path.join(aoti_output_dir, "Makefile"), "w") as f:
        f.write(f"all:\n\tpython3 {build_package_filename}\n")

    if config.aot_inductor.output_path.endswith(".so"):
        raise RuntimeError(
            "Unable to save package as a .so. It should be a .pt2 format or a directory."
        )
    elif config.aot_inductor.output_path.endswith(".pt2"):
        # Save using the PT2 packaging format
        # (https://docs.google.com/document/d/1jLPp8MN8Whs0-VW9PmJ93Yg02W85tpujvHrTa1pc5x8/edit#heading=h.v2y2jgnwc56a)
        archive_path = config.aot_inductor.output_path

        with PT2ArchiveWriter(archive_path) as archive_writer:
            package_files = glob.glob(f"{aoti_output_dir}/*")

            for path in package_files:
                filename = os.path.basename(path)
                archive_writer.write_file(f"{AOTINDUCTOR_DIR}{filename}", path)

        return archive_path

    else:
        # Directly put the files into the directory, without any archiving
        return aoti_output_dir


def load_package(path: str, device: str) -> Callable:  # type: ignore[type-arg]
    if path.endswith(".so"):
        raise RuntimeError(
            "Unable to load .so. It should be a .pt2 format or a directory."
        )

    elif path.endswith(".pt2"):
        so_path = os.path.splitext(path)[0]
        with PT2ArchiveReader(path) as archive_reader:
            file_names = archive_reader.get_file_names()

            with tempfile.TemporaryDirectory() as tmp_dir:
                archive_reader.extractall(tmp_dir)
                file_names = archive_reader.get_file_names()
                aoti_files = [
                    file for file in file_names if file.startswith(AOTINDUCTOR_DIR)
                ]

                so_path = compile_so(tmp_dir, aoti_files, so_path)

    else:
        assert os.path.isdir(path), "Must specify a directory or a .pt2 file"
        aoti_files = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(path)
            for file in files
        ]
        so_path = compile_so(path, aoti_files, path)

    if device == "cpu":
        runner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)  # type: ignore[call-arg]
    elif device == "cuda" or device.startswith("cuda:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)  # type: ignore[assignment, call-arg]
    else:
        raise RuntimeError("Unsupported device " + device)

    def optimized(*args, **kwargs):  # type: ignore[no-untyped-def]
        call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized
