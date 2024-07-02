import glob
import io
import json
import os
import re
import shlex
import subprocess
import tempfile
import zipfile
from typing import Any, Callable, Dict, List, Union

import torch
import torch._inductor
import torch.utils._pytree as pytree
from torch._inductor import config, exc
from torch._utils_internal import get_file_path_2
from torch.export._tree_utils import reorder_kwargs


ARCHIVE_VERSION = 0

with open(
    get_file_path_2(os.path.dirname(__file__), "pt2_archive_constants.json")
) as f:
    PT2_ARCHIVE_CONSTANTS = json.load(f)

AOTINDUCTOR_DIR = PT2_ARCHIVE_CONSTANTS["AOTINDUCTOR_DIR"]
CONSTANTS_DIR = PT2_ARCHIVE_CONSTANTS["CONSTANTS_DIR"]


class PT2ArchiveWriter:
    def __init__(self, archive_path: str):
        self.archive_file = zipfile.ZipFile(archive_path, "w")
        self.write_string("version", str(ARCHIVE_VERSION))
        self.write_string("archive_format", "pt2")

    def __enter__(self) -> "PT2ArchiveWriter":
        return self

    def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
        pass

    def write_string(self, name: str, data: Union[bytes, str]) -> None:
        self.archive_file.writestr(name, data)

    def write_file(self, name: str, file_path: str) -> None:
        """
        Copy a file into the archive.
        name: The destination file inside the archive.
        file_path: The source file on disk.
        """
        assert os.path.isfile(file_path), f"{file_path} is not a valid file path"
        self.archive_file.write(file_path, arcname=name)


class PT2ArchiveReader:
    def __init__(self, archive_path: str):
        self.archive_file = zipfile.ZipFile(archive_path, "r")

    def __enter__(self) -> "PT2ArchiveReader":
        return self

    def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
        pass

    def read_string(self, name: str) -> bytes:
        return self.archive_file.read(name)

    def extract_to_path(self, member: str, path: str) -> str:
        return self.archive_file.extract(member, path)

    def extractall(self, path: str) -> None:
        self.archive_file.extractall(path)

    def get_file_names(self) -> List[str]:
        return self.archive_file.namelist()


def _run_command_and_check(cmd: str) -> None:
    cmd = shlex.split(cmd)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise exc.CppCompileError(cmd, e.output) from e


def compile_so(archive_path: str, device: str, so_path: str) -> str:  # type: ignore[type-arg]
    with PT2ArchiveReader(archive_path) as archive_reader:
        file_names = archive_reader.get_file_names()
        aoti_files = [file for file in file_names if file.startswith(AOTINDUCTOR_DIR)]

        def get_aoti_file_with_suffix(suffix: str) -> str:
            for file in aoti_files:
                if file.endswith(suffix):
                    return file
            raise RuntimeError(f"Unable to find file with suffix {suffix}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_reader.extractall(tmp_dir)

            # Compile all the files into a .so
            build_command = (
                "$CPP_COMPILER $INP_FILES $SHARED $WARNING_ALL_FLAG $CPP_FLAGS "
                "$GLIBCXX_ABI_BUILD_FLAGS $IPATHS $LPATHS $LIBS $BUILD_ARCH_FLAGS "
                "$MACROS $LINKER_PATHS $CLANG_FLAGS $OPTIMIZATION_FLAGS "
                "$CPP_WRAPPER_FLAGS $CUSTOM_GENERATED_MACROS "
                "$STANDARD_SYS_DIR_HEADERS $COMPILE_ONLY $EXTRA_FLAGS -o $OUT_FILES"
            )

            cpp_file = os.path.join(tmp_dir, get_aoti_file_with_suffix(".cpp"))
            consts_o = os.path.join(tmp_dir, get_aoti_file_with_suffix(".o"))

            output_o = os.path.splitext(cpp_file)[0] + ".o"

            # Parse compile flags and build the .o file
            with open(os.path.splitext(cpp_file)[0] + "_compile_flags") as f:
                compile_flags = {}
                for line in f:
                    match = re.match(r'(\w+)\s*=\s*"([^"]*)"', line)
                    assert match is not None
                    assert len(match.groups()) == 2, f"Failed to parse {line}"
                    compile_flags[match.group(1)] = match.group(2)

            compile_flags["INP_FILES"] = cpp_file
            compile_flags["OUT_FILES"] = output_o
            compile_command = re.sub(
                r"\$(\w+)", lambda m: compile_flags[m.group(1)], build_command
            )
            _run_command_and_check(compile_command)

            # Parse linker flags and build the .so file
            with open(os.path.splitext(cpp_file)[0] + "_linker_flags") as f:
                linker_flags = {}
                for line in f:
                    match = re.match(r'(\w+)\s*=\s*"([^"]*)"', line)
                    assert match is not None
                    assert len(match.groups()) == 2, f"Failed to parse {line}"
                    linker_flags[match.group(1)] = match.group(2)

            linker_flags["INP_FILES"] = f"{output_o} {consts_o}"
            linker_flags["OUT_FILES"] = so_path
            linker_command = re.sub(
                r"\$(\w+)", lambda m: linker_flags[m.group(1)], build_command
            )
            _run_command_and_check(linker_command)

            # TODO: load constants w/ mmap
            # tmp_file = os.path.join(tmp_dir, os.path.join(tmp_dir, CONSTANTS_DIR, "constants.pt"))
            # constants = torch.load(tmp_file, mmap=True)
            # print(constants.keys())

            # serialized_weights = b"".join(
            #     _to_bytes(constant) for constant in constants.values()
            # )

            # with open(output_so, "a+b") as f_so:
            #     so_size = f_so.tell()
            #     # Page align the weights
            #     f_so.write(b" " * (16384 - so_size % 16384))
            #     f_so.write(serialized_weights)
            #     f_so.write(struct.pack("q", magic_number))

    return so_path


def package_aoti(aoti_output_dir: str, constants: Dict[str, Any]) -> str:
    """
    Saves the AOTInductor generated files to the PT2Archive format.
    """
    if config.aot_inductor.output_path.endswith(".so"):
        raise RuntimeError(
            "Unable to save package as a .so. It should be a .pt2 format"
        )
    elif config.aot_inductor.output_path.endswith(".pt2"):
        archive_path = config.aot_inductor.output_path
    else:
        archive_path = os.path.splitext(aoti_output_dir)[0] + ".pt2"

    with PT2ArchiveWriter(archive_path) as archive_writer:
        package_files = glob.glob(f"{aoti_output_dir}/*")

        for path in package_files:
            filename = os.path.basename(path)
            archive_writer.write_file(f"{AOTINDUCTOR_DIR}{filename}", path)

        buffer = io.BytesIO()
        torch.save(constants, buffer, _use_new_zipfile_serialization=True)
        archive_writer.write_string(
            os.path.join(CONSTANTS_DIR, "constants.pt"), buffer.getvalue()
        )

    return archive_path


def load_package(path: str, device: str) -> Callable:  # type: ignore[type-arg]
    so_path = os.path.splitext(path)[0] + ".so"
    so_path = compile_so(path, device, so_path)

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
