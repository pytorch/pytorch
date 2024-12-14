import io
import json
import logging
import os
import shlex
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch._inductor
import torch.utils._pytree as pytree
from torch._inductor import exc
from torch._inductor.cpp_builder import BuildOptionsBase, CppBuilder
from torch.export._tree_utils import reorder_kwargs

from .pt2_archive_constants import AOTINDUCTOR_DIR, ARCHIVE_VERSION


log = logging.getLogger(__name__)


class PT2ArchiveWriter:
    def __init__(self, archive_path: Union[str, io.BytesIO]) -> None:
        self.archive_path: Union[str, io.BytesIO] = archive_path
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


def package_aoti(
    archive_file: Union[str, io.BytesIO],
    aoti_files: Union[List[str], Dict[str, List[str]]],
) -> Union[str, io.BytesIO]:
    """
    Saves the AOTInductor generated files to the PT2Archive format.

    Args:
        archive_file: The file name to save the package to.
        aoti_files: This can either be a singular path to a directory containing
        the AOTInductor files, or a dictionary mapping the model name to the
        path to its AOTInductor generated files.
    """
    if isinstance(aoti_files, list):
        aoti_files = {"model": aoti_files}

    assert isinstance(aoti_files, dict), (
        "Please pass a list of AOTI generated files to be packaged or "
        "a dictionary mapping model names to their list of AOTI generated "
        "files. You can get this list of files through calling "
        "`torch._inductor.aot_compile(..., options={aot_inductor.package=True})`"
    )
    assert isinstance(archive_file, io.BytesIO) or (
        isinstance(archive_file, str) and archive_file.endswith(".pt2")
    ), f"Expect archive file to be a file ending in .pt2, or is a buffer. Instead got {archive_file}"

    # Save using the PT2 packaging format
    # (https://docs.google.com/document/d/1jLPp8MN8Whs0-VW9PmJ93Yg02W85tpujvHrTa1pc5x8/edit#heading=h.v2y2jgnwc56a)

    with PT2ArchiveWriter(archive_file) as archive_writer:
        for model_name, files in aoti_files.items():
            num_so_files = 0
            num_cpp_files = 0

            for file in files:
                if file == "":
                    continue

                if file.endswith(".so"):
                    num_so_files += 1
                    if num_so_files > 1:
                        raise RuntimeError(
                            f"Multiple .so files found in {files}. "
                            "You might need to clear your cache "
                            "directory before calling aoti_compile again."
                        )
                if file.endswith(".cpp"):
                    num_cpp_files += 1
                    if num_so_files > 1:
                        raise RuntimeError(
                            f"Multiple .cpp files found in {files}. "
                            "You might need to clear your cache "
                            "directory before calling aoti_compile again."
                        )

                filename = os.path.basename(file)
                new_filepath = os.path.join(AOTINDUCTOR_DIR, model_name, filename)
                log.debug(
                    "Saving AOTI generated file %s to archive in %s", file, new_filepath
                )
                archive_writer.write_file(
                    str(new_filepath),
                    file,
                )

    if isinstance(archive_file, io.BytesIO):
        archive_file.seek(0)
    return archive_file


class AOTICompiledModel:
    """
    Callable AOT Inductor loaded model from a .pt2
    """

    def __init__(self, loader: torch._C._aoti.AOTIModelPackageLoader) -> None:
        self.loader = loader

    def __call__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        call_spec = self.loader.get_call_spec()  # type: ignore[attr-defined]
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
        flat_outputs = self.loader.run(flat_inputs)  # type: ignore[attr-defined]
        return pytree.tree_unflatten(flat_outputs, out_spec)

    def get_metadata(self) -> Dict[str, str]:
        return self.loader.get_metadata()  # type: ignore[attr-defined]

    def load_constants(
        self,
        constants_map: Dict[str, torch.Tensor],
        *,
        check_full_update: bool,
    ) -> None:
        """
        Given a mapping of constant fqns to tensors, load the constants into the model.
        You can use ``get_constant_fqns`` to get the list of constant fqns that
        are needed in the compiled model.

        Args:
            constants_map: A mapping of constant fqns to tensors.
            check_full_update: Whether to add check to see if all the constants
            are updated and have values.
        """
        self.loader.load_constants(constants_map, False, check_full_update)  # type: ignore[attr-defined]

    def get_constant_fqns(self) -> List[str]:
        return self.loader.get_constant_fqns()  # type: ignore[attr-defined]


def load_package(
    path: Union[str, io.BytesIO], model_name: str = "model"
) -> AOTICompiledModel:
    assert isinstance(path, io.BytesIO) or (
        isinstance(path, str) and path.endswith(".pt2")
    ), f"Unable to load package. Path must be a buffer or a file ending in .pt2. Instead got {path}"

    if isinstance(path, io.BytesIO):
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            # TODO(angelayi): We shouldn't need to do this -- miniz should
            # handle reading the buffer. This is just a temporary workaround
            f.write(path.read())
            path.seek(0)
            log.debug("Writing buffer to tmp file located at %s.", f.name)
            loader = torch._C._aoti.AOTIModelPackageLoader(f.name, model_name)  # type: ignore[call-arg]
            return AOTICompiledModel(loader)

    loader = torch._C._aoti.AOTIModelPackageLoader(path, model_name)  # type: ignore[call-arg]
    return AOTICompiledModel(loader)
