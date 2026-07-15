import io
import json
import logging
import os
import tempfile
from typing import IO

import torch
from torch._inductor import config
from torch._inductor.cpp_builder import BuildOptionsBase, CppBuilder
from torch.export.pt2_archive._package import (
    AOTI_FILES,
    AOTICompiledModel,
    load_pt2,
    package_pt2,
)
from torch.types import FileLike


log = logging.getLogger(__name__)


def compile_so(aoti_dir: str, aoti_files: list[str], so_path: str) -> str:
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

    compile_options = BuildOptionsBase(
        **compile_flags, use_relative_path=config.is_fbcode()
    )
    object_builder = CppBuilder(
        name=file_name,
        sources=cpp_file,
        BuildOption=compile_options,
    )
    output_o = object_builder.get_target_file_path()
    object_builder.build()

    # Parse linker flags and build the .so file
    with open(file_name + "_linker_flags.json") as f:
        linker_flags = json.load(f)

    linker_options = BuildOptionsBase(
        **linker_flags, use_relative_path=config.is_fbcode()
    )
    so_builder = CppBuilder(
        name=os.path.split(so_path)[-1],
        sources=[output_o, consts_o],
        BuildOption=linker_options,
        output_dir=so_path,
    )
    output_so = so_builder.get_target_file_path()
    so_builder.build()

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
    archive_file: FileLike,
    aoti_files: AOTI_FILES,
) -> FileLike:
    """
    Saves the AOTInductor generated files to the PT2Archive format.

    Args:
        archive_file: The file name to save the package to.
        aoti_files: This can either be a singular path to a directory containing
        the AOTInductor files, or a dictionary mapping the model name to the
        path to its AOTInductor generated files.
    """

    return package_pt2(
        archive_file,
        aoti_files=aoti_files,
    )


def load_package(
    path: FileLike,
    model_name: str = "model",
    run_single_threaded: bool = False,
    num_runners: int = 1,
    device_index: int = -1,
) -> AOTICompiledModel:
    try:
        pt2_contents = load_pt2(
            path,
            run_single_threaded=run_single_threaded,
            num_runners=num_runners,
            device_index=device_index,
        )
        if model_name not in pt2_contents.aoti_runners:
            raise RuntimeError(f"Model {model_name} not found in package")
        return pt2_contents.aoti_runners[model_name]
    except RuntimeError:
        log.warning("Loading outdated pt2 file. Please regenerate your package.")

    if isinstance(path, (io.IOBase, IO)):
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            # TODO(angelayi): We shouldn't need to do this -- miniz should
            # handle reading the buffer. This is just a temporary workaround
            path.seek(0)
            f.write(path.read())
            log.debug("Writing buffer to tmp file located at %s.", f.name)
            loader = torch._C._aoti.AOTIModelPackageLoader(
                f.name, model_name, run_single_threaded, num_runners, device_index
            )
            return AOTICompiledModel(loader)

    path = os.fspath(path)  # AOTIModelPackageLoader expects (str, str)
    loader = torch._C._aoti.AOTIModelPackageLoader(
        path, model_name, run_single_threaded, num_runners, device_index
    )
    return AOTICompiledModel(loader)
