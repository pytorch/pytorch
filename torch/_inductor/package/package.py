import io
import logging
import os
import tempfile
from typing import IO

import torch
from torch.export.pt2_archive._package import (
    AOTI_FILES,
    AOTICompiledModel,
    load_pt2,
    package_pt2,
)
from torch.types import FileLike


log = logging.getLogger(__name__)
AOTI_PACKAGE_DEVICE_ERROR = "Cannot load AOTInductor package for device"


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
    use_stream_affinity: bool = False,
) -> AOTICompiledModel:
    """Load an AOTI model from a PT2 package or an extracted package directory."""
    is_directory = isinstance(path, (str, os.PathLike)) and os.path.isdir(path)
    if not is_directory:
        try:
            pt2_contents = load_pt2(
                path,
                run_single_threaded=run_single_threaded,
                num_runners=num_runners,
                device_index=device_index,
                use_stream_affinity=use_stream_affinity,
            )
            if model_name not in pt2_contents.aoti_runners:
                raise RuntimeError(f"Model {model_name} not found in package")
            return pt2_contents.aoti_runners[model_name]
        except RuntimeError as e:
            if AOTI_PACKAGE_DEVICE_ERROR in str(e):
                raise
            log.warning("Loading outdated pt2 file. Please regenerate your package.")

    if isinstance(path, (io.IOBase, IO)):
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            # TODO(angelayi): We shouldn't need to do this -- miniz should
            # handle reading the buffer. This is just a temporary workaround
            path.seek(0)
            f.write(path.read())
            log.debug("Writing buffer to tmp file located at %s.", f.name)
            loader = torch._C._aoti.AOTIModelPackageLoader(
                f.name,
                model_name,
                run_single_threaded,
                num_runners,
                device_index,
                use_stream_affinity,
            )
            return AOTICompiledModel(loader)

    path = os.fspath(path)  # AOTIModelPackageLoader expects (str, str)
    loader = torch._C._aoti.AOTIModelPackageLoader(
        path,
        model_name,
        run_single_threaded,
        num_runners,
        device_index,
        use_stream_affinity,
    )
    return AOTICompiledModel(loader)
