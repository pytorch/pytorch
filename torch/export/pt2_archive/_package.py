import io
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, IO, Optional, Union
from typing_extensions import Self

import torch
import torch._inductor
import torch.utils._pytree as pytree
from torch._export.serde.serialize import deserialize, serialize, SerializedArtifact
from torch.export._tree_utils import reorder_kwargs
from torch.export.exported_program import ExportedProgram
from torch.export.pt2_archive._constants import (
    AOTINDUCTOR_DIR,
    ARCHIVE_VERSION,
    CONSTANTS_DIR,
    CUSTOM_OBJ_FILENAME_PREFIX,
    EXTRA_DIR,
    MODELS_DIR,
    MODELS_FILENAME_FORMAT,
    SAMPLE_INPUTS_FILENAME_FORMAT,
    WEIGHTS_DIR,
)
from torch.types import FileLike


DEFAULT_PICKLE_PROTOCOL = 2


log = logging.getLogger(__name__)


class PT2ArchiveWriter:
    def __init__(self, archive_path: FileLike) -> None:
        self.archive_path: FileLike = archive_path
        self.archive_file: Optional[zipfile.ZipFile] = None

    def __enter__(self) -> Self:
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
    def __init__(self, archive_path: FileLike) -> None:
        self.archive_path: FileLike = archive_path
        self.archive_file: Optional[zipfile.ZipFile] = None

    def __enter__(self) -> Self:
        self.archive_file = zipfile.ZipFile(
            self.archive_path, "r", compression=zipfile.ZIP_STORED
        )

        version = int(self.read("version"))
        if version != ARCHIVE_VERSION:
            raise RuntimeError(
                f"Saved archive version {version} does not match our current "
                f"archive version {ARCHIVE_VERSION}."
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

    def get_file_names(self) -> list[str]:
        assert self.archive_file is not None
        return self.archive_file.namelist()


def _package_aoti_files(
    archive_writer: PT2ArchiveWriter,
    aoti_files: Optional[Union[list[str], dict[str, list[str]]]],
) -> None:
    if aoti_files is None:
        return

    if isinstance(aoti_files, list):
        aoti_files = {"model": aoti_files}

    assert isinstance(aoti_files, dict)

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
            if filename.startswith(CUSTOM_OBJ_FILENAME_PREFIX):
                new_filepath = os.path.join(CONSTANTS_DIR, filename)
            else:
                new_filepath = os.path.join(AOTINDUCTOR_DIR, model_name, filename)
            log.debug(
                "Saving AOTI generated file %s to archive in %s", file, new_filepath
            )
            archive_writer.write_file(
                str(new_filepath),
                file,
            )


def _package_exported_programs(
    archive_writer: PT2ArchiveWriter,
    exported_programs: Optional[Union[ExportedProgram, dict[str, ExportedProgram]]],
    opset_version: Optional[dict[str, int]] = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    if exported_programs is None:
        return

    if isinstance(exported_programs, ExportedProgram):
        exported_programs = {"model", exported_programs}  # type: ignore[assignment]

    assert isinstance(exported_programs, dict)

    for model_name, ep in exported_programs.items():
        artifact: SerializedArtifact = serialize(ep, opset_version, pickle_protocol)

        archive_writer.writestr(
            MODELS_FILENAME_FORMAT.format(model_name), artifact.exported_program
        )
        archive_writer.writestr(f"{WEIGHTS_DIR}{model_name}.pt", artifact.state_dict)
        archive_writer.writestr(f"{CONSTANTS_DIR}{model_name}.pt", artifact.constants)
        archive_writer.writestr(
            SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name),
            artifact.example_inputs,
        )


def _package_extra_files(
    archive_writer: PT2ArchiveWriter, extra_files: Optional[dict[str, Any]]
) -> None:
    if extra_files is None:
        return

    for extra_file_name, content in extra_files.items():
        encoded_content = content.encode("utf-8")
        archive_writer.writestr(f"{EXTRA_DIR}{extra_file_name}", encoded_content)


def package_pt2(
    f: FileLike,
    *,
    exported_programs: Optional[
        Union[ExportedProgram, dict[str, ExportedProgram]]
    ] = None,
    aoti_files: Optional[Union[list[str], dict[str, list[str]]]] = None,
    extra_files: Optional[dict[str, Any]] = None,
    opset_version: Optional[dict[str, int]] = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> FileLike:
    """
    Saves the artifacts to a PT2Archive format
    (https://docs.google.com/document/d/1RQ4cmywilnFUT1VE-4oTGxwXdc8vowCSZsrRgo3wFA8/edit?tab=t.0#heading=h.v2y2jgnwc56a).
    The artifact can then be loaded using ``load_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]) A file-like object (has to
         implement write and flush) or a string containing a file name.

        exported_programs (Union[ExportedProgram, dict[str, ExportedProgram]]):
         The exported program to save, or a dictionary mapping model name to an
         exported program to save. The exported program will be saved under
         models/*.json. If only one ExportedProgram is specified, this will
         automatically be named "model".

        aoti_files (Union[list[str], dict[str, list[str]]): A list of files
         generated by AOTInductor via
         ``torch._inductor.aot_compile(..., {"aot_inductor.package": True})``,
         or a dictionary mapping model name to its AOTInductor generated files.
         If only one set of files is specified, this will automatically be named
         "model".

        extra_files (Optional[Dict[str, Any]]): Map from filename to contents
         which will be stored as part of the pt2.

        opset_version (Optional[Dict[str, int]]): A map of opset names
         to the version of this opset

        pickle_protocol: can be specified to override the default protocol

    """
    assert not (
        exported_programs is None and aoti_files is None and extra_files is None
    ), (
        "No value passed in for `exported_programs`, `aoti_files`, and "
        "`extra_files`, implying that you do not plan on saving anything."
    )

    if not (
        (isinstance(f, (io.IOBase, IO)) and f.writable() and f.seekable())
        or (isinstance(f, (str, os.PathLike)) and os.fspath(f).endswith(".pt2"))
    ):
        # TODO: turn this into an error
        log.warning(
            "Expect archive file to be a file ending in .pt2, or is a buffer. "
            "Instead got {%s}",
            f,
        )

    with PT2ArchiveWriter(f) as archive_writer:
        _package_exported_programs(archive_writer, exported_programs)
        _package_aoti_files(archive_writer, aoti_files)
        _package_extra_files(archive_writer, extra_files)

    if isinstance(f, (io.IOBase, IO)):
        f.seek(0)
    return f


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
        flat_outputs = self.loader.boxed_run(flat_inputs)  # type: ignore[attr-defined]
        return pytree.tree_unflatten(flat_outputs, out_spec)

    def get_metadata(self) -> dict[str, str]:
        return self.loader.get_metadata()  # type: ignore[attr-defined]

    def load_constants(
        self,
        constants_map: dict[str, torch.Tensor],
        *,
        check_full_update: bool,
        user_managed: bool = False,
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
        self.loader.load_constants(  # type: ignore[attr-defined]
            constants_map, False, check_full_update, user_managed
        )

    def get_constant_fqns(self) -> list[str]:
        return self.loader.get_constant_fqns()  # type: ignore[attr-defined]

    def __deepcopy__(self, memo: Optional[dict[Any, Any]]) -> "AOTICompiledModel":
        log.warning(
            "AOTICompiledModel deepcopy warning: AOTICompiledModel.loader is not deepcopied."
        )
        return AOTICompiledModel(self.loader)  # type: ignore[attr-defined]


@dataclass
class PT2ArchiveContents:
    exported_programs: dict[str, ExportedProgram]
    aoti_runners: dict[str, AOTICompiledModel]
    extra_files: dict[str, Any]


def _load_exported_programs(
    archive_reader: PT2ArchiveReader,
    file_names: list[str],
    expected_opset_version: Optional[dict[str, int]],
) -> dict[str, ExportedProgram]:
    exported_program_files = [
        file for file in file_names if file.startswith(MODELS_DIR)
    ]
    exported_programs = {}
    for file in exported_program_files:
        prefix, suffix = MODELS_FILENAME_FORMAT.split(
            "{}"
        )  # split "models/{}.json" into "models/" and "json"
        model_name = file[
            len(prefix) : -len(suffix)
        ]  # given "models/foo.json" we can now get "foo"

        weights_file = f"{WEIGHTS_DIR}{model_name}.pt"
        constants_file = f"{CONSTANTS_DIR}{model_name}.pt"
        sample_inputs_file = SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name)

        serialized_exported_program = archive_reader.read(file)
        serialized_weights = archive_reader.read(weights_file)
        serialized_constants = archive_reader.read(constants_file)
        serialized_sample_inputs = archive_reader.read(sample_inputs_file)

        artifact: SerializedArtifact = SerializedArtifact(
            serialized_exported_program,
            serialized_weights,
            serialized_constants,
            serialized_sample_inputs,
        )

        # Deserialize ExportedProgram
        ep = deserialize(artifact, expected_opset_version)
        exported_programs[model_name] = ep

    return exported_programs


def _load_extra_files(
    archive_reader: PT2ArchiveReader, file_names: list[str]
) -> dict[str, Any]:
    extra_files = [file for file in file_names if file.startswith(EXTRA_DIR)]

    extra_file_contents: dict[str, Any] = {}
    for file in extra_files:
        contents = archive_reader.read(file)
        extra_file_contents[file[len(EXTRA_DIR) :]] = contents.decode("utf-8")

    return extra_file_contents


def load_pt2(
    f: FileLike,
    *,
    expected_opset_version: Optional[dict[str, int]] = None,
    run_single_threaded: bool = False,
    num_runners: int = 1,
) -> PT2ArchiveContents:  # type: ignore[type-arg]
    """
    Loads all the artifacts previously saved with ``package_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
         to expected opset versions

        num_runners (int): Number of runners to load AOTInductor artifacts

    Returns:
        A ``PT2ArchiveContents`` object which contains all the objects in the PT2.
    """

    if not (
        (isinstance(f, (io.IOBase, IO)) and f.readable() and f.seekable())
        or (isinstance(f, (str, os.PathLike)) and os.fspath(f).endswith(".pt2"))
    ):
        # TODO: turn this into an error
        log.warning(
            "Unable to load package. f must be a buffer or a file ending in "
            ".pt2. Instead got {%s}",
            f,
        )

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    with PT2ArchiveReader(f) as archive_reader:
        file_names = archive_reader.get_file_names()

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_reader.extractall(tmp_dir)
            file_names = archive_reader.get_file_names()

            exported_programs = _load_exported_programs(
                archive_reader, file_names, expected_opset_version
            )
            extra_files = _load_extra_files(archive_reader, file_names)

            # Get a list of AOTI model names
            aoti_model_names = set()
            for file in file_names:
                if file.startswith(AOTINDUCTOR_DIR):
                    file = file[
                        len(AOTINDUCTOR_DIR) :
                    ]  # remove data/aotinductor/ prefix
                    model_name = file.split("/")[
                        0
                    ]  # split "model_name/...cpp" into "model_name"
                    aoti_model_names.add(model_name)

    if isinstance(f, (io.IOBase, IO)) and len(aoti_model_names) > 0:
        # Workaround for AOTIModelPackageLoader not reading buffers
        with tempfile.NamedTemporaryFile(suffix=".pt2") as tf:
            f.seek(0)
            tf.write(f.read())
            f.seek(0)
            log.debug("Writing buffer to tmp file located at %s.", tf.name)

            aoti_runners = {
                model_name: AOTICompiledModel(
                    torch._C._aoti.AOTIModelPackageLoader(
                        tf.name, model_name, run_single_threaded, num_runners
                    )  # type: ignore[call-arg]
                )
                for model_name in aoti_model_names
            }

    else:
        aoti_runners = {
            model_name: AOTICompiledModel(
                torch._C._aoti.AOTIModelPackageLoader(
                    f, model_name, run_single_threaded, num_runners
                )  # type: ignore[call-arg]
            )
            for model_name in aoti_model_names
        }

    return PT2ArchiveContents(exported_programs, aoti_runners, extra_files)
