import glob
import io
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, IO, Optional, TYPE_CHECKING, Union
from typing_extensions import TypeAlias

import torch
import torch.utils._pytree as pytree
from torch._export.serde.serialize import deserialize, serialize, SerializedArtifact
from torch.export._tree_utils import reorder_kwargs
from torch.export.exported_program import ExportedProgram
from torch.export.pt2_archive._package_weights import (
    get_complete,
    group_weights,
    Weights,
)
from torch.export.pt2_archive.constants import (
    AOTINDUCTOR_DIR,
    ARCHIVE_FORMAT_PATH,
    ARCHIVE_FORMAT_VALUE,
    ARCHIVE_VERSION_PATH,
    ARCHIVE_VERSION_VALUE,
    CONSTANTS_DIR,
    CUSTOM_OBJ_FILENAME_PREFIX,
    EXTRA_DIR,
    MODELS_DIR,
    MODELS_FILENAME_FORMAT,
    SAMPLE_INPUTS_FILENAME_FORMAT,
    WEIGHT_FILENAME_PREFIX,
    WEIGHTS_DIR,
)
from torch.types import FileLike


if TYPE_CHECKING:
    from torch.utils._ordered_set import OrderedSet


DEFAULT_PICKLE_PROTOCOL = 2
AOTI_FILES: TypeAlias = Union[
    list[Union[str, Weights]], dict[str, list[Union[str, Weights]]]
]


logger: logging.Logger = logging.getLogger(__name__)


def is_pt2_package(serialized_model: Union[bytes, str]) -> bool:
    """
    Check if the serialized model is a PT2 Archive package.
    """
    try:
        zip_reader = zipfile.ZipFile(
            io.BytesIO(serialized_model)
            if isinstance(serialized_model, bytes)
            else serialized_model
        )
        root_folder = zip_reader.namelist()[0].split(os.path.sep)[0]
        archive_format_path = f"{root_folder}/{ARCHIVE_FORMAT_PATH}"
        if archive_format_path in zip_reader.namelist():
            return zip_reader.read(archive_format_path) == b"pt2"
    except Exception as ex:
        logger.info("Model is not a PT2 package: %s", str(ex))
    return False


class PT2ArchiveWriter:
    """
    Context manager for writing a PT2 archive.
    """

    def __init__(self, archive_path_or_buffer: FileLike):
        self.archive_file = torch._C.PyTorchFileWriter(archive_path_or_buffer)  # type: ignore[arg-type]
        # NOTICE: version here is different from the archive_version
        # this is the version of zip file format, which is used by PyTorchFileWriter, which write to /.data/version
        # archive_version is the version of the PT2 archive spec, which write to /archive_version
        self.archive_file.set_min_version(6)

    def __enter__(self) -> "PT2ArchiveWriter":
        return self

    def __exit__(self, *args: Any) -> None:
        if not self.has_record(ARCHIVE_FORMAT_PATH):
            self.write_string(ARCHIVE_FORMAT_PATH, ARCHIVE_FORMAT_VALUE)

        if not self.has_record(ARCHIVE_VERSION_PATH):
            self.write_string(ARCHIVE_VERSION_PATH, ARCHIVE_VERSION_VALUE)

        self.close()

    def has_record(self, name: str) -> bool:
        """
        Check if a record exists in the archive.
        """
        return name in self.archive_file.get_all_written_records()

    def count_prefix(self, prefix: str) -> int:
        """
        Count the number of records that start with a given prefix.
        """
        return sum(
            1
            for record in self.archive_file.get_all_written_records()
            if record.startswith(prefix)
        )

    def write_bytes(self, name: str, data: bytes) -> None:
        """
        Write a bytes object to the archive.
        name: The destination file inside the archive.
        data: The bytes object to write.
        """
        assert isinstance(data, bytes), f"Expected bytes but got {type(data)}"
        self.archive_file.write_record(name, data, len(data))

    def write_string(self, name: str, data: str) -> None:
        """
        Write a string object to the archive.
        name: The destination file inside the archive.
        data: The string object to write.
        """
        assert isinstance(data, str), f"Expected string but got {type(data)}"
        data_bytes = data.encode()
        self.write_bytes(name, data_bytes)

    def write_file(self, name: str, file_path: str) -> None:
        """
        Copy a file into the archive.
        name: The destination file inside the archive.
        file_path: The source file on disk.
        """
        assert os.path.isfile(file_path), f"{file_path} is not a valid file path"

        with open(file_path, "rb") as f:
            file_bytes = f.read()
            self.write_bytes(name, file_bytes)

    def write_folder(self, archive_dir: str, folder_dir: str) -> None:
        """
        Copy a folder into the archive.
        archive_dir: The destination folder inside the archive.
        folder_dir: The source folder on disk.
        """
        assert os.path.isdir(folder_dir), f"{folder_dir} is not a valid directory path"

        file_paths = filter(
            os.path.isfile, glob.glob(f"{folder_dir}/**", recursive=True)
        )
        for file_path in file_paths:
            filename = os.path.relpath(file_path, folder_dir)
            archive_path = os.path.join(archive_dir, filename)
            self.write_file(archive_path, file_path)

    def close(self) -> None:
        """
        Close the archive.
        """
        self.archive_file.write_end_of_file()


class PT2ArchiveReader:
    """
    Context manager for reading a PT2 archive.
    """

    def __init__(self, archive_path_or_buffer: FileLike):
        self.archive_file = torch._C.PyTorchFileReader(archive_path_or_buffer)  # type: ignore[arg-type]
        assert self.read_string(ARCHIVE_FORMAT_PATH) == ARCHIVE_FORMAT_VALUE, (
            "Invalid archive format"
        )

    def __enter__(self) -> "PT2ArchiveReader":
        return self

    def __exit__(self, *args: Any) -> None:
        # torch._C.PyTorchFileReader doesn't have a close method
        pass

    def read_bytes(self, name: str) -> bytes:
        """
        Read a bytes object from the archive.
        name: The source file inside the archive.
        """
        return self.archive_file.get_record(name)

    def read_string(self, name: str) -> str:
        """
        Read a string object from the archive.
        name: The source file inside the archive.
        """
        data = self.read_bytes(name)
        return data.decode()

    def archive_version(self) -> int:
        """
        Get the archive version.
        """
        try:
            archive_version = self.read_string(ARCHIVE_VERSION_PATH)
        except Exception:
            # if archive_version is not found, it means the archive is older than version 0.
            # In this case, we assume the archive is version 0.
            archive_version = "0"

        return int(archive_version)

    def get_file_names(self) -> list[str]:
        """
        Get the file names in the archive.
        """
        return self.archive_file.get_all_records()


def _package_aoti_files(
    archive_writer: PT2ArchiveWriter,
    aoti_files: Optional[AOTI_FILES],
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    if aoti_files is None:
        return

    if isinstance(aoti_files, list):
        aoti_files = {"model": aoti_files}

    assert isinstance(aoti_files, dict)

    all_weights: dict[str, Weights] = {}  # model_name -> weight
    weights_configs: dict[
        str, dict[str, Any]
    ] = {}  # model_name -> (weight_name -> (filename, shape, stride, offset))

    for model_name, files in aoti_files.items():
        num_so_files = 0
        weights_configs[model_name] = {}

        for file in files:
            if file == "":
                continue

            if isinstance(file, Weights):
                all_weights[model_name] = file
                continue

            if file.endswith(".so"):
                num_so_files += 1
                if num_so_files > 1:
                    raise RuntimeError(
                        f"Multiple .so files found in {files}. "
                        "You might need to clear your cache "
                        "directory before calling aoti_compile again."
                    )

            filename = os.path.basename(file)
            if filename.startswith(CUSTOM_OBJ_FILENAME_PREFIX):
                new_filepath = os.path.join(CONSTANTS_DIR, filename)
            else:
                new_filepath = os.path.join(AOTINDUCTOR_DIR, model_name, filename)
            logger.debug(
                "Saving AOTI generated file %s to archive in %s", file, new_filepath
            )
            archive_writer.write_file(
                str(new_filepath),
                file,
            )

    if len(all_weights) > 0:
        # Dedup weights
        grouped_tensors: list[OrderedSet[tuple[str, str]]] = group_weights(all_weights)
        for idx, group in enumerate(grouped_tensors):
            filename = f"{WEIGHT_FILENAME_PREFIX}{idx}"
            model_name, weight_name = get_complete(group, all_weights)
            complete_tensor, _ = all_weights[model_name].get_weight(weight_name)
            buffer = io.BytesIO()
            torch.save(complete_tensor, buffer, pickle_protocol=pickle_protocol)
            archive_writer.write_bytes(
                os.path.join(WEIGHTS_DIR, filename), buffer.getvalue()
            )
            for model_name, weight_name in group:
                _, w_property = all_weights[model_name].get_weight(weight_name)
                weights_configs[model_name][weight_name] = (
                    filename,
                    w_property.shape,
                    w_property.stride,
                    w_property.offset,
                )

        for model_name, weights_config in weights_configs.items():
            archive_writer.write_string(
                os.path.join(AOTINDUCTOR_DIR, model_name, "weights_config.json"),
                json.dumps(weights_config),
            )
            logger.debug("packaging weights_config for model %s", model_name)
            logger.debug(weights_config)


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

        archive_writer.write_bytes(
            MODELS_FILENAME_FORMAT.format(model_name), artifact.exported_program
        )
        # TODO:Consider dedup this with the weights saved in package_aoti_files
        archive_writer.write_bytes(f"{WEIGHTS_DIR}{model_name}.pt", artifact.state_dict)
        archive_writer.write_bytes(
            f"{CONSTANTS_DIR}{model_name}.pt", artifact.constants
        )
        archive_writer.write_bytes(
            SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name),
            artifact.example_inputs,
        )


def _package_extra_files(
    archive_writer: PT2ArchiveWriter, extra_files: Optional[dict[str, Any]]
) -> None:
    if extra_files is None:
        return

    for extra_file_name, content in extra_files.items():
        archive_writer.write_string(f"{EXTRA_DIR}{extra_file_name}", content)


def package_pt2(
    f: FileLike,
    *,
    exported_programs: Optional[
        Union[ExportedProgram, dict[str, ExportedProgram]]
    ] = None,
    aoti_files: Optional[AOTI_FILES] = None,
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
        logger.warning(
            "Expect archive file to be a file ending in .pt2, or is a buffer. "
            "Instead got {%s}",
            f,
        )

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    with PT2ArchiveWriter(f) as archive_writer:
        _package_exported_programs(
            archive_writer, exported_programs, pickle_protocol=pickle_protocol
        )
        _package_aoti_files(
            archive_writer,
            aoti_files,
            pickle_protocol=pickle_protocol,
        )
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
        call_spec = self.loader.get_call_spec()
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_inputs = [x for x in flat_inputs if isinstance(x, torch.Tensor)]
        flat_outputs = self.loader.boxed_run(flat_inputs)
        return pytree.tree_unflatten(flat_outputs, out_spec)

    def get_metadata(self) -> dict[str, str]:
        return self.loader.get_metadata()

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
        self.loader.load_constants(
            constants_map, False, check_full_update, user_managed
        )

    def get_constant_fqns(self) -> list[str]:
        return self.loader.get_constant_fqns()

    def __deepcopy__(self, memo: Optional[dict[Any, Any]]) -> "AOTICompiledModel":
        logger.warning(
            "AOTICompiledModel deepcopy warning: AOTICompiledModel.loader is not deepcopied."
        )
        return AOTICompiledModel(self.loader)


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

        serialized_exported_program = archive_reader.read_bytes(file)
        serialized_weights = archive_reader.read_bytes(weights_file)
        serialized_constants = archive_reader.read_bytes(constants_file)
        serialized_sample_inputs = archive_reader.read_bytes(sample_inputs_file)

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
        contents = archive_reader.read_string(file)
        extra_file_contents[file[len(EXTRA_DIR) :]] = contents

    return extra_file_contents


def load_pt2(
    f: FileLike,
    *,
    expected_opset_version: Optional[dict[str, int]] = None,
    run_single_threaded: bool = False,
    num_runners: int = 1,
    device_index: int = -1,
    load_weights_from_disk: bool = False,
) -> PT2ArchiveContents:  # type: ignore[type-arg]
    """
    Loads all the artifacts previously saved with ``package_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        expected_opset_version (Optional[Dict[str, int]]): A map of opset names
         to expected opset versions

        num_runners (int): Number of runners to load AOTInductor artifacts

        run_single_threaded (bool): Whether the model should be run without
            thread synchronization logic. This is useful to avoid conflicts with
            CUDAGraphs.

        device_index (int): The index of the device to which the PT2 package is
            to be loaded. By default, `device_index=-1` is used, which corresponds
            to the device `cuda` when using CUDA. Passing `device_index=1` would
            load the package to `cuda:1`, for example.

    Returns:
        A ``PT2ArchiveContents`` object which contains all the objects in the PT2.
    """

    if not (
        (isinstance(f, (io.IOBase, IO)) and f.readable() and f.seekable())
        or (isinstance(f, (str, os.PathLike)) and os.fspath(f).endswith(".pt2"))
    ):
        # TODO: turn this into an error in 2.9
        logger.warning(
            "Unable to load package. f must be a buffer or a file ending in "
            ".pt2. Instead got {%s}",
            f,
        )

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    weights = {}
    weight_maps = {}
    with PT2ArchiveReader(f) as archive_reader:
        version = archive_reader.read_string(ARCHIVE_VERSION_PATH)
        if version != ARCHIVE_VERSION_VALUE:
            raise ValueError(
                f"Saved archive version {version} does not match our current "
                f"archive version {ARCHIVE_VERSION_VALUE}."
            )

        file_names = archive_reader.get_file_names()

        exported_programs = _load_exported_programs(
            archive_reader, file_names, expected_opset_version
        )
        extra_files = _load_extra_files(archive_reader, file_names)

        # Get a list of AOTI model names
        aoti_model_names: set[str] = set()
        for file in file_names:
            if file.startswith(AOTINDUCTOR_DIR):
                file_end = file[
                    len(AOTINDUCTOR_DIR) :
                ]  # remove data/aotinductor/ prefix
                model_name = file_end.split("/")[
                    0
                ]  # split "model_name/...cpp" into "model_name"
                aoti_model_names.add(model_name)
                if load_weights_from_disk and file.endswith("weights_config.json"):
                    weight_map = json.loads(archive_reader.read_string(file))
                    weight_maps[model_name] = weight_map
            elif load_weights_from_disk and file.startswith(WEIGHTS_DIR):
                weight_file_name = file[
                    len(WEIGHTS_DIR) :
                ]  # remove data/weights/ prefix
                weight_bytes = archive_reader.read_bytes(file)
                loaded_weight = torch.load(io.BytesIO(weight_bytes))
                weights[weight_file_name] = loaded_weight

    if isinstance(f, (io.IOBase, IO)):
        if len(aoti_model_names) > 0:
            # Workaround for AOTIModelPackageLoader not reading buffers
            with tempfile.NamedTemporaryFile(suffix=".pt2") as tf:
                f.seek(0)
                tf.write(f.read())
                f.seek(0)
                logger.debug("Writing buffer to tmp file located at %s.", tf.name)

                aoti_runners = {
                    model_name: AOTICompiledModel(
                        torch._C._aoti.AOTIModelPackageLoader(
                            tf.name,
                            model_name,
                            run_single_threaded,
                            num_runners,
                            device_index,
                        )
                    )
                    for model_name in aoti_model_names
                }
        else:
            aoti_runners = {}
    else:
        aoti_runners = {
            model_name: AOTICompiledModel(
                torch._C._aoti.AOTIModelPackageLoader(
                    f, model_name, run_single_threaded, num_runners, device_index
                )
            )
            for model_name in aoti_model_names
        }

    if weight_maps:
        for model_name in aoti_model_names:
            model_weights = {}
            for weight_name, (file, shape, stride, storage_offset) in weight_maps[
                model_name
            ].items():
                weight = weights[file]
                model_weights[weight_name] = weight.as_strided(
                    shape, stride, storage_offset
                )

            # user_managed=True ensures the weights updates are shared by all runners.
            aoti_runners[model_name].load_constants(
                model_weights, check_full_update=True, user_managed=True
            )

    return PT2ArchiveContents(exported_programs, aoti_runners, extra_files)


def load_weights_to_pt2_contents(
    pt2_contents: PT2ArchiveContents, weights_map: dict[str, Any]
) -> None:
    """
    Load weights into the models in PT2 archive contents

    Args:
        pt2_contents (PT2ArchiveContents): The contents of the PT2 archive.
    """
    for model_name, weights in weights_map.items():
        if model_name not in pt2_contents.aoti_runners:
            raise RuntimeError(f"Model {model_name} not found in PT2 archive contents.")
        pt2_contents.aoti_runners[model_name].load_constants(
            weights, check_full_update=True, user_managed=True
        )
