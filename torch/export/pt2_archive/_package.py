import glob
import io
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Any, IO, TYPE_CHECKING, TypeAlias

import torch
import torch.utils._pytree as pytree
from torch._export.serde import schema
from torch._export.serde.serialize import (
    _dataclass_to_dict,
    _dict_to_dataclass,
    deserialize_device,
    deserialize_scalar_type,
    deserialize_size,
    deserialize_storage_offset,
    deserialize_stride,
    ExportedProgramDeserializer,
    serialize,
    serialize_tensor_meta,
    SerializedArtifact,
)
from torch._inductor.cpp_builder import normalize_path_separator
from torch._subclasses.fake_tensor import FakeTensor
from torch.export import ExportedProgram
from torch.export._tree_utils import reorder_kwargs
from torch.export.pt2_archive._package_weights import (
    get_complete,
    group_weights,
    TensorProperties,
    Weights,
)
from torch.export.pt2_archive.constants import (
    AOTINDUCTOR_DIR,
    ARCHIVE_FORMAT_PATH,
    ARCHIVE_FORMAT_VALUE,
    ARCHIVE_VERSION_PATH,
    ARCHIVE_VERSION_VALUE,
    CONSTANTS_CONFIG_FILENAME_FORMAT,
    CONSTANTS_DIR,
    CUSTOM_OBJ_FILENAME_PREFIX,
    EXECUTORCH_DIR,
    EXTRA_DIR,
    MODELS_DIR,
    MODELS_FILENAME_FORMAT,
    SAMPLE_INPUTS_FILENAME_FORMAT,
    TENSOR_CONSTANT_FILENAME_PREFIX,
    WEIGHT_FILENAME_PREFIX,
    WEIGHTS_CONFIG_FILENAME_FORMAT,
    WEIGHTS_DIR,
)
from torch.types import FileLike


if TYPE_CHECKING:
    from torch.utils._ordered_set import OrderedSet


DEFAULT_PICKLE_PROTOCOL = 2
AOTI_FILES: TypeAlias = list[str | Weights] | dict[str, list[str | Weights]]


logger: logging.Logger = logging.getLogger(__name__)


def is_pt2_package(serialized_model: bytes | str) -> bool:
    """
    Check if the serialized model is a PT2 Archive package.
    """
    try:
        with zipfile.ZipFile(
            io.BytesIO(serialized_model)
            if isinstance(serialized_model, bytes)
            else serialized_model
        ) as zip_reader:
            root_folder = zip_reader.namelist()[0].split(os.path.sep)[0]
            archive_format_path = f"{root_folder}/{ARCHIVE_FORMAT_PATH}"
            if archive_format_path in zip_reader.namelist():
                return zip_reader.read(archive_format_path) == b"pt2"
    except Exception:
        logger.info("Model is not a PT2 package")
    return False


class PT2ArchiveWriter:
    """
    Context manager for writing a PT2 archive.
    """

    def __init__(self, archive_path_or_buffer: FileLike):
        if isinstance(archive_path_or_buffer, str):
            archive_path_or_buffer = normalize_path_separator(archive_path_or_buffer)
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
        if not isinstance(data, bytes):
            raise AssertionError(f"Expected bytes but got {type(data)}")
        self.archive_file.write_record(name, data, len(data))

    def write_string(self, name: str, data: str) -> None:
        """
        Write a string object to the archive.
        name: The destination file inside the archive.
        data: The string object to write.
        """
        if not isinstance(data, str):
            raise AssertionError(f"Expected string but got {type(data)}")
        data_bytes = data.encode()
        self.write_bytes(name, data_bytes)

    def write_file(self, name: str, file_path: str) -> None:
        """
        Copy a file into the archive.
        name: The destination file inside the archive.
        file_path: The source file on disk.
        """
        if not os.path.isfile(file_path):
            raise AssertionError(f"{file_path} is not a valid file path")

        with open(file_path, "rb") as f:
            file_bytes = f.read()
            self.write_bytes(name, file_bytes)

    def write_folder(self, archive_dir: str, folder_dir: str) -> None:
        """
        Copy a folder into the archive.
        archive_dir: The destination folder inside the archive.
        folder_dir: The source folder on disk.
        """
        if not os.path.isdir(folder_dir):
            raise AssertionError(f"{folder_dir} is not a valid directory path")

        file_paths = filter(
            os.path.isfile, glob.glob(f"{folder_dir}/**", recursive=True)
        )
        for file_path in file_paths:
            # pyrefly: ignore [no-matching-overload]
            filename = os.path.relpath(file_path, folder_dir)
            archive_path = os.path.join(archive_dir, filename)
            # pyrefly: ignore [bad-argument-type]
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
        if isinstance(archive_path_or_buffer, str):
            archive_path_or_buffer = normalize_path_separator(archive_path_or_buffer)
        self.archive_file = torch._C.PyTorchFileReader(archive_path_or_buffer)  # type: ignore[arg-type]
        if self.read_string(ARCHIVE_FORMAT_PATH) != ARCHIVE_FORMAT_VALUE:
            raise AssertionError("Invalid archive format")

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


is_pt2_package.__module__ = "torch.export.pt2_archive"
PT2ArchiveWriter.__module__ = "torch.export.pt2_archive"
PT2ArchiveReader.__module__ = "torch.export.pt2_archive"


def _package_aoti_files(
    archive_writer: PT2ArchiveWriter,
    aoti_files: AOTI_FILES | None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    if aoti_files is None:
        return

    if isinstance(aoti_files, list):
        aoti_files = {"model": aoti_files}

    if not isinstance(aoti_files, dict):
        raise AssertionError(
            f"Expected aoti_files to be a dict, but got {type(aoti_files)}"
        )

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


def _is_fake_tensor(t: torch.Tensor) -> bool:
    return isinstance(t, FakeTensor)


def _is_tensor_subclass(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and type(t.data) is not torch.Tensor


def _get_raw_tensor_bytes(value: torch.Tensor) -> bytes:
    """
    Get the raw bytes of a tensor. This is used to save the tensor in pt2 archive.
    """
    # NOTE: don't chain .cpu() with .data_ptr(). If an HtoD copy needs to be
    # performed, the CPU copy needs to be kept alive when its underlying
    # memory is accessed.
    import ctypes

    if _is_fake_tensor(value):
        value_bytes = b""
    elif value.data_ptr():
        cpu_tensor = value.cpu()
        value_untyped_storage = cpu_tensor.untyped_storage()
        # we store the raw bytes the untyped storage. Tensor metadata is stored separately
        value_bytes = bytes(
            ctypes.cast(
                value_untyped_storage.data_ptr(),
                ctypes.POINTER(ctypes.c_ubyte * value_untyped_storage.size()),
            ).contents
        )
    else:
        # for empty tensor
        value_bytes = b""
    return value_bytes


def _should_use_pickle(t: torch.Tensor) -> bool:
    return _is_tensor_subclass(t) and not _is_fake_tensor(t)


def _save_pickled_tensors(
    pickled_items: list[tuple[str, torch.Tensor]],
    archive_writer: PT2ArchiveWriter,
    config: dict[str, schema.PayloadMeta],
    directory: str,
    filename_prefix: str,
    idx: int,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> int:
    """Save pickled tensors and update config. Returns updated index."""
    for item_fqn, tensor in pickled_items:
        path_name = f"{filename_prefix}{idx}"
        archive_path = os.path.join(directory, path_name)
        buffer = io.BytesIO()
        torch.save(tensor, buffer, pickle_protocol=pickle_protocol)
        archive_writer.write_bytes(archive_path, buffer.getvalue())

        config[item_fqn] = schema.PayloadMeta(
            path_name=path_name,
            is_param=isinstance(tensor, torch.nn.Parameter),
            use_pickle=True,
            tensor_meta=serialize_tensor_meta(tensor),
        )
        idx += 1
    return idx


def _save_raw_tensors(
    raw_items: dict[str, tuple[torch.Tensor, TensorProperties]],
    model_name: str,
    archive_writer: PT2ArchiveWriter,
    config: dict[str, schema.PayloadMeta],
    directory: str,
    filename_prefix: str,
    idx: int,
) -> int:
    """Save deduplicated raw tensor bytes and update config. Returns updated index."""
    if not raw_items:
        return idx

    weights_dict = {model_name: Weights(raw_items)}
    storage_groups = group_weights(weights_dict)

    for group in storage_groups:
        # Find the complete tensor that covers all others in this storage group
        model_name, complete_item_name = get_complete(group, weights_dict)
        complete_tensor, _ = weights_dict[model_name].get_weight(complete_item_name)

        path_name = f"{filename_prefix}{idx}"
        archive_path = os.path.join(directory, path_name)
        tensor_bytes = _get_raw_tensor_bytes(complete_tensor)
        archive_writer.write_bytes(archive_path, tensor_bytes)
        idx += 1

        for _, item_fqn in group:
            tensor, _ = weights_dict[model_name].get_weight(item_fqn)
            config[item_fqn] = schema.PayloadMeta(
                path_name=path_name,
                is_param=isinstance(tensor, torch.nn.Parameter),
                use_pickle=False,
                tensor_meta=serialize_tensor_meta(tensor),
            )

    return idx


def _package_state_dict(
    model_name: str,
    exported_program: ExportedProgram,
    archive_writer: PT2ArchiveWriter,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> schema.PayloadConfig:
    weights_config: dict[str, schema.PayloadMeta] = {}

    pickled_weights: list[tuple[str, torch.Tensor]] = []
    raw_weights: dict[str, tuple[torch.Tensor, TensorProperties]] = {}

    # Categorize weights
    for weight_fqn, weight_tensor in exported_program.state_dict.items():
        if not isinstance(weight_tensor, torch.Tensor):
            raise AssertionError("only torch.Tensor is allowed in state_dict")
        if _should_use_pickle(weight_tensor):
            pickled_weights.append((weight_fqn, weight_tensor))
        else:
            raw_weights[weight_fqn] = (weight_tensor, TensorProperties(weight_tensor))

    idx = archive_writer.count_prefix(os.path.join(WEIGHTS_DIR, WEIGHT_FILENAME_PREFIX))

    # Save weights in pickle format
    idx = _save_pickled_tensors(
        pickled_weights,
        archive_writer,
        weights_config,
        WEIGHTS_DIR,
        WEIGHT_FILENAME_PREFIX,
        idx,
        pickle_protocol,
    )

    # Save weights in raw bytes format
    _save_raw_tensors(
        raw_weights,
        model_name,
        archive_writer,
        weights_config,
        WEIGHTS_DIR,
        WEIGHT_FILENAME_PREFIX,
        idx,
    )

    return schema.PayloadConfig(config=weights_config)


def _package_constants(
    model_name: str,
    exported_program: ExportedProgram,
    archive_writer: PT2ArchiveWriter,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> schema.PayloadConfig:
    constants_config: dict[str, schema.PayloadMeta] = {}

    pickled_constants: list[tuple[str, torch.Tensor]] = []
    raw_constants: dict[str, tuple[torch.Tensor, TensorProperties]] = {}
    custom_objects: list[tuple[str, torch._C.ScriptObject]] = []

    # Categorize constants
    for constant_fqn, constant in exported_program.constants.items():
        if isinstance(constant, torch.Tensor):
            if _should_use_pickle(constant):
                pickled_constants.append((constant_fqn, constant))
            else:
                raw_constants[constant_fqn] = (constant, TensorProperties(constant))

        elif isinstance(constant, torch._C.ScriptObject):
            custom_objects.append((constant_fqn, constant))

        else:
            raise RuntimeError(f"Unsupported constant type: {type(constant)}")

    tensor_idx = archive_writer.count_prefix(
        os.path.join(CONSTANTS_DIR, TENSOR_CONSTANT_FILENAME_PREFIX)
    )
    custom_obj_idx = archive_writer.count_prefix(
        os.path.join(CONSTANTS_DIR, CUSTOM_OBJ_FILENAME_PREFIX)
    )

    # Save constants in pickle format
    tensor_idx = _save_pickled_tensors(
        pickled_constants,
        archive_writer,
        constants_config,
        CONSTANTS_DIR,
        TENSOR_CONSTANT_FILENAME_PREFIX,
        tensor_idx,
        pickle_protocol,
    )

    # Save constants in raw bytes format
    _save_raw_tensors(
        raw_constants,
        model_name,
        archive_writer,
        constants_config,
        CONSTANTS_DIR,
        TENSOR_CONSTANT_FILENAME_PREFIX,
        tensor_idx,
    )

    # Handle custom objects
    for constant_fqn, constant in custom_objects:
        path_name = f"{CUSTOM_OBJ_FILENAME_PREFIX}{custom_obj_idx}"
        archive_path = os.path.join(CONSTANTS_DIR, path_name)
        custom_obj_bytes = torch._C._pickle_save(constant)
        archive_writer.write_bytes(archive_path, custom_obj_bytes)

        constants_config[constant_fqn] = schema.PayloadMeta(
            path_name=path_name,
            is_param=False,
            use_pickle=True,
            tensor_meta=None,
        )
        custom_obj_idx += 1

    return schema.PayloadConfig(config=constants_config)


def _package_payload_config(
    archive_writer: PT2ArchiveWriter,
    payload_config: schema.PayloadConfig,
    config_file: str,
) -> None:
    """
    Save the payload config as json file in the archive.
    """
    archive_writer.write_string(
        config_file, json.dumps(_dataclass_to_dict(payload_config))
    )


def _package_exported_programs(
    archive_writer: PT2ArchiveWriter,
    exported_programs: ExportedProgram | dict[str, ExportedProgram] | None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
) -> None:
    if exported_programs is None:
        return

    if isinstance(exported_programs, ExportedProgram):
        exported_programs = {"model": exported_programs}

    if not isinstance(exported_programs, dict):
        raise AssertionError(
            f"Expected exported_programs to be a dict, but got {type(exported_programs)}"
        )

    for model_name, ep in exported_programs.items():
        weights_config = _package_state_dict(
            model_name, ep, archive_writer, pickle_protocol
        )
        weights_config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
        _package_payload_config(archive_writer, weights_config, weights_config_file)

        constants_config = _package_constants(
            model_name, ep, archive_writer, pickle_protocol
        )
        constants_config_file = CONSTANTS_CONFIG_FILENAME_FORMAT.format(model_name)
        _package_payload_config(archive_writer, constants_config, constants_config_file)

        artifact: SerializedArtifact = serialize(
            ep,
            opset_version,
            pickle_protocol,
        )

        archive_writer.write_bytes(
            MODELS_FILENAME_FORMAT.format(model_name), artifact.exported_program
        )
        archive_writer.write_bytes(
            SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name),
            artifact.example_inputs,
        )


def _package_extra_files(
    archive_writer: PT2ArchiveWriter, extra_files: dict[str, Any] | None
) -> None:
    if extra_files is None:
        return

    for extra_file_name, content in extra_files.items():
        archive_writer.write_string(f"{EXTRA_DIR}{extra_file_name}", content)


def _package_executorch_files(
    archive_writer: PT2ArchiveWriter, executorch_files: dict[str, bytes] | None
) -> None:
    if executorch_files is None:
        return

    for file_name, content in executorch_files.items():
        archive_writer.write_bytes(f"{EXECUTORCH_DIR}{file_name}", content)


def package_pt2(
    f: FileLike,
    *,
    exported_programs: ExportedProgram | dict[str, ExportedProgram] | None = None,
    aoti_files: AOTI_FILES | None = None,
    extra_files: dict[str, Any] | None = None,
    opset_version: dict[str, int] | None = None,
    pickle_protocol: int = DEFAULT_PICKLE_PROTOCOL,
    executorch_files: dict[str, bytes] | None = None,
) -> FileLike:
    r"""
    Saves the artifacts to a PT2Archive format. The artifact can then be loaded
    using ``load_pt2``.

    Args:
        f (str | os.PathLike[str] | IO[bytes]): A file-like object (has to
         implement write and flush) or a string containing a file name.

        exported_programs (Union[ExportedProgram, dict[str, ExportedProgram]]):
         The exported program to save, or a dictionary mapping model name to an
         exported program to save. The exported program will be saved under
         models/\*.json. If only one ExportedProgram is specified, this will
         automatically be named "model".

        aoti_files (Union[list[str], dict[str, list[str]]]): A list of files
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

        executorch_files (Optional[dict[str, bytes]]): Optional executorch
         artifacts to save.

    """
    if exported_programs is None and aoti_files is None and extra_files is None:
        raise AssertionError(
            "No value passed in for `exported_programs`, `aoti_files`, and "
            "`extra_files`, implying that you do not plan on saving anything."
        )

    if not (
        (isinstance(f, (io.IOBase, IO)) and f.writable() and f.seekable())
        or (isinstance(f, (str, os.PathLike)) and os.fspath(f).endswith(".pt2"))
        or (isinstance(f, tempfile._TemporaryFileWrapper) and f.name.endswith(".pt2"))
    ):
        # TODO: turn this into an error
        logger.warning(
            "Expect archive file to be a file ending in .pt2, or is a buffer. "
            "Instead got {%s}",
            f,
        )

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

    # pyrefly: ignore [bad-argument-type]
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
        _package_executorch_files(archive_writer, executorch_files)

    if isinstance(f, (io.IOBase, IO)):
        f.seek(0)
    # pyrefly: ignore [bad-return]
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

    def __deepcopy__(self, memo: dict[Any, Any] | None) -> "AOTICompiledModel":
        logger.warning(
            "AOTICompiledModel deepcopy warning: AOTICompiledModel.loader is not deepcopied."
        )
        return AOTICompiledModel(self.loader)


@dataclass
class PT2ArchiveContents:
    exported_programs: dict[str, ExportedProgram]
    aoti_runners: dict[str, AOTICompiledModel]
    extra_files: dict[str, Any]


def _create_flat_tensor_from_bytes(
    tensor_bytes: bytes,
    tensor_meta: schema.TensorMeta,
) -> torch.Tensor:
    """
    Create a flat tensor from raw bytes with dtype, device and requires_grad.
    It will be re-strided based on size, stride, and storage_offset later.
    """
    dtype = deserialize_scalar_type(tensor_meta.dtype)
    size = deserialize_size(tensor_meta.sizes)
    device = deserialize_device(tensor_meta.device)

    if len(tensor_bytes) != 0:
        tensor = torch.frombuffer(
            tensor_bytes, dtype=dtype, requires_grad=tensor_meta.requires_grad
        ).to(device)
    else:
        # cannot call torch.frombuffer() on empty bytes
        logger.warning(
            "Cannot call torch.frombuffer() on empty bytes. "
            "Creating a tensor with zeros as workaround."
        )
        tensor = torch.zeros(size, dtype=dtype, device=device)

    return tensor


def _build_file_map(
    archive_reader: PT2ArchiveReader,
    config: schema.PayloadConfig,
    base_dir: str,
) -> dict[str, torch.Tensor]:
    """
    Build a map from file path to the payload in flat tensor format.
    """
    file_map: dict[str, torch.Tensor] = {}
    for payload_meta in config.config.values():
        # skip pickled objects
        if payload_meta.use_pickle:
            continue
        # skip files that already exist in the map
        if payload_meta.path_name in file_map:
            continue

        tensor_bytes = archive_reader.read_bytes(
            os.path.join(base_dir, payload_meta.path_name)
        )
        if payload_meta.tensor_meta is None:
            raise AssertionError("payload_meta.tensor_meta cannot be None")
        tensor = _create_flat_tensor_from_bytes(tensor_bytes, payload_meta.tensor_meta)
        file_map[payload_meta.path_name] = tensor

    return file_map


def _load_payload_config(
    archive_reader: PT2ArchiveReader,
    config_file: str,
) -> schema.PayloadConfig:
    """
    Load and parse a payload config from the archive.
    """
    return _dict_to_dataclass(
        schema.PayloadConfig,
        json.loads(archive_reader.read_string(config_file)),
    )


def _load_state_dict(
    archive_reader: PT2ArchiveReader,
    model_name: str,
) -> dict[str, torch.Tensor] | bytes:
    # Make it BC compatible with legacy weight files
    legacy_weights_file = f"{WEIGHTS_DIR}{model_name}.pt"
    if legacy_weights_file in archive_reader.get_file_names():
        logger.warning(
            "You are loading weight from the legacy format. "
            "Please generate a new pt2 file using torch.export.save()."
        )
        return archive_reader.read_bytes(legacy_weights_file)
    else:
        weights_config_file = WEIGHTS_CONFIG_FILENAME_FORMAT.format(model_name)
        if weights_config_file not in archive_reader.get_file_names():
            raise AssertionError(f"{weights_config_file} not found in PT2 archive")
        weights_config = _load_payload_config(archive_reader, weights_config_file)
        # construct the mapping from file name (e.g. weight_0) to flat weight payload
        state_dict_file_map = _build_file_map(
            archive_reader, weights_config, WEIGHTS_DIR
        )
        # chain the mapping weight FQN -> weight file name -> strided weight payload
        # so that the aliasing of weights is preserved
        state_dict: dict[str, torch.Tensor] = {}
        for weight_fqn, payload_meta in weights_config.config.items():
            if payload_meta.use_pickle:
                weight_bytes = archive_reader.read_bytes(
                    os.path.join(WEIGHTS_DIR, payload_meta.path_name)
                )
                state_dict[weight_fqn] = torch.load(
                    io.BytesIO(weight_bytes), weights_only=False
                )
            else:
                tensor_meta = payload_meta.tensor_meta
                if tensor_meta is None:
                    raise AssertionError(
                        "tensor_meta cannot be None for non-pickled weight"
                    )
                weight_tensor = torch.as_strided(
                    input=state_dict_file_map[payload_meta.path_name],
                    size=deserialize_size(tensor_meta.sizes),
                    stride=deserialize_stride(tensor_meta.strides),
                    storage_offset=deserialize_storage_offset(
                        tensor_meta.storage_offset
                    ),
                )
                if payload_meta.is_param:
                    state_dict[weight_fqn] = torch.nn.Parameter(
                        weight_tensor, requires_grad=tensor_meta.requires_grad
                    )
                else:
                    state_dict[weight_fqn] = weight_tensor

        return state_dict


def _load_constants(
    archive_reader: PT2ArchiveReader,
    model_name: str,
) -> dict[str, torch.Tensor] | bytes:
    # Make it BC compatible with legacy constant files
    legacy_constants_file = f"{CONSTANTS_DIR}{model_name}.pt"
    if legacy_constants_file in archive_reader.get_file_names():
        logger.warning(
            "You are loading constant from the legacy format. "
            "Please generate a new pt2 file using torch.export.save()."
        )
        return archive_reader.read_bytes(legacy_constants_file)
    else:
        constants_config_file = CONSTANTS_CONFIG_FILENAME_FORMAT.format(model_name)
        if constants_config_file not in archive_reader.get_file_names():
            raise AssertionError(f"{constants_config_file} not found in PT2 archive")
        constants_config = _load_payload_config(archive_reader, constants_config_file)
        # construct the mapping from file name (e.g. constant_0) to constant payload
        constant_file_map = _build_file_map(
            archive_reader, constants_config, CONSTANTS_DIR
        )
        # chain the mapping constant FQN -> constant file name -> strided constant payload
        # so that the aliasing of constants is preserved
        constants: dict[str, torch.Tensor] = {}
        for constant_fqn, payload_meta in constants_config.config.items():
            path_name = payload_meta.path_name
            if path_name.startswith(TENSOR_CONSTANT_FILENAME_PREFIX):
                if payload_meta.use_pickle:
                    constant_bytes = archive_reader.read_bytes(
                        os.path.join(CONSTANTS_DIR, path_name)
                    )
                    constants[constant_fqn] = torch.load(
                        io.BytesIO(constant_bytes), weights_only=False
                    )
                else:
                    tensor_meta = payload_meta.tensor_meta
                    if tensor_meta is None:
                        raise AssertionError(
                            "tensor_meta cannot be None for non-pickled constant"
                        )
                    constant_tensor = torch.as_strided(
                        input=constant_file_map[path_name],
                        size=deserialize_size(tensor_meta.sizes),
                        stride=deserialize_stride(tensor_meta.strides),
                        storage_offset=deserialize_storage_offset(
                            tensor_meta.storage_offset
                        ),
                    )
                    constants[constant_fqn] = constant_tensor

            elif path_name.startswith(CUSTOM_OBJ_FILENAME_PREFIX):
                constant_bytes = archive_reader.read_bytes(
                    os.path.join(CONSTANTS_DIR, path_name)
                )
                constants[constant_fqn] = torch._C._pickle_load_obj(constant_bytes)

            else:
                raise RuntimeError(f"Unsupported constant type: {path_name}")

        return constants


def _load_exported_programs(
    archive_reader: PT2ArchiveReader,
    file_names: list[str],
    expected_opset_version: dict[str, int] | None,
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

        sample_inputs_file = SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name)
        serialized_sample_inputs = archive_reader.read_bytes(sample_inputs_file)

        from torch._export.serde.serialize import _bytes_to_dataclass

        exported_program_bytes = archive_reader.read_bytes(file)
        serialized_exported_program = _bytes_to_dataclass(
            schema.ExportedProgram, exported_program_bytes
        )
        state_dict = _load_state_dict(archive_reader, model_name)
        constants = _load_constants(archive_reader, model_name)

        ep = ExportedProgramDeserializer(expected_opset_version).deserialize(
            serialized_exported_program,
            state_dict,
            constants,
            serialized_sample_inputs,
        )

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


def _load_aoti(
    file: str,
    model_name: str,
    run_single_threaded: bool,
    num_runners: int,
    device_idx: int,
) -> AOTICompiledModel:
    loaded_metadata = torch._C._aoti.AOTIModelPackageLoader.load_metadata_from_package(  # type: ignore[attr-defined]
        file, model_name
    )

    device = loaded_metadata["AOTI_DEVICE_KEY"]
    current_device_info = torch._inductor.codecache.get_device_information(device)

    for k, v in current_device_info.items():
        if k in loaded_metadata:
            if v != loaded_metadata[k]:
                logger.warning(
                    "Device information mismatch for %s: %s vs %s. "
                    "This could cause some issues when loading the AOTInductor compiled artifacts.",
                    k,
                    v,
                    loaded_metadata[k],
                )

    aoti_compiled_model = AOTICompiledModel(
        torch._C._aoti.AOTIModelPackageLoader(
            file,
            model_name,
            run_single_threaded,
            num_runners,
            device_idx,
        )
    )

    return aoti_compiled_model


def load_pt2(
    f: FileLike,
    *,
    expected_opset_version: dict[str, int] | None = None,
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

    from torch._inductor.cpp_builder import normalize_path_separator

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
    # pyrefly: ignore [bad-argument-type]
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
                file_end = normalize_path_separator(
                    file_end
                )  # Win32 need normalize path before split.
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
                    model_name: _load_aoti(
                        tf.name,
                        model_name,
                        run_single_threaded,
                        num_runners,
                        device_index,
                    )
                    for model_name in aoti_model_names
                }
        else:
            aoti_runners = {}
    else:
        aoti_runners = {
            model_name: _load_aoti(
                f,
                model_name,
                run_single_threaded,
                num_runners,
                device_index,
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
