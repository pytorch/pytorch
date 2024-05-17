import glob
import os
import pathlib
from typing import Callable, List, Optional

import torch
import torch._inductor
import torch.utils._pytree as pytree
from torch._export.serde.serialize import deserialize, serialize, SerializedArtifact
from torch._inductor import config
from torch.export import ExportedProgram
from torch.export._tree_utils import reorder_kwargs

from .pt2_archive_constants import (
    AOTINDUCTOR_DIR,
    ARCHIVE_ROOT_NAME,
    CONSTANTS_DIR,
    MODELS_FILENAME_FORMAT,
    SAMPLE_INPUTS_DIR,
    WEIGHTS_DIR,
)


ARCHIVE_VERSION = 0


class PT2ArchiveWriter:
    def __init__(self, archive_path: str):
        self.archive_file = torch._C.PyTorchFileWriter(archive_path)
        self.archive_file.set_min_version(ARCHIVE_VERSION)
        self.write_string("archive_format", "pt2")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def write_bytes(self, name: str, data: bytes) -> None:
        assert isinstance(data, bytes), f"Expected bytes but got {type(data)}"
        self.archive_file.write_record(name, data, len(data))

    def write_string(self, name: str, data: str) -> None:
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

    def close(self) -> None:
        self.archive_file.write_end_of_file()


class PT2ArchiveReader:
    def __init__(self, archive_path: str):
        self.archive_file = torch._C.PyTorchFileReader(archive_path)
        assert self.read_string("archive_format") == "pt2", "Invalid archive format"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # torch._C.PyTorchFileReader doesn't have a close method
        pass

    def read_bytes(self, name: str) -> bytes:
        return self.archive_file.get_record(name)

    def read_string(self, name: str) -> str:
        data = self.read_bytes(name)
        return data.decode()

    def get_file_names(self) -> List[str]:
        return self.archive_file.get_all_records()  # type: ignore[attr-defined]


def _package_exported_program(
    archive_writer: PT2ArchiveWriter, exported_program: ExportedProgram
) -> None:
    exported_artifact: SerializedArtifact = serialize(exported_program)
    archive_writer.write_bytes(
        MODELS_FILENAME_FORMAT.format("model"), exported_artifact.exported_program
    )
    archive_writer.write_bytes(
        os.path.join(WEIGHTS_DIR, "weights.pt"), exported_artifact.state_dict
    )
    archive_writer.write_bytes(
        os.path.join(CONSTANTS_DIR, "constants.pt"), exported_artifact.constants
    )
    archive_writer.write_bytes(
        os.path.join(SAMPLE_INPUTS_DIR, "example_inputs.pt"),
        exported_artifact.example_inputs,
    )


def _package_aoti_files(archive_writer: PT2ArchiveWriter, so_path: str):
    cpp_file_path = so_path[:-3] + ".cpp"
    extern_nodes_file_path = so_path[:-3] + ".json"
    work_dir = pathlib.Path(so_path).parent
    cubin_file_paths = glob.glob(f"{work_dir}/*.cubin")

    package_files = [so_path, cpp_file_path]
    package_files.extend(cubin_file_paths)

    if os.path.isfile(extern_nodes_file_path):
        package_files.append(extern_nodes_file_path)

    for path in package_files:
        filename = os.path.basename(path)
        archive_writer.write_file(f"{AOTINDUCTOR_DIR}{filename}", path)


def _extract_exported_program(archive_reader: PT2ArchiveReader) -> ExportedProgram:
    exported_program_bytes = archive_reader.read_bytes(
        MODELS_FILENAME_FORMAT.format("model")
    )
    state_dict_bytes = archive_reader.read_bytes(
        os.path.join(WEIGHTS_DIR, "weights.pt")
    )
    constants_bytes = archive_reader.read_bytes(
        os.path.join(CONSTANTS_DIR, "constants.pt")
    )
    example_inputs_bytes = archive_reader.read_bytes(
        os.path.join(SAMPLE_INPUTS_DIR, "example_inputs.pt")
    )

    artifact: SerializedArtifact = SerializedArtifact(
        exported_program_bytes,
        state_dict_bytes,
        constants_bytes,
        example_inputs_bytes,
    )

    deserialized_exported_program = deserialize(artifact)
    return deserialized_exported_program


def _extract_so(archive_reader: PT2ArchiveReader, device: str) -> Callable:  # type: ignore[type-arg]
    tmp_output_dir = pathlib.Path("/tmp/aotinductor_loaded_model")
    tmp_output_dir.mkdir(exist_ok=True)

    file_names = archive_reader.get_file_names()
    aoti_files = [file for file in file_names if file.startswith(AOTINDUCTOR_DIR)]

    so_path = None
    for file in aoti_files:
        filename = os.path.basename(file)
        with open(tmp_output_dir / filename, "wb") as f:
            f.write(archive_reader.read_bytes(file))
            if file.endswith(".so"):
                assert so_path is None
                so_path = tmp_output_dir / filename
    assert so_path is not None
    so_path = str(so_path)

    if device == "cpu":
        runner = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1)  # type: ignore[call-arg]
    elif device == "cuda" or device.startswith("cuda:"):
        runner = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device)  # type: ignore[assignment, call-arg]
    else:
        raise RuntimeError("Unsupported device " + device)

    def optimized(*args, **kwargs):
        call_spec = runner.get_call_spec()  # type: ignore[attr-defined]
        in_spec = pytree.treespec_loads(call_spec[0])
        out_spec = pytree.treespec_loads(call_spec[1])
        flat_inputs = pytree.tree_flatten((args, reorder_kwargs(kwargs, in_spec)))[0]
        flat_outputs = runner.run(flat_inputs)  # type: ignore[attr-defined]
        return pytree.tree_unflatten(flat_outputs, out_spec)

    return optimized


def save_package(
    so_path: str,
    exported_program: Optional[ExportedProgram] = None,
) -> str:
    """
    Saves the AOTInductor generated files and the exported program to the PT2Archive
    format.

    Args:
        so_path: The path to AOTInductor's generated .so. We assume that the
        other AOTInductor generated files are in the same directory.

        exported_program: Exported program

    Returns:
        The path to the archive.
    """
    work_dir = config.aot_inductor.output_path or pathlib.Path(so_path).parent
    archive_path = os.path.join(work_dir, f"{ARCHIVE_ROOT_NAME}.zip")

    with PT2ArchiveWriter(archive_path) as archive_writer:
        if exported_program is not None:
            _package_exported_program(archive_writer, exported_program)
        if so_path is not None:
            _package_aoti_files(archive_writer, so_path)

    return archive_path


def load_package(path: str, device: str) -> Callable:  # type: ignore[type-arg]
    with PT2ArchiveReader(path) as archive_reader:
        optimized = _extract_so(archive_reader, device)

    return optimized
