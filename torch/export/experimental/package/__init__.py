import ctypes
import io
import logging
import os
from typing import Any, Dict, Optional, Tuple

###############################################################################
#
# This file contains the code to package a model for Sigmoid in open source.
# Please do not introduce fbcode dependencies here. (e.g. aiplatform, fbgemm, thrift)
#
###############################################################################
import torch

from torch.export.experimental.package.pt2_archive import PT2ArchiveWriter
from torch._C.nativert.pt2_archive_constants import (  # @manual=//sigmoid/core/package:pt2_archive_constants_pybind
    CONSTANTS_DIR,
    CUSTOM_OBJ_FILENAME_PREFIX,
    MODELS_FILENAME_FORMAT,
    SAMPLE_INPUTS_FILENAME_FORMAT,
    TENSOR_CONSTANT_FILENAME_PREFIX,
    WEIGHT_FILENAME_PREFIX,
    WEIGHTS_DIR,
)
from torch._export.serde.schema import Model, Program

from torch._export.serde.serialize import (
    _enable_graph_inputs_of_type_nn_module,
    _to_json_bytes,
    ExportedProgramSerializer,
)
from torch.export import ExportedProgram
from torch.utils import _pytree as pytree

logger = logging.getLogger(__name__)


def get_raw_tensor_bytes(value: torch.Tensor) -> bytes:
    # NOTE: don't chain .cpu() with .data_ptr(). If an HtoD copy needs to be
    # performed, the CPU copy needs to be kept alive when its underlying
    # memory is accessed.
    if value.data_ptr():
        cpu_tensor = value.cpu().contiguous()
        # we store the raw bytes of tensor. Tensor metadata is stored separately
        value_bytes = bytes(
            ctypes.cast(
                cpu_tensor.data_ptr(),
                ctypes.POINTER(ctypes.c_ubyte * value.element_size() * value.numel()),
            ).contents
        )
    else:
        # for empty tensor
        value_bytes = bytes()
    return value_bytes


def _package_state_dict(
    exported_program: ExportedProgram,
    zip_file: PT2ArchiveWriter,
) -> Dict[str, str]:
    idx = zip_file.count_prefix(os.path.join(WEIGHTS_DIR, WEIGHT_FILENAME_PREFIX))

    qual_name_to_id = {}  # Map from tensor name to its name in xl_weights folder

    for name, tensor in exported_program.state_dict.items():
        if tensor.is_meta:
            logger.error(
                f"Skipping state_dict packing of {name} since it's a meta tensor"
            )
            continue

        param_name = f"{WEIGHT_FILENAME_PREFIX}{idx}"
        idx += 1

        qual_name_to_id[name] = param_name

        archive_path = os.path.join(WEIGHTS_DIR, param_name)
        tensor_bytes = get_raw_tensor_bytes(tensor)
        zip_file.write_bytes(archive_path, tensor_bytes)

    return qual_name_to_id


def _package_constants(
    exported_program: ExportedProgram,
    zip_file: PT2ArchiveWriter,
) -> Dict[str, Any]:
    tensor_idx = zip_file.count_prefix(
        os.path.join(CONSTANTS_DIR, TENSOR_CONSTANT_FILENAME_PREFIX)
    )
    custom_obj_idx = zip_file.count_prefix(
        os.path.join(CONSTANTS_DIR, CUSTOM_OBJ_FILENAME_PREFIX)
    )

    qual_name_to_id = {}  # Map from constant name to its name in constants folder

    for name, constant in exported_program.constants.items():
        if isinstance(constant, torch.Tensor):
            # Save the constant tensors the same way we save weights
            tensor_name = f"{TENSOR_CONSTANT_FILENAME_PREFIX}{tensor_idx}"
            tensor_idx += 1

            qual_name_to_id[name] = tensor_name

            archive_path = os.path.join(CONSTANTS_DIR, tensor_name)
            tensor_bytes = get_raw_tensor_bytes(constant)
            zip_file.write_bytes(archive_path, tensor_bytes)

        elif isinstance(constant, torch._C.ScriptObject):
            # CustomClassHolder objects implement their own pickle saving
            # functions.
            logger.info(f"saving script object {name}")

            custom_obj_name = f"{CUSTOM_OBJ_FILENAME_PREFIX}{custom_obj_idx}"
            custom_obj_idx += 1

            qual_name_to_id[name] = custom_obj_name

            archive_path = os.path.join(CONSTANTS_DIR, custom_obj_name)

            custom_obj_bytes = torch._C._pickle_save(constant)
            zip_file.write_bytes(archive_path, custom_obj_bytes)

        else:
            raise RuntimeError(f"Serializing constant type {type(constant)} nyi")

    return qual_name_to_id


# `sample_inputs` will be pytree_flatten as a python list, and saved via `torch.save()`
# in the zip archive as "data/sample_inputs/<model_name>.pt".
# In C++, this can be loaded via `torch::pickle_load`.
# See sigmoid::ModelRunner::loadSampleInputs() for more details.
def _package_sample_inputs(
    sample_args: Tuple[pytree.PyTree, ...],
    sample_kwargs: Dict[str, pytree.PyTree],
    zip_file: PT2ArchiveWriter,
    model_name: str,
) -> str:
    sample_inputs_path = SAMPLE_INPUTS_FILENAME_FORMAT.format(model_name)
    buffer = io.BytesIO()

    # Convert torch.nn.Parameter to torch.Tensor
    # This is needed because torch::pickle_load() doesn't support torch.nn.Parameter
    def get_tensor(x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.nn.Parameter):
            return x.data
        else:
            return x

    # args must be a tuple, not a list
    sample_args = tuple(pytree.tree_map(get_tensor, sample_args))

    # kwargs must be a dict
    sample_kwargs = pytree.tree_map(get_tensor, sample_kwargs)

    torch.save((sample_args, sample_kwargs), buffer)

    zip_file.write_bytes(sample_inputs_path, buffer.getvalue())

    return sample_inputs_path


def package_model(
    exported_program: ExportedProgram,
    model_name: str,
    zip_file: PT2ArchiveWriter,
    delegates: Optional[Dict[str, ExportedProgram]] = None,
) -> None:
    """
    Saving in the format that's compatible with sigmoid ModelRunner.
    """

    def _make_program(
        ep: ExportedProgram,
    ) -> Program:
        with _enable_graph_inputs_of_type_nn_module(ep.example_inputs):
            return Program(
                methods={
                    "forward": ExportedProgramSerializer()
                    .serialize(ep)
                    .exported_program
                },
            )

    if delegates is None:
        delegates = {}

    # Packaging for Weight
    tensor_path_map = _package_state_dict(exported_program, zip_file)
    # Packaging for Constants (tensor constants, custom class obj)
    constant_path_map = _package_constants(exported_program, zip_file)
    example_args, example_kwargs = exported_program.example_inputs

    # Packaging for input samples
    assert (
        example_args is not None or example_kwargs is not None
    ), "PT2 Archive requires sample inputs to be present"
    sample_inputs_path = _package_sample_inputs(  # noqa
        example_args, example_kwargs, zip_file, model_name
    )

    model_json = Model(
        name=model_name,
        tensorPaths=tensor_path_map,
        program=_make_program(exported_program),
        delegates={key: _make_program(ep) for key, ep in delegates.items()},
        deviceAllocationMap={},
        constantPaths=constant_path_map,
    )
    model_bytes: bytes = _to_json_bytes(model_json)

    # Packaging for model
    zip_file.write_bytes(MODELS_FILENAME_FORMAT.format(model_name), model_bytes)

    # Include readable graph for debugging
    zip_file.write_string(
        f"models/debug/{model_name}_readable.txt", str(exported_program)
    )
    zip_file.write_string(
        f"models/debug/{model_name}_device_annotation.txt",
        exported_program.graph_module.print_readable(
            print_output=False, include_device=True
        ),
    )
