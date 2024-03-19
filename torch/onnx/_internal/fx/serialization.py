from __future__ import annotations

import io
import logging
import os
from typing import Tuple, TYPE_CHECKING, Union

import torch
from torch.onnx import _type_utils as jit_type_utils
from torch.onnx._internal import _beartype

if TYPE_CHECKING:
    import onnx

log = logging.getLogger(__name__)


@_beartype.beartype
def _create_tensor_proto_with_external_data(
    tensor: torch.Tensor, name: str, location: str, basepath: str
) -> onnx.TensorProto:  # type: ignore[name-defined]
    """Create a TensorProto with external data from a PyTorch tensor.
    The external data is saved to os.path.join(basepath, location).

    Args:
        tensor: Tensor to be saved.
        name: Name of the tensor (i.e., initializer name in ONNX graph).
        location: Relative location of the external data file
            (e.g., "/tmp/initializers/weight_0" when model is "/tmp/model_name.onnx").
        basepath: Base path of the external data file (e.g., "/tmp/external_data" while model must be in "/tmp").


    Reference for ONNX's external data format:
        How to load?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L187
        How to save?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L43
        How to set ONNX fields?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L88
    """
    # FIXME: Avoid importing onnx into torch.onnx.
    import onnx

    tensor_proto = onnx.TensorProto()  # type: ignore[attr-defined]
    tensor_proto.name = name
    tensor_proto.data_type = jit_type_utils.JitScalarType.from_dtype(
        tensor.dtype
    ).onnx_type()
    tensor_proto.dims.extend(tensor.shape)
    tensor_proto.data_location = onnx.TensorProto.EXTERNAL  # type: ignore[attr-defined]

    # Settings for saving one tensor per file.
    # Offset is zero because there is no other tensor in the same file.
    key_value_pairs = {
        "location": location,
        "offset": 0,
        "length": tensor.untyped_storage().nbytes(),
    }
    for k, v in key_value_pairs.items():
        entry = tensor_proto.external_data.add()
        entry.key = k
        entry.value = str(v)

    # Actual path to write content of tensor.
    external_data_file_path = os.path.join(basepath, location)
    if os.path.exists(external_data_file_path):
        os.remove(external_data_file_path)

    # Create external data's folder if not exists.
    external_data_dir_path = os.path.dirname(external_data_file_path)
    if not os.path.exists(external_data_dir_path):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs(external_data_dir_path)

    # Create a fresh file.
    with open(external_data_file_path, "xb") as data_file:
        # No need to call "seek" because offset is 0.
        # data_file.seek(0)
        # Write tensor content to the file.
        data_file.write(tensor.numpy(force=True).tobytes())

    return tensor_proto


def _convert_safetensors_to_torch_format(safetensors_file):
    # It this function is called, safetensors is guaranteed to exist
    # because the HF model with safetensors was already loaded and exported to ONNX
    from safetensors import safe_open  # type: ignore[import-not-found]

    tensors = {}
    with safe_open(safetensors_file, framework="pt", device="cpu") as f:  # type: ignore[attr-defined]
        for k in f.keys():
            tensors[k] = f.get_tensor(k).cpu()
    return tensors


# TODO: generalize to allow more checkpoints formats (torch or gguf)
@_beartype.beartype
def save_model_with_external_data(
    basepath: str,
    model_location: str,
    initializer_location: str,
    torch_state_dicts: Tuple[Union[dict, str, io.BytesIO], ...],
    onnx_model: onnx.ModelProto,  # type: ignore[name-defined]
    rename_initializer: bool = False,
) -> None:
    """Load PyTorch tensors from files and add to "onnx_model" as external initializers.

    Output files:
        ONNX model file path:
        ONNX initializer folder: os.path.join(basepath, initializer_location)

    After running this function, you can do
        ort_sess = onnxruntime.InferenceSession(os.path.join(basepath, model_location))
    to execute the model.

    Arguments:
        basepath: Base path of the ONNX external data file (e.g., "/path/to/large_model/").
        model_location: Relative location of the ONNX model file.
            E.g., "model.onnx" so that the model file is saved to
            "<basepath>/model.onnx".
        initializer_location: Relative location of the ONNX initializer folder.
            E.g., "initializers" so that the initializers are saved to
            "<basepath>/initializers/".
            Note: When initializers are >2GB, must be the same as `model_location`.
        torch_state_dicts: Dictionaries or files which contain PyTorch tensors to be saved
            as ONNX initializers. For non-dict arguments, `torch.load` will be used to load them from file-like objects.
        onnx_model: ONNX model to be saved with external initializers.
            If an input name matches a tensor loaded from "torch_state_dicts",
            the tensor will be saved as that input's external initializer.
        rename_initializer: Replaces "." by "_" for all ONNX initializer names.
            Not needed by the official torch.onnx.dynamo_export. This is a hack
            for supporting `FXSymbolicTracer` tracer with fake tensor mode.
            In short, `FXSymbolicTracer` lifts FX parameters (self.linear_weight)
            as inputs (`def forward(self, linear_weight)`) and therefore, `.` cannot be used.
    """
    # FIXME: Avoid importing onnx into torch.onnx.
    import onnx

    onnx_model_with_initializers = onnx.ModelProto()  # type: ignore[attr-defined]
    onnx_model_with_initializers.CopyFrom(onnx_model)
    onnx_input_names = {input.name for input in onnx_model.graph.input}
    for el in torch_state_dicts:
        if isinstance(el, dict):
            # Useful for when state_dict is loaded with torch.load(..., mmap=True, map_location="cpu") by the user
            # Using torch.save wouldn't leverage mmap, leading to higher memory usage
            state_dict = el
        else:
            if isinstance(el, str) and el.endswith(".safetensors"):
                state_dict = _convert_safetensors_to_torch_format(el)
            else:
                try:
                    # Loads checkpoint using memory-map on CPU to support really large models
                    # The underlying torch.UntypedStorage is memory mapped, so state_dict is lazy loaded
                    state_dict = torch.load(el, map_location="cpu", mmap=True)
                except (RuntimeError, ValueError) as e:
                    if "mmap can only be used with files saved with" in str(
                        e
                    ) or isinstance(el, io.BytesIO):
                        log.warning(
                            "Failed to load the checkpoint with memory-map enabled, retrying without memory-map."
                            "Consider updating the checkpoint with mmap by using torch.save() on PyTorch version >= 1.6."
                        )
                        if isinstance(el, io.BytesIO):
                            el.seek(0)  # torch.load from `try:` has read the file.
                        state_dict = torch.load(el, map_location="cpu")
                    else:
                        raise e
        for name, tensor in state_dict.items():
            if rename_initializer:
                # Basically, "transformer.attention.self.query.weight" is mapped
                # to "transformer_attention_self_query_weight" for mimicking the
                # name-modifying code in FX-to-ONNX exporter.
                # See function _replace_get_attr_with_placeholder for details.
                name = name.replace(".", "_")

            # This block tries to match the onnx initializer name with torch parameter/buffer
            #  e.g. A pytorch buffer 'transformer.h.0.attn.bias' can be named 'h.0.attn.bias' in a ONNX initializer
            # For each PyTorch tensor name loaded by torch.load,
            #  1.  Search its best match in ONNX model. E.g., the match of
            #       "transformer_attention_weight" could be "attention_weight".
            #  2.  Set "tensor" as the initializer of the matched ONNX input.
            #      E.g., "tensor" is stored as the initializer of "attention_weight".
            # Step 1 is required because sometimes, tensor names are stored with prefix the dictionary
            # loaded by torch.load.
            if name in onnx_input_names:
                # Same input name shouldn't be matched again
                onnx_input_names.remove(name)
            else:
                for onnx_input_name in onnx_input_names:
                    if onnx_input_name.endswith(name) or name.endswith(onnx_input_name):
                        # Find a match. Change name to the matched ONNX input name, so that we
                        # create initializer with the right ONNX name.
                        name = onnx_input_name
                        onnx_input_names.remove(onnx_input_name)
                        break

            relative_tensor_file_path = os.path.join(initializer_location, name)
            # Create one file per tensor.
            # tensor_proto.raw_data is stored to external file at
            # os.path.join(basepath, relative_tensor_file_path).
            tensor_proto = _create_tensor_proto_with_external_data(
                tensor, name, relative_tensor_file_path, basepath
            )
            # Add the tensor_proto to the ONNX model as an initializer with external data.
            onnx_model_with_initializers.graph.initializer.append(tensor_proto)

    # model_location should be a pure file name such as "file_name.onnx", not "folder/file_name.onnx".
    onnx.save(onnx_model_with_initializers, os.path.join(basepath, model_location))  # type: ignore[attr-defined]
