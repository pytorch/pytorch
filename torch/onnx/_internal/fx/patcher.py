import copy
import io
from typing import List, Union

import torch

# TODO: Remove after https://github.com/huggingface/safetensors/pull/318
try:
    # safetensors is not an exporter requirement, but needed for some huggingface models
    import safetensors  # type: ignore[import]  # noqa: F401
    import transformers  # type: ignore[import]
    from safetensors import torch as safetensors_torch  # noqa: F401

    has_safetensors_and_transformers = True
except ImportError:
    has_safetensors_and_transformers = False


class ONNXTorchPatcher:
    """Context manager to temporarily patch PyTorch during FX-to-ONNX export.

    This class is a collection of "patches" required by FX-to-ONNX exporter.

    This context overrides several torch functions to support symbolic
    export of large scale models.

    torch.load:
        This function is patched to record the files PyTorch stores model
        parameters and buffers. Downstream FX-to-ONNX exporter can create
        initializers from these files.
    torch._util._rebuild_tensor:
        This function is patched to avoid creating real tensors during
        model loading. FakeTensor's are created instead. Real tensors
        cannot be fitted into single machine's memory for the targeted
        model scale.
    torch.fx._symbolic_trace._wrapped_methods_to_patch:
        This list is extended with (torch.Tensor, "__getitem__") so that
        weight[x, :, y] becomes exportable with torch.fx.symbolic_trace.
    safetensors.torch.load_file:
        This function is patached to allow safetensors to be loaded within
        FakeTensorMode. Remove after https://github.com/huggingface/safetensors/pull/318

    Search for ONNXTorchPatcher in test_fx_to_onnx_with_onnxruntime.py for
    example usage.

    TODO: Should this really be a global patcher? Can we make it a local patcher?
        A reason for splitting this into several patchers is to patch one part of the code
        as a collateral damage of patching another part of the code. For example, we
        for tracing model with torch._dynamo.export, we don't need to patch
        `torch.fx._symbolic_trace._wrapped_methods_to_patch`
    """

    def __init__(self):
        # List of file paths processed by torch.load.
        self.paths: List[Union[str, io.BufferedIOBase]] = []

        def torch_load_wrapper(f, *args, **kwargs):
            # Record path.
            self.paths.append(f)
            # Then, call the original torch.load.
            return self.torch_load(f, *args, **kwargs)

        def torch__util__rebuild_tensor_wrapper(storage, storage_offset, size, stride):
            from torch._subclasses.fake_tensor import FakeTensorMode
            from torch.utils._mode_utils import no_dispatch
            from torch.utils._python_dispatch import _get_current_dispatch_mode

            def _rebuild_real_tensor(storage, storage_offset, size, stride):
                t = torch.tensor(
                    [], dtype=storage.dtype, device=storage._untyped_storage.device
                )
                return t.set_(storage._untyped_storage, storage_offset, size, stride)

            mode = _get_current_dispatch_mode()
            if isinstance(mode, FakeTensorMode):
                # Create a real tensor and then convert it to FakeTensor.
                # We cannot directly create a FakeTensor because it tensor.set_(...)
                # is not supported in FakeTensorMode dispatcher.

                with no_dispatch():
                    t = _rebuild_real_tensor(storage, storage_offset, size, stride)
                return mode.from_tensor(t)

            return _rebuild_real_tensor(storage, storage_offset, size, stride)

        # Original version of torch.load.
        self.torch_load = torch.load
        self.torch__util_rebuild_tensor = torch._utils._rebuild_tensor

        # Wrapper or modified version of torch functions.
        self.torch_load_wrapper = torch_load_wrapper
        self.torch__util_rebuild_tensor_wrapper = torch__util__rebuild_tensor_wrapper

        if has_safetensors_and_transformers:

            def safetensors_load_file_wrapper(filename, device="cpu"):
                result = {}
                with safetensors.torch.safe_open(
                    filename, framework="pt", device=device
                ) as f:
                    for k in f.keys():
                        fake_mode = torch._guards.detect_fake_mode()
                        if not fake_mode:
                            result[k] = f.get_tensor(k)
                        else:
                            empty_tensor = f.get_slice(k)
                            result[k] = torch.empty(
                                tuple(empty_tensor.get_shape()),
                                dtype=safetensors.torch._getdtype(
                                    empty_tensor.get_dtype()
                                ),
                            )
                return result

            self.safetensors_torch_load_file = safetensors.torch.load_file
            self.safetensors_torch_load_file_wrapper = safetensors_load_file_wrapper
            self.transformers_modeling_utils_safe_load_file = (
                transformers.modeling_utils.safe_load_file
            )

    def __enter__(self):
        torch.load = self.torch_load_wrapper
        torch._utils._rebuild_tensor = self.torch__util_rebuild_tensor_wrapper

        self.torch_fx__symbolic_trace__wrapped_methods_to_patch = (
            torch.fx._symbolic_trace._wrapped_methods_to_patch
        )
        desired_wrapped_methods = copy.deepcopy(
            torch.fx._symbolic_trace._wrapped_methods_to_patch
        )
        if (torch.Tensor, "__getitem__") not in desired_wrapped_methods:
            # Adding `__getitem__` to the patching list will make tensor indexing traceable via
            # torch.fx.symbolic_trace. Otherwise, `tensor[x, :, y]` cannot be traced.
            # This happens because `__getitem__` is neither under torch domain nor an aten operator,
            # so the patching (or similar Proxy-generating mechanism) doesn't happen automatically.
            # Note that torch.fx.symbolic_trace defines FX_PATCH_GETITEM environment variable for
            # enabling the line below for patching.
            desired_wrapped_methods.append((torch.Tensor, "__getitem__"))
        torch.fx._symbolic_trace._wrapped_methods_to_patch = desired_wrapped_methods

        if has_safetensors_and_transformers:
            safetensors.torch.load_file = self.safetensors_torch_load_file_wrapper
            transformers.modeling_utils.safe_load_file = (
                self.safetensors_torch_load_file_wrapper
            )

    def __exit__(self, exc_type, exc_value, traceback):
        torch.load = self.torch_load
        torch._utils._rebuild_tensor = self.torch__util_rebuild_tensor
        torch.fx._symbolic_trace._wrapped_methods_to_patch = (
            self.torch_fx__symbolic_trace__wrapped_methods_to_patch
        )
        if has_safetensors_and_transformers:
            safetensors.torch.load_file = self.safetensors_torch_load_file
            transformers.modeling_utils.safe_load_file = (
                self.transformers_modeling_utils_safe_load_file
            )
