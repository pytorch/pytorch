# mypy: allow-untyped-defs
import copy
import functools
from typing import List, TYPE_CHECKING, Union

import torch


if TYPE_CHECKING:
    import io


# TODO: Remove after https://github.com/huggingface/safetensors/pull/318
@functools.lru_cache(None)
def has_safetensors_and_transformers():
    try:
        # safetensors is not an exporter requirement, but needed for some huggingface models
        import safetensors  # type: ignore[import]  # noqa: F401
        import transformers  # type: ignore[import]  # noqa: F401
        from safetensors import torch as safetensors_torch  # noqa: F401

        return True
    except ImportError:
        return False


class ONNXTorchPatcher:
    """Context manager to temporarily patch PyTorch during FX-to-ONNX export.

    This class is a collection of "patches" required by FX-to-ONNX exporter.

    This context overrides several torch functions to support symbolic
    export of large scale models.

    torch.load:
        This function is patched to record the files PyTorch stores model
        parameters and buffers. Downstream FX-to-ONNX exporter can create
        initializers from these files.
    torch.fx._symbolic_trace._wrapped_methods_to_patch:
        This list is extended with (torch.Tensor, "__getitem__") so that
        weight[x, :, y] becomes exportable with torch.fx.symbolic_trace.
    safetensors.torch.load_file:
        This function is patched to allow safetensors to be loaded within
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
            # Record path for later serialization into ONNX proto
            self.paths.append(f)
            # Then, call the original torch.load.
            return self.torch_load(f, *args, **kwargs)

        # Original version of torch.load.
        self.torch_load = torch.load

        # Wrapper or modified version of torch functions.
        self.torch_load_wrapper = torch_load_wrapper

        if has_safetensors_and_transformers():
            import safetensors
            import transformers

            def safetensors_load_file_wrapper(filename, device="cpu"):
                # Record path for later serialization into ONNX proto
                self.paths.append(filename)
                result = {}
                with safetensors.torch.safe_open(  # type: ignore[attr-defined]
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

        if has_safetensors_and_transformers():
            import safetensors
            import transformers

            safetensors.torch.load_file = self.safetensors_torch_load_file_wrapper
            transformers.modeling_utils.safe_load_file = (
                self.safetensors_torch_load_file_wrapper
            )

    def __exit__(self, exc_type, exc_value, traceback):
        torch.load = self.torch_load
        torch.fx._symbolic_trace._wrapped_methods_to_patch = (
            self.torch_fx__symbolic_trace__wrapped_methods_to_patch
        )
        if has_safetensors_and_transformers():
            import safetensors
            import transformers

            safetensors.torch.load_file = self.safetensors_torch_load_file
            transformers.modeling_utils.safe_load_file = (
                self.transformers_modeling_utils_safe_load_file
            )
