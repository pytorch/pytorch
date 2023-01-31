from typing import List

import torch


class FxToOnnxContext:
    """Context manager to make PyTorch friendly to FX-to-ONNX exporter.

    This context overwrides severl torch functions to support symbolic
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

    Search for FxToOnnxContext in test_fx_to_onnx_with_onnxruntime.py for
    example usage.
    """

    def __init__(self):
        # List of file paths processed by torch.load.
        self.paths: List[str] = []

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

    def __enter__(self):
        torch.load = self.torch_load_wrapper
        torch._utils._rebuild_tensor = self.torch__util_rebuild_tensor_wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        torch.load = self.torch_load
        torch._utils._rebuild_tensor = self.torch__util_rebuild_tensor
