import importlib
import os
import tempfile

import torch
from .common import device_from_inputs, fake_tensor_unsupported
from .registry import register_backend

try:
    import numpy as np

    _np_dtype = {
        torch.float16: np.float16,
        torch.float32: np.float32,
        torch.float64: np.float64,
        torch.uint8: np.uint8,
        torch.int8: np.int8,
        torch.int16: np.int16,
        torch.int32: np.int32,
        torch.int64: np.longlong,
        torch.bool: np.bool_,
    }

except ImportError:
    _np_dtype = None


def default_provider(device_type):
    if "ONNXRT_PROVIDER" in os.environ:
        return os.environ["ONNXRT_PROVIDER"]
    return {
        "cpu": "CPUExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        # "TensorrtExecutionProvider" is another option
    }[device_type]


def has_onnxruntime():
    try:
        importlib.import_module("onnxruntime")
        return True
    except ImportError:
        return False


@register_backend
@fake_tensor_unsupported
def onnxrt(gm, example_inputs, *, filename=None, provider=None):
    if filename is None:
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            return onnxrt(gm, example_inputs, filename=tmp.name)

    import onnxruntime  # type: ignore[import]

    assert _np_dtype, "requires numpy"

    device_type = device_from_inputs(example_inputs).type
    example_outputs = gm(*example_inputs)
    output_spec = [
        (o.shape, o.dtype, o.layout, o.device, o.requires_grad) for o in example_outputs
    ]
    input_names = [f"i{i}" for i in range(len(example_inputs))]
    output_names = [f"o{x}" for x in range(len(example_outputs))]

    torch.onnx.export(
        torch.jit.script(gm),
        example_inputs,
        filename,
        input_names=input_names,
        output_names=output_names,
    )
    del example_inputs, example_outputs

    if provider is None:
        provider = default_provider(device_type)
    assert provider in onnxruntime.get_available_providers()
    session = onnxruntime.InferenceSession(filename, providers=[provider])

    def _call(*initial_args):
        binding = session.io_binding()
        args = [a.contiguous() for a in initial_args]
        for name, value in zip(input_names, args):
            dev = value.device
            binding.bind_input(
                name,
                dev.type,
                dev.index or 0,
                _np_dtype[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        outputs = [
            torch.empty(
                shape,
                dtype=dtype,
                layout=layout,
                device=device,
                requires_grad=requires_grad,
            )
            for shape, dtype, layout, device, requires_grad in output_spec
        ]

        for name, value in zip(output_names, outputs):
            dev = value.device
            binding.bind_output(
                name,
                dev.type,
                dev.index or 0,
                _np_dtype[value.dtype],
                value.size(),
                value.data_ptr(),
            )
        session.run_with_iobinding(binding)
        if device_type == "cpu":
            binding.copy_outputs_to_cpu()
        return outputs

    return _call
