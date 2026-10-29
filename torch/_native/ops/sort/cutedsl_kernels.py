"""CuTeDSL kernels for ``torch.sort(x, out=(values, indices))`` with narrow indices.

Each kernel lives in its own module; this module is the single ``build(spec)``
entry point the AOT export imports.
"""

from . import cutedsl_inplace, cutedsl_register


_KERNELS = {"inplace": cutedsl_inplace, "register": cutedsl_register}


def build(spec: dict) -> dict:
    """AOT builder: one grid point -> kernel object, fake operands and sidecar."""
    module = _KERNELS[spec["kernel"]]
    kernel, fake_args = module.compile_args(spec)
    return {
        "prefix": module.spec_name(spec),
        "fn": kernel,
        "fake_args": fake_args,
        "tensor_args": [
            {
                "name": "mX",
                "dynamic_sizes": [0, 1],
                "dynamic_strides": [0],
                "read_only": True,
            },
            {"name": "mValues", "dynamic_sizes": [0, 1], "dynamic_strides": [0]},
            {"name": "mIndices", "dynamic_sizes": [0, 1], "dynamic_strides": [0]},
        ],
    }
