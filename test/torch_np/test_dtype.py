# Owner(s): ["module: dynamo"]

import numpy as np
import pytest

import torch._numpy as tnp

dtype_names = [
    "bool_",
    *[f"int{w}" for w in [8, 16, 32, 64]],
    "uint8",
    *[f"float{w}" for w in [16, 32, 64]],
    *[f"complex{w}" for w in [64, 128]],
]
np_dtype_params = []
np_dtype_params.append(pytest.param("bool", "bool", id="'bool'"))
np_dtype_params.append(
    pytest.param(
        "bool",
        np.dtype("bool"),
        id="np.dtype('bool')",
        marks=pytest.mark.xfail(reason="XXX: np.dtype() objects not supported"),
    )
)
for name in dtype_names:
    np_dtype_params.append(pytest.param(name, name, id=repr(name)))
    np_dtype_params.append(
        pytest.param(
            name,
            getattr(np, name),
            id=f"np.{name}",
            marks=pytest.mark.xfail(reason="XXX: namespaced dtypes not supported"),
        )
    )
    np_dtype_params.append(
        pytest.param(
            name,
            np.dtype(name),
            id=f"np.dtype({name!r})",
            marks=pytest.mark.xfail(reason="XXX: np.dtype() objects not supported"),
        )
    )


@pytest.mark.parametrize("name, np_dtype", np_dtype_params)
def test_convert_np_dtypes(name, np_dtype):
    tnp_dtype = tnp.dtype(np_dtype)
    if name == "bool_":
        assert tnp_dtype == tnp.bool_
    elif tnp_dtype.name == "bool_":
        assert name.startswith("bool")
    else:
        assert tnp_dtype.name == name


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
