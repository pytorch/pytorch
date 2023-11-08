# Owner(s): ["module: dynamo"]

from unittest import expectedFailure as xfail

import numpy

import torch._numpy as tnp

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    subtest,
    TestCase,
)


dtype_names = [
    "bool_",
    *[f"int{w}" for w in [8, 16, 32, 64]],
    "uint8",
    *[f"float{w}" for w in [16, 32, 64]],
    *[f"complex{w}" for w in [64, 128]],
]

np_dtype_params = []

np_dtype_params = [
    subtest(("bool", "bool"), name="bool"),
    subtest(
        ("bool", numpy.dtype("bool")),
        name="numpy.dtype('bool')",
        decorators=[xfail],  # reason="XXX: np.dtype() objects not supported"),
    ),
]


for name in dtype_names:
    np_dtype_params.append(subtest((name, name), name=repr(name)))

    np_dtype_params.append(
        subtest((name, getattr(numpy, name)), name=f"numpy.{name}", decorators=[xfail])
    )  # numpy namespaced dtypes not supported
    np_dtype_params.append(
        subtest((name, numpy.dtype(name)), name=f"numpy.{name!r}", decorators=[xfail])
    )


@instantiate_parametrized_tests
class TestConvertDType(TestCase):
    @parametrize("name, np_dtype", np_dtype_params)
    def test_convert_np_dtypes(self, name, np_dtype):
        tnp_dtype = tnp.dtype(np_dtype)
        if name == "bool_":
            assert tnp_dtype == tnp.bool_
        elif tnp_dtype.name == "bool_":
            assert name.startswith("bool")
        else:
            assert tnp_dtype.name == name


if __name__ == "__main__":
    run_tests()
