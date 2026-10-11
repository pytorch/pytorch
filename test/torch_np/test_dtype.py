# Owner(s): ["module: dynamo"]

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
    *[f"uint{w}" for w in [8, 16, 32, 64]],
    *[f"float{w}" for w in [16, 32, 64]],
    *[f"complex{w}" for w in [64, 128]],
]

np_dtype_params = [
    subtest(("bool", "bool"), name="bool"),
    subtest(("bool", numpy.dtype("bool")), name="numpy_dtype('bool')"),
]


for name in dtype_names:
    np_dtype_params.append(subtest((name, name), name=repr(name)))
    np_dtype_params.append(subtest((name, getattr(numpy, name)), name=f"numpy_{name}"))
    np_dtype_params.append(
        subtest((name, numpy.dtype(name)), name=f"numpy_{name!r}")
    )


@instantiate_parametrized_tests
class TestConvertDType(TestCase):
    @parametrize("name, np_dtype", np_dtype_params)
    def test_convert_np_dtypes(self, name, np_dtype):
        tnp_dtype = tnp.dtype(np_dtype)
        if name == "bool_":
            if tnp_dtype != tnp.bool_:
                raise AssertionError(
                    f"Expected tnp_dtype == tnp.bool_, got {tnp_dtype}"
                )
        elif tnp_dtype.name == "bool_":
            if not name.startswith("bool"):
                raise AssertionError(
                    f"Expected name to start with 'bool', got '{name}'"
                )
        else:
            if tnp_dtype.name != name:
                raise AssertionError(
                    f"Expected tnp_dtype.name == '{name}', got '{tnp_dtype.name}'"
                )

    def test_astype_accepts_numpy_dtype(self):
        x = tnp.arange(4)
        y = x.astype(tnp.float32)
        z = x.astype(numpy.float32)
        if y.dtype != z.dtype:
            raise AssertionError(f"Expected matching dtypes, got {y.dtype} vs {z.dtype}")
        empty = tnp.empty((1, 1), dtype=numpy.float32)
        if empty.dtype != tnp.float32:
            raise AssertionError(f"Expected float32, got {empty.dtype}")

    def test_non_numpy_dtype_name_lookalike_raises(self):
        class float32:
            pass

        with self.assertRaises(TypeError):
            tnp.dtype(float32)

        class _Lookalike:
            name = "float32"

        with self.assertRaises(TypeError):
            tnp.dtype(_Lookalike())


if __name__ == "__main__":
    run_tests()
