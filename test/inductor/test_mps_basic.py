# Owner(s): ["module: mps"]
import importlib
import os
import sys

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import get_all_dtypes
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MACOS_VERSION,
    parametrize,
)


MPS_UNSUPPORTED_TYPES = [torch.double, torch.cdouble] + (
    [torch.bfloat16] if MACOS_VERSION < 14.0 else []
)
MPS_DTYPES = [t for t in get_all_dtypes() if t not in MPS_UNSUPPORTED_TYPES]

importlib.import_module("filelock")

pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from inductor.test_torchinductor import (  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    check_model_gpu,
    CommonTemplate,
    TestCase,
)


# TODO: Remove this file.
# This tests basic MPS compile functionality


class MPSBasicTests(TestCase):
    common = check_model_gpu
    device = "mps"

    test_add_const_int = CommonTemplate.test_add_const_int

    @parametrize("dtype", MPS_DTYPES)
    def test_add(self, dtype):
        self.common(
            lambda a, b: a + b,
            (
                make_tensor(1024, device="mps", dtype=dtype),
                make_tensor(1024, dtype=dtype, device="mps"),
            ),
        )

    def test_acos(self):
        self.common(lambda x: x.acos(), (torch.rand(1024),))

    def test_sliced_input(self):
        self.common(
            lambda x: x[:, ::2].sin() + x[:, 1::2].cos(), (torch.rand(32, 1024),)
        )

    def test_where(self):
        def foo(x):
            rc = x.abs().sqrt()
            rc[x < 0] = -5
            return rc

        self.common(foo, (torch.rand(1024),))

    @parametrize("dtype", MPS_DTYPES)
    def test_cast(self, dtype):
        self.common(lambda a: a.to(dtype), (torch.rand(1024),))

    def test_broadcast(self):
        self.common(torch.add, (torch.rand(32, 1024), torch.rand(1024)))

    def test_inplace(self):
        def inc_(x):
            x += 1
            return x

        self.common(inc_, (torch.rand(1024),))


instantiate_parametrized_tests(MPSBasicTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if torch.backends.mps.is_available():
        run_tests(needs="filelock")
