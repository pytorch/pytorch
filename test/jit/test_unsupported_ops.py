# Owner(s): ["oncall: jit"]

import os
import sys
import unittest

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# NOTE: FIXING FAILING TESTS
# If you are seeing a test failure from this file, congrats, you improved
# parity between JIT and Python API. Before you fix the test, you must also update
# the corresponding section in documentation that states the unsupported behavior.
# see: `jit_unsupported.rst`


class TestUnsupportedOps(JitTestCase):
    def test_factory_ops_requires_grad_fail(self):
        # Keyword argument {name} unknown is a JIT-only error message,
        # so these functions are succeeding in eager and failing in JIT

        # Complete issue and set of ops is https://github.com/pytorch/pytorch/issues/30761
        # only testing some because they should be fixed all at once
        def ones():
            return torch.ones([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.ones"
        ):
            torch.jit.script(ones)

        def randn():
            return torch.randn([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.randn"
        ):
            torch.jit.script(randn)

        def zeros():
            return torch.zeros([2], requires_grad=True)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Keyword argument requires_grad unknown", "torch.zeros"
        ):
            torch.jit.script(zeros)

    @unittest.skipIf(not torch._C.has_lapack, "PyTorch compiled without Lapack")
    def test_init_ops(self):
        def calculate_gain():
            return torch.nn.init.calculate_gain("leaky_relu", 0.2)

        def eye_():
            return torch.nn.init.eye_(torch.zeros([2, 2]))

        def dirac_():
            return torch.nn.init.dirac_(torch.empty(3, 16, 5, 5))

        def kaiming_uniform_():
            return torch.nn.init.kaiming_normal_(torch.empty(3, 5))

        def orthogonal_():
            return torch.nn.init.orthogonal_(torch.empty(3, 5))

        def sparse():
            return torch.nn.init.sparse_(torch.empty(3, 5), sparsity=0.1)

        for func in [
            calculate_gain,
            eye_,
            dirac_,
            kaiming_uniform_,
            orthogonal_,
            sparse,
        ]:
            # doesn't error in eager
            func()
            with self.assertRaisesRegex(Exception, ""):
                torch.jit.script(func)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
