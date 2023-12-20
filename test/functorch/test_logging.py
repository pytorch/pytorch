# Owner(s): ["module: dynamo"]
import torch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test
from torch._functorch.aot_autograd import aot_function
from torch._functorch.compilers import nop
import logging

class TestAOTLogging(LoggingTestCase):

    @make_logging_test(aot=logging.DEBUG)
    def test_logging(self, records):
        def f(x):
            return torch.sin(x)
        compiled_f = aot_function(
            f,
            fw_compiler=nop,
            bw_compiler=nop
        )
        compiled_f(torch.randn(3))
        self.assertGreater(len(records), 0)


if __name__ == '__main__':
    run_tests()
