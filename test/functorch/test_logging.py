import torch
from torch.testing._internal.logging_utils import LoggingTestCase
from torch.testing._internal.dynamo_logging_utils import make_test
from torch._functorch.aot_autograd import aot_function
from torch._functorch.compilers import nop

class TestAOTLogging(LoggingTestCase):

    @make_test("aot")
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
