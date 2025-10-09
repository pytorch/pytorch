# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch.testing._internal.common_utils import requires_cuda


class TestStreams(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    @requires_cuda
    def test_run_opcheck(self):
        from torch._dynamo.variables.streams import fork_stream, join_stream
        from torch.library import opcheck

        sample_inputs = [
            (0, torch.device("cuda:0"), 1, torch.device("cuda:1")),
            (2, torch.device("cuda:2"), 3, torch.device("cuda:1")),
        ]
        for args in sample_inputs:
            opcheck(fork_stream, args)
            opcheck(join_stream, args)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
