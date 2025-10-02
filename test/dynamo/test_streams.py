# Owner(s): ["module: dynamo"]
import weakref

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class TestStreams(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_stream_weakref(self):
        s = torch.Stream()
        weakref.ref(s)

    def test_event_weakref(self):
        e = torch.Event()
        weakref.ref(e)

    def test_run_opcheck(self):
        from torch._dynamo.variables.streams import fork_stream_, join_stream_
        from torch.library import opcheck

        sample_inputs = [
            (1, torch.device("cuda:0"), 1, [torch.randn(3), torch.randn(3)]),
            (
                2,
                torch.device("cuda:0"),
                0,
                [torch.randn(2, 3, device="cuda"), torch.randn(2, 3, device="cuda")],
            ),
        ]
        for args in sample_inputs:
            opcheck(fork_stream_, args)
            opcheck(join_stream_, args)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
