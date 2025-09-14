# Owner(s): ["module: dynamo"]
import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class TestCudaStreams(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    def test_stream_enter_exit(self):
        pass

    def test_nested_stream_enter_exit(self):
        pass


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
