# Owner(s): ["module: dynamo"]
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

    def test_stream_enter_exit(self):
        s2 = torch.Stream()

        @torch.compile()
        def fn(x, y):
            s1 = torch.cuda.Stream()
            with s2:
                z1 = torch.add(x, y)
            with s1:
                z = torch.add(x, y)
                y = z + 2 + z1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        fn_opt = torch.compile(fn, fullgraph=True)
        actual = fn_opt(*inp)
        self.assertEqual(expected, actual)

    def test_nested_stream_enter_exit(self):
        pass

    def test_stream_enter_exit_graph_break(self):
        pass

    def test_nested_stream_enter_exit_graph_break(self):
        pass

    def test_local_stream_enter_exit(self):
        pass

    def test_local_stream_nested_enter_exit(self):
        pass

    def test_stream_with_mutation(self):
        pass

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
