# Owner(s): ["module: dynamo"]
import weakref

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

    def test_stream_weakref(self):
        s = torch.Stream()
        weakref.ref(s)

    def test_event_weakref(self):
        e = torch.Event()
        weakref.ref(e)

    def test_stream_enter_exit(self):
        def fn(x, y):
            s2 = torch.Stream()
            s1 = torch.Stream()
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        fn_opt = torch.compile(fn, fullgraph=True)
        actual = fn_opt(*inp)
        self.assertEqual(expected, actual)

    def test_stream_context_graph_break(self):
        def fn(x, y):
            s2 = torch.Stream()
            s1 = torch.Stream()
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1
                torch._dynamo.graph_break()
                y = y + 1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        fn_opt = torch.compile(fn)
        actual = fn_opt(*inp)
        self.assertEqual(expected, actual)

    def test_stream_input(self):
        def fn(x, y, s):
            z = torch.add(x, y)
            y = z + 2
            return y, s

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2), torch.Stream(device="cuda"))
        expected = fn(*inp)
        fn_opt = torch.compile(fn, fullgraph=True)
        actual = fn_opt(*inp)
        self.assertEqual(expected, actual)

    def test_local_stream_return(self):
        def fn(x, y):
            s = torch.Stream()
            z = torch.add(x, y)
            y = z + 2
            return y, s

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        fn_opt = torch.compile(fn, fullgraph=True)
        _, s0 = fn_opt(*inp)
        _, s1 = fn_opt(*inp)
        # Streams will be different values for each invocation
        # so don't check for equality
        self.assertIsInstance(s0, torch.Stream)
        # Stream should be newly allocated on each call
        self.assertNotEqual(s0, s1)

    def test_get_current_stream_return(self):
        def fn(x, s):
            with s:
                s0 = torch.accelerator.current_stream()
            return x, s0

        s_inp = torch.Stream(device="cuda")
        inp = (torch.ones(2, 2) + 1, s_inp)
        fn_opt = torch.compile(fn, fullgraph=True)
        _, s0 = fn_opt(*inp)
        _, s1 = fn_opt(*inp)
        self.assertEqual(s_inp, s0)
        self.assertEqual(s0, s1)

    def test_get_current_stream_return_different_device(self):
        def fn(x, s0, s1):
            with s1:
                with s0:
                    s = torch.accelerator.current_stream(torch.device("cuda:1"))
            return s

        s0 = torch.Stream(device="cuda:0")
        s1 = torch.Stream(device="cuda:1")
        inp = (torch.ones(2, 2) + 1, s0, s1)
        fn_opt = torch.compile(fn, fullgraph=True)
        s_act = fn_opt(*inp)
        s_exp = fn(*inp)
        self.assertEqual(s_act, s_exp)

    def test_get_current_stream_return_no_index(self):
        def fn(x, s0, s1):
            with s1:
                with s0:
                    s = torch.accelerator.current_stream(torch.device("cuda"))
            return s

        s0 = torch.Stream(device="cuda:0")
        s1 = torch.Stream(device="cuda:1")
        inp = (torch.ones(2, 2) + 1, s0, s1)
        fn_opt = torch.compile(fn, fullgraph=True)
        s_act = fn_opt(*inp)
        s_exp = fn(*inp)
        self.assertEqual(s_act, s_exp)

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
