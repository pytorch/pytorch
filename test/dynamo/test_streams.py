# Owner(s): ["module: dynamo"]
import functools
import re
import unittest
import weakref

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.graph_bytecode_inputs import (
    reset_user_object_tracking,
    store_user_object_weakrefs,
)
from torch._dynamo.testing import extract_graph, remove_trailing_space
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_utils import requires_cuda


requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)


def remove_file_comment(gm_str: str) -> str:
    return remove_trailing_space(re.sub(r"File.*\n", "\n", gm_str))


def print_graph(graph: torch.fx.GraphModule) -> str:
    return remove_file_comment(graph.print_readable())


class TestStreams(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

    @requires_cuda
    def test_stream_weakref(self):
        s = torch.Stream()
        weakref.ref(s)

    @requires_cuda
    def test_event_weakref(self):
        e = torch.Event()
        weakref.ref(e)

    @requires_cuda
    def test_stream_enter_exit(self):
        def fn(x, y, s1, s2):
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2), torch.Stream(), torch.Stream())
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 0}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 1}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2);  add_1 = None
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_2, add);  add_2 = add = None
        return (add_3,)
""",
        )

    @requires_cuda
    @unittest.skip("Needs graph break support with annotation context")
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
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(expected, actual)
        self.assertEqual(len(fw_graphs), 2)
        self.assertExpectedInline(print_graph(fw_graphs[0]), """""")
        self.assertExpectedInline(print_graph(fw_graphs[1]), """""")

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
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

    @requires_cuda
    @requires_multigpu()
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

    @requires_cuda
    @requires_multigpu()
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

    @requires_cuda
    def test_nested_stream_enter_exit(self):
        def fn(x, y, s0, s1, s2):
            with s1:
                with s2:
                    z1 = torch.add(x, y)
            with s0:
                z0 = torch.add(x, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (
            torch.ones(2, 2) + 1,
            torch.ones(2, 2),
            torch.Stream(),
            torch.Stream(),
            torch.Stream(),
        )
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 2}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 2);  add = None
        return (add_1, add_2)
""",
        )

    @unittest.skip("Needs graph break support with annotation context")
    def test_stream_enter_exit_graph_break(self):
        pass

    @unittest.skip("Needs graph break support with annotation context")
    def test_nested_stream_enter_exit_graph_break(self):
        pass

    @requires_cuda
    def test_local_stream_enter_exit(self):
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
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 0}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 0}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2);  add_1 = None
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_2, add);  add_2 = add = None
        return (add_3,)
""",
        )

    @requires_cuda
    def test_local_stream_nested_enter_exit(self):
        def fn(x, y):
            s2 = torch.Stream()
            s1 = torch.Stream()
            s0 = torch.Stream()
            with s1:
                with s2:
                    z1 = torch.add(x, y)
            with s0:
                z0 = torch.add(x, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 0}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 2}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 0}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 2);  add = None
        return (add_1, add_2)
""",
        )

    @requires_cuda
    def test_stream_with_mutation(self):
        def fn(x, y):
            s2 = torch.Stream()
            s1 = torch.Stream()
            s0 = torch.Stream()
            with s1:
                with s2:
                    x.add_(y)
            with s0:
                z1 = torch.add(y, y)
                z0 = torch.add(z1, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 0}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 2}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg1_1, arg1_1)

        # Annotation: {'stream': 2}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, arg1_1);  arg1_1 = None

        # Annotation: {'stream': 0}
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2);  add_1 = None

        #
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
        return (add_2, add_3)
""",
        )

    @requires_cuda
    def test_stream_backward(self) -> None:
        def fn(x, y):
            s2 = torch.Stream()
            s0 = torch.Stream()
            with s0:
                y0 = 2 * x + y
            with s2:
                z = 2 * x + y

            return y0, z

        inp = (
            torch.ones(2, 2, requires_grad=True) + 1,
            torch.ones(2, 2, requires_grad=True),
        )
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2);  primals_1 = None
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2)

        # Annotation: {'stream': 0}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
        return (add, add_1)
""",
        )

        actual[1].sum().backward()
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 0}
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, 2)

        #
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, tangents_1);  tangents_2 = None

        # Annotation: {'stream': 1}
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None

        #
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        return (add_3, add_2)
""",
        )

    @requires_cuda
    def test_run_opcheck_fork_join(self):
        from torch._dynamo.variables.streams import fork_stream, join_stream
        from torch.library import opcheck

        original_stream = torch.accelerator.current_stream()
        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            store_user_object_weakrefs(s0, s1)

            sample_inputs = [
                (0, 1),
                (1, 0),
            ]
            for args in sample_inputs:
                opcheck(fork_stream, args)
                opcheck(join_stream, args)
        finally:
            torch.accelerator.set_stream(original_stream)
            reset_user_object_tracking()

    @requires_cuda
    def test_run_opcheck_wait_record(self):
        from torch._dynamo.variables.streams import record_event, wait_event
        from torch.library import opcheck

        original_stream = torch.accelerator.current_stream()
        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            e0 = torch.Event()
            e1 = torch.Event()
            store_user_object_weakrefs(s0, s1, e0, e1)

            sample_inputs = [
                (2, 0),
                (3, 1),
            ]
            for args in sample_inputs:
                opcheck(wait_event, args)
                opcheck(record_event, args)
        finally:
            torch.accelerator.set_stream(original_stream)
            reset_user_object_tracking()

    def test_is_marked_side_effectful(self):
        self.assertIn(
            torch.ops.streams.fork.default, torch.fx.node._side_effectful_functions
        )
        self.assertIn(
            torch.ops.streams.join.default, torch.fx.node._side_effectful_functions
        )
        self.assertIn(
            torch.ops.streams.wait_event.default,
            torch.fx.node._side_effectful_functions,
        )
        self.assertIn(
            torch.ops.streams.record_event.default,
            torch.fx.node._side_effectful_functions,
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
