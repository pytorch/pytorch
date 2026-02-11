# Owner(s): ["module: dynamo"]
import functools
import re
import unittest
import weakref
from unittest.mock import patch

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
    return remove_file_comment(graph.print_readable(print_output=False))


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
    @requires_multigpu()
    def test_new_event_api(self) -> None:
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import new_event

        def event_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            e0_ind = new_event()
            with torch.Stream(device="cuda:1"):
                get_external_object_by_index(e0_ind).record()
            e1_ind = new_event()
            self.assertNotEqual(e0_ind, e1_ind)
            self.assertNotEqual(
                get_external_object_by_index(e0_ind),
                get_external_object_by_index(e1_ind),
            )
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(1,), kwargs={}
                )
            return gm

        @torch.compile(backend=event_generation_backend)
        def fn(x):
            return x + 1

        fn(torch.ones(2, 2, device="cuda:0"))

    @requires_cuda
    def test_new_stream_api(self) -> None:
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import new_stream

        def stream_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            s0_ind = new_stream()
            s1_ind = new_stream()
            self.assertNotEqual(s0_ind, s1_ind)
            self.assertNotEqual(
                get_external_object_by_index(s0_ind),
                get_external_object_by_index(s1_ind),
            )
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(1,), kwargs={}
                )
            return gm

        @torch.compile(backend=stream_generation_backend)
        def fn(x):
            return x + 1

        fn(torch.ones(2, 2, device="cuda:0"))

    @requires_cuda
    def test_current_stream_api(self) -> None:
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import get_current_stream

        cur_stream = torch.accelerator.current_stream()
        s0 = None

        def stream_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal s0
            s0_ind = get_current_stream(torch.device("cuda:0"))
            self.assertEqual(get_external_object_by_index(s0_ind), cur_stream)
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(s0_ind,), kwargs={}
                )
                gm.graph.call_function(
                    lambda x: self.assertEqual(
                        cur_stream, get_external_object_by_index(x)
                    ),
                    args=(s0_ind,),
                    kwargs={},
                )
            return gm

        @torch.compile(backend=stream_generation_backend)
        def fn(x):
            return x + 1

        fn(torch.ones(2, 2, device="cuda:0"))

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

        # Annotation: {'stream': 0}
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
        return (add_2, add_3)
""",
        )

    @requires_cuda
    def test_stream_backward_simple(self) -> None:
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

        # Annotation: {'stream': 0}
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = None

        # No stacktrace found for following nodes
        record_event_default = torch.ops.streams.record_event.default(2, 0);  record_event_default = None
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(2, 1, mul_3);  mul_3 = sync_dealloc_default = None
        return (add_3, add_2)
""",
        )

    @requires_cuda
    def test_stream_backward_sync(self) -> None:
        def fn(x, y):
            s2 = torch.Stream()
            s0 = torch.Stream()
            with s0:
                y0 = 2 * x + y
            with s2:
                z = 2 * x + y

            return y0, z

        inp = (
            torch.ones(2, 2, device="cuda:0", requires_grad=True) + 1,
            torch.ones(2, 2, device="cuda:0", requires_grad=True),
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

        # No stacktrace found for following nodes
        record_event_default = torch.ops.streams.record_event.default(2, 1);  record_event_default = None
        wait_event_default = torch.ops.streams.wait_event.default(2, 0);  wait_event_default = None

        # Annotation: {'stream': 0}
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = None

        # No stacktrace found for following nodes
        record_event_default_1 = torch.ops.streams.record_event.default(3, 0);  record_event_default_1 = None
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(3, 1, mul_3);  mul_3 = sync_dealloc_default = None
        return (add_3, add_2)
""",
        )

    @requires_cuda
    def test_event_tracing(self):
        def fn(x) -> None:
            e = torch.Event()
            e.record()
            x.add_(1)
            return x

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        #
        record_event = torch.ops.streams.record_event.default(0, 1);  record_event = None

        #
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 1)
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = None
        return (copy_,)
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

    @requires_cuda
    def test_run_opcheck_wait_record_stream(self):
        from torch._dynamo.variables.streams import wait_stream
        from torch.library import opcheck

        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            s2 = torch.Stream()
            store_user_object_weakrefs(s0, s1, s2)

            sample_inputs = [
                (0, 1),
                (2, 0),
            ]
            for args in sample_inputs:
                opcheck(wait_stream, args)
        finally:
            reset_user_object_tracking()

    @requires_cuda
    def test_record_stream_problem_basic(self):
        # see https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
        # for what this tests/solves for
        # We expect there to be a sync_dealloc op added to the graph for y
        # synchronizing the first stream w/ the second stream after the second stream is finished
        def fn(x):
            e = torch.Event()
            with torch.Stream(device="cuda:0"):
                y = torch.ones(2, 2, device="cuda:0")
                e.record()
                z = y * x

            with torch.Stream(device="cuda:0"):
                e.wait()
                z0 = y * 2 * x

            return z0, z

        inp = (torch.ones(2, 2, device="cuda", requires_grad=True),)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual[1].sum().backward()

        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        ones: "f32[2, 2]" = torch.ops.aten.ones.default([2, 2], device = device(type='cuda', index=0), pin_memory = False)

        # Annotation: {'stream': 2}
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(ones, 2)
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, mul_1);  tangents_1 = mul_1 = None

        # Annotation: {'stream': 1}
        mul_4: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, ones);  tangents_2 = ones = None

        # No stacktrace found for following nodes
        record_event_default = torch.ops.streams.record_event.default(3, 1);  record_event_default = None
        wait_event_default = torch.ops.streams.wait_event.default(3, 2);  wait_event_default = None

        # Annotation: {'stream': 2}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_3, mul_4);  mul_3 = None

        # No stacktrace found for following nodes
        record_event_default_1 = torch.ops.streams.record_event.default(4, 2);  record_event_default_1 = None
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(4, 1, mul_4);  mul_4 = sync_dealloc_default = None
        return (add,)
""",
        )

    @requires_cuda
    def test_record_stream_problem_interleaved(self):
        # see https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
        # for what this tests/solves for
        # This will have interleaved computation where y is
        # first allocated on the first stream used on the second stream
        # used on the first stream again then finally used on the last stream
        def fn(x):
            e = torch.Event()
            with torch.Stream(device="cuda:0"):
                y = torch.ones(2, 2, device="cuda:0")
                z = y * x
                e.record()

            with torch.Stream(device="cuda:0"):
                e.wait()
                z0 = y * 2 * z
                e.record()

            with torch.Stream(device="cuda:0"):
                e.wait()
                z1 = y * x * z0
                e.record()

            with torch.Stream(device="cuda:0"):
                e.wait()
                z2 = y * 4 * z1
                e.record()

            e.wait()
            return z, z1, z2

        inp = (torch.ones(2, 2, device="cuda", requires_grad=True),)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual[1].sum().backward()

        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", mul: "f32[2, 2]", tangents_1: "f32[2, 2]", \
tangents_2: "f32[2, 2]", tangents_3: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        ones: "f32[2, 2]" = torch.ops.aten.ones.default([2, 2], device = device(type='cuda', index=0), pin_memory = False)

        # Annotation: {'stream': 4}
        mul_5: "f32[2, 2]" = torch.ops.aten.mul.Tensor(ones, 4)
        mul_7: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_3, mul_5);  tangents_3 = mul_5 = None

        # No stacktrace found for following nodes
        record_event_default = torch.ops.streams.record_event.default(6, 4);  record_event_default = None
        wait_event_default = torch.ops.streams.wait_event.default(6, 3);  wait_event_default = None

        # Annotation: {'stream': 3}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, mul_7);  tangents_2 = None

        # No stacktrace found for following nodes
        record_event_default_4 = torch.ops.streams.record_event.default(10, 3);  record_event_default_4 = None
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(10, 4, mul_7);  mul_7 = sync_dealloc_default = None

        # Annotation: {'stream': 3}
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(ones, primals_1);  primals_1 = None
        mul_8: "f32[2, 2]" = torch.ops.aten.mul.Tensor(add, mul_3);  mul_3 = None

        # No stacktrace found for following nodes
        record_event_default_1 = torch.ops.streams.record_event.default(7, 3);  record_event_default_1 = None

        # Annotation: {'stream': 2}
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(ones, 2)
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(mul_1, mul);  mul = None

        # Annotation: {'stream': 3}
        mul_9: "f32[2, 2]" = torch.ops.aten.mul.Tensor(add, mul_2);  add = mul_2 = None
        mul_10: "f32[2, 2]" = torch.ops.aten.mul.Tensor(mul_9, ones);  mul_9 = None

        # No stacktrace found for following nodes
        wait_event_default_1 = torch.ops.streams.wait_event.default(7, 2);  wait_event_default_1 = None

        # Annotation: {'stream': 2}
        mul_11: "f32[2, 2]" = torch.ops.aten.mul.Tensor(mul_8, mul_1);  mul_1 = None

        # No stacktrace found for following nodes
        record_event_default_5 = torch.ops.streams.record_event.default(11, 2);  record_event_default_5 = None
        sync_dealloc_default_1 = torch.ops.streams.sync_dealloc.default(11, 3, mul_8);  mul_8 = sync_dealloc_default_1 = None
        record_event_default_2 = torch.ops.streams.record_event.default(8, 2);  record_event_default_2 = None
        wait_event_default_2 = torch.ops.streams.wait_event.default(8, 1);  wait_event_default_2 = None

        # Annotation: {'stream': 1}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_1, mul_11);  tangents_1 = None

        # No stacktrace found for following nodes
        record_event_default_6 = torch.ops.streams.record_event.default(12, 1);  record_event_default_6 = None
        sync_dealloc_default_2 = torch.ops.streams.sync_dealloc.default(12, 2, mul_11);  mul_11 = sync_dealloc_default_2 = None

        # Annotation: {'stream': 1}
        mul_12: "f32[2, 2]" = torch.ops.aten.mul.Tensor(add_1, ones);  add_1 = ones = None

        # No stacktrace found for following nodes
        record_event_default_3 = torch.ops.streams.record_event.default(9, 1);  record_event_default_3 = None
        wait_event_default_3 = torch.ops.streams.wait_event.default(9, 3);  wait_event_default_3 = None

        # Annotation: {'stream': 3}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_10, mul_12);  mul_10 = None

        # No stacktrace found for following nodes
        record_event_default_7 = torch.ops.streams.record_event.default(13, 3);  record_event_default_7 = None
        sync_dealloc_default_3 = torch.ops.streams.sync_dealloc.default(13, 1, mul_12);  mul_12 = sync_dealloc_default_3 = None
        return (add_2,)
""",
        )

    @requires_cuda
    def test_epilogue_copy_streams_inference(self):
        def fn(x):
            with torch.Stream(device="cuda:0"):
                with torch.no_grad():
                    x.add_(2)

            return x

        x = torch.ones(2, 2, requires_grad=True, device="cuda:0")

        inp = (x,)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual.sum().backward()
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        # Annotation: {'stream': 0}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 2)
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = None
        return (copy_,)
""",
        )

    @requires_cuda
    def test_epilogue_copy_streams_external(self):
        @torch.compile(backend="eager")
        def fn(x):
            with torch.Stream(device="cuda:0"):
                x.mul_(3)
            return x.sin()

        x = torch.ones(2, 2, requires_grad=True, device="cuda:0")
        inp = (x.clone(),)
        with self.assertRaisesRegex(
            RuntimeError,
            "Mutations on inputs with user-specified streams are not yet supported",
        ):
            extract_graph(fn, *inp)

    @requires_cuda
    def test_epilogue_copy_stream_tracking(self):
        """
        Test that epilogue copies for mutated inputs use the correct stream.
        This verifies that ViewAndMutationMeta.mutated_inp_stream_indices is
        properly populated and used at runtime.
        Uses a custom autograd.Function where the backward mutates a saved
        tensor on a specific stream.
        """

        class BwMutationWithStream(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x)
                ctx.s1 = torch.Stream(device="cuda:0")
                ctx.s2 = torch.Stream(device="cuda:0")
                # Do computation on stream s2
                with ctx.s2:
                    result = x * 2 + y
                return result

            @staticmethod
            def backward(ctx, grad_output):
                (x,) = ctx.saved_tensors
                # Mutate saved tensor x on stream s1 in backward
                with ctx.s1:
                    x.mul_(2)
                # Compute gradients on stream s2
                with ctx.s2:
                    grad_x = grad_output * 2
                    grad_y = grad_output.clone()
                return grad_x, grad_y, None, None

        def fn(x, y):
            result = BwMutationWithStream.apply(x, y)
            return result

        x = torch.ones(2, 2, requires_grad=True, device="cuda:0")
        y = torch.ones(2, 2, requires_grad=True, device="cuda:0")
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, x.clone(), y.clone())
        self.assertEqual(len(fw_graphs), 1)
        # Forward graph should show computation on stream 1 (s2)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None
        return (add, primals_1, mul)
""",
        )
        # Run backward and check that the epilogue copy uses stream 0 (s1)
        actual.sum().backward()
        # The backward graph should show:
        # 1. Mutation happening on stream 0 (s1)
        # 2. Gradient computation on stream 1 (s2)
        # 3. Epilogue copy for the mutated tensor on stream 0 (s1)
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", mul: "f32[2, 2]", tangents_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2)

        # Annotation: {'stream': 1}
        clone: "f32[2, 2]" = torch.ops.aten.clone.default(tangents_1);  tangents_1 = None

        # Annotation: {'stream': 0} No stacktrace found for following nodes
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(primals_1, mul);  primals_1 = mul = copy_ = None
        return (mul_2, clone)
""",
        )

    @requires_cuda
    def test_inductor_lowering(self):
        with patch("torch._inductor.config.implicit_fallbacks", False):

            @torch.compile()
            def fn(x):
                e = torch.Event()
                x += x + 1
                e.record()
                return x

            inp = (torch.ones(2, 2, device="cuda"),)
            fn(*inp)

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
