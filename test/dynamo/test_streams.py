# Owner(s): ["module: dynamo"]
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
from torch.testing._internal.common_utils import requires_cuda


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
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 2}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 2}
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
    def test_cuda_current_stream_attrs(self):
        """Verify that torch.cuda.current_stream() attributes are accessible
        under torch.compile and match eager behavior."""

        def fn_cuda_stream(x):
            return torch.cuda.current_stream().cuda_stream

        x = torch.zeros(1, device="cuda")
        compiled = torch.compile(fn_cuda_stream, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn_cuda_stream(x))

    @requires_cuda
    def test_cuda_current_stream_with_entered_stream(self):
        """Verify that torch.cuda.current_stream().cuda_stream returns the
        correct value when inside a stream context for a user-created stream."""

        def fn(x, s):
            with s:
                return torch.cuda.current_stream().cuda_stream

        s = torch.cuda.Stream()
        x = torch.zeros(1, device="cuda")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, s), fn(x, s))

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
        # Annotation: {'stream': 2}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 2}
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
        # Annotation: {'stream': 2}
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
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 2);  add = None
        return (add_1, add_2)
""",
        )

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
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg1_1, arg1_1)

        # Annotation: {'stream': 3}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, arg1_1);  arg1_1 = None

        # Annotation: {'stream': 1}
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2);  add_1 = None

        # Annotation: {'stream': 1}
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
        # Annotation: {'stream': 2}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2);  primals_1 = None
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2)

        # Annotation: {'stream': 1}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None
        return (add, add_1, mul, add_1)
""",
        )

        actual[1].sum().backward()
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, mul: "f32[2, 2]", add_1: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, 2)

        #
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, tangents_1);  tangents_2 = None

        # Annotation: {'stream': 2}
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None

        # Annotation: {'stream': 1}
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_2, mul_3)

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps = torch.ops.higher_order.control_deps((mul, add_1, mul_2, add_3, add_2), subgraph_record_event_default, add_1, add_3, add_2);  mul = add_1 = mul_2 = add_3 = add_2 = subgraph_record_event_default = None

        #
        getitem_2: "f32[2, 2]" = control_deps[3]

        # Annotation: {'stream': 1}
        getitem_1: "f32[2, 2]" = control_deps[2];  control_deps = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(3, 2, mul_3);  mul_3 = sync_dealloc_default = None
        return (getitem_1, getitem_2)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]", dep_2: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(3, 1)
            return (record_event_default, dep_0, dep_1, dep_2)
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
        # Annotation: {'stream': 2}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2);  primals_1 = None
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2)

        # Annotation: {'stream': 1}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2);  primals_2 = None
        return (add, add_1, mul, add, add_1)
""",
        )

        actual[1].sum().backward()
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, mul: "f32[2, 2]", add: "f32[2, 2]", add_1: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, 2)

        #
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, tangents_1);  tangents_2 = None

        # Annotation: {'stream': 2}
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps = torch.ops.higher_order.control_deps((mul, add, mul_3, add_2), subgraph_record_event_default, add, mul_3, add_2);  add = mul_3 = add_2 = subgraph_record_event_default = None

        #
        getitem_2: "f32[2, 2]" = control_deps[3]

        # Annotation: {'stream': 2}
        getitem_1: "f32[2, 2]" = control_deps[2]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_1 = torch.ops.higher_order.control_deps((control_deps, mul, add_1, mul_2), subgraph_wait_event_default, add_1, mul_2);  control_deps = mul = add_1 = mul_2 = subgraph_wait_event_default = None

        # Annotation: {'stream': 1}
        getitem_4: "f32[2, 2]" = control_deps_1[2];  control_deps_1 = None

        # Annotation: {'stream': 1}
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_4, getitem_1);  getitem_4 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_2 = torch.ops.higher_order.control_deps((add_3,), subgraph_record_event_default_1, add_3);  add_3 = subgraph_record_event_default_1 = None

        # Annotation: {'stream': 1}
        getitem_5: "f32[2, 2]" = control_deps_2[1];  control_deps_2 = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(4, 2, getitem_1);  getitem_1 = sync_dealloc_default = None
        return (getitem_5, getitem_2)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]", dep_2: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(3, 2)
            return (record_event_default, dep_0, dep_1, dep_2)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(3, 1)
            return (wait_event_default, dep_0, dep_1)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(4, 1)
            return (record_event_default, dep_0)
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
        record_event = torch.ops.streams.record_event.default(1, 0);  record_event = None

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
    def forward(self, getitem: "f32[2, 2]", mul: "f32[2, 2]", mul_1: "f32[2, 2]", mul_2: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 3}
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, mul_1);  tangents_1 = None

        # Annotation: {'stream': 2}
        mul_4: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, getitem);  tangents_2 = getitem = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps_2 = torch.ops.higher_order.control_deps((mul, mul_4), subgraph_record_event_default, mul, mul_4);  mul = mul_4 = subgraph_record_event_default = None

        # Annotation: {'stream': 2}
        getitem_2: "f32[2, 2]" = control_deps_2[2]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_3 = torch.ops.higher_order.control_deps((control_deps_2, mul_1, mul_2, mul_3), subgraph_wait_event_default, mul_2, mul_3);  control_deps_2 = mul_1 = mul_2 = mul_3 = subgraph_wait_event_default = None

        # Annotation: {'stream': 3}
        getitem_4: "f32[2, 2]" = control_deps_3[2];  control_deps_3 = None

        # Annotation: {'stream': 3}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_4, getitem_2);  getitem_4 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_4 = torch.ops.higher_order.control_deps((add,), subgraph_record_event_default_1, add);  add = subgraph_record_event_default_1 = None

        # Annotation: {'stream': 3}
        getitem_5: "f32[2, 2]" = control_deps_4[1];  control_deps_4 = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(5, 2, getitem_2);  getitem_2 = sync_dealloc_default = None
        return (getitem_5,)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(4, 2)
            return (record_event_default, dep_0, dep_1)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(4, 3)
            return (wait_event_default, dep_0, dep_1)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(5, 3)
            return (record_event_default, dep_0)
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
    def forward(self, getitem: "f32[2, 2]", getitem_3: "f32[2, 2]", getitem_2: "f32[2, 2]", getitem_4: "f32[2, 2]", getitem_6: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]", tangents_3: "f32[2, 2]"):
        # Annotation: {'stream': 5}
        mul_7: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_3, getitem_6);  tangents_3 = getitem_6 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps_8 = torch.ops.higher_order.control_deps((mul_7,), subgraph_record_event_default, mul_7);  mul_7 = subgraph_record_event_default = None

        # Annotation: {'stream': 5}
        getitem_8: "f32[2, 2]" = control_deps_8[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_9 = torch.ops.higher_order.control_deps((control_deps_8,), subgraph_wait_event_default);  control_deps_8 = subgraph_wait_event_default = control_deps_9 = None

        # Annotation: {'stream': 4}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, getitem_8);  tangents_2 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_4 = self.subgraph_record_event_default_4
        control_deps_10 = torch.ops.higher_order.control_deps((add,), subgraph_record_event_default_4, add);  add = subgraph_record_event_default_4 = None

        # Annotation: {'stream': 4}
        getitem_9: "f32[2, 2]" = control_deps_10[1];  control_deps_10 = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(10, 5, getitem_8);  getitem_8 = sync_dealloc_default = None

        # Annotation: {'stream': 4}
        mul_8: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_9, getitem_4);  getitem_4 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_11 = torch.ops.higher_order.control_deps((mul_8,), subgraph_record_event_default_1, mul_8);  mul_8 = subgraph_record_event_default_1 = None

        # Annotation: {'stream': 4}
        getitem_10: "f32[2, 2]" = control_deps_11[1]
        mul_9: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_9, getitem_3);  getitem_9 = getitem_3 = None
        mul_10: "f32[2, 2]" = torch.ops.aten.mul.Tensor(mul_9, getitem)

        # No stacktrace found for following nodes
        subgraph_wait_event_default_1 = self.subgraph_wait_event_default_1
        control_deps_12 = torch.ops.higher_order.control_deps((control_deps_11,), subgraph_wait_event_default_1);  control_deps_11 = subgraph_wait_event_default_1 = control_deps_12 = None

        # Annotation: {'stream': 3}
        mul_11: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_10, getitem_2);  getitem_2 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_5 = self.subgraph_record_event_default_5
        control_deps_13 = torch.ops.higher_order.control_deps((mul_11,), subgraph_record_event_default_5, mul_11);  mul_11 = subgraph_record_event_default_5 = None

        # Annotation: {'stream': 3}
        getitem_11: "f32[2, 2]" = control_deps_13[1];  control_deps_13 = None

        # No stacktrace found for following nodes
        sync_dealloc_default_1 = torch.ops.streams.sync_dealloc.default(11, 4, getitem_10);  getitem_10 = sync_dealloc_default_1 = None
        record_event_default_2 = torch.ops.streams.record_event.default(8, 3);  record_event_default_2 = None
        wait_event_default_2 = torch.ops.streams.wait_event.default(8, 2);  wait_event_default_2 = None

        # Annotation: {'stream': 2}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_1, getitem_11);  tangents_1 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_6 = self.subgraph_record_event_default_6
        control_deps_14 = torch.ops.higher_order.control_deps((add_1,), subgraph_record_event_default_6, add_1);  add_1 = subgraph_record_event_default_6 = None

        # Annotation: {'stream': 2}
        getitem_12: "f32[2, 2]" = control_deps_14[1];  control_deps_14 = None

        # No stacktrace found for following nodes
        sync_dealloc_default_2 = torch.ops.streams.sync_dealloc.default(12, 3, getitem_11);  getitem_11 = sync_dealloc_default_2 = None

        # Annotation: {'stream': 2}
        mul_12: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_12, getitem);  getitem_12 = getitem = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_3 = self.subgraph_record_event_default_3
        control_deps_15 = torch.ops.higher_order.control_deps((mul_12,), subgraph_record_event_default_3, mul_12);  mul_12 = subgraph_record_event_default_3 = None

        # Annotation: {'stream': 2}
        getitem_13: "f32[2, 2]" = control_deps_15[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default_3 = self.subgraph_wait_event_default_3
        control_deps_16 = torch.ops.higher_order.control_deps((control_deps_15, mul_9, mul_10), subgraph_wait_event_default_3, mul_10);  control_deps_15 = mul_9 = mul_10 = subgraph_wait_event_default_3 = None

        # Annotation: {'stream': 4}
        getitem_14: "f32[2, 2]" = control_deps_16[1];  control_deps_16 = None

        # Annotation: {'stream': 4}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_14, getitem_13);  getitem_14 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_7 = self.subgraph_record_event_default_7
        control_deps_17 = torch.ops.higher_order.control_deps((add_2,), subgraph_record_event_default_7, add_2);  add_2 = subgraph_record_event_default_7 = None

        # Annotation: {'stream': 4}
        getitem_15: "f32[2, 2]" = control_deps_17[1];  control_deps_17 = None

        # No stacktrace found for following nodes
        sync_dealloc_default_3 = torch.ops.streams.sync_dealloc.default(13, 2, getitem_13);  getitem_13 = sync_dealloc_default_3 = None
        return (getitem_15,)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(6, 5)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(6, 4)
            return wait_event_default

    class subgraph_record_event_default_4(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(10, 4)
            return (record_event_default, dep_0)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(7, 4)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default_1(torch.nn.Module):
        def forward(self):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(7, 3)
            return wait_event_default

    class subgraph_record_event_default_5(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(11, 3)
            return (record_event_default, dep_0)

    class subgraph_record_event_default_6(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(12, 2)
            return (record_event_default, dep_0)

    class subgraph_record_event_default_3(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(9, 2)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default_3(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(9, 4)
            return (wait_event_default, dep_0)

    class subgraph_record_event_default_7(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(13, 4)
            return (record_event_default, dep_0)
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
        # Annotation: {'stream': 1}
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
    def test_control_deps_wrapping_record_event(self) -> None:
        """Test wrapping record_event with control_deps on a two-stream graph.

        The producer stream has work before the record_event, so control_deps
        should capture those same-stream nodes as dependencies.
        """

        def fn(x) -> torch.Tensor:
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            e = torch.Event()

            with s1:
                y = x + 1
                z = y * 2
                e.record()

            with s2:
                e.wait()
                w = z + 3

            return w

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        import operator

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        wrap_all_sync_nodes_with_control_deps(gm)

        # record_event has same-stream deps, and wait_event gets a
        # cross-event dependency on the record's control_deps node.
        control_deps_nodes = list(
            graph.find_nodes(op="call_function", target=control_deps)
        )
        self.assertEqual(len(control_deps_nodes), 2)
        record_ctrl_node = control_deps_nodes[0]

        # Verify getitem nodes were created for pass-through dependencies
        getitem_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target == operator.getitem
            and n.args[0] is record_ctrl_node
        ]
        self.assertGreaterEqual(len(getitem_nodes), 1)

        # Verify the wait's control_deps depends on the record's control_deps
        wait_ctrl_node = control_deps_nodes[1]
        self.assertIn(record_ctrl_node, wait_ctrl_node.args[0])

    @requires_cuda
    def test_control_deps_wrapping_wait_event(self) -> None:
        """Test wrapping wait_event with control_deps on a two-stream graph.

        The consumer stream has the wait_event, and there should be same-stream
        nodes before it that become dependencies.
        """

        def fn(x) -> torch.Tensor:
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            e = torch.Event()

            with s1:
                y = x + 1
                e.record()

            with s2:
                a = x * 3
                e.wait()
                z = y * a

            return z

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        wrap_all_sync_nodes_with_control_deps(gm)

        # Both record_event (deps: [y]) and wait_event (deps: [a]) have
        # same-stream deps, so both get wrapped.
        control_deps_nodes = list(
            graph.find_nodes(op="call_function", target=control_deps)
        )
        self.assertEqual(len(control_deps_nodes), 2)

    @requires_cuda
    def test_control_deps_prevents_invalid_reordering(self) -> None:
        """
        Test that control_deps creates proper data dependencies that prevent invalid reordering.

        This test:
        1. Creates a two-stream graph with control_deps wrapping
        2. Manually moves a node to violate the ordering
        3. Verifies graph.lint() catches the invalid ordering
        """

        def fn(x) -> torch.Tensor:
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            e = torch.Event()

            with s1:
                y = x + 1
                z = y * 2
                e.record()

            with s2:
                e.wait()
                w = z + 3

            return w

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        wrap_all_sync_nodes_with_control_deps(gm)

        control_deps_node = next(
            iter(graph.find_nodes(op="call_function", target=control_deps))
        )

        # Find add_1 (the node after the sync that uses getitem output)
        add_1_node = next(n for n in graph.nodes if n.name == "add_1")

        # Verify valid ordering: control_deps before add_1
        original_order = [n.name for n in graph.nodes if n.op == "call_function"]
        self.assertLess(
            original_order.index("control_deps"),
            original_order.index("add_1"),
            "control_deps should be before add_1 in valid ordering",
        )

        graph.lint()  # Should not raise

        # Manually move add_1 BEFORE control_deps (violates dependency)
        control_deps_node.prepend(add_1_node)

        # Verify the order is now wrong (add_1 comes before control_deps)
        violated_order = [n.name for n in graph.nodes if n.op == "call_function"]
        self.assertLess(
            violated_order.index("add_1"),
            violated_order.index("control_deps"),
            "add_1 should be before control_deps after manual move (violating order)",
        )

        # graph.lint() should catch the invalid ordering
        with self.assertRaises(RuntimeError):
            graph.lint()

    @requires_cuda
    def test_cross_event_deps_multiple_events(self) -> None:
        """Stress test: multiple events across three streams.

        Verifies that each wait_event's control_deps node depends on exactly
        the matching record_event's control_deps node (by event index), not
        on some other event's node.

        Graph layout:
          s1: x+1 -> record(e1) -> x+5 -> record(e2)
          s2: wait(e1) -> x*3
          s3: wait(e2) -> x*7
        """

        def fn(x) -> torch.Tensor:
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            s3 = torch.Stream(device="cuda")
            e1 = torch.Event()
            e2 = torch.Event()

            with s1:
                y = x + 1
                e1.record()
                z = x + 5
                e2.record()

            with s2:
                e1.wait()
                a = x * 3

            with s3:
                e2.wait()
                b = x * 7

            return a + b + y + z

        inp = (torch.ones(2, 2, device="cuda"),)
        # Patch out wrapping so we get the raw graph to manually wrap below.
        with patch(
            "torch._functorch._aot_autograd.graph_capture.wrap_all_sync_nodes_with_control_deps"
        ):
            (
                _,
                _,
                fw_graphs,
                _,
            ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        # Capture event indices before wrapping so we can verify correspondence.
        # Each sync node has args (event_index, stream_index).
        record_event_indices = []
        wait_event_indices = []
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is torch.ops.streams.record_event.default:
                record_event_indices.append(node.args[0])
            elif node.target is torch.ops.streams.wait_event.default:
                wait_event_indices.append(node.args[0])

        self.assertEqual(len(record_event_indices), 2)
        self.assertEqual(len(wait_event_indices), 2)
        # Each wait corresponds to exactly one record by event index
        self.assertEqual(set(record_event_indices), set(wait_event_indices))

        wrap_all_sync_nodes_with_control_deps(gm)

        # 2 records + 2 waits = 4 control_deps nodes. Each record has
        # same-stream deps, and each wait gets a cross-event dep.
        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        self.assertEqual(len(ctrl_nodes), 4)

        # Graph order follows program order: both records (s1 block) before
        # both waits (s2, s3 blocks).
        record_e1_ctrl = ctrl_nodes[0]
        record_e2_ctrl = ctrl_nodes[1]
        wait_e1_ctrl = ctrl_nodes[2]
        wait_e2_ctrl = ctrl_nodes[3]

        # wait(e1) depends on record(e1), not record(e2)
        self.assertIn(record_e1_ctrl, wait_e1_ctrl.args[0])
        self.assertNotIn(record_e2_ctrl, wait_e1_ctrl.args[0])

        # wait(e2) depends on record(e2), not record(e1)
        self.assertIn(record_e2_ctrl, wait_e2_ctrl.args[0])
        self.assertNotIn(record_e1_ctrl, wait_e2_ctrl.args[0])

        graph.lint()

    @requires_cuda
    def test_cross_event_deps_event_reuse(self) -> None:
        """Test that reusing an event updates the cross-event dependency.

        When the same event is recorded twice with work in between, the wait
        should depend on the LAST record's control_deps node (matching CUDA
        semantics where record() overwrites the event).
        """

        def fn(x) -> torch.Tensor:
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            e = torch.Event()

            with s1:
                y = x + 1
                e.record()  # first record
                z = y * 2
                e.record()  # second record, reuses same event

            with s2:
                e.wait()
                w = x + 3

            return w + z

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        # 2 records + 1 wait = 3 control_deps nodes
        self.assertEqual(len(ctrl_nodes), 3)

        record1_ctrl = ctrl_nodes[0]
        record2_ctrl = ctrl_nodes[1]
        wait_ctrl = ctrl_nodes[2]

        # The wait depends on the SECOND record (last write wins), not the first
        self.assertIn(record2_ctrl, wait_ctrl.args[0])
        self.assertNotIn(record1_ctrl, wait_ctrl.args[0])

        graph.lint()

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
        # Annotation: {'stream': 2}
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
        # Annotation: {'stream': 2}
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2)

        # Annotation: {'stream': 2}
        clone: "f32[2, 2]" = torch.ops.aten.clone.default(tangents_1);  tangents_1 = None

        # Annotation: {'stream': 1} No stacktrace found for following nodes
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
        self.assertIn(
            torch.ops.streams.synchronize_event.default,
            torch.fx.node._side_effectful_functions,
        )

    @requires_cuda
    def test_backward_sync_control_deps_e2e(self) -> None:
        """
        End-to-end test verifying that backward sync nodes are wrapped with control_deps.

        This test:
        1. Creates a function with multiple streams
        2. Compiles it and runs backward
        3. Verifies control_deps nodes are present in the backward graph
        """
        from torch._inductor.fx_passes.control_dependencies import control_deps

        def fn(x, y):
            s0 = torch.Stream()
            s1 = torch.Stream()
            with s0:
                z0 = 2 * x + y
            with s1:
                z1 = 3 * x + y
            return z0, z1

        inp = (
            torch.ones(2, 2, device="cuda:0", requires_grad=True) + 1,
            torch.ones(2, 2, device="cuda:0", requires_grad=True),
        )

        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        # Run backward to trigger backward graph capture
        actual[0].sum().backward()

        # Check that backward graph has control_deps nodes
        self.assertGreaterEqual(len(bw_graphs), 1)
        bw_graph = bw_graphs[0]

        control_deps_nodes = list(
            bw_graph.graph.find_nodes(op="call_function", target=control_deps)
        )

        # Should have at least one control_deps node wrapping sync operations
        self.assertGreater(
            len(control_deps_nodes),
            0,
            "Expected control_deps nodes in backward graph for stream synchronization",
        )

    def test_sync_dealloc_has_fake_impl(self):
        """Test that sync_dealloc has a registered fake impl.

        Without a fake impl, Inductor's backward compilation crashes when the
        backward graph contains cross-stream sync_dealloc ops.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            t = torch.randn(4)
            # Should not raise "no fake impl registered"
            torch.ops.streams.sync_dealloc.default(0, 1, t)

    def test_record_stream_has_fake_impl(self):
        """Test that record_stream's fake impl has the correct signature."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            t = torch.randn(4)
            # Should not raise due to signature mismatch
            torch.ops.streams.record_stream.default(t, 0)

    @requires_cuda
    def test_record_stream(self):
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        def fn(x):
            s = torch.Stream()
            x.record_stream(s)
            return x

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        compiled(torch.randn(4, device="cuda"))

        self.assertEqual(len(backend.graphs), 1)
        found = any(
            node.target is torch.ops.streams.record_stream
            for node in backend.graphs[0].graph.nodes
        )
        self.assertTrue(found, "record_stream op not found in graph")

    @requires_cuda
    def test_event_record_after_input_mutation_errors(self):
        def fn(x):
            s = torch.Stream()
            e = torch.Event()
            with s:
                x.add_(1)
                e.record()
            return e

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )

    @requires_cuda
    def test_event_record_after_input_mutation_stack_traces(self):
        def fn(x):
            s = torch.Stream()
            e = torch.Event()
            with s:
                x.add_(1)
                e.record()
            return e

        try:
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            msg = str(e)
            self.assertIn("Input mutation occurred here:", msg)
            self.assertIn("x.add_(1)", msg)
            self.assertIn("Event record occurred here:", msg)
            self.assertIn("e.record()", msg)

    @requires_cuda
    def test_event_record_after_input_mutation_record_event(self):
        def fn(x):
            s = torch.Stream()
            with s:
                x.add_(1)
                e = s.record_event()
            return e

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )

    @requires_cuda
    def test_event_record_after_input_mutation_through_view(self):
        def fn(x):
            s = torch.Stream()
            e = torch.Event()
            v = x.view(-1)
            with s:
                v.add_(1)
                e.record()
            return e

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )

    @requires_cuda
    def test_event_record_after_input_mutation_input_event(self):
        def fn(x, e):
            s = torch.Stream()
            with s:
                x.add_(1)
                e.record()
            return x

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda"),
                torch.Event(),
            )

    @requires_cuda
    def test_event_record_before_input_mutation_no_error(self):
        def fn(x):
            s = torch.Stream()
            e = torch.Event()
            with s:
                e.record()
                x.add_(1)
            return e

        torch.compile(fn, backend="eager", fullgraph=True)(
            torch.ones(2, 2, device="cuda")
        )

    @requires_cuda
    def test_event_record_on_different_stream_no_error(self):
        def fn(x):
            s0 = torch.Stream()
            s1 = torch.Stream()
            e = torch.Event()
            with s0:
                x.add_(1)
            with s1:
                e.record()
            return e

        torch.compile(fn, backend="eager", fullgraph=True)(
            torch.ones(2, 2, device="cuda")
        )

    @requires_cuda
    def test_event_not_returned_no_error(self):
        def fn(x):
            s = torch.Stream()
            e = torch.Event()
            with s:
                x.add_(1)
                e.record()
            return x

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )

    @requires_cuda
    @unittest.skip("https://github.com/pytorch/pytorch/issues/177771")
    def test_cuda_event_record_on_stream(self):
        """torch.cuda.Event should be accepted by torch.Stream.record_event (C++ type check)."""
        s = torch.Stream(device="cuda")
        e = torch.cuda.Event()
        # This hits THPStream_record_event in Stream.cpp which does a type check
        s.record_event(e)

    @requires_cuda
    def test_event_synchronize_tracing(self):
        def fn(x):
            e = torch.Event()
            e.record()
            x = x + 1
            e.synchronize()
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
        record_event = torch.ops.streams.record_event.default(1, 0);  record_event = None

        #
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None

        # No stacktrace found for following nodes
        subgraph_synchronize_event = self.subgraph_synchronize_event
        control_deps = torch.ops.higher_order.control_deps((add,), subgraph_synchronize_event, add);  subgraph_synchronize_event = control_deps = None
        return (add,)

    class subgraph_synchronize_event(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            #
            synchronize_event_default = torch.ops.streams.synchronize_event.default(1)
            return (synchronize_event_default, dep_0)
""",
        )

    @requires_cuda
    def test_event_synchronize_inductor_lowering(self):
        with patch("torch._inductor.config.implicit_fallbacks", False):

            @torch.compile()
            def fn(x):
                e = torch.Event()
                x = x + 1
                e.record()
                e.synchronize()
                return x

            inp = (torch.ones(2, 2, device="cuda"),)
            fn(*inp)

    @requires_cuda
    def test_control_deps_wrapping_synchronize_event(self) -> None:
        """Test that synchronize_event threads recorded ops' values through.

        After record_event wraps ops in control_deps and produces getitem
        pass-throughs, synchronize_event must also thread those through so
        that subsequent consumers depend on the synchronize.
        """

        def fn(x) -> torch.Tensor:
            e = torch.Event()
            y = x + 1
            e.record()
            e.synchronize()
            # z uses y which was produced before the record — its value must
            # be threaded through both record and synchronize control_deps.
            z = y * 2
            return z

        inp = (torch.ones(2, 2, device="cuda"),)
        # Patch out wrapping so we get the raw graph to manually wrap below.
        with patch(
            "torch._functorch._aot_autograd.graph_capture.wrap_all_sync_nodes_with_control_deps"
        ):
            (
                _,
                _,
                fw_graphs,
                _,
            ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        import operator

        from torch._functorch._aot_autograd.streams import (
            set_stream,
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        # extract_graph doesn't annotate streams, so set stream metadata on
        # compute nodes to match the record_event's stream index.
        record_node = next(
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.streams.record_event.default
        )
        stream_idx = record_node.args[1]
        for n in graph.nodes:
            if (
                n.op == "call_function"
                and "val" in n.meta
                and n.target
                not in (
                    torch.ops.streams.record_event.default,
                    torch.ops.streams.synchronize_event.default,
                )
            ):
                set_stream(n, stream_idx)

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        # record_event + synchronize_event = 2 control_deps nodes
        self.assertEqual(len(ctrl_nodes), 2)
        record_ctrl = ctrl_nodes[0]
        sync_ctrl = ctrl_nodes[1]

        # synchronize_event's control_deps should depend on record's ctrl
        self.assertIn(record_ctrl, sync_ctrl.args[0])

        # The record should thread through the add (y = x + 1)
        record_getitems = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target == operator.getitem
            and n.args[0] is record_ctrl
        ]
        self.assertGreaterEqual(len(record_getitems), 1)

        # Those getitems should be passed through synchronize's control_deps
        # as additional args (the passthrough deps)
        sync_passthrough_args = sync_ctrl.args[2:]  # skip (deps_tuple, subgraph)
        for getitem in record_getitems:
            self.assertIn(
                getitem,
                sync_passthrough_args,
                "record_event's getitem should be threaded through synchronize_event",
            )

        # The mul (z = y * 2) should consume a getitem from synchronize's
        # control_deps, not directly from record's.
        sync_getitems = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target == operator.getitem
            and n.args[0] is sync_ctrl
        ]
        self.assertGreaterEqual(len(sync_getitems), 1)

        # Find the mul node and verify it uses a sync getitem
        mul_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.mul.Tensor
        ]
        self.assertEqual(len(mul_nodes), 1)
        mul_args = set(mul_nodes[0].args)
        self.assertTrue(
            mul_args & set(sync_getitems),
            "mul should depend on synchronize_event's getitem, not record_event's",
        )

    @requires_cuda
    def test_external_event_synchronize_threads_inputs(self) -> None:
        """When the event was recorded externally, synchronize threads graph inputs through."""

        def fn(x):
            e = torch.Event()
            y = x + 1
            e.record()
            e.synchronize()
            z = y * 2
            return z

        inp = (torch.ones(2, 2, device="cuda"),)
        # Patch out wrapping so we get the raw graph to manually wrap below.
        with patch(
            "torch._functorch._aot_autograd.graph_capture.wrap_all_sync_nodes_with_control_deps"
        ):
            (
                _,
                _,
                fw_graphs,
                _,
            ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            set_stream,
            wrap_all_sync_nodes_with_control_deps,
        )

        # Remove the record_event to simulate an externally-recorded event.
        record_node = next(
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.streams.record_event.default
        )
        stream_idx = record_node.args[1]
        graph.erase_node(record_node)

        # Set stream metadata on compute nodes.
        for n in graph.nodes:
            if (
                n.op == "call_function"
                and "val" in n.meta
                and n.target is not torch.ops.streams.synchronize_event.default
            ):
                set_stream(n, stream_idx)

        wrap_all_sync_nodes_with_control_deps(gm)
        gm.recompile()

        self.assertExpectedInline(
            print_graph(gm),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        # Annotation: {'stream': 0}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 1)

        # No stacktrace found for following nodes
        subgraph_synchronize_event = self.subgraph_synchronize_event
        control_deps = torch.ops.higher_order.control_deps((arg0_1, add), subgraph_synchronize_event, add);  arg0_1 = add = subgraph_synchronize_event = None

        # Annotation: {'stream': 0}
        getitem: "f32[2, 2]" = control_deps[1];  control_deps = None

        # Annotation: {'stream': 0}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem, 2);  getitem = None
        return (mul,)

    class subgraph_synchronize_event(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            #
            synchronize_event_default = torch.ops.streams.synchronize_event.default(1)
            return (synchronize_event_default, dep_0)
""",
        )

    @requires_cuda
    def test_event_synchronize_control_deps_e2e(self):
        """E2E: compute → record → synchronize → use result through torch.compile."""

        def f(x):
            e = torch.Event()
            y = x + 1
            e.record()
            e.synchronize()
            z = y * 2
            return z

        inp = torch.ones(2, 2, device="cuda")
        eager_result = f(inp)
        compiled_result = torch.compile(f)(inp)
        self.assertEqual(eager_result, compiled_result)

    @requires_cuda
    def test_event_synchronize_e2e(self):
        def f(a_list):
            a_cpu_list = []
            a_to_cpu_event_list = []
            for a in a_list:
                a_cpu = a.to(device="cpu", non_blocking=True)
                e = torch.Event()
                e.record()
                a_cpu_list.append(a_cpu)
                a_to_cpu_event_list.append(e)

            for e in a_to_cpu_event_list:
                e.synchronize()

            return torch.cat(a_cpu_list)

        f_compiled = torch.compile(f)
        inputs = [
            torch.rand(100, dtype=torch.float16, device="cuda") for _ in range(10)
        ]
        eager_result = f(inputs)
        compiled_result = f_compiled(inputs)
        self.assertEqual(eager_result, compiled_result)

    @requires_cuda
    def test_event_record_wait_on_default_stream(self):
        e = torch.cuda.Event()

        def f(x):
            y = x + 1
            e.record()
            e.wait()
            return y + 1

        f_compiled = torch.compile(f)
        x = torch.randn(10, device="cuda")
        eager_result = f(x)
        compiled_result = f_compiled(x)
        self.assertEqual(eager_result, compiled_result)

    @requires_cuda
    def test_record_stream_inductor_output_code(self) -> None:
        """Verify record_stream is ordered between the producing kernel and the
        consuming kernel in inductor-generated wrapper code."""
        from torch._inductor.utils import run_and_get_code
        from torch.testing import FileCheck

        def fn(x):
            s = torch.Stream(device="cuda")
            y = x + 1
            y.record_stream(s)
            z = y * 2
            return z

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(1024, device="cuda")
        result, (code,) = run_and_get_code(compiled, x)
        self.assertEqual(result, (x + 1) * 2)

        # record_stream must appear after the kernel that produces the tensor
        # and before the return.
        FileCheck().check(".run(").check(
            "torch.ops.streams.record_stream.default("
        ).check("return").run(code)

    @requires_cuda
    def test_del_multi_stream_sync_dealloc(self):
        def fn(x, y):
            s = torch.Stream()
            e = torch.Event()
            z0 = x + 1
            with s:
                z = torch.add(x, y)
                e.record()
            e.wait()
            del x
            return z0, z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertIn("sync_dealloc", graph_str)
        self.assertIn("record_event", graph_str)

    @requires_cuda
    def test_del_same_stream_no_sync_dealloc(self):
        def fn(x, y):
            s = torch.Stream()
            e = torch.Event()
            with s:
                z = torch.add(x, y)
                del x
                e.record()
            e.wait()
            return z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertNotIn("sync_dealloc", graph_str)

    @requires_cuda
    def test_del_single_stream_no_sync_dealloc(self):
        def fn(x, y):
            z = torch.add(x, y)
            del x
            return z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertNotIn("sync_dealloc", graph_str)

    @requires_cuda
    def test_del_attr_multi_stream_sync_dealloc(self):
        class Holder:
            pass

        def fn(x, y):
            s = torch.Stream()
            e = torch.Event()
            h = Holder()
            h.tensor = x
            z0 = x + 1
            with s:
                z = torch.add(h.tensor, y)
                e.record()
            e.wait()
            del h.tensor
            return z0, z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertIn("sync_dealloc", graph_str)
        self.assertIn("record_event", graph_str)

    @requires_cuda
    def test_del_subscr_multi_stream_sync_dealloc(self):
        def fn(x, y):
            s = torch.Stream()
            e = torch.Event()
            d = {"t": x}
            z0 = x + 1
            with s:
                z = torch.add(d["t"], y)
                e.record()
            e.wait()
            del d["t"]
            return z0, z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertIn("sync_dealloc", graph_str)
        self.assertIn("record_event", graph_str)

    @requires_cuda
    def test_stream_pointer_extraction_edge_cases(self):
        def get_ptrs(stream_a, stream_b, default_stream):
            return (
                stream_a.cuda_stream,
                stream_b.cuda_stream,
                default_stream.cuda_stream,
            )

        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        default_s = torch.cuda.default_stream()
        expected_s1, expected_s2 = s1.cuda_stream, s2.cuda_stream

        self.assertNotEqual(expected_s1, expected_s2)
        self.assertGreater(expected_s1, 1000)

        opt_get_ptrs = torch.compile(get_ptrs, backend="inductor")

        s3 = torch.cuda.Stream()
        with torch.cuda.stream(s3):
            actual_s1, actual_s2, actual_default = opt_get_ptrs(s1, s2, default_s)

        self.assertEqual(actual_s1, expected_s1)
        self.assertEqual(actual_s2, expected_s2)
        self.assertEqual(actual_default, default_s.cuda_stream)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
