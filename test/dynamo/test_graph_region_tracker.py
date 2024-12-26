# Owner(s): ["module: dynamo"]
import contextlib

import torch
import torch.fx
from torch._dynamo.test_case import TestCase
from torch._dynamo.testing import extract_graph_and_tracker
from torch.utils._pytree import tree_map


def get_nodes_by_name(graph, names):
    nodes = []
    for node in graph.nodes:
        if node.name in names:
            nodes.append(node)

    return nodes


unique_ind = 0


def track_same_nodes(names, graph, region_tracker):
    global unique_ind
    unique_ind += 1
    # find nodes in graph with names and track them
    # as if they were at the same code location
    nodes = get_nodes_by_name(graph, names)
    for node in nodes:
        region_tracker.track_node("x", unique_ind, node)


class GraphRegionTrackerTests(TestCase):
    def setUp(self):
        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(
            torch._dynamo.config.patch("track_nodes_for_deduplication", True)
        )
        super().setUp()

    def tearDown(self):
        self.exit_stack.close()
        super().tearDown()

    def get_result(self, fn, *args, **kwargs):
        graph, region_tracker = extract_graph_and_tracker(fn, *args, **kwargs)
        region_groups = region_tracker.get_identical_regions(graph)
        region_groups = tree_map(lambda n: n.name, region_groups)
        return str(region_groups)

    def test_get_regions_single_region_group(self):
        def inner_fn(x, y):
            x0 = x + 1
            y0 = y + 2
            z = x0.sum() + y0.sum()
            return z

        def fn(x, y):
            _o0 = inner_fn(x, y)
            o1 = torch.sin(y)
            o2 = inner_fn(x, o1)
            o3 = inner_fn(x, y)
            o4 = o3 * o3
            return o2 * o4

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.ones(10, 20),
            ),
            """[[['x0', 'y0', 'sum_1', 'sum_2', 'z'], ['x0_1', 'y0_1', 'sum_3', 'sum_4', 'z_1'],\
 ['x0_2', 'y0_2', 'sum_5', 'sum_6', 'z_2']]]""",
        )

    def test_get_regions_multiple_region_groups(self):
        def inner_fn(x, y):
            x1 = x + 1
            y1 = y + 2
            z = x1.sum() + y1.sum()
            return z

        def inner_fn2(a, b):
            a += 2
            b += 3
            c = a * b.cos().sum()
            return c

        def fn(x, y):
            x0 = torch.cos(x)
            y0 = torch.sin(y)
            o1 = inner_fn2(x0, y0)
            o0 = inner_fn(x, y)
            o1 = torch.sin(o0)
            o2 = inner_fn(x, y0)
            o2 = inner_fn2(x0, y0)
            o3 = inner_fn(x, y)
            return o1 * o2 + o3

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.ones(10, 20),
            ),
            """[[['x1', 'y1', 'sum_2', 'sum_3', 'z'], ['x1_1', 'y1_1', 'sum_4', 'sum_5', 'z_1'],\
 ['x1_2', 'y1_2', 'sum_7', 'sum_8', 'z_2']], [['a', 'b', 'cos_1', 'sum_1', 'c'], ['a_1', 'b_1', 'cos_2', 'sum_6', 'c_1']]]""",
        )

    def test_no_single_node_regions(self):
        def inner_fn(x):
            return x + 1

        def fn(x):
            o0 = inner_fn(x)
            o1 = inner_fn(x)
            o2 = inner_fn(x)
            return o0 + o1 + o2

        self.assertExpectedInline(self.get_result(fn, torch.ones(10, 10)), """[]""")

    def test_mismatched_arg_shapes(self):
        def inner_fn(x, y):
            x1 = x + 1
            y1 = y + 2
            z = x1.sum() + y1.sum()
            return z

        def inner_fn2(a, b):
            a += 2
            b += 3
            c = a * b.cos().sum()
            return c

        def fn(x, y):
            x0 = torch.cos(x)
            y0 = torch.sin(y)
            o1 = inner_fn2(x0, y0)
            o0 = inner_fn(x, o1)
            o1 = torch.sin(o0)
            o2 = inner_fn(x, y0)
            o2 = inner_fn2(o2, y0)
            o3 = inner_fn(x, y)
            return o1 * o2 + o3

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.ones(10, 20),
            ),
            """[[['y1_1', 'sum_5'], ['y1_2', 'sum_8']], [['x1', 'sum_2', 'z'], ['x1_1', 'sum_4', 'z_1'], \
['x1_2', 'sum_7', 'z_2']], [['b', 'cos_1', 'sum_1'], ['b_1', 'cos_2', 'sum_6']]]""",
        )

    def test_mismatched_dtypes(self):
        def inner_fn(x, y):
            x1 = x * 1
            y1 = y + 1
            return x1 + y1.sum()

        def fn(x, y):
            x0 = torch.sin(x)
            y0 = torch.cos(y)
            o0 = inner_fn(x0, y0)
            o2 = inner_fn(x0, y0)
            o4 = inner_fn(x0, y0)
            o5 = inner_fn(x0, y0)
            o1 = inner_fn(x0.to(torch.bfloat16), y0.to(torch.bfloat16))
            o3 = o1 + o2
            return o3 * o0 + o4 + o5

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.ones(10, 20),
            ),
            """[[['x1', 'y1', 'sum_1', 'o0'], ['x1_1', 'y1_1', 'sum_2', 'o2'], \
['x1_2', 'y1_2', 'sum_3', 'o4'], ['x1_3', 'y1_3', 'sum_4', 'o5']]]""",
        )

    def test_nested_args(self):
        def inner_fn(xs, ys):
            out = torch._foreach_add(xs, ys)
            return out[0] + out[1].sum()

        def fn(x, y, z):
            x0 = torch.sin(x)
            y0 = torch.cos(y)
            z0 = torch.sin(z)
            o0 = inner_fn([x0, z0], [x0, y0])
            o2 = inner_fn([x0, z0], [x0, y0])
            o4 = inner_fn([x0, z0], [x0, y0])
            o5 = inner_fn([x0, z0], [x0, y0])
            o1 = inner_fn(
                [x0.to(torch.bfloat16), z0.to(torch.bfloat16)],
                [x0.to(torch.bfloat16), y0.to(torch.bfloat16)],
            )
            o3 = o1 + o2
            return o3 * o0 + o4 + o5

        self.assertExpectedInline(
            self.get_result(
                fn,
                torch.rand(10, 10),
                torch.rand(10, 20),
                torch.ones(10, 20),
            ),
            """[[['_foreach_add', 'getitem', 'getitem_1', 'sum_1', 'o0'], ['_foreach_add_1', \
'getitem_2', 'getitem_3', 'sum_2', 'o2'], ['_foreach_add_2', 'getitem_4', 'getitem_5', 'sum_3', \
'o4'], ['_foreach_add_3', 'getitem_6', 'getitem_7', 'sum_4', 'o5']]]""",
        )

    def test_mismatched_global_state(self):
        def inner_fn(x, y):
            x1 = x * 1
            y1 = y + 1
            return x1 + y1.sum()

        def fn(x, y, c):
            x0 = torch.sin(x)
            y0 = torch.cos(y)
            o4 = inner_fn(x0, y0)
            o5 = inner_fn(x0, y0)
            if isinstance(c, tuple):
                c[0]()
                o0 = inner_fn(x0, y0)
                o2 = inner_fn(x0, y0)
                c[1]()
            else:
                with c():
                    o0 = inner_fn(x0, y0)
                    o2 = inner_fn(x0, y0)
            return o0 + o2 + o4 + o5

        def create_toggle_fns(property):
            old_value = getattr(torch.backends.cuda.matmul, property)

            def toggle_property():
                setattr(torch.backends.cuda.matmul, property, not old_value)

            def reset_property():
                setattr(torch.backends.cuda.matmul, property, old_value)

            return toggle_property, reset_property

        old_dtype = torch.get_default_dtype()

        def set_default_dtype_bfloat16():
            torch.set_default_dtype(torch.bfloat16)

        def reset_default_dtype():
            torch.set_default_dtype(old_dtype)

        for ctx in [
            lambda: torch.set_grad_enabled(False),
            torch.autograd.grad_mode.inference_mode,
            lambda: torch.autograd.graph.disable_saved_tensors_hooks(
                "This is not supported"
            ),
            # lambda: torch.set_num_threads(2), : Unsupported
            (set_default_dtype_bfloat16, reset_default_dtype),
            (
                lambda: torch.use_deterministic_algorithms(True),
                lambda: torch.use_deterministic_algorithms(False),
            ),
            # (lambda: torch.use_deterministic_algorithms(True, warn_only=True),
            # lambda: torch.use_deterministic_algorithms(False)), : Unsupported
            create_toggle_fns("allow_bf16_reduced_precision_reduction"),
            create_toggle_fns("allow_fp16_reduced_precision_reduction"),
            create_toggle_fns("allow_tf32"),
        ]:
            self.assertExpectedInline(
                self.get_result(fn, torch.rand(10, 10), torch.ones(10, 20), ctx),
                """[[['x1_2', 'y1_2', 'sum_3', 'o0'], ['x1_3', 'y1_3', 'sum_4', 'o2']], \
[['x1', 'y1', 'sum_1', 'o4'], ['x1_1', 'y1_1', 'sum_2', 'o5']]]""",
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
