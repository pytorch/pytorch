# Owner(s): ["module: dynamo"]
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_adam in OptimizerTests)
"""

import functools
import gc
import itertools
import weakref
from copy import deepcopy

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.variables.optimizer import OptimizerVariable
from torch._guards import TracingContext
from torch.nn import Parameter
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    disable_gc,
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
)


class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {})

    def _init_group(self, params, group):
        any_complex = False
        for p in group["params"]:
            params.append(p)
            any_complex |= p.is_complex()
        return any_complex

    def step(self):
        for group in self.param_groups:
            params = []
            any_complex = self._init_group(params, group)
            if any_complex:
                params[0] -= 1
            else:
                params[0] += 1


@instantiate_parametrized_tests
class End2EndTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.GENERIC

    # https://github.com/pytorch/torchdynamo/issues/1604
    def test_optimizing_over_tensor_with_requires_grad(self):
        class Net(torch.nn.Module):
            def forward(self, x, y):
                z = torch.bmm(x, y)
                z = torch.flatten(z, 1)
                return z

        def training_iter_fn(batch, model, optimizer):
            optimizer.zero_grad()
            out = model(**batch)
            target = torch.tensor([0, 7])
            loss = torch.nn.CrossEntropyLoss()(out, target)
            loss.backward()
            optimizer.step()
            return loss

        net = Net()
        input1 = torch.randn(2, 1, 4)
        input2 = torch.randn(2, 4, 8, requires_grad=True)
        optimizer = torch.optim.Adam([input2], lr=0.1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_training_iter_fn = torch.compile(training_iter_fn, backend=cnts)
        batch = {"x": input1, "y": input2}
        for _ in range(2):
            opt_training_iter_fn(batch, net, optimizer)
        self.assertEqual(cnts.frame_count, 2)

    def test_state_dict(self):
        @torch.compile(backend="eager")
        def _test_state_dict(weight, bias, input):
            def fn_base(optimizer, weight, bias):
                optimizer.zero_grad()
                i = input
                loss = (weight.mv(i) + bias).pow(2).sum()
                loss.backward()
                return loss

            optimizer = torch.optim.Adagrad([weight, bias])
            fn = functools.partial(fn_base, optimizer, weight, bias)
            return optimizer, fn

        optimizer, fn = _test_state_dict(
            Parameter(torch.randn(10, 5)),
            Parameter(torch.randn(10)),
            torch.randn(5, requires_grad=True),
        )
        optimizer.step(fn)

    def test_init_group(self):
        for dtype in [torch.float32, torch.cfloat]:
            tensor = torch.randn(5, 5, dtype=dtype)
            params = Parameter(tensor.detach().clone(), requires_grad=False)
            opt_params = Parameter(tensor.detach().clone(), requires_grad=False)

            optim = MyOptimizer([params])
            optim.step()

            opt_optim = MyOptimizer([opt_params])
            opt_step = torch.compile(backend="eager", fullgraph=True)(opt_optim.step)
            opt_step()

            self.assertEqual(params, opt_params)

    @parametrize(
        "backend,num_breaks,prefix_graph",
        [
            *itertools.product(("eager", "aot_eager"), (0, 1, 2), (False, True)),
            ("inductor", 2, True),
        ],
    )
    def test_optimizer_cleanup_after_graph_break(
        self, backend, num_breaks, prefix_graph
    ):
        compiler = torch._dynamo.testing.CompileCounterWithBackend(backend)
        graphs = compiler.graphs

        def scope():
            p = Parameter(torch.ones(2, 2))
            p.grad = torch.ones_like(p)
            optimizer = torch.optim.Adam([p], lr=0.1, foreach=True)

            def step(x):
                group = optimizer.param_groups[0]
                params, grads, avgs, sqs, max_sqs, steps = [], [], [], [], [], []
                optimizer._init_group(group, params, grads, avgs, sqs, max_sqs, steps)
                if prefix_graph:
                    x = x + 1
                if num_breaks:
                    torch._dynamo.graph_break()
                params[0].add_(grads[0], alpha=-0.1)
                avgs[0].add_(grads[0])
                if num_breaks == 2:
                    torch._dynamo.graph_break()
                    params[0].add_(grads[0], alpha=-0.1)
                    avgs[0].add_(grads[0])
                return x

            with torch.no_grad():
                result = torch.compile(step, backend=compiler)(torch.ones(1))
            self.assertEqual(result, torch.full((1,), 2.0 if prefix_graph else 1.0))
            updates = 2 if num_breaks == 2 else 1
            self.assertEqual(p, torch.full_like(p, 1 - 0.1 * updates))
            average = optimizer.state[p]["exp_avg"]
            self.assertEqual(average, torch.full_like(average, updates))
            self.assertEqual(
                compiler.frame_count,
                1 if num_breaks == 0 else num_breaks + int(prefix_graph),
            )
            self.assertEqual(
                sum(
                    any(tensor is p for tensor in gm._parameters.values())
                    and any(tensor is average for tensor in gm._buffers.values())
                    for gm in graphs
                ),
                updates,
            )
            return weakref.ref(optimizer), weakref.ref(p), weakref.ref(average)

        with disable_gc():
            optimizer_ref, parameter_ref, state_ref = scope()
            self.assertIsNone(optimizer_ref())
            self.assertIsNone(parameter_ref())
            self.assertIsNone(state_ref())

    def test_optimizer_cleanup_preserves_live_optimizer(self):
        compiler = torch._dynamo.testing.EagerAndRecordGraphs()
        graphs = compiler.graphs

        def scope():
            p = Parameter(torch.ones(2, 2))
            optimizer = MyOptimizer([p])

            def step():
                params = []
                optimizer._init_group(params, optimizer.param_groups[0])
                torch._dynamo.graph_break()
                params[0].add_(1)

            return optimizer, torch.compile(step, backend=compiler), weakref.ref(p)

        with disable_gc():
            optimizer, compiled, parameter_ref = scope()
            with torch.no_grad():
                compiled()
                compiled()
            self.assertEqual(parameter_ref(), torch.full((2, 2), 3.0))
            self.assertEqual(len(graphs), 1)
            del compiled, optimizer
            self.assertIsNone(parameter_ref())

    def test_optimizer_cleanup_preserves_other_static_tensors(self):
        compiler = torch._dynamo.testing.EagerAndRecordGraphs()
        graphs = compiler.graphs

        def scope():
            p = Parameter(torch.ones(2, 2))
            other = torch.ones(2, 2)
            torch._dynamo.mark_static_address(other, guard=True)
            optimizer = MyOptimizer([p])

            def step():
                params = []
                optimizer._init_group(params, optimizer.param_groups[0])
                torch._dynamo.graph_break()
                params[0].add_(other)

            with torch.no_grad():
                torch.compile(step, backend=compiler)()
            self.assertEqual(p, torch.full_like(p, 2))
            self.assertEqual(len(graphs), 1)
            self.assertTrue(
                any(tensor is other for tensor in graphs[0]._buffers.values())
            )
            return optimizer, weakref.ref(p), weakref.ref(other)

        with disable_gc():
            optimizer, parameter_ref, other_ref = scope()
            del optimizer
            self.assertIsNone(parameter_ref())
            self.assertIsNotNone(other_ref())

    @parametrize("owner_kind", ("default", "unhashable", "equal"))
    def test_optimizer_cleanup_is_per_optimizer(self, owner_kind):
        class UnhashableOptimizer(MyOptimizer):
            __hash__ = None

        class EqualOptimizer(MyOptimizer):
            def __hash__(self):
                return 0

            def __eq__(self, other):
                return isinstance(other, EqualOptimizer)

        optimizer_class = {
            "default": MyOptimizer,
            "unhashable": UnhashableOptimizer,
            "equal": EqualOptimizer,
        }[owner_kind]
        compiler = torch._dynamo.testing.EagerAndRecordGraphs()
        graphs = compiler.graphs

        def scope():
            p1 = Parameter(torch.ones(2, 2))
            p2 = Parameter(torch.ones(2, 2))
            optimizer1 = optimizer_class([p1])
            optimizer2 = optimizer_class([p2])

            def step():
                params1, params2 = [], []
                optimizer1._init_group(params1, optimizer1.param_groups[0])
                optimizer2._init_group(params2, optimizer2.param_groups[0])
                torch._dynamo.graph_break()
                params1[0].add_(1)
                params2[0].add_(2)

            with torch.no_grad():
                torch.compile(step, backend=compiler)()
            self.assertEqual(p1, torch.full_like(p1, 2))
            self.assertEqual(p2, torch.full_like(p2, 3))
            return optimizer1, optimizer2, weakref.ref(p1), weakref.ref(p2)

        with disable_gc():
            optimizer1, optimizer2, parameter1_ref, parameter2_ref = scope()
            del optimizer1
            self.assertIsNone(parameter1_ref())
            self.assertEqual(len(graphs), 1)
            self.assertTrue(
                any(
                    tensor is parameter2_ref()
                    for gm in graphs
                    for tensor in gm._parameters.values()
                )
            )
            del optimizer2
            self.assertIsNone(parameter2_ref())

    def test_optimizer_cleanup_reused_resume_with_shared_parameter(self):
        compiler = torch._dynamo.testing.EagerAndRecordGraphs()
        graphs = compiler.graphs

        def step(optimizer):
            params = []
            optimizer._init_group(params, optimizer.param_groups[0])
            torch._dynamo.graph_break()
            params[0].add_(1)

        with disable_gc():
            p = Parameter(torch.ones(2, 2))
            parameter_ref = weakref.ref(p)
            optimizer1 = MyOptimizer([p])
            optimizer2 = MyOptimizer([p])
            compiled = torch.compile(step, backend=compiler)
            with torch.no_grad():
                compiled(optimizer1)
                self.assertEqual(len(graphs), 1)
                compiled(optimizer2)
                self.assertEqual(len(graphs), 1)
                del optimizer1
                compiled(optimizer2)
                self.assertEqual(len(graphs), 2)
                compiled(optimizer2)
                self.assertEqual(len(graphs), 2)
            self.assertEqual(p, torch.full_like(p, 5))
            del compiled, optimizer2, p
            self.assertIsNone(parameter_ref())

    def test_optimizer_cleanup_resume_ignores_unrelated_owners(self):
        compiler = torch._dynamo.testing.EagerAndRecordGraphs()
        graphs = compiler.graphs

        def step(optimizer):
            params = []
            optimizer._init_group(params, optimizer.param_groups[0])
            torch._dynamo.graph_break()
            params[0].add_(1)

        with disable_gc():
            p1, p2 = Parameter(torch.ones(2, 2)), Parameter(torch.ones(2, 2))
            optimizer1, optimizer2 = MyOptimizer([p1]), MyOptimizer([p2])
            compiled = torch.compile(step, backend=compiler)
            with torch.no_grad():
                compiled(optimizer1)
                compiled(optimizer2)
                self.assertEqual(len(graphs), 2)
                del optimizer1
                compiled(optimizer2)
                self.assertEqual(len(graphs), 2)
            self.assertEqual(p2, torch.full_like(p2, 3))

    def test_optimizer_cleanup_shared_parameter_after_shape_recompile(self):
        compiler = torch._dynamo.testing.EagerAndRecordGraphs()
        graphs = compiler.graphs

        def step(optimizer, x):
            params = []
            optimizer._init_group(params, optimizer.param_groups[0])
            torch._dynamo.graph_break()
            params[0].add_(x.sum())

        with disable_gc():
            p = Parameter(torch.ones(2, 2))
            parameter_ref = weakref.ref(p)
            optimizer1, optimizer2 = MyOptimizer([p]), MyOptimizer([p])
            compiled = torch.compile(step, backend=compiler, dynamic=False)
            with torch.no_grad():
                compiled(optimizer1, torch.ones(2))
                compiled(optimizer2, torch.ones(3))
                self.assertEqual(len(graphs), 2)
                del optimizer1
                compiled(optimizer2, torch.ones(3))
                self.assertEqual(len(graphs), 2)
            self.assertEqual(p, torch.full_like(p, 9))
            del compiled, optimizer2, p
            self.assertIsNone(parameter_ref())

    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    def test_optimizer_cleanup_does_not_retain_graph(self, backend):
        graph_refs, context_refs = [], []

        def compiler(gm, inputs):
            graph_refs.append(weakref.ref(gm))
            context_refs.append(weakref.ref(TracingContext.get()))
            return lookup_backend(backend)(gm, inputs)

        def step(optimizer, x):
            params = []
            optimizer._init_group(params, optimizer.param_groups[0])
            torch._dynamo.graph_break()
            params[0].add_(x.sum())

        p = Parameter(torch.ones(2, 2))
        optimizer = MyOptimizer([p])
        compiled = torch.compile(step, backend=compiler, dynamic=False)
        with torch.no_grad():
            compiled(optimizer, torch.ones(2))
            compiled(optimizer, torch.ones(3))
        self.assertEqual(p, torch.full_like(p, 6))
        self.assertEqual(len(graph_refs), 2)
        del compiled
        torch._dynamo.reset()
        gc.collect()
        for graph_ref in graph_refs:
            self.assertIsNone(graph_ref())
        for context_ref in context_refs:
            self.assertIsNone(context_ref())
        self.assertIs(optimizer.param_groups[0]["params"][0], p)

    @parametrize("backend", ("aot_eager", "inductor"))
    @parametrize("lift_attributes", (False, True))
    def test_optimizer_cleanup_tracing_context(self, backend, lift_attributes):
        graphs, contexts = [], []

        def compiler(gm, inputs):
            graphs.append(gm)
            contexts.append(TracingContext.get())
            compiled = lookup_backend(backend)(gm, inputs)
            if lift_attributes:
                gm._parameters.clear()
                gm._buffers.clear()
            return compiled

        def scope():
            p1, p2 = Parameter(torch.ones(2, 2)), Parameter(torch.ones(2, 2))
            optimizer1, optimizer2 = MyOptimizer([p1]), MyOptimizer([p2])

            def step():
                params1, params2 = [], []
                optimizer1._init_group(params1, optimizer1.param_groups[0])
                optimizer2._init_group(params2, optimizer2.param_groups[0])
                torch._dynamo.graph_break()
                params1[0].add_(1)
                params2[0].add_(2)

            with torch.no_grad():
                torch.compile(step, backend=compiler)()
            self.assertEqual(p1, torch.full_like(p1, 2))
            self.assertEqual(p2, torch.full_like(p2, 3))
            return optimizer1, optimizer2, weakref.ref(p1), weakref.ref(p2)

        with disable_gc():
            optimizer1, optimizer2, parameter1_ref, parameter2_ref = scope()
            self.assertEqual(len(contexts), 1)
            tc = contexts[0]
            self.assertEqual(len(tc.params_flat), 2)
            self.assertEqual(len(tc.params_flat_unwrap_subclasses), 2)
            survivor_compiler = torch._dynamo.testing.CompileCounterWithBackend(backend)
            compiled_step = torch.compile(
                optimizer2.step, backend=survivor_compiler, fullgraph=True
            )
            with torch.no_grad():
                compiled_step()
            self.assertEqual(survivor_compiler.frame_count, 1)
            del optimizer1
            self.assertIsNone(parameter1_ref())
            self.assertEqual(len(tc.params_flat), 2)
            self.assertEqual(len(tc.params_flat_unwrap_subclasses), 2)
            self.assertEqual(sum(tensor is None for tensor in tc.params_flat), 1)
            self.assertEqual(
                sum(tensor is None for tensor in tc.params_flat_unwrap_subclasses), 1
            )
            self.assertTrue(
                any(tensor is parameter2_ref() for tensor in tc.params_flat)
            )
            self.assertTrue(
                any(
                    tensor is parameter2_ref()
                    for tensor in tc.params_flat_unwrap_subclasses
                )
            )
            with torch.no_grad():
                compiled_step()
            self.assertEqual(survivor_compiler.frame_count, 1)
            self.assertEqual(parameter2_ref(), torch.full((2, 2), 5.0))
            del compiled_step, optimizer2
            self.assertIsNone(parameter2_ref())
            self.assertEqual(tc.params_flat, [None, None])
            self.assertEqual(tc.params_flat_unwrap_subclasses, [None, None])

    @parametrize("backend", ("eager", "aot_eager", "inductor"))
    @parametrize(
        "optimizer_class",
        (torch.optim.SGD, torch.optim.Adam),
        name_fn=lambda optimizer_class: optimizer_class.__name__,
    )
    def test_optimizer_cleanup_parameter_alias_across_step(
        self, backend, optimizer_class
    ):
        compiler = torch._dynamo.testing.CompileCounterWithBackend(backend)
        graphs = compiler.graphs

        def scope():
            parameter = Parameter(torch.ones(2))
            parameter.grad = torch.ones_like(parameter)
            optimizer = optimizer_class([parameter], lr=0.1)
            expected = Parameter(torch.ones(2))
            expected.grad = torch.ones_like(expected)
            reference = optimizer_class([expected], lr=0.1, foreach=False)

            def step():
                params = optimizer.param_groups[0]["params"]
                optimizer.step()
                with torch.no_grad():
                    params[0].add_(1)

            compiled = torch.compile(step, backend=compiler)
            for _ in range(2):
                compiled()
                reference.step()
                with torch.no_grad():
                    expected.add_(1)
                self.assertEqual(parameter, expected)
            self.assertTrue(
                any(
                    any(tensor is parameter for tensor in gm._parameters.values())
                    and any(
                        node.op == "call_method"
                        and node.target == "add_"
                        and len(node.args) > 1
                        and isinstance(node.args[1], int)
                        and node.args[1] == 1
                        for node in gm.graph.nodes
                    )
                    for gm in graphs
                )
            )
            return weakref.ref(optimizer), weakref.ref(parameter)

        with disable_gc():
            optimizer_ref, parameter_ref = scope()
            self.assertIsNone(optimizer_ref())
            self.assertIsNone(parameter_ref())

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_optimizer_cleanup_real_adam_step(self, backend):
        compiler = torch._dynamo.testing.CompileCounterWithBackend(backend)
        graphs = compiler.graphs

        def scope():
            p, expected = Parameter(torch.ones(2, 2)), Parameter(torch.ones(2, 2))
            p.grad = torch.full_like(p, 0.25)
            expected.grad = torch.full_like(expected, 0.25)
            optimizer = torch.optim.Adam([p], lr=0.01, foreach=True)
            reference = torch.optim.Adam([expected], lr=0.01, foreach=False)

            def step():
                optimizer.step()
                torch._dynamo.graph_break()
                optimizer.step()

            compiled = torch.compile(step, backend=compiler)
            with torch.no_grad():
                compiled()
                compiled()
                for _ in range(4):
                    reference.step()
            self.assertEqual(p, expected)
            self.assertTrue(
                any(
                    any(tensor is p for tensor in gm._parameters.values())
                    and any(
                        tensor is optimizer.state[p]["exp_avg"]
                        for tensor in gm._buffers.values()
                    )
                    for gm in graphs
                )
            )
            return (
                weakref.ref(optimizer),
                weakref.ref(p),
                weakref.ref(optimizer.state[p]["exp_avg"]),
            )

        with disable_gc():
            optimizer_ref, parameter_ref, state_ref = scope()
            self.assertIsNone(optimizer_ref())
            self.assertIsNone(parameter_ref())
            self.assertIsNone(state_ref())

    @parametrize("backend", ("aot_eager", "inductor"))
    def test_optimizer_cleanup_shared_parameter_separate_states(self, backend):
        graphs, contexts = [], []

        def compiler(gm, inputs):
            graphs.append(gm)
            contexts.append(TracingContext.get())
            return lookup_backend(backend)(gm, inputs)

        def scope():
            p = Parameter(torch.ones(2, 2))
            p.grad = torch.ones_like(p)
            optimizer1 = torch.optim.Adam([p], foreach=True)
            optimizer2 = torch.optim.Adam([p], foreach=True)

            def step():
                params1, grads1, avgs1, sqs1, max_sqs1, steps1 = [], [], [], [], [], []
                params2, grads2, avgs2, sqs2, max_sqs2, steps2 = [], [], [], [], [], []
                optimizer1._init_group(
                    optimizer1.param_groups[0],
                    params1,
                    grads1,
                    avgs1,
                    sqs1,
                    max_sqs1,
                    steps1,
                )
                optimizer2._init_group(
                    optimizer2.param_groups[0],
                    params2,
                    grads2,
                    avgs2,
                    sqs2,
                    max_sqs2,
                    steps2,
                )
                torch._dynamo.graph_break()
                params1[0].add_(avgs1[0] + avgs2[0] + 1)
                avgs1[0].add_(1)
                avgs2[0].add_(2)

            with torch.no_grad():
                torch.compile(step, backend=compiler)()
            self.assertEqual(p, torch.full_like(p, 2))
            self.assertEqual(
                optimizer1.state[p]["exp_avg"],
                torch.full_like(optimizer1.state[p]["exp_avg"], 1),
            )
            self.assertEqual(
                optimizer2.state[p]["exp_avg"],
                torch.full_like(optimizer2.state[p]["exp_avg"], 2),
            )
            self.assertIsNot(
                optimizer1.state[p]["exp_avg"], optimizer2.state[p]["exp_avg"]
            )
            return (
                optimizer1,
                optimizer2,
                weakref.ref(p),
                weakref.ref(optimizer1.state[p]["exp_avg"]),
                weakref.ref(optimizer2.state[p]["exp_avg"]),
            )

        with disable_gc():
            optimizer1, optimizer2, parameter_ref, state1_ref, state2_ref = scope()
            self.assertEqual(len(contexts), 1)
            del optimizer1
            self.assertIsNone(state1_ref())
            for tensors in (
                contexts[0].params_flat,
                contexts[0].params_flat_unwrap_subclasses,
            ):
                self.assertEqual(len(tensors), 3)
                self.assertEqual(sum(tensor is None for tensor in tensors), 1)
                self.assertTrue(any(tensor is parameter_ref() for tensor in tensors))
                self.assertTrue(any(tensor is state2_ref() for tensor in tensors))
            del optimizer2
            self.assertIsNone(parameter_ref())
            self.assertIsNone(state2_ref())

    @torch._dynamo.config.patch(nested_graph_breaks=False)
    @parametrize("backend", ("eager", "aot_eager"))
    @parametrize("reset_before_recapture", (False, True))
    def test_optimizer_cleanup_in_callee(self, backend, reset_before_recapture):
        compiler = torch._dynamo.testing.CompileCounterWithBackend(backend)

        def helper(params):
            torch._dynamo.graph_break()
            params[0].add_(1)

        def scope():
            p = Parameter(torch.ones(2, 2))
            optimizer = MyOptimizer([p])

            def step():
                params = []
                optimizer._init_group(params, optimizer.param_groups[0])
                helper(params)

            with torch.no_grad():
                torch.compile(step, backend=compiler)()
            self.assertEqual(p, torch.full_like(p, 2))
            self.assertEqual(compiler.frame_count, 1)
            if reset_before_recapture:
                torch._dynamo.reset()
                self.assertEqual(p._dynamo_static_input_type, "guarded")
                # This fresh frame sees only the marked tensor, not its optimizer.
                with torch.no_grad():
                    torch.compile(helper, backend=compiler)([p])
                self.assertEqual(compiler.frame_count, 2)
                self.assertEqual(p, torch.full_like(p, 3))
            self.assertTrue(
                all(
                    any(tensor is p for tensor in gm._parameters.values())
                    for gm in compiler.graphs
                )
            )
            return weakref.ref(optimizer), weakref.ref(p)

        with disable_gc():
            optimizer_ref, parameter_ref = scope()
            self.assertIsNone(optimizer_ref())
            self.assertIsNone(parameter_ref())

    def test_optimizer_cleanup_does_not_retain_other_static_tensor_after_reset(self):
        graph_refs, context_refs, static_tensor_refs = [], [], []

        def compiler(gm, inputs):
            graph_refs.append(weakref.ref(gm))
            tc = TracingContext.get()
            context_refs.append(weakref.ref(tc))
            compiled = lookup_backend("aot_eager")(gm, inputs)
            self.assertEqual(len(tc.params_flat), 2)
            for tensor_ref in static_tensor_refs:
                self.assertTrue(
                    any(tensor is tensor_ref() for tensor in tc.params_flat)
                )
            return compiled

        def scope():
            p = Parameter(torch.ones(2, 2))
            other = torch.ones(2, 2)
            torch._dynamo.mark_static_address(other, guard=True)
            optimizer = MyOptimizer([p])
            static_tensor_refs.extend((weakref.ref(p), weakref.ref(other)))

            def step():
                params = optimizer.param_groups[0]["params"]
                params[0].add_(other)

            with torch.no_grad():
                torch.compile(step, backend=compiler, fullgraph=True)()
            self.assertEqual(p, torch.full_like(p, 2))
            self.assertEqual(len(context_refs), 1)
            return optimizer, weakref.ref(p), weakref.ref(other)

        optimizer, parameter_ref, other_ref = scope()
        torch._dynamo.reset()
        gc.collect()
        self.assertIsNone(graph_refs[0]())
        self.assertIsNone(context_refs[0]())
        self.assertIsNone(other_ref())
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter_ref())
        del optimizer
        gc.collect()
        self.assertIsNone(parameter_ref())

    @parametrize("tensor_kind", ("parameter", "subclass"))
    def test_optimizer_cleanup_preserves_tracing_context_list_aliases(
        self, tensor_kind
    ):
        from torch.testing._internal.two_tensor import TwoTensor

        graph_refs, context_refs = [], []
        retained_lists, owned_indices, other_indices, lengths = {}, {}, {}, {}

        def compiler(gm, inputs):
            graph_refs.append(weakref.ref(gm))
            tc = TracingContext.get()
            context_refs.append(weakref.ref(tc))
            compiled = lookup_backend("aot_eager")(gm, inputs)
            self.assertEqual(len(tc.params_flat), 2)
            retained_lists["unwrapped"] = tc.params_flat_unwrap_subclasses
            if tensor_kind == "parameter":
                retained_lists["flat"] = tc.params_flat
            return compiled

        def scope():
            if tensor_kind == "subclass":
                p = Parameter(
                    TwoTensor(torch.ones(2, 2), torch.ones(2, 2)), requires_grad=False
                )
            else:
                p = Parameter(torch.ones(2, 2))
            other = torch.ones(2, 2)
            torch._dynamo.mark_static_address(other, guard=True)
            optimizer = MyOptimizer([p])

            def step():
                params = optimizer.param_groups[0]["params"]
                params[0].add_(other)

            with torch.no_grad():
                torch.compile(step, backend=compiler, fullgraph=True)()
            if tensor_kind == "subclass":
                self.assertEqual(p.a, torch.full((2, 2), 2.0))
                self.assertEqual(p.b, torch.full((2, 2), 2.0))
            else:
                self.assertEqual(p, torch.full_like(p, 2))
            self.assertEqual(len(context_refs), 1)
            self.assertEqual(
                len(retained_lists["unwrapped"]), 3 if tensor_kind == "subclass" else 2
            )
            for name, tensors in retained_lists.items():
                lengths[name] = len(tensors)
                owned_indices[name] = [
                    index
                    for index, tensor in enumerate(tensors)
                    if tensor is p
                    or (tensor_kind == "subclass" and (tensor is p.a or tensor is p.b))
                ]
                other_indices[name] = [
                    index for index, tensor in enumerate(tensors) if tensor is other
                ]
                self.assertEqual(
                    len(owned_indices[name]), 2 if tensor_kind == "subclass" else 1
                )
                self.assertEqual(len(other_indices[name]), 1)
            return optimizer, weakref.ref(p), weakref.ref(other)

        optimizer, parameter_ref, other_ref = scope()
        torch._dynamo.reset()
        gc.collect()
        self.assertIsNone(graph_refs[0]())
        self.assertIsNone(context_refs[0]())
        # The subclass case retains only the real unwrapped list. Its entries
        # must still be cleared after AOT's outer parameter list has died.
        for name, tensors in retained_lists.items():
            self.assertEqual(len(tensors), lengths[name])
            self.assertTrue(
                all(tensors[index] is not None for index in owned_indices[name])
            )
            self.assertIs(tensors[other_indices[name][0]], other_ref())
        self.assertIs(optimizer.param_groups[0]["params"][0], parameter_ref())
        del optimizer
        gc.collect()
        self.assertIsNone(parameter_ref())
        self.assertIsNotNone(other_ref())
        for name, tensors in retained_lists.items():
            self.assertEqual(len(tensors), lengths[name])
            self.assertTrue(
                all(tensors[index] is None for index in owned_indices[name])
            )
            self.assertIs(tensors[other_indices[name][0]], other_ref())

    def test_optimizer_cleanup_after_backend_failure(self):
        graphs = []

        def compiler(gm, inputs):
            graphs.append(gm)
            raise RuntimeError("expected optimizer backend failure")

        def scope():
            p = Parameter(torch.ones(2, 2))
            optimizer = MyOptimizer([p])

            def step():
                params = []
                optimizer._init_group(params, optimizer.param_groups[0])
                torch._dynamo.graph_break()
                params[0].add_(1)

            with (
                torch.no_grad(),
                self.assertRaisesRegex(
                    torch._dynamo.exc.BackendCompilerFailed,
                    "expected optimizer backend failure",
                ),
            ):
                torch.compile(step, backend=compiler)()
            self.assertEqual(len(graphs), 1)
            return weakref.ref(optimizer), weakref.ref(p)

        with disable_gc():
            optimizer_ref, parameter_ref = scope()
            self.assertIsNone(optimizer_ref())
            self.assertIsNone(parameter_ref())

    def test_stock_optimizer_init_group_after_disable_patch(self):
        # TorchPatcher wraps stock Optimizer._init_group with compiler.disable.
        # Compiling the inner step must still constant-fold _init_group via
        # OptimizerVariable.tp_methods, not inline the disable wrapper.
        torch._dynamo.eval_frame.TorchPatcher.patch()

        # Adadelta is the inductor compiled-optimizer shard that failed; Adamax
        # is the dynamo-wrapped test_optim case. Both wrap _init_group.
        for opt_cls in (torch.optim.Adadelta, torch.optim.Adamax):
            with self.subTest(opt_cls=opt_cls.__name__):
                p = Parameter(torch.ones(2, 2), requires_grad=True)
                opt = opt_cls([p])
                p.grad = torch.ones_like(p)

                step_fn = opt.step.__wrapped__.__wrapped__

                @torch.compile(backend="eager", fullgraph=True)
                def compiled_step():
                    step_fn(opt)

                before = p.detach().clone()
                with torch.no_grad():
                    compiled_step()
                self.assertFalse(torch.equal(p, before))
                torch._dynamo.reset()

    def test_capturable_predicate_covers_runtime_devices(self):
        from torch.optim.optimizer import _get_capturable_supported_devices

        class FakeParam:
            def __init__(self, device_type):
                self.device = type("Device", (), {"type": device_type})()

        opt = torch.optim.Adam([torch.randn(2)], lr=1e-2)
        ov = OptimizerVariable(opt)

        def fake_group(device_type):
            return {"params": [FakeParam(device_type)], "capturable": False}

        for device_type in _get_capturable_supported_devices(supports_xla=True):
            with self.subTest(device=device_type):
                self.assertTrue(
                    ov._safe_to_set_capturable(fake_group(device_type)),
                    f"{device_type} should be capturable",
                )

        for device_type in ("cpu", "mps"):
            with self.subTest(device=device_type):
                self.assertFalse(
                    ov._safe_to_set_capturable(fake_group(device_type)),
                    f"{device_type} should NOT be capturable",
                )


class CapturableOptimizerTests(torch._dynamo.test_case.TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_capturable_optim_preserves_user_setting(self, device):
        from torch.optim.optimizer import _get_capturable_supported_devices

        p = torch.randn(8, device=device).requires_grad_(True)
        p.grad = torch.full_like(p, 1.0)
        opt = torch.optim.Adam([p], lr=1e-2)

        original_capturable = opt.param_groups[0]["capturable"]
        self.assertFalse(original_capturable)

        ov = OptimizerVariable(opt)
        self.assertEqual(
            ov._safe_to_set_capturable(opt.param_groups[0]),
            torch.device(device).type in _get_capturable_supported_devices(),
        )

        opt.step()
        self.assertEqual(opt.param_groups[0]["capturable"], original_capturable)

        sd = deepcopy(opt.state_dict())
        self.assertEqual(
            sd["param_groups"][0].get("capturable", False),
            original_capturable,
        )


instantiate_device_type_tests(
    CapturableOptimizerTests, globals(), allow_mps=True, allow_xpu=True
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
