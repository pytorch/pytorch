# Owner(s): ["oncall: profiler"]
import functools
import gc
import itertools as it
import textwrap
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
from torch._C._profiler import _EventType, _TensorMetadata
from torch.profiler import _memory_profiler, _utils
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.utils import _pytree as pytree


profile = functools.partial(
    torch.profiler.profile, record_shapes=True, profile_memory=True, with_stack=True
)


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestMemoryProfiler(TestCase):
    def test_config_check(self) -> None:
        with torch.profiler.profile() as prof:
            pass

        pattern = r"record_shapes=True, profile_memory=True, with_stack=True"
        with self.assertRaisesRegex(ValueError, pattern):
            prof._memory_profile()

        with torch.profiler.profile(record_shapes=True, with_stack=True) as prof:
            pass

        pattern = r"^profile_memory=True required for memory profiling\.$"
        with self.assertRaisesRegex(ValueError, pattern):
            prof._memory_profile()

        with profile() as prof:
            pass

        self.assertIsInstance(prof._memory_profile(), _memory_profiler.MemoryProfile)


class ScaleLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.rand(()), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class LazyLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x) -> torch.Tensor:
        if getattr(self, "weight", None) is None:
            self.weight = torch.nn.Parameter(
                torch.empty((self.out_features, self.in_features))
            )
            self.bias = torch.nn.Parameter(torch.empty(self.out_features))

        return torch.nn.functional.linear(x, self.weight, self.bias)


class RecordInputOutputDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        self.results = []

    def mark_region(self, name: str):
        self.results.append((name, (), ()))

    @staticmethod
    def flat_ids(args):
        flat_args = pytree.tree_leaves(args)
        return tuple(
            (t._cdata, t.storage().data_ptr())
            for t in flat_args
            if isinstance(t, torch.Tensor) and t.storage()
        )

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        args = args or []
        kwargs = kwargs or {}
        flat_inputs = self.flat_ids(args) + self.flat_ids(kwargs)
        out = func(*args, **kwargs)
        flat_outputs = self.flat_ids(out)
        if (
            flat_inputs or flat_outputs
        ) and "_record_function_enter" not in func.name():
            self.results.append((func.name(), flat_inputs, flat_outputs))
        return out


@skipIfTorchDynamo("TorchDynamo changes Python calls that memory profiling relies on.")
class TestIdentifyGradients(TestCase):
    def gradient_detected(
        self,
        prof: torch.profiler.profile,
        ctx: _EventType,
        grad_tensor: torch.Tensor,
        parameter: Optional[torch.Tensor] = None,
    ) -> None:
        # This is not an exhaustive check, but for the purpose of unit testing
        # it is sufficient.
        def key_matches_tensor(key, tensor) -> bool:
            # Vacuous case.
            if tensor is None:
                return True

            if key is None:
                return False

            return tensor.storage().data_ptr() == key.storage.ptr

        tree = prof.profiler.kineto_results.experimental_event_tree()
        for node in _utils.traverse_dfs(tree):
            for p_key, p_grad_key in _memory_profiler.extract_gradients(node):
                if node.tag == ctx and key_matches_tensor(p_grad_key, grad_tensor):
                    if parameter is None:
                        return True  # Don't need to check parameter; we're done.

                    elif p_key is not None:
                        # For a complex workflow a gradient could correspond to
                        # different parameters at different points in a trace.
                        # However this will not happen in the relatively simple
                        # cases tested here, so if `extract_gradients` identifies
                        # the parameter corresponding to a particular gradient it
                        # must be the one we expect.
                        self.assertTrue(key_matches_tensor(p_key, parameter))
                        return True

        return False

    def assertGradientDetected(self, name: str, *args, **kwargs) -> None:
        self.assertTrue(
            self.gradient_detected(*args, **kwargs),
            f"Failed to identify gradient `{name}` from profile.",
        )

    def assertOnlyGradients(
        self, prof: torch.profiler.profile, tensors: Iterator[torch.Tensor]
    ) -> None:
        allowed_set = {t.storage().data_ptr() for t in tensors}

        tree = prof.profiler.kineto_results.experimental_event_tree()
        for node in _utils.traverse_dfs(tree):
            for _, p_grad_key in _memory_profiler.extract_gradients(node):
                self.assertTrue(
                    p_grad_key.storage.ptr in allowed_set,
                    f"Tensor wrongly marked as gradient: {node.name}: {p_grad_key}",
                )

    def test_extract_gradients_low_level(self) -> None:
        x = torch.ones((1,))
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)

        def check(cold_start: bool):
            self.assertEqual(w0.grad is None, cold_start)
            self.assertEqual(w1.grad is None, cold_start)
            with profile() as prof:
                z = x.expand(4) * w0
                (z * w1).sum().backward()

            # Gradient detection through op inspection does not provide a
            # reference to the parameter corresponding to the gradient.
            self.assertGradientDetected("w0", prof, _EventType.TorchOp, w0.grad)
            self.assertGradientDetected("w1", prof, _EventType.TorchOp, w1.grad)
            self.assertOnlyGradients(prof, (w0.grad, w1.grad))

        check(cold_start=True)
        check(cold_start=False)

    def test_extract_gradients_from_module(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), ScaleLayer())
        named_parameters = dict(model.named_parameters())
        self.assertEqual(len(named_parameters), 3)

        def assert_only_gradients(prof: torch.profiler.profile):
            gradients = tuple(i.grad for i in named_parameters.values())
            self.assertFalse(any(i is None for i in gradients))
            self.assertOnlyGradients(prof, gradients)

        def check(cold_start: bool):
            x = torch.ones((2, 2))
            with profile() as prof:
                model(x).sum().backward()

            for name, p in named_parameters.items():
                # The first time we run a module none of the `.grad` fields
                # have been initialized. This is fine; in that case we can
                # detect everything we need in the profiled section.
                self.assertNotEqual(
                    self.gradient_detected(prof, _EventType.PyCall, p.grad, p),
                    cold_start,
                    name,
                )

                # Op based detection should still identify the gradients.
                self.assertGradientDetected(name, prof, _EventType.TorchOp, p.grad)
            assert_only_gradients(prof)

            # We can detect gradients even when `.backward()` is not called.
            with profile() as prof:
                model(torch.ones((2, 2)))

            for name, p in named_parameters.items():
                self.assertGradientDetected(name, prof, _EventType.PyCall, p.grad, p)
                self.assertFalse(
                    self.gradient_detected(prof, _EventType.TorchOp, p.grad), name
                )
            assert_only_gradients(prof)

        check(cold_start=True)
        check(cold_start=False)

    def _test_extract_gradients_from_optimizer(self, set_to_none: bool) -> None:
        x = torch.ones((1,))
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)
        optimizer = torch.optim.SGD((w0, w1), lr=0.1, momentum=0.9)

        def check(cold_start: bool):
            self.assertEqual(w0.grad is None, cold_start)
            self.assertEqual(w1.grad is None, cold_start)
            with profile() as prof:
                optimizer.zero_grad(set_to_none=set_to_none)
                z = x.expand(4) * w0
                (z * w1).sum().backward()
                optimizer.step()

            # Optimizer instrumentation runs late in the step, so we can detect
            # gradients for both cold and warm start.
            self.assertGradientDetected("w0", prof, _EventType.PyCall, w0.grad, w0)
            self.assertGradientDetected("w1", prof, _EventType.PyCall, w1.grad, w1)

            self.assertGradientDetected("w0", prof, _EventType.TorchOp, w0.grad)
            self.assertGradientDetected("w1", prof, _EventType.TorchOp, w1.grad)
            self.assertOnlyGradients(prof, (w0.grad, w1.grad))

            with profile() as prof:
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=set_to_none)
                    z = x.expand(4) * w0
                    (z * w1).sum().backward()
                    optimizer.step()

            # Inspected state is cached, so if we replace gradients (as is the
            # case for `set_to_none=True`) our python instrumentation will not
            # see them.
            # TODO(robieta): Should `.step()` be excluded from caching?
            self.assertNotEqual(
                self.gradient_detected(prof, _EventType.PyCall, w0.grad, w0),
                set_to_none,
            )

            self.assertNotEqual(
                self.gradient_detected(prof, _EventType.PyCall, w1.grad, w1),
                set_to_none,
            )

            if set_to_none:
                with self.assertRaisesRegex(AssertionError, "Tensor wrongly marked"):
                    self.assertOnlyGradients(prof, (w0.grad, w1.grad))

        check(cold_start=True)
        check(cold_start=False)

    def test_extract_gradients_from_optimizer(self) -> None:
        self._test_extract_gradients_from_optimizer(set_to_none=False)

    def test_extract_gradients_from_optimizer_set_to_none(self) -> None:
        self._test_extract_gradients_from_optimizer(set_to_none=True)

    def test_extract_gradients_from_module_and_optimizer(self) -> None:
        # Module and optimizer are thoroughly tested individually and should be
        # additive. Thus we can manage with a lightweight check that they don't
        # interact adversely.
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), ScaleLayer())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        with profile() as prof:
            model(torch.ones((2, 2))).sum().backward()
            optimizer.step()

        self.assertGradientDetected(
            "weight", prof, _EventType.PyCall, model[0].weight.grad, model[0].weight
        )


@skipIfTorchDynamo("TorchDynamo removes profiler altogether.")
class TestDataFlow(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.maxDiff = None

    @staticmethod
    def formatSchemas(
        prof: torch.profiler.profile, indent: int = 12
    ) -> Tuple[Tuple[str, Tuple[bool, ...]], ...]:
        tree = prof.profiler.kineto_results.experimental_event_tree()
        out: List[Tuple[str, Tuple[bool, ...]]] = []
        for node in _utils.traverse_dfs(tree):
            if node.tag == _EventType.TorchOp:
                e = node.extra_fields
                schemas = _memory_profiler.SchemaMatcher.match_schemas(e)
                name = node.name
                if len(schemas) == 1:
                    name = f"{name}.{schemas[0].overload_name}"
                elif len(schemas) > 1:
                    name = f"{name}.{{{', '.join(s.overload_name for s in schemas)}}}"

                out.append((name, _memory_profiler.SchemaMatcher.inputs_are_mutable(e)))
        return tuple(out)

    @staticmethod
    def _run_and_format_data_flow(
        inputs: Dict[str, torch.Tensor],
        f: Callable[..., Optional[Dict[str, torch.Tensor]]],
        indent: int = 12,
    ) -> str:
        with profile() as prof:
            outputs = f(**inputs) or {}
            gc.collect()

        memory_profile = prof._memory_profile()
        graph = memory_profile._data_flow_graph
        storage_to_id = {key.storage.ptr: key.id for key in graph._active_version}

        lines: List[str] = []
        for name, t in it.chain(inputs.items(), outputs.items()):
            lines.append(f"{name + ':':<8} T{storage_to_id[t.storage().data_ptr()]}")
            if t.grad is not None:
                grad_id = storage_to_id[t.grad.storage().data_ptr()]
                lines.append(f"{name + '.grad:':<9} T{grad_id}")

        if lines:
            lines.append("")

        for node in graph.flow_nodes:
            destroyed = {k for k, v in node._edges.items() if v.is_deletion}

            inputs: List[str] = []
            for key, (_, v) in node.inputs.items():
                inputs.append(f"T{key.id}(v{v}{'*' if key in destroyed else ''})")

            outputs = [f"T{key.id}(v{v})" for key, v in node.outputs.items()]
            if inputs or outputs:
                event_name = node._event.name.replace("torch::autograd::", "")
                lines.append(
                    f"{event_name:<25} {', '.join(inputs):<15}  ->  {', '.join(outputs)}"
                )

        return textwrap.indent("\n".join([l.rstrip() for l in lines]), " " * indent)

    def test_match_schemas(self) -> None:
        with profile() as prof:
            x = torch.ones((1,)).mul(2).add_(2)
            _ = torch.sin(x, out=torch.empty_like(x))

        self.assertEqual(
            self.formatSchemas(prof),
            (
                ("aten::ones.", (False,) * 5),
                ("aten::empty.memory_format", (False,) * 6),
                #
                # fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
                ("aten::fill_.Scalar", (True, False)),
                ("aten::mul.Tensor", (False, False)),
                ("aten::to.dtype", (False,) * 5),
                ("aten::_to_copy.", (False,) * 7),
                ("aten::empty_strided.", (False,) * 6),
                #
                # copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
                ("aten::copy_.", (True, False, False)),
                #
                # add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
                ("aten::add_.Tensor", (True, False, False)),
                ("aten::to.dtype", (False,) * 5),
                ("aten::_to_copy.", (False,) * 7),
                ("aten::empty_strided.", (False,) * 6),
                #
                # copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
                ("aten::copy_.", (True, False, False)),
                ("aten::empty_like.", (False,) * 6),
                ("aten::empty_strided.", (False,) * 6),
                #
                # sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
                ("aten::sin.out", (False, True)),
            ),
        )

    def test_match_schemas_backward(self) -> None:
        x = torch.ones((1,))
        w = torch.ones((1,), requires_grad=True)
        with profile() as prof:
            torch.mul(x, w).backward()

        self.assertEqual(
            self.formatSchemas(prof),
            (
                ("aten::mul.Tensor", (False, False)),
                ("aten::ones_like.", (False,) * 6),
                ("aten::empty_like.", (False,) * 6),
                ("aten::empty_strided.", (False,) * 6),
                #
                # fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
                ("aten::fill_.Scalar", (True, False)),
                ("autograd::engine::evaluate_function: MulBackward0", ()),
                ("MulBackward0", (None,)),
                ("aten::mul.Tensor", (False, False)),
                (
                    "autograd::engine::evaluate_function: torch::autograd::AccumulateGrad",
                    (),
                ),
                ("torch::autograd::AccumulateGrad", (None,)),
                ("aten::detach.", (False,)),
                ("detach", (None,)),
            ),
        )

    def test_match_schemas_tensorlist(self) -> None:
        x = torch.ones((1,))
        y = torch.ones((1,))
        with profile() as prof:
            torch.cat([x, y], axis=0)

        self.assertEqual(
            self.formatSchemas(prof),
            (("aten::cat.", (False, False)),),
        )

    def test_data_flow_graph_with_annotations(self) -> None:
        def f(x, y):
            # torch._C._jit_get_schemas_for_operator will reject any name that
            # is missing a namespace. (denoted by the presence of "::") We want
            # to check that we skip both annotations which have no schema
            # (return empty tuple from SchemaMatcher.lookup_schemas) and
            # annotations which cannot have schema (return None from
            # SchemaMatcher.lookup_schemas).
            with torch.profiler.record_function("Namespaced::Annotation"):
                with torch.profiler.record_function("My Annotation"):
                    x.zero_()
                    y.zero_()
                    return {"x0": torch.ones_like(x), "y0": torch.zeros_like(y)}

        inputs = {"x": torch.ones((1,)), "y": torch.ones((1,))}
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f),
            """\
            x:       T0
            y:       T1
            x0:      T2
            y0:      T3

            aten::zero_               T0(v0)           ->  T0(v1)
            aten::zero_               T1(v0)           ->  T1(v1)
            aten::ones_like           T0(v1)           ->  T2(v0)
            aten::zeros_like          T1(v1)           ->  T3(v0)""",
        )

    def test_data_flow_graph_non_op_allocations(self) -> None:
        def f(x):
            x.mul(2)

        # The python arg parser will convert the python scalar `2` to a Tensor
        # to pass to `aten::mul`. As a result there is no op that "owns" the
        # allocation. The Tensor deletions also do not happen in an op; they
        # are collected as a result of the Python objects going out of scope.
        self.assertExpectedInline(
            self._run_and_format_data_flow({"x": torch.ones((1,))}, f),
            """\
            x:       T1

            [memory]                                   ->  T0(v0)
            aten::mul                 T0(v0), T1(v0)   ->
            [memory]                  T0(v0*)          ->""",
        )

    def test_data_flow_graph_simple(self) -> None:
        inputs = {"x": torch.ones((25,)), "y": torch.ones((25,), requires_grad=True)}

        def f0(x, y):
            z = x.mul(y)
            return {"z": z.view_as(z)}

        def f1(x, y):
            with torch.no_grad():
                return f0(x, y)

        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f0),
            """\
            x:       T0
            y:       T1
            z:       T2

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::view_as             T2(v0)           ->""",
        )

        # Out of place is identical regardless of Autograd.
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f0),
            """\
            x:       T0
            y:       T1
            z:       T2

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::view_as             T2(v0)           ->""",
        )

    def test_data_flow_graph_simple_inplace(self) -> None:
        inputs = {"x": torch.ones((25,)), "y": torch.ones((25,), requires_grad=True)}

        def f0(x, y):
            x.mul_(y)

        def f1(x, y):
            with torch.no_grad():
                return f0(x, y)

        # When Autograd is enabled a second Tensor `T2` is created to store
        # the values of T0(v0) which are needed for backwards.
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f0),
            """\
            x:       T0
            y:       T1

            aten::mul_                T0(v0), T1(v0)   ->  T0(v1), T2(v0)""",
        )

        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f1),
            """\
            x:       T0
            y:       T1

            aten::mul_                T0(v0), T1(v0)   ->  T0(v1)""",
        )

    def test_data_flow_graph_simple_backward(self) -> None:
        inputs = {
            "x": torch.ones((1,)),
            "w": torch.ones((1,), requires_grad=True),
        }
        self.assertExpectedInline(
            self._run_and_format_data_flow(
                inputs, lambda x, w: (x * w).sin().backward()
            ),
            """\
            x:       T0
            w:       T1
            w.grad:   T7

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::sin                 T2(v0)           ->  T3(v0)
            aten::ones_like           T3(v0)           ->  T4(v0)
            SinBackward0              T2(v0), T4(v0)   ->  T6(v0)
            [memory]                  T2(v0*)          ->
            MulBackward0              T0(v0), T6(v0)   ->  T7(v0)
            [memory]                  T6(v0*)          ->
            AccumulateGrad            T7(v0)           ->
            [memory]                  T4(v0*)          ->
            [memory]                  T3(v0*)          ->""",
        )

    def test_data_flow_graph_complicated(self) -> None:
        def f():
            x = torch.ones((25,))
            y = x.mul(2).add_(2)
            z = torch.sin(y, out=torch.empty_like(y))
            return {"x": x, "y": y, "z": z}

        # T1 is the `2` in `.mul(2)`. The Python arg parser automatically
        # converts Scalar arguments to Tensors. The same is true for `T4`
        # and `.add_(2)`.
        self.assertExpectedInline(
            self._run_and_format_data_flow({}, f),
            """\
            x:       T0
            y:       T3
            z:       T6

            aten::ones                                 ->  T0(v0)
            [memory]                                   ->  T1(v0)
            aten::mul                 T0(v0), T1(v0)   ->  T3(v0)
            [memory]                  T1(v0*)          ->
            [memory]                                   ->  T4(v0)
            aten::add_                T3(v0), T4(v0)   ->  T3(v1)
            [memory]                  T4(v0*)          ->
            aten::empty_like          T3(v1)           ->  T6(v0)
            aten::sin                 T3(v1), T6(v0)   ->  T6(v1)""",
        )

        with profile() as prof:
            f()

        # `aten::mul` creates a temporary Tensor (T2), which is why the output
        # is has ID three rather than two.
        mul_node = prof._memory_profile()._data_flow_graph.flow_nodes[2]
        self.assertEqual(mul_node._event.name, "aten::mul")
        self.assertEqual(len(mul_node.intermediates), 1)
        self.assertEqual(mul_node.intermediates[0].id, 2)

    def test_data_flow_graph_stacked(self) -> None:
        inputs = {
            "x": torch.ones((25,)),
            "w0": torch.ones((1,), requires_grad=True),
            "w1": torch.ones((1,), requires_grad=True),
        }

        def f(x, w0, w1):
            return x.mul(w0).relu().mul(w1).relu().sum()

        def f_fwd(**kwargs):
            with torch.no_grad():
                return {"loss": f(**kwargs)}

        def f_fwd_bwd(**kwargs):
            loss = f(**kwargs)
            loss.backward()
            return {"loss": loss}

        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f_fwd),
            """\
            x:       T0
            w0:      T1
            w1:      T4
            loss:    T7

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::relu                T2(v0)           ->  T3(v0)
            [memory]                  T2(v0*)          ->
            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)
            [memory]                  T3(v0*)          ->
            aten::relu                T5(v0)           ->  T6(v0)
            [memory]                  T5(v0*)          ->
            aten::sum                 T6(v0)           ->  T7(v0)
            [memory]                  T6(v0*)          ->""",
        )

        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f_fwd_bwd),
            """\
            x:       T0
            w0:      T1
            w0.grad:  T15
            w1:      T4
            w1.grad:  T12
            loss:    T7

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::relu                T2(v0)           ->  T3(v0)
            [memory]                  T2(v0*)          ->
            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)
            aten::relu                T5(v0)           ->  T6(v0)
            [memory]                  T5(v0*)          ->
            aten::sum                 T6(v0)           ->  T7(v0)
            aten::ones_like           T7(v0)           ->  T8(v0)
            SumBackward0              T8(v0)           ->
            ReluBackward0             T6(v0), T8(v0)   ->  T9(v0)
            [memory]                  T6(v0*)          ->
            MulBackward0              T3(v0), T4(v0), T9(v0)  ->  T10(v0), T11(v0)
            aten::sum                 T10(v0)          ->  T12(v0)
            [memory]                  T10(v0*)         ->
            [memory]                  T9(v0*)          ->
            AccumulateGrad            T12(v0)          ->
            ReluBackward0             T3(v0), T11(v0)  ->  T13(v0)
            [memory]                  T11(v0*)         ->
            [memory]                  T3(v0*)          ->
            MulBackward0              T0(v0), T13(v0)  ->  T14(v0)
            aten::sum                 T14(v0)          ->  T15(v0)
            [memory]                  T14(v0*)         ->
            [memory]                  T13(v0*)         ->
            AccumulateGrad            T15(v0)          ->
            [memory]                  T8(v0*)          ->""",
        )

        # Second time grads are already initialized.
        self.assertExpectedInline(
            self._run_and_format_data_flow(inputs, f_fwd_bwd),
            """\
            x:       T0
            w0:      T1
            w0.grad:  T17
            w1:      T4
            w1.grad:  T13
            loss:    T7

            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::relu                T2(v0)           ->  T3(v0)
            [memory]                  T2(v0*)          ->
            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)
            aten::relu                T5(v0)           ->  T6(v0)
            [memory]                  T5(v0*)          ->
            aten::sum                 T6(v0)           ->  T7(v0)
            aten::ones_like           T7(v0)           ->  T8(v0)
            SumBackward0              T8(v0)           ->
            ReluBackward0             T6(v0), T8(v0)   ->  T9(v0)
            [memory]                  T6(v0*)          ->
            MulBackward0              T3(v0), T4(v0), T9(v0)  ->  T10(v0), T11(v0)
            aten::sum                 T10(v0)          ->  T12(v0)
            [memory]                  T10(v0*)         ->
            [memory]                  T9(v0*)          ->
            AccumulateGrad            T12(v0*), T13(v0)  ->  T13(v1)
            ReluBackward0             T3(v0), T11(v0)  ->  T14(v0)
            [memory]                  T11(v0*)         ->
            [memory]                  T3(v0*)          ->
            MulBackward0              T0(v0), T14(v0)  ->  T15(v0)
            aten::sum                 T15(v0)          ->  T16(v0)
            [memory]                  T15(v0*)         ->
            [memory]                  T14(v0*)         ->
            AccumulateGrad            T16(v0*), T17(v0)  ->  T17(v1)
            [memory]                  T8(v0*)          ->""",
        )

        return

        x = torch.ones((25,))
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)

        with profile() as prof_no_grad:
            with torch.no_grad():
                x.mul(w0).relu().mul(w1).relu().sum()

        # TODO: one with `.logsumexp(dim=0)`

        self.assertExpectedInline(
            self._format_graph(prof_no_grad),
            """\
            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::relu                T2(v0)           ->  T3(v0)
            [memory]                  T2(v0*)          ->
            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)
            [memory]                  T3(v0*)          ->
            aten::relu                T5(v0)           ->  T6(v0)
            [memory]                  T5(v0*)          ->
            aten::sum                 T6(v0)           ->  T7(v0)
            [memory]                  T6(v0*)          ->
            [memory]                  T7(v0*)          ->""",
        )

        with profile() as prof_grad:
            loss = x.mul(w0).relu().mul(w1).relu().sum()
            loss.backward()

        self.assertExpectedInline(
            self._format_graph(prof_grad),
            """\
            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::relu                T2(v0)           ->  T3(v0)
            [memory]                  T2(v0*)          ->
            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)
            aten::relu                T5(v0)           ->  T6(v0)
            [memory]                  T5(v0*)          ->
            aten::sum                 T6(v0)           ->  T7(v0)
            aten::ones_like           T7(v0)           ->  T8(v0)
            SumBackward0              T8(v0)           ->  T8(v1)
            ReluBackward0             T6(v0), T8(v1)   ->  T8(v2), T9(v0)
            [memory]                  T6(v0*)          ->
            MulBackward0              T3(v0), T4(v0), T9(v0)  ->  T9(v1), T10(v0), T11(v0)
            aten::sum                 T10(v0)          ->  T12(v0)
            [memory]                  T10(v0*)         ->
            [memory]                  T9(v1*)          ->
            AccumulateGrad            T12(v0)          ->  T12(v1)
            ReluBackward0             T3(v0), T11(v0)  ->  T11(v1), T13(v0)
            [memory]                  T11(v1*)         ->
            [memory]                  T3(v0*)          ->
            MulBackward0              T0(v0), T13(v0)  ->  T13(v1), T14(v0)
            aten::sum                 T14(v0)          ->  T15(v0)
            [memory]                  T14(v0*)         ->
            [memory]                  T13(v1*)         ->
            AccumulateGrad            T15(v0)          ->  T15(v1)
            [memory]                  T8(v2*)          ->""",
        )

        # Second time grads are already initialized.
        with profile() as prof_grad:
            loss = x.mul(w0).relu().mul(w1).relu().sum()
            loss.backward()

        self.assertExpectedInline(
            self._format_graph(prof_grad),
            """\
            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)
            aten::relu                T2(v0)           ->  T3(v0)
            [memory]                  T2(v0*)          ->
            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)
            aten::relu                T5(v0)           ->  T6(v0)
            [memory]                  T5(v0*)          ->
            aten::sum                 T6(v0)           ->  T7(v0)
            aten::ones_like           T7(v0)           ->  T8(v0)
            SumBackward0              T8(v0)           ->  T8(v1)
            ReluBackward0             T6(v0), T8(v1)   ->  T8(v2), T9(v0)
            [memory]                  T6(v0*)          ->
            MulBackward0              T3(v0), T4(v0), T9(v0)  ->  T9(v1), T10(v0), T11(v0)
            aten::sum                 T10(v0)          ->  T12(v0)
            [memory]                  T10(v0*)         ->
            [memory]                  T9(v1*)          ->
            AccumulateGrad            T12(v0*), T13(v0)  ->  T13(v1)
            ReluBackward0             T3(v0), T11(v0)  ->  T11(v1), T14(v0)
            [memory]                  T11(v1*)         ->
            [memory]                  T3(v0*)          ->
            MulBackward0              T0(v0), T14(v0)  ->  T14(v1), T15(v0)
            aten::sum                 T15(v0)          ->  T16(v0)
            [memory]                  T15(v0*)         ->
            [memory]                  T14(v1*)         ->
            AccumulateGrad            T16(v0*), T17(v0)  ->  T17(v1)
            [memory]                  T8(v2*)          ->""",
        )


@skipIfTorchDynamo("TorchDynamo changes Python calls that memory profiling relies on.")
class TestMemoryProfilerE2E(TestCase):
    @staticmethod
    def _lookup_tensor_categories(
        t: torch.Tensor, memory_profile: _memory_profiler.MemoryProfile
    ) -> Dict[_memory_profiler.TensorAndID, Optional[_memory_profiler.Category]]:
        storage = t.storage()
        if storage is None:
            raise ValueError("Cannot look up uninitialized Tensor.")

        snapshot = memory_profile._category_snapshot()
        ids = {
            key.storage.allocation_id
            for key, _ in snapshot
            if key.storage.ptr == storage.data_ptr() and key.device == storage.device
        }

        return {
            (key, version): category
            for (key, version), category in memory_profile._category_snapshot().items()
            #
            # If a Tensor is live we want the most recent ID
            if key.storage.allocation_id == max(ids | {-1})
        }

    def _run_and_check_parameters_and_gradients(
        self, inner_fn, model, grads_none: bool = False
    ):
        with profile() as prof:
            inner_fn()

        memory_profile = prof._memory_profile()

        def assert_category(
            t: torch.Tensor,
            category: _memory_profiler.Category,
            should_be_none: bool = False,
        ):
            if should_be_none:
                assert t is None, "tensor should be None but is not."
                return
            self.assertIsNotNone(t)
            categories = self._lookup_tensor_categories(t, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(all(c == category for c in categories.values()), categories)

        for p in model.parameters():
            assert_category(p, _memory_profiler.Category.PARAMETER)
            assert_category(p.grad, _memory_profiler.Category.GRADIENT, grads_none)

        # Rely on internal asserts
        _ = memory_profile.timeline

    def _run_and_format_categories(self, fn, indent=12):
        """Generate summary of assigned categories for expecttest."""

        # Use `__torch_dispatch__` to collect ground truth.
        with RecordInputOutputDispatchMode() as record_ops, profile() as prof:
            fn(lambda name: record_ops.mark_region(f"-- {name} ".ljust(105, "-")))

        memory_profile = prof._memory_profile()
        ptr_pair_to_key: Dict[Tuple[int, int], _memory_profiler.TensorKey] = {}
        snapshot = memory_profile._category_snapshot()

        # Build map from observed live Tensors to the memory profiler's
        # TensorKey representation.
        for op in memory_profile._op_tree.dfs():
            if op.typed[0] == _EventType.TorchOp:
                inputs = pytree.tree_leaves(op.typed[1].inputs)
                for t in (i for i in inputs if isinstance(i, _TensorMetadata)):
                    key = _memory_profiler.TensorKey.from_tensor(t)
                    if key:
                        ptr_pair_to_key[(t.impl_ptr, t.storage_data_ptr)] = key

        def format_categories(ptr_pair: int):
            target_key = ptr_pair_to_key.get(ptr_pair, None)
            if target_key is None:
                return "???"

            matches = tuple(
                (version, category.name if category else "???")
                for (key, version), category in snapshot.items()
                if key == target_key
            )
            assert matches, "Failed to lookup Tensor"

            # Deduplicate version bumps which don't change the category.
            categories = [matches[0][1]]
            for _, category in matches:
                if category != categories[-1]:
                    categories.append(category)

            return f"{target_key.storage.allocation_id} ({','.join(categories)})"

        out: List[str] = []
        for name, inputs, outputs in record_ops.results:
            if inputs or outputs:
                # PyTorch ops
                inputs_str = ", ".join(format_categories(i) for i in inputs)
                outputs_str = ", ".join(format_categories(i) for i in outputs)
                out.append(f"{name:<40} {inputs_str:<45} -> {outputs_str}")

            else:
                # Marked regions.
                out.append(f"\n{name}")

        return textwrap.indent("\n".join(out), " " * indent)

    def test_parameters_and_gradients(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2), ScaleLayer(), torch.nn.Linear(2, 1), ScaleLayer()
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def fwd_only():
            _ = model(torch.ones((2, 2)))

        def fwd_bwd_step():
            optimizer.zero_grad()
            y = model(torch.ones((2, 2)))
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
            optimizer.step()

        # If we profile the first step then gradients will not have been
        # created when we call `model.forward`, so if we don't call `.backward`
        # then gradients are never created.
        self._run_and_check_parameters_and_gradients(
            inner_fn=fwd_only, model=model, grads_none=True
        )

        # On the first step we must rely on `AccumulateGrad`, since gradients
        # did not exist when `model.forward` was called.
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)

        # After one step the python tracer will also flag gradients.
        self.assertTrue(not any(p.grad is None for p in model.parameters()))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)

        # The parameter gradients are not used but we still detect them with
        # the python tracer.
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_only, model=model)

    def test_parameters_and_gradients_set_to_none(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def fwd_bwd_step():
            for _ in range(3):
                # zero grads at the start so gradients are still live to be
                # checked.
                optimizer.zero_grad(set_to_none=True)

                y = model(torch.ones((2, 2)))
                torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
                optimizer.step()

        fwd_bwd_step()
        self.assertTrue(not any(p.grad is None for p in model.parameters()))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)

        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)

    def test_inputs_fwd(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        inputs = [torch.ones((2, 2)) for _ in range(2)]

        with profile() as prof:
            # Inputs which were allocated before profiling began
            for x in inputs:
                _ = model(x)

            # Inputs which were allocated after profiling began
            for _ in range(2):
                x = torch.ones((2, 2))
                inputs.append(x)
                _ = model(x)

        memory_profile = prof._memory_profile()
        for x in inputs:
            categories = self._lookup_tensor_categories(x, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(
                all(i == _memory_profiler.Category.INPUT for i in categories.values()),
                categories,
            )

        snapshot = memory_profile._category_snapshot()
        self.assertTrue(_memory_profiler.Category.INPUT in snapshot.values())

    def test_inputs_fwd_lazy(self):
        model = torch.nn.Sequential(LazyLinear(2, 2), LazyLinear(2, 1))
        inputs = [torch.ones((2, 2)) for _ in range(2)]

        with profile() as prof:
            # Inputs which were allocated before profiling began
            for x in inputs:
                _ = model(x)

            # Inputs which were allocated after profiling began
            for _ in range(2):
                x = torch.ones((2, 2))
                inputs.append(x)
                _ = model(x)

        # For now we can't make any meaningful statements without a backward
        # pass. Here we simply ensure that passes don't generate false positive
        # category classifications.
        memory_profile = prof._memory_profile()
        for x in inputs:
            categories = self._lookup_tensor_categories(x, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(all(i is None for i in categories.values()), categories)

        snapshot = memory_profile._category_snapshot()
        self.assertFalse(_memory_profiler.Category.INPUT in snapshot.values())

    def test_inputs_fwd_bwd(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs_targets = [(torch.ones((2, 2)), torch.rand((2, 1))) for _ in range(2)]

        def fwd_bwd_step(x, targets):
            y = model(x)
            torch.nn.functional.mse_loss(y, targets).backward()
            optimizer.step()
            optimizer.zero_grad()

        with profile() as prof:
            # Inputs which were allocated before profiling began
            for x, targets in inputs_targets:
                fwd_bwd_step(x, targets)

            # Inputs which were allocated after profiling began
            for _ in range(2):
                x = torch.ones((2, 2))
                targets = torch.rand((2, 1))
                inputs_targets.append((x, targets))
                fwd_bwd_step(x, targets)

        memory_profile = prof._memory_profile()

        def check(t):
            categories = self._lookup_tensor_categories(t, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(
                all(i == _memory_profiler.Category.INPUT for i in categories.values())
            )

        for x, targets in inputs_targets:
            check(x)
            check(targets)

    def test_lazily_initialized(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.ReLU(),
            LazyLinear(2, 2),
            torch.nn.ReLU(),
            torch.nn.Linear(2, 1),
        )

        self.assertEqual(len(list(model.parameters())), 4)

        def inner_fn():
            y = model(torch.ones((2, 2)))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
            optimizer.step()

        self._run_and_check_parameters_and_gradients(inner_fn=inner_fn, model=model)
        self.assertEqual(len(list(model.parameters())), 6)

    def test_manual_optimizer_step(self) -> None:
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))

        def inner_fn():
            y = model(torch.ones((2, 2)))
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()

            with torch.no_grad():
                for p in model.parameters():
                    grad = p.grad
                    self.assertIsNotNone(grad)
                    p.add_(grad, alpha=-0.1)

        self._run_and_check_parameters_and_gradients(inner_fn=inner_fn, model=model)

    def test_categories_e2e_simple_fwd(self) -> None:
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)

        def step_fn(_):
            x = torch.ones((2, 2))
            y = torch.cat([x * w0, x * w1], dim=1)

        # NOTE: We expect that all unknown categories. This is simply a sanity
        #       check to ensure that we do not over-label.
        self.assertExpectedInline(
            self._run_and_format_categories(step_fn),
            """\
            aten::ones                                                                             -> 1 (???)
            aten::mul.Tensor                         1 (???), 2 (???)                              -> 3 (???)
            aten::mul.Tensor                         1 (???), 4 (???)                              -> 5 (???)
            aten::cat                                3 (???), 5 (???)                              -> ???""",
        )

    def test_categories_e2e_simple_fwd_bwd(self) -> None:
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)

        def step_fn(mark_region):
            x = torch.ones((2, 2))
            targets = torch.ones((2, 4))

            mark_region("Forward & loss")
            y = torch.cat([x * w0, x * w1], dim=1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y, targets)

            mark_region("Backward")
            loss.backward()

        self.assertExpectedInline(
            self._run_and_format_categories(step_fn),
            """\
            aten::ones                                                                             -> 1 (INPUT)
            aten::ones                                                                             -> 2 (INPUT)

            -- Forward & loss ---------------------------------------------------------------------------------------
            aten::mul.Tensor                         1 (INPUT), 3 (INPUT)                          -> 4 (INPUT)
            aten::mul.Tensor                         1 (INPUT), 5 (INPUT)                          -> 6 (INPUT)
            aten::cat                                4 (INPUT), 6 (INPUT)                          -> 7 (INPUT)
            aten::binary_cross_entropy_with_logits   7 (INPUT), 2 (INPUT)                          -> 11 (INPUT)

            -- Backward ---------------------------------------------------------------------------------------------
            aten::ones_like                          11 (INPUT)                                    -> 14 (INPUT)
            aten::sigmoid                            7 (INPUT)                                     -> 15 (TEMPORARY)
            aten::sub.Tensor                         15 (TEMPORARY), 2 (INPUT)                     -> 16 (TEMPORARY)
            aten::mul.Tensor                         16 (TEMPORARY), 14 (INPUT)                    -> 17 (AUTOGRAD_DETAIL)
            aten::div_.Scalar                        17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::slice.Tensor                       17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::slice.Tensor                       17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::mul.Tensor                         17 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 20 (AUTOGRAD_DETAIL)
            aten::sum.dim_IntList                    20 (AUTOGRAD_DETAIL)                          -> 21 (GRADIENT)
            aten::view                               21 (GRADIENT)                                 -> 21 (GRADIENT)
            aten::detach                             21 (GRADIENT)                                 -> 21 (GRADIENT)
            aten::detach                             21 (GRADIENT)                                 -> ???
            aten::mul.Tensor                         17 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 22 (AUTOGRAD_DETAIL)
            aten::sum.dim_IntList                    22 (AUTOGRAD_DETAIL)                          -> 23 (GRADIENT)
            aten::view                               23 (GRADIENT)                                 -> 23 (GRADIENT)
            aten::detach                             23 (GRADIENT)                                 -> 23 (GRADIENT)
            aten::detach                             23 (GRADIENT)                                 -> ???""",
        )

    def test_categories_e2e_simple_fwd_bwd_step(self) -> None:
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)
        optimizer = torch.optim.SGD([w0, w1], lr=0.1)

        def step_fn(mark_region):
            x = torch.ones((2, 2))
            targets = torch.ones((2, 4))

            mark_region("Forward & loss")
            y = torch.cat([x * w0, x * w1], dim=1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y, targets)

            mark_region("Backward")
            loss.backward()

            mark_region("Optimizer")
            optimizer.step()
            optimizer.zero_grad()

        self.assertExpectedInline(
            self._run_and_format_categories(step_fn),
            """\
            aten::ones                                                                             -> 1 (INPUT)
            aten::ones                                                                             -> 2 (INPUT)

            -- Forward & loss ---------------------------------------------------------------------------------------
            aten::mul.Tensor                         1 (INPUT), 3 (PARAMETER)                      -> 4 (ACTIVATION)
            aten::mul.Tensor                         1 (INPUT), 5 (PARAMETER)                      -> 6 (ACTIVATION)
            aten::cat                                4 (ACTIVATION), 6 (ACTIVATION)                -> 7 (ACTIVATION)
            aten::binary_cross_entropy_with_logits   7 (ACTIVATION), 2 (INPUT)                     -> 11 (ACTIVATION)

            -- Backward ---------------------------------------------------------------------------------------------
            aten::ones_like                          11 (ACTIVATION)                               -> 14 (ACTIVATION)
            aten::sigmoid                            7 (ACTIVATION)                                -> 15 (TEMPORARY)
            aten::sub.Tensor                         15 (TEMPORARY), 2 (INPUT)                     -> 16 (TEMPORARY)
            aten::mul.Tensor                         16 (TEMPORARY), 14 (ACTIVATION)               -> 17 (AUTOGRAD_DETAIL)
            aten::div_.Scalar                        17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::slice.Tensor                       17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::slice.Tensor                       17 (AUTOGRAD_DETAIL)                          -> 17 (AUTOGRAD_DETAIL)
            aten::mul.Tensor                         17 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 20 (AUTOGRAD_DETAIL)
            aten::sum.dim_IntList                    20 (AUTOGRAD_DETAIL)                          -> 21 (GRADIENT)
            aten::view                               21 (GRADIENT)                                 -> 21 (GRADIENT)
            aten::detach                             21 (GRADIENT)                                 -> 21 (GRADIENT)
            aten::detach                             21 (GRADIENT)                                 -> 21 (GRADIENT)
            aten::mul.Tensor                         17 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 22 (AUTOGRAD_DETAIL)
            aten::sum.dim_IntList                    22 (AUTOGRAD_DETAIL)                          -> 23 (GRADIENT)
            aten::view                               23 (GRADIENT)                                 -> 23 (GRADIENT)
            aten::detach                             23 (GRADIENT)                                 -> 23 (GRADIENT)
            aten::detach                             23 (GRADIENT)                                 -> 23 (GRADIENT)

            -- Optimizer --------------------------------------------------------------------------------------------
            aten::add_.Tensor                        3 (PARAMETER), 23 (GRADIENT)                  -> 3 (PARAMETER)
            aten::add_.Tensor                        5 (PARAMETER), 21 (GRADIENT)                  -> 5 (PARAMETER)""",
        )

    def test_categories_e2e_simple_module_fwd(self) -> None:
        model = torch.nn.Linear(2, 4, bias=True)
        self.assertExpectedInline(
            self._run_and_format_categories(lambda _: model(torch.ones((2, 2)))),
            """\
            aten::ones                                                                             -> 1 (INPUT)
            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)
            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)""",
        )

    def test_categories_e2e_simple_module_fwd_bwd(self) -> None:
        model = torch.nn.Linear(2, 1, bias=True)

        def step_fn(mark_region):
            mark_region("Forward & loss")
            loss = model(torch.ones((2, 2))).sum()

            mark_region("Backward")
            loss.backward()

        self.assertExpectedInline(
            self._run_and_format_categories(step_fn),
            """\

            -- Forward & loss ---------------------------------------------------------------------------------------
            aten::ones                                                                             -> 1 (INPUT)
            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)
            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)
            aten::sum                                4 (ACTIVATION)                                -> 5 (ACTIVATION)

            -- Backward ---------------------------------------------------------------------------------------------
            aten::ones_like                          5 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::expand                             6 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::t                                  6 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::mm                                 6 (ACTIVATION), 1 (INPUT)                     -> 7 (GRADIENT)
            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)
            aten::sum.dim_IntList                    6 (ACTIVATION)                                -> 9 (GRADIENT)
            aten::view                               9 (GRADIENT)                                  -> 9 (GRADIENT)
            aten::detach                             9 (GRADIENT)                                  -> 9 (GRADIENT)
            aten::detach                             9 (GRADIENT)                                  -> ???
            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)
            aten::detach                             7 (GRADIENT)                                  -> 7 (GRADIENT)
            aten::detach                             7 (GRADIENT)                                  -> ???""",
        )

    def test_categories_e2e_simple_module_fwd_bwd_step(self) -> None:
        model = torch.nn.Linear(2, 1, bias=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        def step_fn(mark_region):
            mark_region("Forward & loss")
            loss = model(torch.ones((2, 2))).sum()

            mark_region("Backward")
            loss.backward()

            mark_region("Optimizer")
            optimizer.step()
            optimizer.zero_grad()

        self.assertExpectedInline(
            self._run_and_format_categories(step_fn),
            """\

            -- Forward & loss ---------------------------------------------------------------------------------------
            aten::ones                                                                             -> 1 (INPUT)
            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)
            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)
            aten::sum                                4 (ACTIVATION)                                -> 5 (ACTIVATION)

            -- Backward ---------------------------------------------------------------------------------------------
            aten::ones_like                          5 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::expand                             6 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::t                                  6 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::mm                                 6 (ACTIVATION), 1 (INPUT)                     -> 7 (GRADIENT)
            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)
            aten::sum.dim_IntList                    6 (ACTIVATION)                                -> 9 (GRADIENT)
            aten::view                               9 (GRADIENT)                                  -> 9 (GRADIENT)
            aten::detach                             9 (GRADIENT)                                  -> 9 (GRADIENT)
            aten::detach                             9 (GRADIENT)                                  -> 9 (GRADIENT)
            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)
            aten::detach                             7 (GRADIENT)                                  -> 7 (GRADIENT)
            aten::detach                             7 (GRADIENT)                                  -> 7 (GRADIENT)

            -- Optimizer --------------------------------------------------------------------------------------------
            aten::clone                              7 (GRADIENT)                                  -> 10 (OPTIMIZER_STATE)
            aten::detach                             10 (OPTIMIZER_STATE)                          -> 10 (OPTIMIZER_STATE)
            aten::detach                             10 (OPTIMIZER_STATE)                          -> 10 (OPTIMIZER_STATE)
            aten::add_.Tensor                        2 (PARAMETER), 10 (OPTIMIZER_STATE)           -> 2 (PARAMETER)
            aten::clone                              9 (GRADIENT)                                  -> 11 (OPTIMIZER_STATE)
            aten::detach                             11 (OPTIMIZER_STATE)                          -> 11 (OPTIMIZER_STATE)
            aten::detach                             11 (OPTIMIZER_STATE)                          -> 11 (OPTIMIZER_STATE)
            aten::add_.Tensor                        3 (PARAMETER), 11 (OPTIMIZER_STATE)           -> 3 (PARAMETER)""",
        )

    def test_categories_e2e_sequential_fwd(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Softmax(dim=1),
        )
        self.assertExpectedInline(
            self._run_and_format_categories(lambda _: model(torch.ones((2, 2)))),
            """\
            aten::ones                                                                             -> 1 (INPUT)
            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)
            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)
            aten::relu                               4 (ACTIVATION)                                -> 5 (ACTIVATION)
            aten::detach                             5 (ACTIVATION)                                -> ???
            aten::t                                  6 (PARAMETER)                                 -> 6 (PARAMETER)
            aten::mm                                 5 (ACTIVATION), 6 (PARAMETER)                 -> 7 (ACTIVATION)
            aten::_softmax                           7 (ACTIVATION)                                -> 8 (ACTIVATION)
            aten::detach                             8 (ACTIVATION)                                -> ???""",
        )

    def test_categories_e2e_sequential_fwd_bwd(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 4, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4, bias=False),
            torch.nn.Softmax(dim=1),
        )

        def step_fn(mark_region):
            x = torch.ones((2, 2))
            targets = torch.ones((2, 4))

            mark_region("Forward")
            y = model(x)

            mark_region("Loss")
            loss = torch.sum((y - targets) ** 2).mean()

            mark_region("Backward")
            loss.backward()

        self.assertExpectedInline(
            self._run_and_format_categories(step_fn),
            """\
            aten::ones                                                                             -> 1 (INPUT)
            aten::ones                                                                             -> 2 (INPUT)

            -- Forward ----------------------------------------------------------------------------------------------
            aten::t                                  3 (PARAMETER)                                 -> 3 (PARAMETER)
            aten::addmm                              4 (PARAMETER), 1 (INPUT), 3 (PARAMETER)       -> 5 (ACTIVATION)
            aten::relu                               5 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::detach                             6 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::t                                  7 (PARAMETER)                                 -> 7 (PARAMETER)
            aten::mm                                 6 (ACTIVATION), 7 (PARAMETER)                 -> 8 (ACTIVATION)
            aten::_softmax                           8 (ACTIVATION)                                -> 9 (ACTIVATION)
            aten::detach                             9 (ACTIVATION)                                -> 9 (ACTIVATION)

            -- Loss -------------------------------------------------------------------------------------------------
            aten::sub.Tensor                         9 (ACTIVATION), 2 (INPUT)                     -> 10 (ACTIVATION)
            aten::pow.Tensor_Scalar                  10 (ACTIVATION)                               -> 11 (ACTIVATION)
            aten::sum                                11 (ACTIVATION)                               -> 12 (ACTIVATION)
            aten::mean                               12 (ACTIVATION)                               -> 13 (ACTIVATION)

            -- Backward ---------------------------------------------------------------------------------------------
            aten::ones_like                          13 (ACTIVATION)                               -> 16 (ACTIVATION)
            aten::expand                             16 (ACTIVATION)                               -> 16 (ACTIVATION)
            aten::div.Scalar                         16 (ACTIVATION)                               -> 19 (AUTOGRAD_DETAIL)
            aten::expand                             19 (AUTOGRAD_DETAIL)                          -> 19 (AUTOGRAD_DETAIL)
            aten::pow.Tensor_Scalar                  10 (ACTIVATION)                               -> 20 (TEMPORARY)
            aten::mul.Scalar                         20 (TEMPORARY)                                -> 23 (TEMPORARY)
            aten::mul.Tensor                         19 (AUTOGRAD_DETAIL), 23 (TEMPORARY)          -> 24 (AUTOGRAD_DETAIL)
            aten::detach                             9 (ACTIVATION)                                -> 9 (ACTIVATION)
            aten::_softmax_backward_data             24 (AUTOGRAD_DETAIL), 9 (ACTIVATION)          -> 25 (AUTOGRAD_DETAIL)
            aten::t                                  25 (AUTOGRAD_DETAIL)                          -> 25 (AUTOGRAD_DETAIL)
            aten::mm                                 25 (AUTOGRAD_DETAIL), 6 (ACTIVATION)          -> 26 (GRADIENT)
            aten::t                                  26 (GRADIENT)                                 -> 26 (GRADIENT)
            aten::t                                  7 (PARAMETER)                                 -> 7 (PARAMETER)
            aten::mm                                 25 (AUTOGRAD_DETAIL), 7 (PARAMETER)           -> 27 (AUTOGRAD_DETAIL)
            aten::t                                  26 (GRADIENT)                                 -> 26 (GRADIENT)
            aten::detach                             26 (GRADIENT)                                 -> 26 (GRADIENT)
            aten::detach                             26 (GRADIENT)                                 -> ???
            aten::detach                             6 (ACTIVATION)                                -> 6 (ACTIVATION)
            aten::threshold_backward                 27 (AUTOGRAD_DETAIL), 6 (ACTIVATION)          -> 28 (AUTOGRAD_DETAIL)
            aten::t                                  28 (AUTOGRAD_DETAIL)                          -> 28 (AUTOGRAD_DETAIL)
            aten::mm                                 28 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 29 (GRADIENT)
            aten::t                                  29 (GRADIENT)                                 -> 29 (GRADIENT)
            aten::sum.dim_IntList                    28 (AUTOGRAD_DETAIL)                          -> 30 (GRADIENT)
            aten::view                               30 (GRADIENT)                                 -> 30 (GRADIENT)
            aten::detach                             30 (GRADIENT)                                 -> 30 (GRADIENT)
            aten::detach                             30 (GRADIENT)                                 -> ???
            aten::t                                  29 (GRADIENT)                                 -> 29 (GRADIENT)
            aten::detach                             29 (GRADIENT)                                 -> 29 (GRADIENT)
            aten::detach                             29 (GRADIENT)                                 -> ???""",
        )

    def test_memory_timeline(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512, bias=False),
            torch.nn.Softmax(dim=1),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        with profile() as prof:
            x = torch.ones((1024, 64))
            targets = torch.ones((1024, 512))
            y = model(x)
            loss = torch.nn.functional.mse_loss(y, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        memory_profile = prof._memory_profile()
        timeline = memory_profile.timeline
        times = tuple(t for t, _, _, _ in timeline)
        self.assertTrue(all(t1 >= t0 for t0, t1 in zip(times, times[1:])), times)
        self.assertTrue(
            all(
                (t == -1) if action == _memory_profiler.Action.PREEXISTING else (t > 0)
                for t, action, _, _ in timeline
            )
        )

        def category_name(category):
            return category.name if category else "???"

        def format_action(action, key, version):
            category = memory_profile._categories.get(key, version)
            if action == _memory_profiler.Action.INCREMENT_VERSION:
                new_category = memory_profile._categories.get(key, version + 1)
                if category != new_category:
                    return f"{category_name(category)} -> {category_name(new_category)}"
            return category_name(category)

        def format_size(size: int):
            if size < 1024:
                return f"{size / 1024:3.1f} kB"
            return f"{size // 1024} kB"

        # We generate sequential IDs for Tensors; however platforms vary
        # slightly in the exact computation executed. If this results in
        # tensor creation the IDs will be shifted and the unit test will fail.
        # (Even though the behavior we're testing is unchanged.) To correct for
        # this we assign sequential numbers to the tensors which are actually
        # tested, effectively suppressing the extraneous implementation details.
        id_map = {}

        def id_for_testing(key):
            return id_map.setdefault(key.storage.allocation_id, len(id_map))

        lines = [
            f"{action.name.lower():<25}  {format_action(action, key, version):<25}  "
            f"{id_for_testing(key):>3}(v{version}) {format_size(size):>15}"
            for _, action, (key, version), size in prof._memory_profile().timeline
            # We generally don't care about tiny allocations during memory
            # profiling and they add a lot of noise to the unit test.
            if size > 1024
        ]

        self.assertExpectedInline(
            textwrap.indent("\n".join(lines), " " * 12),
            """\
            preexisting                PARAMETER                    0(v0)          128 kB
            preexisting                PARAMETER                    1(v0)            2 kB
            preexisting                PARAMETER                    2(v0)         1024 kB
            create                     INPUT                        3(v0)          256 kB
            create                     INPUT                        4(v0)         2048 kB
            create                     ACTIVATION                   5(v0)         2048 kB
            create                     ACTIVATION                   6(v0)         2048 kB
            destroy                    ACTIVATION                   5(v0)         2048 kB
            create                     ACTIVATION                   7(v0)         2048 kB
            create                     ACTIVATION                   8(v0)         2048 kB
            destroy                    ACTIVATION                   7(v0)         2048 kB
            create                     ACTIVATION                   9(v0)         2048 kB
            create                     TEMPORARY                   10(v0)         2048 kB
            destroy                    TEMPORARY                   10(v0)         2048 kB
            create                     AUTOGRAD_DETAIL             11(v0)         2048 kB
            create                     AUTOGRAD_DETAIL             12(v0)         2048 kB
            destroy                    AUTOGRAD_DETAIL             11(v0)         2048 kB
            create                     GRADIENT                    13(v0)         1024 kB
            create                     AUTOGRAD_DETAIL             14(v0)         2048 kB
            destroy                    AUTOGRAD_DETAIL             12(v0)         2048 kB
            create                     AUTOGRAD_DETAIL             15(v0)         2048 kB
            destroy                    AUTOGRAD_DETAIL             14(v0)         2048 kB
            destroy                    ACTIVATION                   6(v0)         2048 kB
            create                     GRADIENT                    16(v0)          128 kB
            create                     GRADIENT                    17(v0)            2 kB
            destroy                    AUTOGRAD_DETAIL             15(v0)         2048 kB
            create                     OPTIMIZER_STATE             18(v0)          128 kB
            create                     OPTIMIZER_STATE             19(v0)          128 kB
            create                     OPTIMIZER_STATE             20(v0)            2 kB
            create                     OPTIMIZER_STATE             21(v0)            2 kB
            create                     OPTIMIZER_STATE             22(v0)         1024 kB
            create                     OPTIMIZER_STATE             23(v0)         1024 kB
            increment_version          OPTIMIZER_STATE             18(v0)          128 kB
            increment_version          OPTIMIZER_STATE             19(v0)          128 kB
            increment_version          OPTIMIZER_STATE             19(v1)          128 kB
            create                     ???                         24(v0)          128 kB
            create                     ???                         25(v0)          128 kB
            destroy                    ???                         24(v0)          128 kB
            increment_version          ???                         25(v0)          128 kB
            increment_version          PARAMETER                    0(v0)          128 kB
            increment_version          OPTIMIZER_STATE             20(v0)            2 kB
            increment_version          OPTIMIZER_STATE             21(v0)            2 kB
            increment_version          OPTIMIZER_STATE             21(v1)            2 kB
            create                     ???                         26(v0)            2 kB
            create                     ???                         27(v0)            2 kB
            destroy                    ???                         26(v0)            2 kB
            increment_version          ???                         27(v0)            2 kB
            destroy                    ???                         25(v1)          128 kB
            increment_version          PARAMETER                    1(v0)            2 kB
            increment_version          OPTIMIZER_STATE             22(v0)         1024 kB
            increment_version          OPTIMIZER_STATE             23(v0)         1024 kB
            increment_version          OPTIMIZER_STATE             23(v1)         1024 kB
            create                     ???                         28(v0)         1024 kB
            create                     ???                         29(v0)         1024 kB
            destroy                    ???                         28(v0)         1024 kB
            increment_version          ???                         29(v0)         1024 kB
            destroy                    ???                         27(v1)            2 kB
            increment_version          PARAMETER                    2(v0)         1024 kB
            destroy                    ???                         29(v1)         1024 kB
            destroy                    GRADIENT                    16(v0)          128 kB
            destroy                    GRADIENT                    17(v0)            2 kB
            destroy                    GRADIENT                    13(v0)         1024 kB""",
        )

    def test_memory_timeline_no_id(self) -> None:
        # On CPU the default behavior is to simply forward to malloc. That
        # means that when we free `x` the allocator doesn't actually know how
        # many bytes are in the allocation, and thus there's no point to
        # calling `c10::reportMemoryUsageToProfiler`. So in order to test that
        # memory profiler processes this case correctly we need to use CUDA
        # where we do always keep a record.
        x = torch.ones((1024,), device="cuda" if torch.cuda.is_available() else "cpu")

        with profile() as prof:
            # We never see `x` used so we don't know the storage is for a
            # Tensor, but we do still see the free event.
            del x

            # For empty we see the allocation and free, but not any use.
            # So this also cannot be identified as a Tensor.
            y = torch.empty((64,))
            del y

            z = torch.empty((256,))
            z.view_as(z)  # Show `z` to the profiler
            del z

        memory_profile = prof._memory_profile()

        expected = [
            # x
            (_memory_profiler.Action.PREEXISTING, 4096),
            (_memory_profiler.Action.DESTROY, 4096),
            #
            # y
            (_memory_profiler.Action.CREATE, 256),
            (_memory_profiler.Action.DESTROY, 256),
            #
            # z
            (_memory_profiler.Action.CREATE, 1024),
            (_memory_profiler.Action.DESTROY, 1024),
        ]

        actual = [(action, size) for _, action, _, size in memory_profile.timeline]

        # See above.
        if not torch.cuda.is_available():
            expected = expected[2:]
            for event in expected:
                self.assertTrue(
                    event in actual, f"event: {event} was not found in actual."
                )
        else:
            self.assertEqual(
                actual,
                expected,
                f"expected does not match actual: {actual}",
            )


if __name__ == "__main__":
    run_tests()
