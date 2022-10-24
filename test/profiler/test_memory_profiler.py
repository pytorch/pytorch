# Owner(s): ["oncall: profiler"]
import functools
from typing import Iterator, Optional

import torch
from torch._C._profiler import _EventType
from torch.profiler import _memory_profiler, _utils
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


profile = functools.partial(
    torch.profiler.profile, record_shapes=True, profile_memory=True, with_stack=True
)


class ScaleLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.rand(()), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


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
        named_parameters = {name: p for name, p in model.named_parameters()}
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


if __name__ == "__main__":
    run_tests()
