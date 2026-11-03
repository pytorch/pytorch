# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import copy
import gc
import weakref
from contextlib import nullcontext

from model_registry import MLPModule, MultiInterMediateModel

import torch
from torch.distributed.pipelining._backward import (
    stage_backward,
    stage_backward_input,
    stage_backward_weight,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    HardwareClassification,
    run_tests,
    TestCase,
)


d_hid = 512
batch_size = 256


class CustomMatmul(torch.autograd.Function):
    """Matmul whose optional third input is stored only on the context."""

    @staticmethod
    def forward(ctx, a, b, saved=None):
        ctx.save_for_backward(a, b)
        ctx.saved = saved
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            grad_a = torch.matmul(grad_output, b.transpose(-1, -2))
        if ctx.needs_input_grad[1]:
            grad_b = torch.matmul(a.transpose(-1, -2), grad_output)
        if len(ctx.needs_input_grad) == 2:
            return grad_a, grad_b
        return grad_a, grad_b, None


class HoldAbove(torch.autograd.Function):
    """Identity that retains ``saved`` only while this node stays alive."""

    @staticmethod
    def forward(ctx, activation, saved):
        ctx.saved = saved
        return activation.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _find_grad_node(root, name):
    """Return a node under ``root`` whose type name contains ``name``."""
    pending = [root]
    seen_ids: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen_ids:
            continue
        seen_ids.add(id(current))
        if name in type(current).__name__:
            return current
        pending.extend(fn for fn, _ in current.next_functions)
    raise AssertionError(f"missing {name} under {type(root).__name__}")


class StageBackwardTests(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    def test_stage_backward(self, device):
        # oneDNN is nondeterministic on XPU: https://github.com/intel/torch-xpu-ops/issues/1682
        with DeterministicGuard(True) if self.device_type == "xpu" else nullcontext():
            # MLP as a stage module
            mod = MLPModule(d_hid).to(device)
            x = torch.randn(batch_size, d_hid, device=device)
            # As in a pipeline stage, the inputs to this stage requires gradients
            x.requires_grad_(True)
            target = torch.randn(batch_size, d_hid, device=device)
            loss_fn = torch.nn.MSELoss(reduction="sum")

            # Make a copy
            ref_mod = copy.deepcopy(mod).to(device)
            ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
            ref_target = target.detach().to(device)

            # Forward and backward in stage manner
            out = mod(x)
            loss = loss_fn(out, target)
            grad_inputs = stage_backward(
                stage_output=loss,
                output_grads=None,
                input_values=(x,),
            )

            # Run reference
            ref_out = ref_mod(ref_x)
            ref_loss = loss_fn(ref_out, ref_target)
            ref_loss.backward()

            torch.testing.assert_close(grad_inputs[0], ref_x.grad)

            # Every rank checks gradients
            for name, p in mod.named_parameters():
                ref_p = ref_mod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

    def test_stage_backward_input(self, device):
        # MLP as a stage module
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device)
        # As in a pipeline stage, the inputs to this stage requires gradients
        x.requires_grad_(True)
        target = torch.randn(batch_size, d_hid, device=device)
        loss_fn = torch.nn.MSELoss(reduction="sum")

        # Make a copy
        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
        ref_target = target.detach().to(device)

        # Forward, then backward of loss with respect to inputs
        out = mod(x)
        loss = loss_fn(out, target)
        dinputs, _param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )

        # Run reference
        ref_out = ref_mod(ref_x)
        ref_loss = loss_fn(ref_out, ref_target)
        ref_loss.backward()

        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for _, p in mod.named_parameters():
            # Check that the weight gradients were not updated
            self.assertEqual(p.grad, None)

    def test_stage_backward_input_ignores_non_tensor_inputs(self, device):
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        non_tensor_input = object()

        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().requires_grad_(True).to(device)

        loss = mod(x).sum()
        dinputs, _param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[non_tensor_input, x],
            weights=mod.parameters(),
        )

        ref_mod(ref_x).sum().backward()
        self.assertEqual(dinputs[0], None)
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(dinputs[1], ref_x.grad)

    def test_stage_backward_weight_custom_autograd_function(self, device):
        # Split dI/dW grads for a custom autograd.Function should match a
        # full backward().
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))

            def forward(self, x):
                return CustomMatmul.apply(x, self.weight)

        mod = TestModel().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)

        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().clone().requires_grad_(True)

        out = mod(x)
        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=(out,),
            output_grads=[output_grad],
            input_values=[x],
            weights=mod.parameters(),
        )
        stage_backward_weight(mod.parameters(), param_groups)

        ref_mod(ref_x).backward(output_grad)
        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            torch.testing.assert_close(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_weight_custom_function_under_cpp_root(self, device):
        # .sum() makes the loss root a C++ SumBackward0. The custom Function is
        # only an intermediate, and split backward must still match.
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))

            def forward(self, x):
                return CustomMatmul.apply(x, self.weight).sum()

        mod = TestModel().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().clone().requires_grad_(True)

        loss = mod(x)
        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )
        stage_backward_weight(mod.parameters(), param_groups)

        ref_mod(ref_x).backward()
        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            torch.testing.assert_close(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_input_frees_tensor_above_builtin_intermediate(self, device):
        # A tensor saved only above the built-in matmul must die when the output
        # is detached. The matmul node retained for weight backward does not
        # keep it alive.
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))

            def forward(self, x, saved):
                return HoldAbove.apply(torch.matmul(x, self.weight), saved).sum()

        saved = torch.randn(batch_size, d_hid, device=device)
        saved_ref = weakref.ref(saved)
        mod = TestModel().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        loss = mod(x, saved)
        hold = _find_grad_node(loss.grad_fn, "HoldAbove")
        self.assertIs(hold.saved, saved)
        del saved, hold
        self.assertIsNotNone(saved_ref())

        # Keep the returned groups alive. The saved tensor must already be dead.
        _dinputs, _param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )
        gc.collect()
        self.assertIsNone(saved_ref())

    def test_stage_backward_frees_tensor_above_custom_function(self, device):
        # A tensor saved only above the custom Function must die when the
        # output is detached. Weight backward still matches a full backward.
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))

            def forward(self, x, saved):
                activation = CustomMatmul.apply(x, self.weight).relu()
                return HoldAbove.apply(activation, saved).sum()

        saved = torch.randn(batch_size, d_hid, device=device)
        saved_ref = weakref.ref(saved)
        mod = TestModel().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().clone().requires_grad_(True)
        ref_saved = saved.detach().clone()
        loss = mod(x, saved)
        hold_node = _find_grad_node(loss.grad_fn, "HoldAbove")
        self.assertIs(hold_node.saved, saved)
        del saved, hold_node
        self.assertIsNotNone(saved_ref())

        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=(loss,),
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )
        gc.collect()
        self.assertIsNone(saved_ref())

        stage_backward_weight(mod.parameters(), param_groups)
        ref_mod(ref_x, ref_saved).backward()
        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            torch.testing.assert_close(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_weight_releases_custom_function(self, device):
        # param_groups keeps the custom Function across detach. Weight backward
        # drops it, and a tensor saved only on that context must die with it.
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))

            def forward(self, x, saved):
                return CustomMatmul.apply(x, self.weight, saved)

        saved = torch.randn(batch_size, d_hid, device=device)
        saved_ref = weakref.ref(saved)
        mod = TestModel().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)
        out = mod(x, saved)
        ctx = out.grad_fn
        ctx_ref = weakref.ref(ctx)
        self.assertIs(ctx.saved, saved)
        del saved, ctx

        _dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=(out,),
            output_grads=[output_grad],
            input_values=[x],
            weights=mod.parameters(),
        )
        del out
        gc.collect()
        self.assertIsNotNone(saved_ref())
        self.assertIsNotNone(ctx_ref())

        stage_backward_weight(mod.parameters(), param_groups)
        gc.collect()
        self.assertIsNone(saved_ref())
        self.assertIsNone(ctx_ref())

    def test_stage_backward_weight(self, device):
        # oneDNN is nondeterministic on XPU: https://github.com/intel/torch-xpu-ops/issues/1682
        with DeterministicGuard(True) if self.device_type == "xpu" else nullcontext():
            # MLP as a stage module
            mod = MLPModule(d_hid).to(device)
            x = torch.randn(batch_size, d_hid, device=device)
            # As in a pipeline stage, the inputs to this stage requires gradients
            x.requires_grad_(True)
            target = torch.randn(batch_size, d_hid, device=device)
            loss_fn = torch.nn.MSELoss(reduction="sum")

            # Make a copy
            ref_mod = copy.deepcopy(mod).to(device)
            ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
            ref_target = target.detach().to(device)
            # Forward, then backward of loss with respect to inputs
            out = mod(x)
            loss = loss_fn(out, target)
            _dinputs, param_groups = stage_backward_input(
                stage_outputs_or_loss=(loss,),
                output_grads=None,
                input_values=[x],
                weights=mod.parameters(),
            )

            # backward of loss with respect to weights
            stage_backward_weight(mod.parameters(), param_groups, retain_graph=True)

            # Run reference
            ref_out = ref_mod(ref_x)
            ref_loss = loss_fn(ref_out, ref_target)
            ref_loss.backward()

            # Every rank checks gradients
            for name, p in mod.named_parameters():
                ref_p = ref_mod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

    def test_stage_backward_weight_multiple_iters(self, device):
        # oneDNN is nondeterministic on XPU: https://github.com/intel/torch-xpu-ops/issues/1682
        with DeterministicGuard(True) if self.device_type == "xpu" else nullcontext():
            # MLP as a stage module
            mod = MLPModule(d_hid).to(device)
            inputs = []
            for _ in range(10):
                x = torch.randn(batch_size, d_hid, device=device)
                inputs.append(x)
                # As in a pipeline stage, the inputs to this stage requires gradients
                x.requires_grad_(True)

            target = torch.randn(batch_size, d_hid, device=device)
            loss_fn = torch.nn.MSELoss(reduction="sum")

            # Make a copy
            ref_mod = copy.deepcopy(mod).to(device)
            ref_inputs = []
            for x in inputs:
                ref_x = x.detach().requires_grad_(x.requires_grad).to(device)
                ref_inputs.append(ref_x)
            ref_target = target.detach().to(device)

            # Forward, then backward of loss with respect to inputs
            for x in inputs:
                out = mod(x)
                loss = loss_fn(out, target)
                _dinputs, param_groups = stage_backward_input(
                    stage_outputs_or_loss=(loss,),
                    output_grads=None,
                    input_values=[x],
                    weights=mod.parameters(),
                )

                # backward of loss with respect to weights
                stage_backward_weight(mod.parameters(), param_groups)

            # Run reference
            for ref_x in ref_inputs:
                ref_out = ref_mod(ref_x)
                ref_loss = loss_fn(ref_out, ref_target)
                ref_loss.backward()

            # Every rank checks gradients
            for name, p in mod.named_parameters():
                ref_p = ref_mod.get_parameter(name)
                try:
                    torch.testing.assert_close(p.grad, ref_p.grad)
                except AssertionError:
                    print(f"Gradient test failed for {name}: {p.grad} vs {ref_p.grad}")
                    raise

    def test_stage_backward_weight_grad_validation(self, device):
        test_cases = [
            (
                "size == 2",
                lambda: MultiInterMediateModel([d_hid // 2, d_hid // 2]).to(device),
                lambda: [
                    (
                        torch.randn(batch_size, d_hid // 2, device=device),
                        torch.randn(d_hid // 2, d_hid // 2, device=device),
                    )
                ],
            ),
            (
                "size = 1",
                lambda: MLPModule(d_hid).to(device),
                lambda: [(torch.randn(batch_size, d_hid, device=device),)],
            ),
            (
                "1 grad, 1 None",
                lambda: MultiInterMediateModel([d_hid // 2, d_hid // 2]).to(device),
                lambda: [(torch.randn(batch_size, d_hid // 2, device=device), None)],
            ),
            (
                "1 None, 1 grad",
                lambda: MultiInterMediateModel([d_hid // 2, d_hid // 2]).to(device),
                lambda: [(None, torch.randn(d_hid // 2, d_hid // 2, device=device))],
            ),
        ]

        for description, module_factory, mock_grads_factory in test_cases:
            with self.subTest(description=description):
                mod = module_factory()
                x = torch.randn(batch_size, d_hid, device=device)
                x.requires_grad_(True)
                out = mod(x)
                loss = torch.sum(out)
                dinputs, param_groups = stage_backward_input(
                    stage_outputs_or_loss=[loss],
                    output_grads=None,
                    input_values=[x],
                    weights=mod.parameters(),
                )

                # Set up mock grads
                for param_group in param_groups:
                    param_group["grads"] = mock_grads_factory()

                stage_backward_weight(mod.parameters(), param_groups)

    def test_stage_backward_multi_output_intermediate(self, device):
        mod = MultiInterMediateModel([d_hid // 2, d_hid // 2]).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)

        out = mod(x)
        loss = out.sum()

        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=[loss],
            output_grads=None,
            input_values=[x],
            weights=mod.parameters(),
        )

        stage_backward_weight(mod.parameters(), param_groups)

        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().clone().requires_grad_(True)
        ref_out = ref_mod(ref_x)
        ref_loss = ref_out.sum()
        ref_loss.backward()

        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            ref_p = ref_mod.get_parameter(name)
            torch.testing.assert_close(p.grad, ref_p.grad)

    def test_stage_backward_weight_shared_weights(self, device):
        # oneDNN is nondeterministic on XPU: https://github.com/intel/torch-xpu-ops/issues/1682
        with DeterministicGuard(True) if self.device_type == "xpu" else nullcontext():

            class SharedWeightModule(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.w = torch.nn.Parameter(torch.randn(d_hid, d_hid))

                def forward(self, x):
                    x = torch.matmul(x, self.w)
                    x = torch.relu(x)
                    return torch.matmul(x, self.w)

            mod = SharedWeightModule().to(device)
            x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)

            ref_mod = copy.deepcopy(mod)
            ref_x = x.detach().clone().requires_grad_(True)

            out = mod(x)
            loss = out.sum()

            dinputs, param_groups = stage_backward_input(
                stage_outputs_or_loss=[loss],
                output_grads=None,
                input_values=[x],
                weights=mod.parameters(),
            )
            stage_backward_weight(mod.parameters(), param_groups)

            ref_out = ref_mod(ref_x)
            ref_loss = ref_out.sum()
            ref_loss.backward()

            torch.testing.assert_close(dinputs[0], ref_x.grad)
            for name, p in mod.named_parameters():
                ref_p = ref_mod.get_parameter(name)
                torch.testing.assert_close(p.grad, ref_p.grad)

    def test_stage_backward_from_gradient_edge(self, device):
        """Match tensor-rooted backward after releasing the output."""
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)

        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().clone().requires_grad_(True)

        out = mod(x)
        edge = torch.autograd.graph.get_gradient_edge(out)
        # The edge must not keep the activation alive.
        released = weakref.ref(out)
        del out
        self.assertIsNone(released())

        grad_inputs = stage_backward(
            stage_output=(edge,),
            output_grads=(output_grad,),
            input_values=(x,),
        )

        ref_out = ref_mod(ref_x)
        ref_out.backward(output_grad)

        self.assertEqual(grad_inputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            self.assertEqual(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_mixed_edge_and_tensor_outputs(self, device):
        """Handle edge, tensor, and non-grad outputs together."""

        class TwoOutputModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net1 = torch.nn.Linear(d_hid, d_hid)
                self.net2 = torch.nn.Linear(d_hid, d_hid)

            def forward(self, x):
                return self.net1(x), self.net2(x), torch.ones(1, device=x.device)

        mod = TwoOutputModule().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        grads = (
            torch.randn(batch_size, d_hid, device=device),
            torch.randn(batch_size, d_hid, device=device),
            None,
        )

        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().clone().requires_grad_(True)

        first, second, no_grad_out = mod(x)
        edge = torch.autograd.graph.get_gradient_edge(first)
        del first

        self.assertFalse(no_grad_out.requires_grad)
        grad_inputs = stage_backward(
            stage_output=(edge, second, None),
            output_grads=grads,
            input_values=(x,),
        )

        ref_first, ref_second, _ = ref_mod(ref_x)
        torch.autograd.backward((ref_first, ref_second), (grads[0], grads[1]))

        self.assertEqual(grad_inputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            self.assertEqual(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_edge_keeps_python_autograd_function_alive(self, device):
        """Keep a Python autograd graph alive through its edge."""

        class ScaleBy(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, factor):
                ctx.factor = factor
                return x * factor

            @staticmethod
            def backward(ctx, grad_out):
                return grad_out * ctx.factor, None

        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)

        out = ScaleBy.apply(x, 3.0)
        edge = torch.autograd.graph.get_gradient_edge(out)
        del out

        grad_inputs = stage_backward(
            stage_output=(edge,),
            output_grads=(output_grad,),
            input_values=(x,),
        )
        self.assertEqual(grad_inputs[0], output_grad * 3.0)

    def test_stage_backward_input_from_gradient_edge(self, device):
        """Match tensor-rooted split backward."""
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)

        ref_mod = copy.deepcopy(mod).to(device)
        ref_x = x.detach().clone().requires_grad_(True)

        out = mod(x)
        edge = torch.autograd.graph.get_gradient_edge(out)
        del out

        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=[edge],
            output_grads=[output_grad],
            input_values=[x],
            weights=mod.parameters(),
        )
        stage_backward_weight(mod.parameters(), param_groups)

        ref_mod(ref_x).backward(output_grad)

        self.assertEqual(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            self.assertEqual(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_weight_from_released_cpp_root(self, device):
        # A released output is a GradientEdge. Dropping that edge after input
        # backward must still leave weight backward able to run.
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))

            def forward(self, x):
                return CustomMatmul.apply(x, self.weight).relu()

        mod = TestModel().to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        output_grad = torch.randn(batch_size, d_hid, device=device)
        ref_mod = copy.deepcopy(mod)
        ref_x = x.detach().clone().requires_grad_(True)

        out = mod(x)
        edge = torch.autograd.graph.get_gradient_edge(out)
        released = weakref.ref(out)
        del out
        self.assertIsNone(released())

        dinputs, param_groups = stage_backward_input(
            stage_outputs_or_loss=[edge],
            output_grads=[output_grad],
            input_values=[x],
            weights=mod.parameters(),
        )
        del edge
        gc.collect()
        stage_backward_weight(mod.parameters(), param_groups)

        ref_mod(ref_x).backward(output_grad)
        torch.testing.assert_close(dinputs[0], ref_x.grad)
        for name, p in mod.named_parameters():
            torch.testing.assert_close(p.grad, ref_mod.get_parameter(name).grad)

    def test_stage_backward_input_edge_requires_output_grads(self, device):
        mod = MLPModule(d_hid).to(device)
        x = torch.randn(batch_size, d_hid, device=device, requires_grad=True)
        edge = torch.autograd.graph.get_gradient_edge(mod(x))

        with self.assertRaisesRegex(AssertionError, "requires output gradients"):
            stage_backward_input(
                stage_outputs_or_loss=[edge],
                output_grads=None,
                input_values=[x],
                weights=mod.parameters(),
            )


instantiate_device_type_tests(StageBackwardTests, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
