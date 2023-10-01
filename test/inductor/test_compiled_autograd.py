# Owner(s): ["module: inductor"]
import functools
import re
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

# note: these tests are not run on windows due to inductor_utils.HAS_CPU


def compiler_fn(gm):
    """Same as torch.compile() but counts number of compiles"""

    def inner_compiler(gm_, example_inputs_):
        counters["compiled_autograd"]["compiles"] += 1
        return inductor.compile(gm_, example_inputs_)

    return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)


# TODO(jansel): hooks as lambdas creates recompiles in dynamo, we should fix that
def hook1(grad):
    return grad * 2


def hook2(grads):
    return (grads[0] + 1,)


def hook3(gI, gO):
    return (torch.sin(gI[0]) + gO[0],)


class TestCompiledAutograd(TestCase):
    def check_output_and_recompiles(self, fn, count=1):
        with torch.autograd.set_multithreading_enabled(False):
            torch._dynamo.reset()
            counters["compiled_autograd"].clear()
            torch.manual_seed(123)
            expected = list(fn())
            torch.manual_seed(123)
            with compiled_autograd.enable(compiler_fn):
                actual = list(fn())
            self.assertEqual(expected, actual)
            self.assertEqual(counters["compiled_autograd"]["captures"], count)
            self.assertEqual(counters["compiled_autograd"]["compiles"], count)

    def test_basic(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad

        self.check_output_and_recompiles(fn)

    def test_cache_hit(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])
                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook1(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([2, 4])

                model[0].weight.register_hook(hook1)

                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook2(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.grad_fn.register_prehook(hook2)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook3(self):
        def fn():
            for _ in range(3):
                model = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.grad_fn.register_hook(hook3)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad

        self.check_output_and_recompiles(fn)

    def test_torch_compile(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.Sigmoid(),
            )
            opt_model = torch.compile(model, fullgraph=True)

            for _ in range(3):
                x = torch.randn([1, 4])

                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                model.zero_grad()

        self.check_output_and_recompiles(fn)

    def test_implicit_add(self):
        def fn():
            y = torch.randn(1, 4, requires_grad=True)

            def model(x):
                # y is used multiple times, gradients get added
                return torch.sigmoid(x * y + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                result.backward()
                yield result
                yield y.grad
                y.grad = None

        self.check_output_and_recompiles(fn)

    def test_output_nodes(self):
        def fn():
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))

            for _ in range(3):
                x = torch.randn([1, 4])

                result = model(x).sum()
                gy, gz = torch.autograd.grad(result, [y, z])
                assert y.grad is None
                assert z.grad is None
                yield gy
                yield gz

        self.check_output_and_recompiles(fn)

    def test_dynamic_shapes(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for b in range(10, 100, 10):
                x = torch.randn([b, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()

        # TODO(jansel): we should be able to get this count to 1
        self.check_output_and_recompiles(fn, count=2)

    def test_accumulate_without_zero(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for _ in range(10):
                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad.clone()
                yield model[0].bias.grad.clone()
                yield model[2].weight.grad.clone()
                yield model[2].bias.grad.clone()

        self.check_output_and_recompiles(fn, count=2)

    def test_inplace_grad_update(self):
        def fn():
            model = torch.nn.Sequential(
                torch.nn.Linear(4, 4),
                torch.nn.ReLU(),
            )
            opt_model = torch.compile(model, dynamic=True)

            for _ in range(10):
                w_grad = torch.rand_like(model[0].weight)
                b_grad = torch.rand_like(model[0].bias)
                model[0].weight.grad = w_grad
                model[0].bias.grad = b_grad

                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                assert model[0].weight.grad is w_grad
                assert model[0].bias.grad is b_grad
                yield w_grad.clone()
                yield b_grad.clone()

        self.check_output_and_recompiles(fn, count=1)

    @unittest.skipIf(not HAS_CUDA, "requires cuda")
    def test_issue106555(self):
        DEVICE = torch.device("cuda:0")
        NUM_FEATURES = 256

        def bias_sigmoid_mul(x1, x2, bias):
            x2 = torch.sigmoid(x2 + bias)
            y = x1 * x2
            return y

        bias_sigmoid_mul_jit = torch.compile(bias_sigmoid_mul)

        class ModuleWithJit(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_1 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=True)
                self.linear_2 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=False)
                self.linear_2_bias = nn.Parameter(torch.zeros(NUM_FEATURES))

            def forward(self, input_tensor):
                x1 = self.linear_1(input_tensor)
                x2 = self.linear_2(input_tensor)
                output = bias_sigmoid_mul_jit(x1, x2, self.linear_2_bias)
                return output

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.module_with_jit_1 = ModuleWithJit()
                self.module_with_jit_2 = ModuleWithJit()

            def forward(self, x, gradient_checkpointing: bool):
                if gradient_checkpointing:
                    y = torch.utils.checkpoint.checkpoint(
                        self._forward, x, use_reentrant=True
                    )
                else:
                    y = self._forward(x)
                return y

            def _forward(self, x):
                x = x + self.module_with_jit_1(x)
                x = x + self.module_with_jit_2(x.transpose(-2, -3)).transpose(-2, -3)
                return x

        torch.cuda.set_device(device=DEVICE)
        torch.manual_seed(1234567890)
        model = Model()
        model.train()
        model.to(device=DEVICE)
        model_parameters = list(model.parameters())

        torch.manual_seed(1234567890)
        input_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(device=DEVICE)
        input_tensor.requires_grad = True
        target_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(
            dtype=input_tensor.dtype, device=DEVICE
        )

        for iteration in range(10):
            for param in model_parameters:
                param.grad = None
            output_tensor = model(
                x=input_tensor.clone(),
                gradient_checkpointing=True,
            )
            loss = torch.mean(torch.abs(target_tensor - output_tensor))
            loss.backward()

    def test_keep_graph_simple(self):
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2

        # First backward pass; keep the computation graph
        y.backward(retain_graph=True)
        self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4

        # Note - this will run under both the eager and compiled regime.
        def fn():
            # Reset the gradients
            x.grad = torch.tensor([0.0])
            # Second and Third backward pass; keep the computation graph
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            return x.grad

        self.check_output_and_recompiles(fn, count=1)

    def test_keep_graph_usage_after_compiled(self):
        x = torch.tensor([2.0], requires_grad=True)
        y = x**2

        # First backward pass; keep the computation graph
        def eager_check():
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))  # dy/dx at x=2 is 4
            x.grad = torch.tensor([0.0])

        eager_check()

        for i in range(0, 5):
            with compiled_autograd.enable(compiler_fn):
                eager_check()

            eager_check()


def load_test_module(name):
    testdir = Path(__file__).absolute().parent.parent
    with mock.patch("sys.path", [*sys.path, str(testdir)]):
        return SourceFileLoader(
            name, str(testdir / f"{name.replace('.', '/')}.py")
        ).load_module()


test_autograd = load_test_module("test_autograd")


class EagerAutogradTests(TestCase):
    @classmethod
    def add_test(cls, name, fn):
        @functools.wraps(fn)
        def wrapped(self: EagerAutogradTests):
            torch._dynamo.reset()
            with compiled_autograd.enable(compiler_fn):
                return fn(self)

        if not callable(fn):
            return
        elif known_failures_re.match(name) or name in known_failing_tests:
            setattr(cls, name, unittest.expectedFailure)
        elif name.startswith("test"):
            setattr(cls, name, wrapped)
        else:
            setattr(cls, name, fn)


# These groups of tests aren't supported yet
known_failures_re = re.compile(
    r"^test_(sparse|profiler|gradcheck|checkpoint|named_tensor)"
)

# Bugs needing investigation:
known_failing_tests = {
    "test_current_graph_task_execution_order",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function <
    "test_input_buffer_accum",  # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
    "test_graph_save_on_cpu_cuda",  # AssertionError: 0 not greater than 0
    "test_graph_save_on_cpu",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' raised:
    "test_reentrant_with_leaf_variable_hook",  # torch._dynamo.exc.Unsupported: inline in skipfiles: RemovableHandle.
    "test_reentrant_with_non_leaf_variable_hook",  # torch._dynamo.exc.Unsupported: inline in skipfiles: RemovableHan
    "test_saved_variable_saved_original_inplace_detach",  # AssertionError: RuntimeError not raised
    "test_saving_variable_to_disk",  # Cannot call numel() on tensor with symbolic sizes/strides
    "test_setitem_mask",  # torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: It appears that you're
    "test_tensor_hooks_inplace_over_view",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_tensor_hooks_inplace",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_wrapped_number_saved_variable_hooks",  # RuntimeError: this hook should not be called
    "test_accumulate_grad_posthooks_can_observe_tensor_prehook",  # data dependent operator: aten.allclose.default
    "test_accumulate_grad_tensor_reference",  # backend='inner_compiler' raised:
    "test_anomaly_detect_nan",  # type object 'MyFunc' has no attribute '_compiled_autograd_key'
    "test_anomaly_grad_warnings",  # "one of the variables needed for gradient computation has been modified by an...
    "test_autograd_inplace_views_cross_dtype",  # view_fn not supported by compiled autograd
    "test_autograd_multiple_views_python",  # type object 'ComplexView' has no attribute '_compiled_autograd_key'
    "test_autograd_node_isinstance",  # type object 'Func' has no attribute '_compiled_autograd_key'
    "test_autograd_python_custom_function_inplace",  # type object 'MyAdder' has no attribute '_compiled_autograd_key'
    "test_backward_with_inputs",  # specifying inputs= with .backward() not yet implemented for compiled autograd
    "test_callback_adds_callback",  # type object 'MyFunc' has no attribute '_compiled_autograd_key'
    "test_current_node",  # TorchDispatchMode not yet implemented for compiled autograd
    "test_custom_function_cycle",  # type object 'MyFn' has no attribute '_compiled_autograd_key'
    "test_custom_function_error",  # type object 'BadBw' has no attribute '_compiled_autograd_key'
    "test_custom_function_exception",  # "Simulate error on backward pass" does not match "type object 'SimulateBackwa...
    "test_custom_function_non_tensor_inputs_outputs",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_custom_function_save_for_forward",  # type object 'Func' has no attribute '_compiled_autograd_key'
    "test_custom_function_saved_tensors",  # type object 'MyFn' has no attribute '_compiled_autograd_key'
    "test_custom_function_setup_context_multi_input",  # type object 'MyReshape' has no attribute '_compiled_autograd_key'
    "test_custom_function_setup_context_multi_output",  # type object 'MySquare' has no attribute '_compiled_autograd_key'
    "test_custom_function_setup_context_simple",  # type object 'MySquare' has no attribute '_compiled_autograd_key'
    "test_deep_reentrant",  # type object 'DeepReentrant' has no attribute '_compiled_autograd_key'
    "test_dep_nograd",  # type object 'F2' has no attribute '_compiled_autograd_key'
    "test_dont_materialize_grads",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_function_returns_input",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_function_returns_undefined_tensor",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_grad_batched_grad",  # Cannot access storage of BatchedTensorImpl
    "test_grad_fn_prehooks",  # type object 'Mul2' has no attribute '_compiled_autograd_key'
    "test_grad_fn_prehooks_multiple_outputs",  # type object 'DoubleMul2' has no attribute '_compiled_autograd_key'
    "test_grad_fn_prehooks_remove_hooks",  # type object 'Mul2' has no attribute '_compiled_autograd_key'
    "test_grad_mode_restored_reentrant",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_grad_unreachable_discovery",  # specifying inputs= with .backward() not yet implemented for compiled autograd
    "test_hook_none",  # type object 'NoneGradientFunction' has no attribute '_compiled_autograd_key'
    "test_index_backward_does_not_save_tensor",  # dynamic shape operator: aten.nonzero.default
    "test_invalid_gradients",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_mark_non_differentiable_mixed",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_materialize_grads",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_naughty_anomaly_access",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_no_grad_copy",  # type object 'NonContGradFunc' has no attribute '_compiled_autograd_key'
    "test_post_accumulate_grad_hook_e2e",  # tensor_post_acc_grad_hooks not implemented for compiled autograd
    "test_post_accumulate_grad_hook_gets_cleaned_up",  # tensor_post_acc_grad_hooks not implemented for compiled autograd
    "test_post_accumulate_grad_hook_multiple_hooks",  # tensor_post_acc_grad_hooks not implemented for compiled autograd
    "test_post_accumulate_grad_hook_multiple_tensors",  # tensor_post_acc_grad_hooks not implemented for compiled autograd
    "test_post_accumulate_grad_hook_ordering",  # tensor_post_acc_grad_hooks not implemented for compiled autograd
    "test_post_accumulate_grad_hook_returns_not_None",  # "hooks should return None." does not match
    "test_reentrant_child_error",  # "Simulate error" does not match "type object 'ReentrantFunc' has no attribute...
    "test_reentrant_priority",  # type object 'Reentrant' has no attribute '_compiled_autograd_key'
    "test_reentrant_with_callbacks_both_depths",  # type object 'MyReentrantFunc' has no attribute '_compiled_autograd_key'
    "test_reentrant_with_callbacks_depth_0",  # type object 'MyReentrantFunc' has no attribute '_compiled_autograd_key'
    "test_reentrant_with_callbacks_depth_1",  # type object 'MyReentrantFunc' has no attribute '_compiled_autograd_key'
    "test_retain_grad_cycle",  # retains_grad_hooks not implemented for compiled autograd
    "test_retain_grad_inplace",  # retains_grad_hooks not implemented for compiled autograd
    "test_retain_grad_inplace_over_view",  # retains_grad_hooks not implemented for compiled autograd
    "test_retains_grad_can_always_observe_tensor_prehook",  # retains_grad_hooks not implemented for compiled autograd
    "test_retains_grad_inplace_multiple_outputs",  # retains_grad_hooks not implemented for compiled autograd
    "test_return_leaf",  # type object 'Identity' has no attribute '_compiled_autograd_key'
    "test_return_leaf_inplace",  # type object 'Inplace' has no attribute '_compiled_autograd_key'
    "test_save_none_for_backward",  # type object 'MyFn' has no attribute '_compiled_autograd_key'
    "test_save_output_nr",  # type object 'TestFn' has no attribute '_compiled_autograd_key'
    "test_saved_tensor_hooks_custom_function_intermediates",  # type object 'Func' has no attribute '_compiled_autograd_key'
    "test_saved_variables_deprecated",  # type object 'MyFunction' has no attribute '_compiled_autograd_key'
    "test_set_materialize_non_diff_grads",  # type object 'Func' has no attribute '_compiled_autograd_key'
    "test_setup_context_when_forward_has_default_args",  # type object 'PowFunction' has no attribute '_compiled_autograd_key'
    "test_simple_reentrant",  # type object 'Reenter' has no attribute '_compiled_autograd_key'
    "test_tensor_hooks_inplace_multiple_outputs",  # type object 'DoubleMul' has no attribute '_compiled_autograd_key'
    "test_to_sparse_backward",  # backend='inner_compiler' raised:
    "test_too_many_grads",  # type object 'MyFn' has no attribute '_compiled_autograd_key'
    "test_accumulate_grad",  # RuntimeError: compiled_autograd does not support create_graph
    "test_anomaly_assign_parent_cleanup",  # RuntimeError: compiled_autograd does not support create_graph
    "test_anomaly_mode_no_check_nan",  # RuntimeError: compiled_autograd does not support AnomalyMode
    "test_autograd_simple_views_python",  # AttributeError: type object 'IdOneOutput' has no attribute '_compiled_autograd_key'
    "test_backward_create_graph_warns",  # RuntimeError: compiled_autograd does not support create_graph
    "test_backward_with_nonleaf_inputs",  # RuntimeError: compiled_autograd does not support create_graph
    "test_create_graph_and_full_backward_hook_cycle",  # RuntimeError: compiled_autograd does not support create_graph
    "test_current_graph_task_id",  # torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor int
    "test_custom_autograd_no_early_free",  # AttributeError: type object 'Double' has no attribute '_compiled_autograd_key'
    "test_custom_autograd_repeated_grad_grad",  # RuntimeError: compiled_autograd does not support create_graph
    "test_custom_function_forward_mode_forward_is_no_op",  # AttributeError: type object 'MyFn'
    "test_custom_function_forward_mode_inplace_checks",  # AttributeError: type object 'InplaceFn'
    "test_custom_function_forward_mode_view_checks",  # AttributeError: type object 'ViewFn'
    "test_custom_function_forward_mode_wrong_formula",  # AttributeError: type object 'UserFn'
    "test_default_saved_variable_hooks_double_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_full_backward_hook_double_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_function",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_materialize_grads",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_nonleaf",  # RuntimeError: compiled_autograd does not support create_graph
    "test_grad_nonleaf_many_outputs",  # RuntimeError: compiled_autograd does not support create_graph
    "test_hessian_vector",  # RuntimeError: compiled_autograd does not support create_graph
    "test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_False",  # AttributeError: type object
    "test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_True",  # AttributeError: type object
    "test_hook_edge_case_when_called_with_grad",  # RuntimeError: specifying inputs= with .backward() not yet
    "test_hooks",  # torch._dynamo.exc.Unsupported: inline in skipfiles
    "test_inplace_on_view_backward",  # RuntimeError: compiled_autograd does not support create_graph
    "test_lobpcg",  # AttributeError: type object 'LOBPCGAutogradFunction' has no attribute '_compiled_autograd_key'
    "test_multi_grad_hooks",  # RuntimeError: specifying inputs= with .backward() not yet implemented for compiled autograd
    "test_naughty_autograd_function_stashing_ctx",  # AttributeError: type object 'Id' has no attribute '_compiled_autograd_key'
    "test_nested_anomaly_detect_nan",  # RuntimeError: compiled_autograd does not support create_graph
    "test_nested_anomaly_printstack_cleanup",  # RuntimeError: compiled_autograd does not support create_graph
    "test_no_grad_copy_sparse",  # AttributeError: type object 'MyFunc' has no attribute '_compiled_autograd_key'
    "test_once_differentiable",  # RuntimeError: compiled_autograd does not support create_graph
    "test_prehook_ordering",  # RuntimeError: specifying inputs= with .backward() not yet implemented for compiled autograd
    "test_retain_grad",  # RuntimeError: retains_grad_hooks not implemented for compiled autograd
    "test_return_duplicate",  # AttributeError: type object 'DoubleDuplicate' has no attribute '_compiled_autograd_key'
    "test_return_duplicate_inplace",  # AttributeError: type object 'DoubleInplace' has no attribute '_compiled_autograd_key'
    "test_saved_variable_packing_unpacking_saved_original_with_hooks",  # RuntimeError: compiled_autograd
    "test_select_sum",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_unrelated_inputs",  # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
    "test_will_engine_execute_node",  # RuntimeError: specifying inputs= with .backward() not yet implemented for compiled autograd
}

if not HAS_CUDA:
    # Found Tesla M60 which is too old to be supported by the triton GPU compiler
    known_failing_tests.add("test_type_conversions")

for name, fn in test_autograd.TestAutograd.__dict__.items():
    EagerAutogradTests.add_test(name, fn)


if __name__ == "__main__":
    if HAS_CPU:
        run_tests(needs="filelock")
