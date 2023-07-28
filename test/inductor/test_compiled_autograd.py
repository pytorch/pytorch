# Owner(s): ["module: inductor"]
import functools
import re
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

import torch
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
            try:
                with compiled_autograd.enable(compiler_fn):
                    return fn(self)
            except Exception as e:
                if not_implemented_re.search(str(e)):
                    raise unittest.SkipTest("not implemented")
                raise

        if skip_re.match(name) or name in skips or not callable(fn):
            return
        elif name.startswith("test"):
            setattr(cls, name, wrapped)
        else:
            setattr(cls, name, fn)


not_implemented_re = re.compile(
    r"|".join(
        map(
            re.escape,
            [
                # compiled autograd nyi errors:
                "compiled_autograd does not support",
                "not supported by compiled autograd",
                "not yet implemented for compiled autograd",
                "not implemented for compiled autograd",
                "has no attribute '_compiled_autograd_key'",
                # make_fx() tracing errors:
                "Cannot access storage of BatchedTensorImpl",
                "data dependent operator:",
            ],
        )
    )
)

# These groups of tests aren't supported yet
skip_re = re.compile(r"^test_(sparse|profiler|gradcheck|checkpoint|named_tensor)")

# Bugs needing investigation:
skips = {
    "test_accumulate_grad_tensor_reference",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' rai
    "test_autograd_inplace_views_cross_dtype",  # RuntimeError: compiled_args not implemented: torch::autograd::CopyS
    "test_calculate_shape_util",  # AssertionError: NYI: aten._nested_tensor_from_tensor_list.default
    "test_copy_slices_graph_task_updates",  # AssertionError: "Boom!" does not match "compiled_args not implemented:
    "test_current_graph_task_execution_order",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function <
    "test_current_node",  # RuntimeError: aten::detach() Expected a value of type 'Tensor' for argument 'self' but in
    "test_dont_materialize_grads",  # RuntimeError: compiled_args not implemented: torch::autograd::UndefinedGradBack
    "test_duplicate_backward_root",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/c
    "test_grad_fn_attr_bindings",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/csr
    "test_grad_unreachable_discovery",  # RuntimeError: tensor does not have a device
    "test_grad_unreachable",  # RuntimeError: tensor does not have a device
    "test_graph_save_on_cpu_cuda",  # AssertionError: 0 not greater than 0
    "test_graph_save_on_cpu",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/csrc/dy
    "test_hooks_cpp",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' raised:
    "test_index_backward_does_not_save_tensor",  # RuntimeError: expected int but got i0
    "test_inplace_on_view_weak_grad_fn",  # RuntimeError: compiled_args not implemented: torch::autograd::CopySlices
    "test_input_buffer_accum",  # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
    "test_integer_outputs",  # TypeError: unsupported operand type(s) for +: 'OpOverload' and 'str'
    "test_leaf_assignment",  # RuntimeError: compiled_args not implemented: torch::autograd::CopySlices
    "test_lobpcg",  # RuntimeError: tried to get Double out of SymFloat
    "test_materialize_grads",  # RuntimeError: compiled_args not implemented: torch::autograd::UndefinedGradBackward
    "test_no_unnecessary_save",  # RuntimeError: compiled_args not implemented: torch::autograd::CopyBackwards
    "test_no_unnecessary_unwrapping",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch
    "test_numpy_requires_grad",  # AssertionError: "Can't call numpy\(\) on Tensor that requires grad. Use tensor.det
    "test_pickle",  # TypeError: cannot pickle 'StorageWeakRef' object: a class that defines __slots__ without defini
    "test_reentrant_with_leaf_variable_hook",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytor
    "test_reentrant_with_non_leaf_variable_hook",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/p
    "test_saved_variable_packing_unpacking_saved_original_with_default_hooks",  # RuntimeError: inserted INTERNAL ASS
    "test_saved_variable_packing_unpacking_saved_original_with_hooks",  # RuntimeError: inserted INTERNAL ASSERT FAIL
    "test_saved_variable_saved_original_inplace_detach",  # AssertionError: RuntimeError not raised
    "test_saving_variable_to_disk",  # AttributeError: Can't pickle local object 'WeakValueDictionary.__init__.<local
    "test_setitem_mask",  # torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: It appears that you're
    "test_setitem",  # RuntimeError: compiled_args not implemented: torch::autograd::CopySlices
    "test_setting_default_saved_variable_hooks_twice_should_use_inner",  # RuntimeError: inserted INTERNAL ASSERT FAI
    "test_sharded_grad",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/csrc/dynamo/
    "test_tensor_hooks_inplace_over_view",  # RuntimeError: compiled_args not implemented: torch::autograd::CopySlic
    "test_tensor_hooks_inplace",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_to_sparse_backward",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' raised:
    "test_var_mean_differentiable",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/c
    "test_wrapped_number_saved_variable_hooks",  # RuntimeError: this hook should not be called
}

if not HAS_CUDA:
    # Found Tesla M60 which is too old to be supported by the triton GPU compiler
    skips.add("test_type_conversions")

for name, fn in test_autograd.TestAutograd.__dict__.items():
    EagerAutogradTests.add_test(name, fn)


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
            try:
                with compiled_autograd.enable(compiler_fn):
                    return fn(self)
            except Exception as e:
                if not_implemented_re.search(str(e)):
                    raise unittest.SkipTest("not implemented")
                raise

        if skip_re.match(name) or name in skips or not callable(fn):
            return
        elif name.startswith("test"):
            setattr(cls, name, wrapped)
        else:
            setattr(cls, name, fn)


not_implemented_re = re.compile(
    r"|".join(
        map(
            re.escape,
            [
                # compiled autograd nyi errors:
                "compiled_autograd does not support",
                "not supported by compiled autograd",
                "not yet implemented for compiled autograd",
                "not implemented for compiled autograd",
                "has no attribute '_compiled_autograd_key'",
                # make_fx() tracing errors:
                "Cannot access storage of BatchedTensorImpl",
                "data dependent operator:",
            ],
        )
    )
)

# These groups of tests aren't supported yet
skip_re = re.compile(r"^test_(sparse|profiler|gradcheck|checkpoint|named_tensor)")

# Bugs needing investigation:
skips = {
    "test_accumulate_grad_tensor_reference",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' rai
    "test_autograd_inplace_views_cross_dtype",  # RuntimeError: compiled_args not implemented: torch::autograd::CopyS
    "test_calculate_shape_util",  # AssertionError: NYI: aten._nested_tensor_from_tensor_list.default
    "test_copy_slices_graph_task_updates",  # AssertionError: "Boom!" does not match "compiled_args not implemented:
    "test_current_graph_task_execution_order",  # torch._dynamo.exc.TorchRuntimeError: Failed running call_function <
    "test_current_node",  # RuntimeError: aten::detach() Expected a value of type 'Tensor' for argument 'self' but in
    "test_dont_materialize_grads",  # RuntimeError: compiled_args not implemented: torch::autograd::UndefinedGradBack
    "test_duplicate_backward_root",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/c
    "test_grad_fn_attr_bindings",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/csr
    "test_grad_unreachable_discovery",  # RuntimeError: tensor does not have a device
    "test_grad_unreachable",  # RuntimeError: tensor does not have a device
    "test_graph_save_on_cpu_cuda",  # AssertionError: 0 not greater than 0
    "test_graph_save_on_cpu",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/csrc/dy
    "test_hooks_cpp",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' raised:
    "test_index_backward_does_not_save_tensor",  # RuntimeError: expected int but got i0
    "test_inplace_on_view_weak_grad_fn",  # RuntimeError: compiled_args not implemented: torch::autograd::CopySlices
    "test_input_buffer_accum",  # RuntimeError: Cannot access data pointer of Tensor that doesn't have storage
    "test_integer_outputs",  # TypeError: unsupported operand type(s) for +: 'OpOverload' and 'str'
    "test_leaf_assignment",  # RuntimeError: compiled_args not implemented: torch::autograd::CopySlices
    "test_lobpcg",  # RuntimeError: tried to get Double out of SymFloat
    "test_materialize_grads",  # RuntimeError: compiled_args not implemented: torch::autograd::UndefinedGradBackward
    "test_no_unnecessary_save",  # RuntimeError: compiled_args not implemented: torch::autograd::CopyBackwards
    "test_no_unnecessary_unwrapping",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch
    "test_numpy_requires_grad",  # AssertionError: "Can't call numpy\(\) on Tensor that requires grad. Use tensor.det
    "test_pickle",  # TypeError: cannot pickle 'StorageWeakRef' object: a class that defines __slots__ without defini
    "test_reentrant_with_leaf_variable_hook",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytor
    "test_reentrant_with_non_leaf_variable_hook",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/p
    "test_saved_variable_packing_unpacking_saved_original_with_default_hooks",  # RuntimeError: inserted INTERNAL ASS
    "test_saved_variable_packing_unpacking_saved_original_with_hooks",  # RuntimeError: inserted INTERNAL ASSERT FAIL
    "test_saved_variable_saved_original_inplace_detach",  # AssertionError: RuntimeError not raised
    "test_saving_variable_to_disk",  # AttributeError: Can't pickle local object 'WeakValueDictionary.__init__.<local
    "test_setitem_mask",  # torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: It appears that you're
    "test_setitem",  # RuntimeError: compiled_args not implemented: torch::autograd::CopySlices
    "test_setting_default_saved_variable_hooks_twice_should_use_inner",  # RuntimeError: inserted INTERNAL ASSERT FAI
    "test_sharded_grad",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/csrc/dynamo/
    "test_tensor_hooks_inplace_over_view",  # RuntimeError: compiled_args not implemented: torch::autograd::CopySlic
    "test_tensor_hooks_inplace",  # torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable() [] {}
    "test_to_sparse_backward",  # torch._dynamo.exc.BackendCompilerFailed: backend='inner_compiler' raised:
    "test_var_mean_differentiable",  # RuntimeError: inserted INTERNAL ASSERT FAILED at "/home/jansel/pytorch/torch/c
    "test_wrapped_number_saved_variable_hooks",  # RuntimeError: this hook should not be called
    "test_grad_nonleaf_register_hook",  # segfault
    "test_accumulate_grad_with_zero_numel_grad",  # aten.sym_size
    "test_isolated_node",  # aten.sym_size
}

if not HAS_CUDA:
    # Found Tesla M60 which is too old to be supported by the triton GPU compiler
    skips.add("test_type_conversions")

for name, fn in test_autograd.TestAutograd.__dict__.items():
    EagerAutogradTests.add_test(name, fn)

if __name__ == "__main__":
    if HAS_CPU:
        run_tests(needs="filelock")
