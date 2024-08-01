# Owner(s): ["module: nn"]
import pickle
from copy import deepcopy
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.parametrize as parametrize
from torch import Tensor
from torch.__future__ import get_swap_module_params_on_conversion
from torch.nn import Buffer, Parameter
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import (
    gradcheck,
    instantiate_parametrized_tests,
    run_tests,
    set_default_dtype,
    skipIfNoLapack,
    skipIfTorchDynamo,
    swap,
    TemporaryFileName,
)
from torch.testing._internal.two_tensor import TwoTensor


class TestNNParametrization(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    # torch/nn/utils/parametrize
    @skipIfNoLapack
    @swap([True, False])
    def test_register_and_remove_parametrization(self):
        r"""Test that it is possible to add a few parametrizations
        on a parameter or a buffer and that removing them restores the initial state
        It also tests that backpropagating through them works as expected
        """

        # Define a couple matrix parametrizations
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                # Cayley map
                # If X is skew-symmetric it returns an orthogonal matrix
                Id = torch.eye(X.size(0), device=X.device)
                # We call contiguous because solve returns a tensor with strides that are Fortran-contiguous
                # and autograd raises a performance warning.
                # This happens when we remove the parametrization with leave_parametrized=True,
                # which does a set_ with a non-contiguous tensor while the gradient is contiguous
                return torch.linalg.solve(Id + X, Id - X).contiguous()

        class Resize(nn.Module):
            def forward(self, X):
                return X[[0]]

        class NoResize(nn.Module):
            def forward(self, X):
                return X

        # Define a couple vector parametrizations
        class FirstZero(nn.Module):
            def forward(self, x):
                return torch.cat([x.new_zeros(1), x[1:]])

        class LastZero(nn.Module):
            def forward(self, x):
                return torch.cat([x[:-1], x.new_zeros(1)])

        model = nn.Linear(8, 8)
        initial_weight_id = id(model.weight)
        initial_bias_id = id(model.bias)
        initial_model = deepcopy(model)

        # Test unsafe flag
        with self.assertRaisesRegex(
            ValueError,
            "Registering a parametrization may not change the shape of the tensor",
        ):
            parametrize.register_parametrization(
                model, "weight", Resize()
            )  # default unsafe = False
            model(torch.ones(8, 8))

        # One parametrization with unsafe=True
        parametrize.register_parametrization(model, "weight", Resize(), unsafe=True)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        self.assertTrue(model.weight.shape[0] == 1)
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Two parametrizations with unsafe=True
        parametrize.register_parametrization(model, "weight", Resize(), unsafe=True)
        parametrize.register_parametrization(model, "weight", NoResize(), unsafe=False)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        self.assertTrue(model.weight.shape[0] == 1)
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test unsafe flag doesn't change expected behavior
        parametrize.register_parametrization(model, "weight", Skew(), unsafe=True)
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del A
        # Remove and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test one parametrization
        parametrize.register_parametrization(model, "weight", Skew())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del A
        # Remove and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertEqual(model.__class__, nn.Linear)

        # Test two parametrizations at the same time and removing them
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())
        # Result should be orthogonal
        X = model.weight
        Id = torch.eye(X.size(0), device=X.device)
        self.assertEqual(X.T @ X, Id)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del X
        # Structure tests
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertIn("weight", model.parametrizations)
        self.assertNotIn("weight", model._parameters)
        # Remove
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertEqual(model.weight, initial_model.weight)
        self.assertEqual(id(model.weight), initial_weight_id)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)

        # Add everything
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())
        parametrize.register_parametrization(model, "bias", FirstZero())
        parametrize.register_parametrization(model, "bias", LastZero())

        # Basic tests
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertTrue(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.0)
        self.assertEqual(model.bias[-1].item(), 0.0)
        self.assertEqual(
            len(list(model.parameters())), 2
        )  # Nothing weird has happpened
        # Should not throw

        sgd = torch.optim.SGD(model.parameters(), lr=0.01)

        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)

        # Remove first parametrization.
        # Check that the model is still parametrized and so is the second parameter
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertTrue(parametrize.is_parametrized(model))  # Still parametrized
        self.assertFalse(
            parametrize.is_parametrized(model, "weight")
        )  # Parametrization removed
        self.assertTrue(
            parametrize.is_parametrized(model, "bias")
        )  # Still parametrized
        self.assertEqual(model.bias[0].item(), 0.0)  # Still parametrized
        self.assertEqual(model.bias[-1].item(), 0.0)  # Still parametrized
        self.assertNotEqual(model.weight, initial_model.weight)  # Has been updated
        self.assertEqual(id(model.weight), initial_weight_id)  # Keeps the same id
        self.assertEqual(len(list(model.parameters())), 2)  # Nothing weird has happened
        # Should not throw
        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)

        # Remove the second parametrization.
        # Check that the module is not parametrized
        parametrize.remove_parametrizations(model, "bias", leave_parametrized=False)
        self.assertFalse(parametrize.is_parametrized(model))  # Not parametrized
        self.assertNotEqual(model.bias, initial_model.bias)  # Has been updated
        self.assertNotEqual(model.bias[0].item(), 0.0)  # Not parametrized
        self.assertNotEqual(model.bias[-1].item(), 0.0)  # Not parametrized
        self.assertEqual(id(model.bias), initial_bias_id)  # Keeps the same id
        self.assertFalse(
            hasattr(model, "parametrizations")
        )  # Not parametrized the module
        self.assertEqual(model.__class__, nn.Linear)  # Resores the previous class
        self.assertEqual(len(list(model.parameters())), 2)  # Nothing weird has happeed

        # Should not throw things are updated
        weight_copy = model.weight.clone()
        bias_copy = model.bias.clone()
        sgd.zero_grad()
        (model.weight.T @ model.bias).sum().backward()
        sgd.step()
        self.assertNotEqual(model.weight, weight_copy)
        self.assertNotEqual(model.bias, bias_copy)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del weight_copy, bias_copy

        # Test leave_parametrized=True
        for _ in range(2):
            parametrize.register_parametrization(model, "weight", Skew())
            parametrize.register_parametrization(model, "weight", Orthogonal())
            parametrize.remove_parametrizations(
                model, "weight", leave_parametrized=True
            )
            # We didn't change the dtype nor had multiple inputs, so the id should be the same
            self.assertEqual(id(model.weight), initial_weight_id)
            self.assertEqual(id(model.bias), initial_bias_id)

            # Should not throw. Things are updated
            weight_copy = model.weight.clone()
            bias_copy = model.bias.clone()
            sgd.zero_grad()
            (model.weight.T @ model.bias).sum().backward()
            sgd.step()
            self.assertNotEqual(model.weight, weight_copy)
            self.assertNotEqual(model.bias, bias_copy)
            if get_swap_module_params_on_conversion():
                # When using the swap_tensors path, this is needed so that the autograd
                # graph is not alive anymore.
                del weight_copy, bias_copy

    @swap([True, False])
    def test_register_and_remove_nested_parametrization(self):
        r"""Test that it is possible to nest the parametrizations
        meaning that the original param is parametrized again
        """

        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        model = nn.Linear(8, 8)
        # Add top level parametrization
        parametrize.register_parametrization(model, "weight", Skew())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be skew-symmetric
        A = model.weight
        self.assertEqual(A, -A.T)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del A

        # Add nested parametrization
        param_mod = model.parametrizations.weight
        self.assertFalse(hasattr(param_mod, "parametrizations"))
        self.assertFalse(parametrize.is_parametrized(param_mod))
        self.assertFalse(parametrize.is_parametrized(param_mod, "original"))

        parametrize.register_parametrization(param_mod, "original", Skew())
        self.assertTrue(hasattr(param_mod, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(param_mod))
        self.assertTrue(parametrize.is_parametrized(param_mod, "original"))
        self.assertNotIn("original", param_mod._parameters)
        # Result should be skew-symmetric
        A = param_mod.original
        self.assertEqual(A, -A.T)

        # Remove nested param and check consistency
        parametrize.remove_parametrizations(
            param_mod, "original", leave_parametrized=False
        )
        self.assertFalse(hasattr(param_mod, "parametrizations"))
        self.assertEqual(param_mod.__class__, parametrize.ParametrizationList)

        # Remove top level and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=False)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)

    @swap([True, False])
    def test_register_and_remove_buffer_parametrization(self):
        r"""Test that it is possible to add and remove parametrizations on buffers"""

        # Define a couple vector parametrizations
        class FirstZero(nn.Module):
            def forward(self, x):
                return torch.cat([x.new_zeros(1), x[1:]])

        class LastZero(nn.Module):
            def forward(self, x):
                return torch.cat([x[:-1], x.new_zeros(1)])

        model = nn.Linear(8, 8)

        # Instantiate parametrizations on buffers. It should work as expected
        delattr(model, "bias")
        model.bias = Buffer(torch.ones(8))
        parametrize.register_parametrization(model, "bias", FirstZero())
        parametrize.register_parametrization(model, "bias", LastZero())
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.0)
        self.assertEqual(model.bias[-1].item(), 0.0)
        self.assertTrue((model.bias[1:-1] == torch.ones(6)).all())
        self.assertEqual(len(list(model.parameters())), 1)

        # Remove parametrizations on buffers. It should work as expected
        parametrize.remove_parametrizations(model, "bias", leave_parametrized=True)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertEqual(model.bias[0].item(), 0.0)
        self.assertEqual(model.bias[-1].item(), 0.0)
        self.assertTrue((model.bias[1:-1] == torch.ones(6)).all())
        self.assertEqual(len(list(model.parameters())), 1)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_serialization_parametrization(self):
        r"""Test that it is possible to serialize a parametrized model via state_dict"""

        # A stateful parametrization
        class Orthogonal(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.id = Buffer(torch.eye(n))
                self.B = Buffer(torch.empty(n, n))
                init.orthogonal_(self.B)

            def forward(self, X):
                A = X.triu(1)
                A = A - A.T
                return self.B @ torch.linalg.solve(self.id + A, self.id - A)

        def get_model():
            model = torch.nn.Sequential(
                torch.nn.Linear(5, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1),
            )

            parametrize.register_parametrization(model[0], "weight", Orthogonal(5))
            return model

        model = get_model()

        prev_weight = model[0].weight
        prev_B = model[0].parametrizations.weight[0].B

        new_model = get_model()
        with TemporaryFileName() as fname:
            torch.save(model.state_dict(), fname)
            new_model.load_state_dict(torch.load(fname))

        # Integrity tests
        self.assertTrue(parametrize.is_parametrized(new_model[0], "weight"))
        self.assertEqual(prev_weight, new_model[0].weight)
        self.assertEqual(prev_B, new_model[0].parametrizations.weight[0].B)

        # Trying to save the whole parametrized model raises
        with self.assertRaisesRegex(RuntimeError, "state_dict"):
            with TemporaryFileName() as fname:
                torch.save(model, fname)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_initialization_parametrization(self):
        r"""Test that it is possible to initialize a parametrization when it
        implements a `right_inverse` method
        """

        class Skew(nn.Module):
            def forward(self, X):
                A = X.triu(1)
                return A - A.T

            def is_skew(self, A):
                return torch.allclose(A, -A.T, atol=1e-6)

            def right_inverse(self, X):
                if not self.is_skew(X):
                    raise ValueError("The matrix is not skew-symmetric.")
                return X.triu(1)

        # Implements a Cayley map where right_inverse is not quite the inverse of forward
        class Orthogonal(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.B = Buffer(torch.eye(n))

            def forward(self, X):
                Id = torch.eye(X.size(0))
                return self.B @ torch.linalg.solve(Id + X, Id - X)

            def is_orthogonal(self, X):
                Id = torch.eye(X.size(0))
                return torch.allclose(X.T @ X, Id, atol=1e-4)

            def right_inverse(self, X):
                if not self.is_orthogonal(X):
                    raise ValueError("The input is not orthogonal.")
                # cayley(0) == Id, so B @ cayley(0) == B
                self.B = X
                return torch.zeros_like(X)

        N = 5
        model = nn.Linear(N, N)
        # Register the skew-symmetric constraint. The result is now skew-symmetric
        skew = Skew()
        # Make the weight skew-symmetric before registering the parametrization
        with torch.no_grad():
            model.weight.set_(skew(model.weight))
        parametrize.register_parametrization(model, "weight", skew)
        X = torch.rand(N, N)
        # X is not skew-symmetric, so it throws an error
        with self.assertRaises(ValueError):
            model.weight = X
        # Make X skew-symmetric
        X = X - X.T
        model.weight = X
        self.assertEqual(model.parametrizations.weight.original, X.triu(1))
        self.assertEqual(model.weight, X)

        # Having several parametrizations registered should work in the same way
        parametrize.register_parametrization(model, "weight", Orthogonal(N))
        # Register now the Cayley map. The result is now orthogonal
        X = torch.rand(N, N)
        # X is not orthogonal, so it throws an error
        with self.assertRaises(ValueError):
            model.weight = X
        init.orthogonal_(X)
        model.weight = X
        self.assertEqual(model.weight, X)
        self.assertEqual(model.parametrizations.weight.original, torch.zeros_like(X))

    @swap([True, False])
    def test_errors_unparametrized_tensor_parametrization(self):
        # Test errors when registering a parametrization on an unparametrized tensor
        module = nn.Linear(3, 4)
        weight_init = module.weight.clone()

        class Identity(nn.Module):
            def forward(self, x):
                return x

        # Register a parametrization on a non-existing parameter throws
        with self.assertRaisesRegex(ValueError, "does not have a parameter"):
            parametrize.register_parametrization(module, "foo", Identity())
        self.assertFalse(parametrize.is_parametrized(module))

        # Removing parametrizations from an unparametrized tensor throws
        with self.assertRaisesRegex(ValueError, "does not have a parametrization"):
            parametrize.remove_parametrizations(module, "bias")
        self.assertFalse(parametrize.is_parametrized(module))

        # A correct parametrization with several outputs
        class Sum(nn.Module):
            def forward(self, x, y):
                return x + y

            def right_inverse(self, z):
                return z, torch.zeros_like(z)

        parametrize.register_parametrization(module, "weight", Sum())
        # Cannot remove a parametrization with several outputs with `leave_parametrized=False`
        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            parametrize.remove_parametrizations(
                module, "weight", leave_parametrized=False
            )
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        # A parametrization with an incorrect number of outputs
        class WrongNumberParams(nn.Module):
            def forward(self, x, y, z):
                return x + y + z

            def right_inverse(self, w):
                return w, torch.zeros_like(w)

        # Makes param(*param.right_inverse(X)) fail
        with self.assertRaisesRegex(TypeError, "positional argument"):
            parametrize.register_parametrization(module, "weight", WrongNumberParams())
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization with a right_inverse that does not return a Tensor or Sequence[Tensor]
        class WrongRightInverse(Identity):
            def right_inverse(self, z):
                return None

        # right_inverse should return a Tensor or a Sequence[Tensor]
        with self.assertRaisesRegex(ValueError, "Tensor or a Sequence of"):
            parametrize.register_parametrization(module, "weight", WrongRightInverse())
        self.assertFalse(parametrize.is_parametrized(module))

        # If it's a sequence, it must to be a sequence of tensors
        class WrongRightInverseSequence(nn.Module):
            def forward(self, x, y):
                return x

            def right_inverse(self, z):
                return None, z

        with self.assertRaisesRegex(ValueError, "of the sequence with type"):
            parametrize.register_parametrization(
                module, "weight", WrongRightInverseSequence()
            )
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization from one tensor to one tensor that changes the dtype
        class ChangeDtypeInverse(nn.Module):
            def forward(self, x):
                return x.float()

            def right_inverse(self, w):
                return w.bool()

        # For parametrizations that return one tensor, right_inverse may not change the dtype
        with self.assertRaisesRegex(
            ValueError, "outputs one tensor, it may not change the dtype"
        ):
            parametrize.register_parametrization(module, "weight", ChangeDtypeInverse())
        self.assertFalse(parametrize.is_parametrized(module))

        # Doesn't return a tensor
        class NotTensor(nn.Module):
            def forward(self, x):
                return 2

        # Forward must return a tensor
        with self.assertRaisesRegex(ValueError, "must return a tensor"):
            parametrize.register_parametrization(module, "weight", NotTensor())
        self.assertFalse(parametrize.is_parametrized(module))

        # A parametrization from one tensor to one tensor that changes the dtype
        class ChangeDtype(nn.Module):
            def forward(self, x):
                return x.bool()

        # forward should not change the initial dtype
        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtype())
        self.assertFalse(parametrize.is_parametrized(module))

        # Change shape
        class ChangeShape(nn.Module):
            def forward(self, x):
                return x[:-1]

        # forward should not change the original shape
        with self.assertRaisesRegex(ValueError, "may not change the shape"):
            parametrize.register_parametrization(module, "weight", ChangeShape())
        self.assertFalse(parametrize.is_parametrized(module))

        # Many to one that changes dtype
        class ChangeDtypeMulti(nn.Module):
            def forward(self, x, y):
                return (x + y).bool()

            def right_inverse(self, w):
                return w, w + 1

        # forward should not change the original shape even for parametrizations with many inputs
        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtypeMulti())
        self.assertFalse(parametrize.is_parametrized(module))

        # Returning a sequence of size one, although weird, it's correct
        class SequenceLen1(nn.Module):
            def forward(self, x):
                return x

            def right_inverse(self, w):
                return (w,)

        parametrize.register_parametrization(module, "weight", SequenceLen1())
        self.assertTrue(hasattr(module.parametrizations.weight, "original0"))
        self.assertFalse(hasattr(module.parametrizations.weight, "original1"))
        _ = module.weight  # Does not throw
        self.assertTrue(parametrize.is_parametrized(module))
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)

        # None of the operations above should have altered the weight
        self.assertFalse(parametrize.is_parametrized(module))
        self.assertEqual(module.weight, weight_init)

    @swap([True, False])
    def test_errors_parametrized_tensor_parametrization(self):
        # Test errors when registering a parametrization on a parametrized tensor

        class Identity(nn.Module):
            def forward(self, x):
                return x

        module = nn.Linear(3, 4)
        parametrize.register_parametrization(module, "weight", Identity())

        # Has to return a tensor
        class WrongReturn(nn.Module):
            def forward(self, x):
                return x, x

        with self.assertRaisesRegex(ValueError, "must return a tensor"):
            parametrize.register_parametrization(module, "weight", WrongReturn())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change dtype
        class ChangeDtype(nn.Module):
            def forward(self, x):
                return x.bool()

        with self.assertRaisesRegex(ValueError, "may not change the dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtype())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change shape
        class ChangeShape(nn.Module):
            def forward(self, x):
                return x[:-1]

        with self.assertRaisesRegex(ValueError, "may not change the shape"):
            parametrize.register_parametrization(module, "weight", ChangeShape())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # The following checks are mostly due to bugs in the code of the parametrization

        # right_inverse has to return a tensor
        class WrongReturnInverse(Identity):
            def right_inverse(self, x):
                return x, x

        with self.assertRaisesRegex(ValueError, "right_inverse must return a tensor"):
            parametrize.register_parametrization(module, "weight", WrongReturnInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change dtype
        class ChangeDtypeInverse(Identity):
            def right_inverse(self, x):
                return x.bool()

        with self.assertRaisesRegex(ValueError, "must have the same dtype"):
            parametrize.register_parametrization(module, "weight", ChangeDtypeInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

        # Cannot change shape
        class ChangeShapeInverse(Identity):
            def right_inverse(self, x):
                return x[:-1]

        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            parametrize.register_parametrization(module, "weight", ChangeShapeInverse())
        self.assertTrue(parametrize.is_parametrized(module))
        self.assertEqual(len(module.parametrizations.weight), 1)
        self.assertTrue(isinstance(module.parametrizations.weight[0], Identity))

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_multiple_inputs_parametrization(self):
        # A parametrization with several outputs
        class RankOne(nn.Module):
            def forward(self, x, y):
                # Form a rank-1 matrix from a pair of vectors
                return x.unsqueeze(-1) @ y.unsqueeze(-2)

            def right_inverse(self, Y):
                # We project the given matrix onto the rank 1 matrices
                U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                # S is ordered in a decreasing way.
                s0_sqrt = S[0].sqrt().unsqueeze(-1)
                return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt

        # Simple parametrisation
        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, w):
                return 0.5 * w

        model = nn.Linear(3, 3)
        # Test one parametrization
        parametrize.register_parametrization(model, "weight", RankOne())
        self.assertTrue(hasattr(model, "parametrizations"))
        self.assertTrue(parametrize.is_parametrized(model))
        self.assertTrue(parametrize.is_parametrized(model, "weight"))
        self.assertTrue(hasattr(model.parametrizations.weight, "original0"))
        self.assertIn("original0", model.parametrizations.weight._parameters)
        self.assertTrue(hasattr(model.parametrizations.weight, "original1"))
        self.assertIn("original1", model.parametrizations.weight._parameters)
        self.assertFalse(parametrize.is_parametrized(model, "bias"))
        self.assertNotIn("weight", model._parameters)
        # Result should be rank 1
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)

        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            # Cannot remove a parametrization with multiple inputs and not leave it parametrized
            parametrize.remove_parametrizations(
                model, "weight", leave_parametrized=False
            )
        # Remove parametrization and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=True)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)
        self.assertIn("weight", model._parameters)

        # Registering parametrizations with one input on top of one with multiple inputs should work
        init_weight = model.weight.clone()
        parametrize.register_parametrization(model, "weight", RankOne())
        # Projecting a rank 1 matrix onto the matrices of rank one does not change the matrix
        self.assertEqual(init_weight, model.weight)
        parametrize.register_parametrization(model, "weight", Double())
        # The matrix now is twice the initial matrix
        self.assertEqual(2.0 * init_weight, model.weight)
        # Multiplying by a scalar does not change the rank
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)

        # The model has now three parameters
        self.assertEqual(len(list(model.parameters())), 3)

        sgd = torch.optim.SGD(model.parameters(), lr=0.1)

        # Test backward. Should not throw
        for _ in range(2):
            sgd.zero_grad()
            loss = (model.weight.T @ model.bias).sum()
            loss.backward()
            sgd.step()

        # Same drill as before, removing should work as expected
        with self.assertRaisesRegex(ValueError, "leave_parametrized=False"):
            # Cannot remove a parametrization with multiple inputs and not leave it parametrized
            parametrize.remove_parametrizations(
                model, "weight", leave_parametrized=False
            )
        # Remove parametrization and check consistency
        parametrize.remove_parametrizations(model, "weight", leave_parametrized=True)
        self.assertFalse(hasattr(model, "parametrizations"))
        self.assertEqual(model.__class__, nn.Linear)
        self.assertFalse(parametrize.is_parametrized(model))
        self.assertEqual(torch.linalg.matrix_rank(model.weight).item(), 1)
        self.assertIn("weight", model._parameters)

        # The model has now two parameters
        self.assertEqual(len(list(model.parameters())), 2)

        # Test backward. Should not throw
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(2):
            sgd.zero_grad()
            loss = (model.weight.T @ model.bias).sum()
            loss.backward()
            sgd.step()

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_caching_parametrization(self):
        r"""Test the caching system of a parametrization"""

        # Define a couple matrix parametrizations
        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                Id = torch.eye(X.size(0), device=X.device)
                return torch.linalg.solve(Id + X, Id - X)

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())

        # Test that the caching system works
        with parametrize.cached():
            X = model.weight
            Y = model.weight
            self.assertEqual(id(X), id(Y))

    # FIXME: Rewrite this test using functions not depending on LAPACK
    #        and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_caching_parametrization_with_transfer_parametrizations_and_params(self):
        r"""Test that transferring parametrizations doesn't cause issues with caching"""

        class Skew(nn.Module):
            def forward(self, X):
                X = X.tril(-1)
                return X - X.T

        class Orthogonal(nn.Module):
            def forward(self, X):
                Id = torch.eye(X.size(0), device=X.device)
                return torch.linalg.solve(Id + X, Id - X)

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Skew())
        parametrize.register_parametrization(model, "weight", Orthogonal())

        to_model = nn.Linear(5, 5)
        parametrize.transfer_parametrizations_and_params(model, to_model)

        with parametrize.cached():
            X = model.weight
            Y = model.weight
            self.assertEqual(id(X), id(Y))

            A = to_model.weight
            B = to_model.weight
            self.assertEqual(id(A), id(B))

            # test that the results are distinct objects for each module
            self.assertNotEqual(id(A), id(X))

    @swap([True, False])
    def test_parametrization_same_training_mode(self):
        r"""Test training mode updated on parametrization registration"""

        class Identity(nn.Module):
            def forward(self, X):
                return X

        module = nn.Linear(4, 4)
        module.eval()
        parametrize.register_parametrization(module, "weight", Identity())
        self.assertFalse(module.parametrizations.weight[0].training)
        module.train()
        parametrize.register_parametrization(module, "weight", Identity().eval())
        self.assertTrue(module.parametrizations.weight[0].training)
        self.assertTrue(module.parametrizations.weight[1].training)

    @swap([True, False])
    def test_type_before_parametrizations(self):
        r"""Test that type_before_parametrizations always retrieves original type"""

        class Identity(nn.Module):
            def forward(self, X):
                return X

        model = nn.Linear(5, 5)
        original_type = type(model)
        self.assertTrue(
            parametrize.type_before_parametrizations(model) == original_type
        )
        parametrize.register_parametrization(model, "weight", Identity())
        self.assertTrue(
            parametrize.type_before_parametrizations(model) == original_type
        )

    @swap([True, False])
    def test_deepcopy_after_parametrization(self):
        r"""Test that we are able to create a deepcopy of the module when it's parametrized."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class ModelWithoutDeepcopy(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.tensor([1.0, 1.0, 1.0, 1.0]), requires_grad=True
                )
                self.bias = nn.Parameter(
                    torch.tensor([0.0, 0.0, 0.0, 0.0]), requires_grad=True
                )
                self.attr = [1.0, 2.0, 3.0, 4.0]

        class ActualModel(ModelWithoutDeepcopy):
            # Emulate custom implementation of the deepcopying.
            def __deepcopy__(self, memo):
                result = self.__new__(self.__class__)
                memo[id(self)] = result
                result.__dict__ = deepcopy(self.__dict__, memo)
                return result

        def check_deepcopy(m1: nn.Module, m2: nn.Module):
            w1 = m1.parametrizations.weight.original
            w2 = m2.parametrizations.weight.original
            b1 = (
                m1.parametrizations.bias.original
                if parametrize.is_parametrized(m1, "bias")
                else m1.bias
            )
            b2 = (
                m2.parametrizations.bias.original
                if parametrize.is_parametrized(m2, "bias")
                else m2.bias
            )
            # Weights, biases and attributes should be equal but they must be different objects.
            self.assertEqual(m1.__dict__.keys(), m2.__dict__.keys())
            self.assertIsNot(m1, m2)
            self.assertEqual(w1, w2)
            self.assertIsNot(w1, w2)
            self.assertEqual(b1, b2)
            self.assertIsNot(b1, b2)
            self.assertEqual(m1.attr, m2.attr)
            self.assertIsNot(m1.attr, m2.attr)

        for model in (ModelWithoutDeepcopy(), ActualModel()):
            # General check that we are able to create deepcopy.
            parametrize.register_parametrization(model, "weight", AddOne())
            check_deepcopy(model, deepcopy(model))
            # Check that this works on models with several parametrized tensors.
            parametrize.register_parametrization(model, "bias", AddOne())
            check_deepcopy(model, deepcopy(model))
            # Check that this works on models where tensors have more than one parametrization.
            parametrize.register_parametrization(model, "weight", AddOne())
            check_deepcopy(model, deepcopy(model))

    @swap([True, False])
    def test_transfer_parametrizations_and_params(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, x):
                return 0.5 * x

        class MinusOne(nn.Module):
            def forward(self, x):
                return x - 1.0

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", AddOne())
        parametrize.register_parametrization(model, "weight", Double())
        parametrize.register_parametrization(model, "weight", MinusOne())
        hold_weight = model.weight

        to_model = torch.ao.nn.qat.Linear(
            5, 5, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # checks that final and original value are correct and the to_model is parametrized
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )

        # check that the transfer didn't affect the original value
        self.assertEqual(hold_weight, model.weight)
        if get_swap_module_params_on_conversion():
            # When using the swap_tensors path, this is needed so that the autograd
            # graph is not alive anymore.
            del hold_weight

        # testing that changes to one set of parametrizations do not affect the other
        parametrize.remove_parametrizations(to_model, "weight")
        self.assertFalse(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(model, "weight"))

        # also test that parameters that don't exist in to_model get transferred
        model.test_param = Parameter(torch.randn(5, 5))

        self.assertTrue(not hasattr(to_model, "test_param"))
        parametrize.register_parametrization(model, "test_param", Double())
        hold_test_param = model.test_param
        parametrize.transfer_parametrizations_and_params(model, to_model, "test_param")

        # check that previously missing params got transferred correctly
        self.assertEqual(model.test_param, to_model.test_param)
        self.assertEqual(
            model.parametrizations.test_param.original,
            to_model.parametrizations.test_param.original,
        )

        # check that the new transfer didn't change the value for the from_module
        self.assertEqual(hold_test_param, model.test_param)

    @swap([True, False])
    def test_transfer_parametrizations_and_params_right_inverse(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

            def right_inverse(self, x):
                return 0.5 * x

        model = nn.Linear(5, 5)
        parametrize.register_parametrization(model, "weight", Double())
        hold_weight = model.weight

        to_model = torch.ao.nn.qat.Linear(
            5, 5, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model)

        # check that transfer occurs successfully
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )

        # check that transfer doesn't affect the from_model weight
        self.assertEqual(hold_weight, model.weight)

    @swap([True, False])
    def test_transfer_parametrizations_and_params_single_param(self):
        r"""Test that all parametrizations and their associated parameters are transferred."""

        class AddOne(nn.Module):
            def forward(self, x):
                return x + 1.0

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

        class MinusOne(nn.Module):
            def forward(self, x):
                return x - 1.0

        model = nn.Linear(5, 5, bias=True)
        parametrize.register_parametrization(model, "weight", AddOne())
        parametrize.register_parametrization(model, "weight", Double())
        parametrize.register_parametrization(model, "weight", MinusOne())
        parametrize.register_parametrization(model, "bias", AddOne())
        parametrize.register_parametrization(model, "bias", Double())
        parametrize.register_parametrization(model, "bias", MinusOne())

        to_model = torch.ao.nn.qat.Linear(
            5, 5, bias=True, qconfig=torch.ao.quantization.get_default_qconfig()
        )
        parametrize.transfer_parametrizations_and_params(model, to_model, "weight")

        # check that weight and only weight was transferred
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original,
            to_model.parametrizations.weight.original,
        )
        self.assertTrue("bias" not in to_model.parametrizations)

    # FIXME: Rewrite this test using functions not depending on LAPACK
    # and remove the `@skipIfNoLapack` (see #70995)
    @skipIfNoLapack
    @swap([True, False])
    def test_transfer_parametrizations_and_params_many_to_one(self):
        # A parametrization with several outputs
        class RankOne(nn.Module):
            def forward(self, x, y):
                # Form a rank-1 matrix from a pair of vectors
                return x.unsqueeze(-1) @ y.unsqueeze(-2)

            def right_inverse(self, Y):
                # We project the given matrix onto the rank 1 matrices
                U, S, Vh = torch.linalg.svd(Y, full_matrices=False)
                # S is ordered in a decreasing way.
                s0_sqrt = S[0].sqrt().unsqueeze(-1)
                return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt

        class Double(nn.Module):
            def forward(self, x):
                return 2.0 * x

        model = nn.Linear(3, 3)
        parametrize.register_parametrization(model, "weight", RankOne())
        parametrize.register_parametrization(model, "weight", Double())
        hold_weight = model.weight

        to_model = torch.ao.nn.qat.Linear(
            3, 3, qconfig=torch.ao.quantization.get_default_qconfig()
        )

        parametrize.transfer_parametrizations_and_params(model, to_model)

        # checks that final and original value are correct and the to_model is parametrized
        self.assertTrue(torch.nn.utils.parametrize.is_parametrized(to_model, "weight"))
        self.assertEqual(model.weight, to_model.weight)
        self.assertEqual(
            model.parametrizations.weight.original0,
            to_model.parametrizations.weight.original0,
        )
        self.assertEqual(
            model.parametrizations.weight.original1,
            to_model.parametrizations.weight.original1,
        )

        # check that the transfer didn't affect the original value
        self.assertEqual(hold_weight, model.weight)

        # testing that changes to one set of parametrizations do not affect the other
        model.test_param = Parameter(torch.randn(3, 3))

        self.assertTrue(not hasattr(to_model, "test_param"))
        parametrize.register_parametrization(model, "test_param", RankOne())
        hold_test_param = model.test_param
        parametrize.transfer_parametrizations_and_params(model, to_model, "test_param")

        # also check that previously missing params got transferred correctly
        self.assertEqual(model.test_param, to_model.test_param)
        self.assertEqual(
            model.parametrizations.test_param.original0,
            to_model.parametrizations.test_param.original0,
        )
        self.assertEqual(
            model.parametrizations.test_param.original1,
            to_model.parametrizations.test_param.original1,
        )

        # check that the new transfer didn't change the value for the from_module
        self.assertEqual(hold_test_param, model.test_param)

    @swap([True, False])
    def test_new_spectral_norm(self):
        with set_default_dtype(torch.double):
            input = torch.randn(3, 5)
            m = nn.Linear(5, 7)
            m = torch.nn.utils.parametrizations.spectral_norm(m)
            spectral_norm_m = m.parametrizations.weight[0]

            self.assertEqual(spectral_norm_m._u.size(), torch.Size([m.weight.size(0)]))

            # .parametrizations.weight.original should be trainable
            self.assertTrue(hasattr(m.parametrizations.weight, "original"))
            self.assertTrue("original" in m.parametrizations.weight._parameters)

            # u should be just a reused buffer
            self.assertTrue(hasattr(spectral_norm_m, "_u"))
            self.assertTrue("_u" in spectral_norm_m._buffers)
            self.assertTrue("_v" in spectral_norm_m._buffers)

            # weight should be a plain attribute, not counted as a buffer or a param
            self.assertIsNotNone(m.weight)
            self.assertFalse("weight" in m._buffers)
            self.assertFalse("weight" in m._parameters)

            # it should also be sharing storage as `weight_orig`
            # self.assertEqual(m.parametrizations.weight.original.storage(), m.weight.storage())
            self.assertEqual(m.parametrizations.weight.original.size(), m.weight.size())
            self.assertEqual(
                m.parametrizations.weight.original.stride(), m.weight.stride()
            )

            m = torch.nn.utils.parametrize.remove_parametrizations(m, "weight")

            # spectral_norm is the only parametrization
            self.assertFalse(hasattr(m, "parametrizations"))
            self.assertTrue("weight" in m._parameters)

            # We can register spectral_norm multiple times on the same parameter
            # and on multiple parameters in the same module
            m = torch.nn.utils.parametrizations.spectral_norm(m, "weight")
            m = torch.nn.utils.parametrizations.spectral_norm(m, "weight")
            m = torch.nn.utils.parametrizations.spectral_norm(m, "bias")

            # If we remove the parametrization on bias, weight is still parametrized
            # Removing a parametrization runs forward in eval mode if leave_parametrized=True
            m = torch.nn.utils.parametrize.remove_parametrizations(m, "bias")
            self.assertTrue("bias" in m._parameters)
            self.assertTrue(hasattr(m, "parametrizations"))
            self.assertFalse("weight" in m._parameters)

            m = torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
            # Neither weight and bias are parametrized
            self.assertFalse(hasattr(m, "parametrizations"))
            self.assertTrue("weight" in m._parameters)
            self.assertFalse(torch.nn.utils.parametrize.is_parametrized(m))

            # test correctness in training/eval modes and cpu/multi-gpu settings
            for apply_dp in (True, False):
                if apply_dp:
                    if not TEST_MULTIGPU:
                        continue
                    device = torch.device("cuda:0")

                    def maybe_wrap(m):
                        return torch.nn.DataParallel(m, [0, 1])

                else:
                    device = torch.device("cpu")

                    def maybe_wrap(m):
                        return m

                for requires_grad in (True, False):

                    def get_modules():
                        m = nn.Linear(3, 4).to(device)
                        m.weight.requires_grad_(requires_grad)
                        m = torch.nn.utils.parametrizations.spectral_norm(m)
                        wrapped_m = maybe_wrap(m)
                        spectral_norm_m = m.parametrizations.weight[0]
                        return m, wrapped_m, spectral_norm_m

                    input = torch.randn(2, 3, device=device)

                    m, wrapped_m, spectral_norm_m = get_modules()

                    self.assertTrue(hasattr(spectral_norm_m, "_u"))
                    u0 = spectral_norm_m._u.clone()
                    v0 = spectral_norm_m._v.clone()

                    # TEST TRAINING BEHAVIOR

                    # We perform GD first to modify the initial matrix
                    opt = torch.optim.SGD(wrapped_m.parameters(), lr=0.1)

                    opt.zero_grad()
                    wrapped_m(input).sum().backward()
                    opt.step()

                    out = wrapped_m(input)
                    if requires_grad:
                        # run forward again and assert that u and v are updated
                        self.assertNotEqual(u0, spectral_norm_m._u)
                        self.assertNotEqual(v0, spectral_norm_m._v)

                    # assert that backprop reaches original weight
                    # can't use gradcheck because the function changes as we
                    # activate through it in training mode
                    if requires_grad:
                        torch.autograd.grad(
                            out.sum(), m.parametrizations.weight.original
                        )

                    # test backward works with multiple forwards
                    # it uses training mode so we need to reset `u` and `v` vectors
                    # to same value at beginning for finite difference test to pass
                    saved_u = spectral_norm_m._u.clone()
                    saved_v = spectral_norm_m._v.clone()

                    def fn(input):
                        spectral_norm_m._u.data.copy_(saved_u)
                        spectral_norm_m._v.data.copy_(saved_v)
                        out0 = wrapped_m(input)
                        out1 = wrapped_m(input)
                        return out0 + out1

                    # Make sure we can compute gradients wrt to all the parameters in the case
                    # of double forward
                    fn(input.clone().requires_grad_()).sum().backward()
                    gradcheck(
                        fn, (input.clone().requires_grad_(),), check_batched_grad=False
                    )

                    # test removing
                    # spectral norm module needs to be in eval mode if we'd like to
                    # avoid doing another power iteration
                    m, wrapped_m, _ = get_modules()
                    pre_remove_out = wrapped_m(input)
                    if get_swap_module_params_on_conversion():
                        # When using the swap_tensors path, this is needed so that the autograd
                        # graph is not alive anymore.
                        pre_remove_out_ref = pre_remove_out.detach()
                        del pre_remove_out
                    else:
                        pre_remove_out_ref = pre_remove_out
                    m.eval()
                    m = torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
                    self.assertEqual(wrapped_m(input), pre_remove_out_ref)

                    torch.nn.utils.parametrizations.spectral_norm(m)
                    for _ in range(3):
                        pre_remove_out = wrapped_m(input)
                    if get_swap_module_params_on_conversion():
                        # When using the swap_tensors path, this is needed so that the autograd
                        # graph is not alive anymore.
                        pre_remove_out_ref = pre_remove_out.detach()
                        del pre_remove_out
                    else:
                        pre_remove_out_ref = pre_remove_out
                    m.eval()
                    m = torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
                    self.assertEqual(wrapped_m(input), pre_remove_out_ref)

                    # TEST EVAL BEHAVIOR
                    m, wrapped_m, spectral_norm_m = get_modules()
                    wrapped_m(input)
                    last_train_out = wrapped_m(input)
                    last_train_u = spectral_norm_m._u.clone()
                    last_train_v = spectral_norm_m._v.clone()
                    wrapped_m.zero_grad()
                    wrapped_m.eval()

                    eval_out0 = wrapped_m(input)
                    # assert eval gives same result as last training iteration
                    self.assertEqual(eval_out0, last_train_out)
                    # assert doing more iteartion in eval don't change things
                    self.assertEqual(eval_out0, wrapped_m(input))
                    self.assertEqual(last_train_u, spectral_norm_m._u)
                    self.assertEqual(last_train_v, spectral_norm_m._v)

                    # FIXME: the code below is flaky when executed with DataParallel
                    # see https://github.com/pytorch/pytorch/issues/13818
                    if apply_dp:
                        continue

                    # test backward works with multiple forwards in mixed training
                    # and eval modes
                    # it uses training mode so we need to reset `u` and `v` vectors
                    # to same value at beginning for finite difference test to pass
                    saved_u = spectral_norm_m._u.clone()
                    saved_v = spectral_norm_m._v.clone()

                    def fn(input):
                        spectral_norm_m._u.data.copy_(saved_u)
                        spectral_norm_m._v.data.copy_(saved_v)
                        wrapped_m.train()
                        out0 = wrapped_m(input)
                        wrapped_m.eval()
                        out1 = wrapped_m(input)
                        wrapped_m.train()
                        out2 = wrapped_m(input)
                        wrapped_m.eval()
                        out3 = wrapped_m(input)
                        return out0 + out1 + out2 + out3

                    gradcheck(fn, (input.clone().requires_grad_(),))

                    # assert that backprop reaches weight_orig in eval
                    if requires_grad:

                        def fn(weight):
                            return wrapped_m(input)

                        gradcheck(fn, (m.parametrizations.weight.original,))

    def test_register_parametrization_no_grad(self):
        r"""Test that it is possible to register a parametrization without gradient"""

        class SplitAndCat(nn.Module):
            def right_inverse(self, x):
                # split the tensor in two halfs
                return torch.split(x, x.shape[1] // 2)

            def forward(self, x0, x1):
                return torch.cat([x0, x1])

        model = nn.Linear(8, 8)

        model.weight.requires_grad = False
        parametrize.register_parametrization(model, "weight", SplitAndCat())
        # making sure the parameterized and decomposed Tensors both have requires_grad == False
        self.assertFalse(model.weight.requires_grad)
        self.assertFalse(model.parametrizations.weight.original0.requires_grad)
        self.assertFalse(model.parametrizations.weight.original1.requires_grad)

    @swap([True, False])
    def test_new_spectral_norm_load_state_dict(self):
        for activate_times in (0, 3):
            inp = torch.randn(2, 3)
            m = nn.Linear(3, 5)
            snm = torch.nn.utils.parametrizations.spectral_norm(m)
            snm.train()

            for _ in range(activate_times):
                snm(inp)

            state_dict = deepcopy(snm.state_dict())
            self.assertEqual(
                {
                    "parametrizations.weight.original",
                    "bias",
                    "parametrizations.weight.0._v",
                    "parametrizations.weight.0._u",
                },
                set(state_dict.keys()),
            )

            # test that non-strict loading works
            non_strict_state_dict = deepcopy(state_dict)
            non_strict_state_dict["nonsense"] = "nonsense"
            with self.assertRaisesRegex(
                RuntimeError, r'Unexpected key\(s\) in state_dict: "nonsense"'
            ):
                snm.load_state_dict(non_strict_state_dict, strict=True)
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict["parametrizations.weight.original"]
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict["parametrizations.weight.0._u"]
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict["parametrizations.weight.0._v"]
            snm.load_state_dict(non_strict_state_dict, strict=False)
            non_strict_state_dict[
                "weight"
            ] = snm.weight.detach().clone()  # set W as a buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict._metadata[
                "parametrizations.weight.0"
            ]  # remove metadata info
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict["weight"]  # remove W buffer
            snm.load_state_dict(non_strict_state_dict, strict=False)
            del non_strict_state_dict["bias"]
            snm.load_state_dict(non_strict_state_dict, strict=False)

            # normal state_dict

            # test that re-wrapping does not matter
            m = torch.nn.utils.parametrize.remove_parametrizations(snm, "weight")
            snm = torch.nn.utils.parametrizations.spectral_norm(m)

            snm.load_state_dict(state_dict)
            with torch.no_grad():
                snm.eval()
                out0_eval = snm(inp)
                snm.train()
                out1_train = snm(inp)
                out2_train = snm(inp)
                snm.eval()
                out3_eval = snm(inp)

            # test that re-wrapping does not matter
            m = torch.nn.utils.parametrize.remove_parametrizations(snm, "weight")
            snm = torch.nn.utils.parametrizations.spectral_norm(m)

            # Test normal loading
            snm.load_state_dict(state_dict)
            with torch.no_grad():
                snm.eval()
                self.assertEqual(out0_eval, snm(inp))
                snm.train()
                self.assertEqual(out1_train, snm(inp))
                self.assertEqual(out2_train, snm(inp))
                snm.eval()
                self.assertEqual(out3_eval, snm(inp))

    @swap([True, False])
    def test_new_spectral_norm_dim(self):
        inp = torch.randn(2, 3, 10, 12)
        m = nn.ConvTranspose2d(3, 4, (5, 6))
        m = torch.nn.utils.parametrizations.spectral_norm(m)
        snm = m.parametrizations.weight[0]
        # this should not run into incompatible shapes
        x = m(inp)
        # check that u refers to the same dimension
        self.assertEqual(
            snm._u.shape, m.parametrizations.weight.original[0, :, 0, 0].shape
        )

    @swap([True, False])
    def test_new_spectral_norm_forward(self):
        input = torch.randn(3, 5)
        m = nn.Linear(5, 7)
        m = torch.nn.utils.parametrizations.spectral_norm(m)
        snm = m.parametrizations.weight[0]
        # naive forward
        _weight = m.parametrizations.weight.original
        _bias, _v = m.bias, snm._v
        _weight_mat = _weight.view(_weight.size(0), -1)
        _u = torch.mv(_weight_mat, _v)
        _u = F.normalize(_u, dim=0, eps=1e-12)
        _v = torch.mv(_weight_mat.t(), _u)
        _v = F.normalize(_v, dim=0, eps=1e-12)
        _weight.data /= torch.dot(_u, torch.matmul(_weight_mat, _v))
        out_hat = torch.nn.functional.linear(input, _weight, _bias)
        expect_out = m(input)
        self.assertEqual(expect_out, out_hat)

    @swap([True, False])
    @skipIfTorchDynamo("Test does not work with TorchDynamo")
    def test_new_spectral_norm_value(self):
        # a test that the spectral norm (= top singular value)
        # is in fact properly calculated, using example of a simple diagonal matrix.
        for dtype in (torch.float, torch.cfloat):
            m = nn.Linear(2, 2, dtype=dtype)
            with torch.no_grad():
                # set weight to be diagonal
                x = torch.diagonal(m.weight)
                m.weight = nn.Parameter(torch.diag(x))
                torch.nn.utils.parametrizations.spectral_norm(m)
                # weights should be rescaled by spectral norm, (i.e., largest diagonal element in norm)
                expected = torch.diag(x / x.abs().max())
                self.assertEqual(m.weight.data, expected)

    @skipIfNoLapack
    @swap([True, False])
    def test_orthogonal_parametrization(self):
        # Orthogonal implements 6 algorithms (3x parametrizations times 2 options of use_trivialization)

        def assert_is_orthogonal(X):
            n, k = X.size(-2), X.size(-1)
            if n < k:
                X = X.mT
                n, k = k, n
            Id = torch.eye(k, dtype=X.dtype, device=X.device).expand(
                *(X.size()[:-2]), k, k
            )
            eps = 10 * n * torch.finfo(X.dtype).eps
            torch.testing.assert_close(X.mH @ X, Id, atol=eps, rtol=0.0)

        def assert_weight_allclose_Q(weight, W):
            # Test that weight is equal to the Q part of the QR decomposition of W
            # (or of its transpose if the matrix is wide)
            wide_matrix = W.size(-2) < W.size(-1)
            if wide_matrix:
                W = W.mT
            Q, R = torch.linalg.qr(W)
            Q *= R.diagonal(dim1=-2, dim2=-1).sgn().unsqueeze(-2)
            if wide_matrix:
                Q = Q.mT
            torch.testing.assert_close(Q, weight, atol=1e-5, rtol=0.0)

        for shape, dtype, use_linear in product(
            ((4, 4), (5, 3), (3, 5)),  # square/ tall / wide
            (torch.float32, torch.complex64),
            (True, False),
        ):
            # Conv2d does not support complex yet
            if not use_linear:
                continue

            if use_linear:
                input = torch.randn(3, shape[0], dtype=dtype)
            else:
                input = torch.randn(2, 2, shape[0] + 2, shape[1] + 1, dtype=dtype)

            for parametrization, use_trivialization in product(
                ("matrix_exp", "cayley", "householder"), (False, True)
            ):
                # right_inverse for Cayley and matrix_exp not implemented for use_trivialization=False
                # See Note [right_inverse expm cayley]
                can_initialize = use_trivialization or parametrization == "householder"

                # We generate them every time to always start with fresh weights
                if use_linear:
                    m = nn.Linear(*shape, dtype=dtype)
                else:
                    m = nn.Conv2d(2, 3, shape, dtype=dtype)

                # We do not support householder for complex inputs
                # See Note [Householder complex]

                # When using the swap_tensors path, this is needed so that the autograd
                # graph is not alive anymore.
                if get_swap_module_params_on_conversion():
                    w_init = m.weight.clone().detach()
                else:
                    w_init = m.weight.clone()
                if parametrization == "householder" and m.weight.is_complex():
                    msg = "householder parametrization does not support complex tensors"
                    with self.assertRaisesRegex(ValueError, msg):
                        torch.nn.utils.parametrizations.orthogonal(
                            m,
                            "weight",
                            parametrization,
                            use_trivialization=use_trivialization,
                        )
                    continue

                wide_matrix = w_init.size(-2) < w_init.size(-1)
                torch.nn.utils.parametrizations.orthogonal(
                    m, "weight", parametrization, use_trivialization=use_trivialization
                )
                # Forwards works as expected
                self.assertEqual(w_init.shape, m.weight.shape)
                assert_is_orthogonal(m.weight)
                if can_initialize:
                    assert_weight_allclose_Q(m.weight, w_init)

                # Intializing with a given orthogonal matrix works
                X = torch.randn_like(m.weight)
                if wide_matrix:
                    X = X.mT
                w_new = torch.linalg.qr(X).Q
                if wide_matrix:
                    w_new = w_new.mT
                if can_initialize:
                    m.weight = w_new
                    torch.testing.assert_close(w_new, m.weight, atol=1e-5, rtol=0.0)
                else:
                    msg = (
                        "assign to the matrix exponential or the Cayley parametrization"
                    )
                    with self.assertRaisesRegex(NotImplementedError, msg):
                        m.weight = w_new

                # Intializing with a non-orthogonal matrix makes m.weight be the Q part of the given matrix
                w_new = torch.randn_like(m.weight)
                if can_initialize:
                    m.weight = w_new
                    assert_weight_allclose_Q(m.weight, w_new)
                else:
                    msg = (
                        "assign to the matrix exponential or the Cayley parametrization"
                    )
                    with self.assertRaisesRegex(NotImplementedError, msg):
                        m.weight = w_new

                opt = torch.optim.SGD(m.parameters(), lr=0.1)
                for _ in range(2):
                    opt.zero_grad()
                    m(input).norm().backward()
                    grad = m.parametrizations.weight.original.grad
                    self.assertIsNotNone(grad)
                    # We do not update the upper triangular part of the matrix if tall tril if wide
                    if grad.size(-2) >= grad.size(-1):
                        zeros_grad = grad.triu(1)
                    else:
                        zeros_grad = grad.tril(-1)
                    self.assertEqual(zeros_grad, torch.zeros_like(zeros_grad))
                    # The gradient in the diagonal can only be imaginary because a skew-Hermitian
                    # matrix has imaginary diagonal
                    diag_grad = grad.diagonal(dim1=-2, dim2=-1)
                    if grad.is_complex():
                        diag_grad = diag_grad.real
                    self.assertEqual(diag_grad, torch.zeros_like(diag_grad))
                    opt.step()
                    assert_is_orthogonal(m.weight)

    @skipIfNoLapack
    @swap([True, False])
    def test_orthogonal_errors(self):
        m = nn.Linear(3, 4)
        with self.assertRaisesRegex(ValueError, "has to be one of"):
            torch.nn.utils.parametrizations.orthogonal(m, "weight", "foo")

        with self.assertRaisesRegex(ValueError, "Expected a matrix"):
            torch.nn.utils.parametrizations.orthogonal(m, "bias")

        torch.nn.utils.parametrizations.orthogonal(m, "weight")
        with self.assertRaisesRegex(ValueError, "matrices of shape"):
            m.weight = torch.randn(5, 5)
        torch.nn.utils.parametrize.remove_parametrizations(m, "weight")

    @swap([True, False])
    def test_weight_norm_state_dict_compat(self):
        m = nn.Linear(4, 5)
        m = torch.nn.utils.weight_norm(m)
        old_dict = m.state_dict()

        m2 = nn.Linear(4, 5)
        m2 = torch.nn.utils.parametrizations.weight_norm(m2)
        m2.load_state_dict(old_dict)

        input = torch.randn(3, 4)
        self.assertEqual(m(input), m2(input))

    @swap([True, False])
    def test_weight_norm_pickle(self):
        m = nn.Linear(4, 5)
        m = torch.nn.utils.parametrizations.weight_norm(m)
        with self.assertRaisesRegex(RuntimeError, "state_dict"):
            pickle.dumps(m)

    @swap([True, False])
    def test_weight_norm_deepcopy(self):
        m = nn.Linear(4, 5)
        m = torch.nn.utils.parametrizations.weight_norm(m)
        m2 = deepcopy(m)
        input = torch.randn(3, 4)
        self.assertEqual(m(input), m2(input))

    @swap([True])
    def test_wrapper_subclass_parametrization(self):
        class Subclassify(nn.Module):
            def forward(self, X):
                return TwoTensor(X, X)

        class UnSubclassify(nn.Module):
            def forward(self, X):
                return X.a

        class IdentityWithRightInverse(nn.Module):
            def forward(self, X):
                return X

            def right_inverse(self, X):
                return TwoTensor(X, X)

        def _check_parametrization(
            parametrization,
            type_before_registration,
            type_after_registration,
            leave_parametrized=False,
            type_after_right_inverse=None,
        ):
            model = nn.Linear(2, 2)
            buf = torch.randn(2, 2)
            model.buf = torch.nn.Buffer(buf)
            if (
                type_before_registration == TwoTensor
                and type_after_registration == Tensor
            ):
                model._apply(lambda t: TwoTensor(t, t))
            initial_weight = model.weight.clone().detach()
            initial_weight_id = id(model.weight)
            initial_buf = model.buf.clone().detach()
            initial_buf_id = id(model.buf)
            type_original_weight = (
                type_before_registration
                if type_after_right_inverse is None
                else type_after_right_inverse
            )
            type_original_buf = (
                Tensor if type_original_weight is nn.Parameter else type_original_weight
            )
            type_after_removal_buf = (
                type_after_registration if leave_parametrized else type_original_buf
            )
            if leave_parametrized:
                if type_after_registration is Tensor:
                    type_after_removal_weight = nn.Parameter
                else:
                    type_after_removal_weight = type_after_registration
            else:
                type_after_removal_weight = type_original_weight

            parametrize.register_parametrization(model, "weight", parametrization())
            parametrize.register_parametrization(model, "buf", parametrization())
            self.assertTrue(hasattr(model, "parametrizations"))
            self.assertTrue(parametrize.is_parametrized(model))
            self.assertFalse(parametrize.is_parametrized(model, "bias"))
            # checks for weight
            self.assertTrue(parametrize.is_parametrized(model, "weight"))
            self.assertTrue(
                isinstance(model.parametrizations.weight.original, nn.Parameter)
            )
            self.assertTrue(
                type(model.parametrizations.weight.original) is type_original_weight
            )
            self.assertNotIn("weight", model._parameters)
            self.assertTrue(type(model.weight) is type_after_registration)
            # checks for buf
            self.assertTrue(parametrize.is_parametrized(model, "buf"))
            self.assertFalse(
                isinstance(model.parametrizations.buf.original, nn.Parameter)
            )
            self.assertTrue(
                type(model.parametrizations.buf.original) is type_original_buf
            )
            self.assertTrue(type(model.buf) is type_after_registration)
            parametrize.remove_parametrizations(
                model, "weight", leave_parametrized=leave_parametrized
            )
            parametrize.remove_parametrizations(
                model, "buf", leave_parametrized=leave_parametrized
            )
            self.assertFalse(hasattr(model, "parametrizations"))
            self.assertEqual(model.__class__, nn.Linear)
            # checks for weight
            self.assertTrue(type(model.weight) is type_after_removal_weight)
            self.assertTrue(isinstance(model.weight, nn.Parameter))
            self.assertEqual(id(model.weight), initial_weight_id)
            # checks for buf
            self.assertTrue(type(model.buf) is type_after_removal_buf)
            self.assertFalse(isinstance(model.buf, nn.Parameter))
            self.assertEqual(id(model.buf), initial_buf_id)
            if not leave_parametrized and type_after_right_inverse is None:
                self.assertEqual(model.weight, initial_weight)
                self.assertEqual(model.buf, initial_buf)

        _check_parametrization(Subclassify, nn.Parameter, TwoTensor)
        _check_parametrization(UnSubclassify, TwoTensor, Tensor)
        _check_parametrization(
            IdentityWithRightInverse,
            nn.Parameter,
            TwoTensor,
            type_after_right_inverse=TwoTensor,
        )
        _check_parametrization(
            Subclassify, nn.Parameter, TwoTensor, leave_parametrized=True
        )
        _check_parametrization(
            UnSubclassify, TwoTensor, Tensor, leave_parametrized=True
        )
        _check_parametrization(
            IdentityWithRightInverse,
            nn.Parameter,
            TwoTensor,
            leave_parametrized=True,
            type_after_right_inverse=TwoTensor,
        )


class TestNNParametrizationDevice(NNTestCase):
    @swap([True, False])
    def test_weight_norm_parametrization(self, device):
        for dtype in [torch.float, torch.bfloat16]:
            input = torch.randn(3, 4, dtype=dtype, device=device)
            m = nn.Linear(4, 5, dtype=dtype, device=device)
            expected_output = m(input)

            # add weight normalization
            m = torch.nn.utils.parametrizations.weight_norm(m)
            self.assertEqual(
                m.parametrizations.weight.original1.size(), m.weight.size()
            )
            self.assertEqual(m.parametrizations.weight.original0.size(), (5, 1))
            self.assertEqual(m(input), expected_output)

            # remove weight norm
            torch.nn.utils.parametrize.remove_parametrizations(m, "weight")
            self.assertFalse(hasattr(m, "parametrizations"))
            self.assertEqual(m(input), expected_output)

            # test with dim=1
            m = torch.nn.utils.parametrizations.weight_norm(m, dim=1)
            self.assertEqual(
                m.parametrizations.weight.original1.size(), m.weight.size()
            )
            self.assertEqual(m.parametrizations.weight.original0.size(), (1, 4))
            self.assertEqual(m(input), expected_output)

            # test with dim=None
            m = nn.Linear(4, 5, dtype=dtype, device=device)
            expected_output = m(input)
            m = torch.nn.utils.parametrizations.weight_norm(m, dim=None)
            self.assertEqual(m(input), expected_output)


only_for = ("cpu", "cuda")
instantiate_device_type_tests(TestNNParametrizationDevice, globals(), only_for=only_for)
instantiate_parametrized_tests(TestNNParametrization)

if __name__ == "__main__":
    run_tests()
