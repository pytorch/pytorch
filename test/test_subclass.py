# Owner(s): ["module: nn"]

import tempfile
import torch
from copy import deepcopy
from torch import nn
from torch.nn.utils.parametrize import register_parametrization, remove_parametrizations
from torch.nn.modules.lazy import LazyModuleMixin
from torch.testing._internal.common_utils import (
    TestCase, run_tests, parametrize, subtest, instantiate_parametrized_tests)
from torch.testing._internal.common_subclass import subclass_db, DiagTensorBelow
from torch.testing._internal.logging_tensor import LoggingTensor
from unittest import expectedFailure

# The current test methodology in this file is to test a variety of real use cases
# with a set of fully-fledged tensor subclasses. In the future, this may change
# to more narrowly specify toy subclasses for each of the specific invariants under
# test, avoiding the need to maintain the set of fully-fledged tensor subclasses.


# Decorator for parametrizing tests across the various tensor classes.
parametrize_tensor_cls = parametrize("tensor_cls", [
    subtest(tensor_cls, name=info.name) for tensor_cls, info in subclass_db.items()])


class TestSubclass(TestCase):
    def _create_tensor(self, tensor_cls):
        return subclass_db[tensor_cls].create_fn(3)

    @parametrize_tensor_cls
    @parametrize("tensor_requires_grad", [False, True])
    def test_param_invariants(self, tensor_cls, tensor_requires_grad):
        x = self._create_tensor(tensor_cls).requires_grad_(tensor_requires_grad)
        param = nn.Parameter(x, requires_grad=(not tensor_requires_grad))

        self.assertTrue(isinstance(param, nn.Parameter))
        # Ensure requires_grad passed to Parameter's constructor takes precedence.
        self.assertEqual(param.requires_grad, not tensor_requires_grad)

        # Ensure original tensor is not mutated by Parameter construction.
        self.assertFalse(isinstance(x, nn.Parameter))
        self.assertEqual(x.requires_grad, tensor_requires_grad)

    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_deepcopy(self, tensor_cls, as_param):
        x = self._create_tensor(tensor_cls)
        if as_param:
            x = nn.Parameter(x)
        x_copy = deepcopy(x)
        self.assertEqual(x, x_copy)
        self.assertEqual(x.__class__, x_copy.__class__)
        self.assertFalse(x is x_copy)
        if as_param:
            # Deepcopy should preserve both custom type and "parameter-ness".
            self.assertTrue(isinstance(x_copy, tensor_cls))
            self.assertTrue(isinstance(x_copy, nn.Parameter))
        else:
            self.assertTrue(type(x_copy) is tensor_cls)

    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_serialization(self, tensor_cls, as_param):
        with tempfile.TemporaryFile() as f:
            x = self._create_tensor(tensor_cls)
            if as_param:
                x = nn.Parameter(x)
            torch.save(x, f)
            f.seek(0)
            x_loaded = torch.load(f)

            self.assertEqual(x, x_loaded)
            self.assertFalse(x is x_loaded)
            if as_param:
                # Serialization should preserve both custom type and "parameter-ness".
                self.assertTrue(isinstance(x_loaded, tensor_cls))
                self.assertTrue(isinstance(x_loaded, nn.Parameter))
            else:
                self.assertTrue(type(x_loaded) is tensor_cls)

    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_repr(self, tensor_cls, as_param):
        x = self._create_tensor(tensor_cls)
        if as_param:
            x = nn.Parameter(x)
        str_repr = x.__repr__()
        if tensor_cls is not torch.Tensor:
            self.assertEqual(str_repr.count(f"{tensor_cls.__name__}("), 1)
        self.assertEqual(str_repr.count("Parameter"), 1 if as_param else 0)

    @parametrize_tensor_cls
    @parametrize("as_param", [False, True])
    def test_type_propagation(self, tensor_cls, as_param):
        x = self._create_tensor(tensor_cls)
        if as_param:
            x = nn.Parameter(x)

        # Call the add operator to produce an output tensor.
        output = x + self._create_tensor(torch.Tensor)

        # Custom type should be propagated across operations, but "parameter-ness" should not be.
        self.assertTrue(output.__class__ is (tensor_cls if as_param else x.__class__))

    @parametrize_tensor_cls
    def test_module_optimization(self, tensor_cls):
        def create_fn():
            return self._create_tensor(tensor_cls)

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.p1 = nn.Parameter(create_fn())

                self.p_list = nn.ParameterList([create_fn() for _ in range(3)])
                self.p_list.append(create_fn())

                self.p_dict = nn.ParameterDict({
                    'foo': create_fn(),
                    'bar': create_fn(),
                })
                self.p_dict['baz'] = create_fn()

                with torch.no_grad():
                    nn.init.normal_(self.p1)
                    for p in self.p_list:
                        nn.init.uniform_(p)
                    for _, p in self.p_dict.items():
                        nn.init.uniform_(p)

            def forward(self, x):
                out = self.p1 + x
                for p in self.p_list:
                    out = p + out

                for _, v in self.p_dict.items():
                    out = v + out

                return out

        m = MyModule()
        self.assertEqual(len(m.state_dict()), 8)

        optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
        m(create_fn()).sum().backward(torch.tensor(1))
        optimizer.step()

    @parametrize("tensor_cls", [
        subtest(tensor_cls, name=info.name) for tensor_cls, info in subclass_db.items()
        # TODO: Either implement set_() properly for these tensor subclasses or apply a
        # more general fix to avoid the need for special set_() handling.
        if tensor_cls not in [LoggingTensor, DiagTensorBelow]])
    @parametrize("leave_parametrized", [False, True])
    def test_parametrization(self, tensor_cls, leave_parametrized):
        def create_fn():
            return self._create_tensor(tensor_cls)

        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(create_fn())

            def forward(self, x):
                return self.weight + x

        class MyParametrization(nn.Module):
            def forward(self, X):
                return -X

        m = MyModule()
        self.assertEqual(len(m.state_dict()), 1)
        register_parametrization(m, 'weight', MyParametrization())
        self.assertIs(type(m.weight), tensor_cls)
        output = m(self._create_tensor(torch.Tensor))
        self.assertIs(type(output), tensor_cls)
        remove_parametrizations(m, 'weight', leave_parametrized=leave_parametrized)

    # Lazy modules with custom tensors are not supported yet.
    @expectedFailure
    @parametrize_tensor_cls
    def test_lazy_module(self, tensor_cls):

        class MyLazyModule(LazyModuleMixin, nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.UninitializedParameter()

            def initialize_parameters(self, input) -> None:  # type: ignore[override]
                if self.has_uninitialized_params():
                    with torch.no_grad():
                        self.param.materialize(input.shape)
                        nn.init.uniform_(self.param)

            def forward(self, x):
                return self.param + x

        m = MyLazyModule()
        self.assertTrue(m.has_uninitialized_params())
        output = m(self._create_tensor(tensor_cls))
        self.assertFalse(m.has_uninitialized_params())
        self.assertIs(type(m.param), tensor_cls)

instantiate_parametrized_tests(TestSubclass)

if __name__ == '__main__':
    run_tests()
