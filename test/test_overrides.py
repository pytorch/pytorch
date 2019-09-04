import torch
import numpy as np
import warnings
import math
import unittest
from torch._overrides import (
    get_overloaded_types_and_args, torch_function_dispatch,
    verify_matching_signatures)
import pickle
import inspect
import sys
from unittest import mock

from common_utils import TestCase


HANDLED_FUNCTIONS = {}

def implements(torch_function):
   "Register an implementation of a torch function for a Tensor-like object."
   def decorator(func):
       HANDLED_FUNCTIONS[torch_function] = func
       return func
   return decorator


class DiagonalTensor:
    """A class with __torch_function__ and a specific diagonal representation"""
    def __init__(self, N, value):
        self._N = N
        self._i = value

    def __repr__(self):
        return f"{self.__class__.__name__}(N={self._N}, value={self._i})"

    def __array__(self):
        return self._i * np.eye(self._N)

    def tensor(self):
        return self._i * torch.eye(self._N)

    def __torch_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __torch_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __eq__(self, other):
        if type(other) is type(self):
            if self._N == other._N and self._i == other._i:
                return True
            else:
                return False
        else:
            return False


@implements(torch.unique)
def unique_diag(mat1):
    "Implementation of torch.unique for DiagonalArray objects"
    return torch.Tensor([0, mat1._i])


class TestOverride(TestCase):

    def test_unique(self):
        t1 = DiagonalTensor(5, 2)
        t2 = torch.eye(5) * 2
        self.assertEqual(t1.tensor(), t2)
        self.assertEqual(torch.unique(t1), torch.unique(t2))


def _return_not_implemented(self, *args, **kwargs):
    return NotImplemented


# need to define this at the top level to test pickling
@torch_function_dispatch(lambda tensor: (tensor,))
def dispatched_one_arg(tensor):
    """Docstring."""
    return 'original'


@torch_function_dispatch(lambda tensor1, tensor2: (tensor1, tensor2))
def dispatched_two_arg(tensor1, tensor2):
    """Docstring."""
    return 'original'


class TestGetImplementingArgs(TestCase):

    def test_tensor(self):
        tensor = torch.tensor(1)

        args = get_overloaded_types_and_args([tensor])
        self.assertEqual(list(args),[[type(tensor)], [tensor]])

        args = get_overloaded_types_and_args([tensor, tensor])
        self.assertEqual(list(args),[[type(tensor)], [tensor]])

        args = get_overloaded_types_and_args([tensor, 1])
        self.assertEqual(list(args),[[type(tensor)], [tensor]])

        args = get_overloaded_types_and_args([1, tensor])
        self.assertEqual(list(args),[[type(tensor)], [tensor]])

    def test_tensor_subclasses(self):
        # Check order in which args are returned: subclasses before superclasses

        class OverrideSub(torch.Tensor):
            __torch_function__ = _return_not_implemented

        class NoOverrideSub(torch.Tensor):
            pass

        tensor = torch.tensor([1])
        override_sub = OverrideSub([2])
        no_override_sub = NoOverrideSub([3])

        args = get_overloaded_types_and_args([tensor, override_sub])
        self.assertEqual(args[1], [override_sub, tensor])

        args = get_overloaded_types_and_args([tensor, no_override_sub])
        self.assertEqual(args[1], [no_override_sub, tensor])

        args = get_overloaded_types_and_args(
            [override_sub, no_override_sub])
        self.assertEqual(args[1], [override_sub, no_override_sub])

    def test_tensor_and_duck_tensor(self):

        class Other(object):
            __torch_function__ = _return_not_implemented

        tensor = torch.tensor(1)
        other = Other()

        args = get_overloaded_types_and_args([other, tensor])
        self.assertEqual(list(args), [[type(other), type(tensor)], [other, tensor]])

        args = get_overloaded_types_and_args([tensor, other])
        self.assertEqual(list(args), [[type(tensor), type(other)], [tensor, other]])

    @unittest.expectedFailure # Tensor.view() is different from ndarray.view()
    def test_tensor_subclass_and_duck_tensor(self):

        class OverrideSub(torch.Tensor):
            __torch_function__ = _return_not_implemented

        class Other(object):
            __torch_function__ = _return_not_implemented

        tensor = torch.tensor(1)
        subtensor = torch.tensor(1).view(OverrideSub)
        other = Other()

        self.assertEqual(get_overloaded_types_and_args([tensor, subtensor, other]),
                     [subtensor, tensor, other])
        self.assertEqual(get_overloaded_types_and_args([tensor, other, subtensor]),
                     [subtensor, tensor, other])

    def test_many_duck_tensors(self):

        class A(object):
            __torch_function__ = _return_not_implemented

        class B(A):
            __torch_function__ = _return_not_implemented

        class C(A):
            __torch_function__ = _return_not_implemented

        class D(object):
            __torch_function__ = _return_not_implemented

        a = A()
        b = B()
        c = C()
        d = D()

        self.assertEqual(get_overloaded_types_and_args([1]), ([],[]))
        self.assertEqual(get_overloaded_types_and_args([a]), ([type(a)], [a]))
        self.assertEqual(get_overloaded_types_and_args([a, 1]), ([type(a)], [a]))
        self.assertEqual(get_overloaded_types_and_args([a, a, a]), ([type(a)], [a]))
        self.assertEqual(get_overloaded_types_and_args([a, d, a]), ([type(a), type(d)], [a, d]))
        self.assertEqual(get_overloaded_types_and_args([a, b]), ([type(a), type(b)], [b, a]))
        self.assertEqual(get_overloaded_types_and_args([b, a]), ([type(b), type(a)], [b, a]))
        self.assertEqual(get_overloaded_types_and_args([a, b, c]), ([type(a), type(b), type(c)], [b, c, a]))
        self.assertEqual(get_overloaded_types_and_args([a, c, b]), ([type(a), type(c), type(b)], [c, b, a]))

    @unittest.expectedFailure
    def test_too_many_duck_tensors(self):
        namespace = dict(__torch_function__=_return_not_implemented)
        types = [type('A' + str(i), (object,), namespace) for i in range(33)]
        relevant_args = [t() for t in types]

        actual = get_overloaded_types_and_args(relevant_args[:32])
        self.assertEqual(actual, relevant_args[:32])

        with self.assertRaisesRegex(TypeError, 'distinct argument types'):
            get_overloaded_types_and_args(relevant_args)


class TestTensorTorchFunction(TestCase):

    @unittest.expectedFailure # Tensor.view() is different from ndarray.view()
    def test_method(self):

        class Other(object):
            __torch_function__ = _return_not_implemented

        class NoOverrideSub(torch.Tensor):
            pass

        class OverrideSub(torch.Tensor):
            __torch_function__ = _return_not_implemented

        tensor = torch.tensor([1])
        other = Other()
        no_override_sub = tensor.view(NoOverrideSub)
        override_sub = tensor.view(OverrideSub)

        result = tensor.__torch_function__(func=dispatched_two_arg,
                                          types=(torch.tensor,),
                                          args=(tensor, 1.), kwargs={})
        self.assertEqual(result, 'original')

        result = tensor.__torch_function__(func=dispatched_two_arg,
                                          types=(torch.tensor, Other),
                                          args=(tensor, other), kwargs={})
        assert result is NotImplemented

        result = tensor.__torch_function__(func=dispatched_two_arg,
                                          types=(torch.tensor, NoOverrideSub),
                                          args=(tensor, no_override_sub),
                                          kwargs={})
        self.assertEqual(result, 'original')

        result = tensor.__torch_function__(func=dispatched_two_arg,
                                          types=(torch.tensor, OverrideSub),
                                          args=(tensor, override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        with self.assertRaisesRegex(TypeError, 'no implementation found'):
            torch.cat((tensor, other))

        expected =torch.cat((tensor, tensor))
        result = torch.cat((tensor, no_override_sub))
        assert_equal(result, expected.view(NoOverrideSub))
        result = torch.cat((tensor, override_sub))
        assert_equal(result, expected.view(OverrideSub))

    @unittest.expectedFailure # TODO
    def test_no_wrapper(self):
        # This shouldn't happen unless a user intentionally calls
        # __torch_function__ with invalid arguments, but check that we raise
        # an appropriate error all the same.
        tensor = torch.tensor(1)
        func = lambda x: x
        with self.assertRaisesRegex(AttributeError, '_implementation'):
            tensor.__torch_function__(func=func, types=(torch.Tensor,),
                                     args=(tensor,), kwargs={})


class TestTorchFunctionDispatch(TestCase):

    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            roundtripped = pickle.loads(
                    pickle.dumps(dispatched_one_arg, protocol=proto))
            assert roundtripped is dispatched_one_arg

    def test_name_and_docstring(self):
        self.assertEqual(dispatched_one_arg.__name__, 'dispatched_one_arg')
        if sys.flags.optimize < 2:
            self.assertEqual(dispatched_one_arg.__doc__, 'Docstring.')

    def test_interface(self):

        class MyTensor(object):
            def __torch_function__(self, func, types, args, kwargs):
                return (self, func, types, args, kwargs)

        original = MyTensor()
        (obj, func, types, args, kwargs) = dispatched_one_arg(original)
        assert obj is original
        assert func is dispatched_one_arg
        self.assertEqual(set(types), {MyTensor})
        # assert_equal uses the overloaded torch.iscomplexobj() internally
        assert args == (original,)
        self.assertEqual(kwargs, {})

    def test_not_implemented(self):

        class MyTensor(object):
            def __torch_function__(self, func, types, args, kwargs):
                return NotImplemented

        tensor = MyTensor()
        with self.assertRaisesRegex(TypeError, 'no implementation found'):
            dispatched_one_arg(tensor)


class TestVerifyMatchingSignatures(TestCase):

    def test_verify_matching_signatures(self):

        verify_matching_signatures(lambda x: 0, lambda x: 0)
        verify_matching_signatures(lambda x=None: 0, lambda x=None: 0)
        verify_matching_signatures(lambda x=1: 0, lambda x=None: 0)

        with self.assertRaises(RuntimeError):
            verify_matching_signatures(lambda a: 0, lambda b: 0)
        with self.assertRaises(RuntimeError):
            verify_matching_signatures(lambda x: 0, lambda x=None: 0)
        with self.assertRaises(RuntimeError):
            verify_matching_signatures(lambda x=None: 0, lambda y=None: 0)
        with self.assertRaises(RuntimeError):
            verify_matching_signatures(lambda x=1: 0, lambda y=1: 0)

    def test_torch_function_dispatch(self):

        with self.assertRaises(RuntimeError):
            @torch_function_dispatch(lambda x: (x,))
            def f(y):
                pass

        # should not raise
        @torch_function_dispatch(lambda x: (x,), verify=False)
        def f(y):
            pass


def _new_duck_type_and_implements():
    """Create a duck tensor type and implements functions."""
    HANDLED_FUNCTIONS = {}

    class MyTensor(object):
        def __torch_function__(self, func, types, args, kwargs):
            if func not in HANDLED_FUNCTIONS:
                return NotImplemented
            if not all(issubclass(t, MyTensor) for t in types):
                return NotImplemented
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def implements(torch_function):
        """Register an __torch_function__ implementations."""
        def decorator(func):
            HANDLED_FUNCTIONS[torch_function] = func
            return func
        return decorator

    return (MyTensor, implements)


class TestTensorFunctionImplementation(TestCase):

    def test_one_arg(self):
        MyTensor, implements = _new_duck_type_and_implements()

        @implements(dispatched_one_arg)
        def _(tensor):
            return 'mytensor'

        self.assertEqual(dispatched_one_arg(1), 'original')
        self.assertEqual(dispatched_one_arg(MyTensor()), 'mytensor')

    def test_optional_args(self):
        MyTensor, implements = _new_duck_type_and_implements()

        @torch_function_dispatch(lambda tensor, option=None: (tensor,))
        def func_with_option(tensor, option='default'):
            return option

        @implements(func_with_option)
        def my_tensor_func_with_option(tensor, new_option='mytensor'):
            return new_option

        # we don't need to implement every option on __torch_function__
        # implementations
        self.assertEqual(func_with_option(1), 'default')
        self.assertEqual(func_with_option(1, option='extra'), 'extra')
        self.assertEqual(func_with_option(MyTensor()), 'mytensor')
        with self.assertRaises(TypeError):
            func_with_option(MyTensor(), option='extra')

        # but new options on implementations can't be used
        result = my_tensor_func_with_option(MyTensor(), new_option='yes')
        self.assertEqual(result, 'yes')
        with self.assertRaises(TypeError):
            func_with_option(MyTensor(), new_option='no')

    def test_not_implemented(self):
        MyTensor, implements = _new_duck_type_and_implements()

        @torch_function_dispatch(lambda tensor: (tensor,), module='my')
        def func(tensor):
            return tensor

        tensor = torch.tensor(1)
        assert func(tensor) is tensor
        self.assertEqual(func.__module__, 'my')

        with self.assertRaisesRegex(
                TypeError, "no implementation found for 'my.func'"):
            func(MyTensor())


class TestTensorMethods(TestCase):

    @unittest.expectedFailure # Tensor.view() is different from ndarray.view()
    def test_repr(self):
        # gh-12162: should still be defined even if __torch_function__ doesn't
        # implement torch.tensor_repr()

        class MyTensor(torch.Tensor):
            def __torch_function__(*args, **kwargs):
                return NotImplemented

        tensor = torch.tensor(1).view(MyTensor)
        self.assertEqual(repr(tensor), 'MyTensor(1)')
        self.assertEqual(str(tensor), '1')


class TestTorchFunctions(TestCase):

    @unittest.expectedFailure # Discuss
    def test_set_module(self):
        self.assertEqual(torch.sum.__module__, 'torch')
        self.assertEqual(torch.char.equal.__module__, 'torch.char')
        self.assertEqual(torch.fft.fft.__module__, 'torch.fft')
        self.assertEqual(torch.linalg.solve.__module__, 'torch.linalg')

    @unittest.expectedFailure # Discuss
    def test_inspect_sum(self):
        signature = inspect.signature(torch.sum)
        assert 'axis' in signature.parameters

    @unittest.expectedFailure # Discuss
    def test_override_sum(self):
        MyTensor, implements = _new_duck_type_and_implements()

        @implements(torch.sum)
        def _(tensor):
            return 'yes'

        self.assertEqual(torch.sum(MyTensor()), 'yes')

    @unittest.expectedFailure # Discuss
    def test_sum_on_mock_tensor(self):

        # We need a proxy for mocks because __torch_function__ is only looked
        # up in the class dict
        class TensorProxy:
            def __init__(self, value):
                self.value = value
            def __tensor_function__(self, *args, **kwargs):
                return self.value.__tensor_function__(*args, **kwargs)
            def __tensor__(self, *args, **kwargs):
                return self.value.__tensor__(*args, **kwargs)

        proxy = TensorProxy(mock.Mock(spec=TensorProxy))
        proxy.value.__torch_function__.return_value = 1
        result = torch.sum(proxy)
        self.assertEqual(result, 1)
        proxy.value.__torch_function__.assert_called_once_with(
            torch.sum, (TensorProxy,), (proxy,), {})
        proxy.value.__tensor__.assert_not_called()


if __name__ == '__main__':
    unittest.main()
