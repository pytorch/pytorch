import torch
import numpy as np
import warnings
import math
import unittest
from common_utils import TestCase, run_tests, TEST_WITH_UBSAN, load_tests, \
    skipIfRocm


HANDLED_FUNCTIONS = {}

class DiagonalTensor:
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


def implements(torch_function):
   "Register an __torch_function__ implementation for DiagonalTensor objects."
   def decorator(func):
       HANDLED_FUNCTIONS[torch_function] = func
       return func
   return decorator

@implements(torch.gemm)
def gemm_diag(mat1, mat2, out=None):
    "Implementation of torch.gemm for DiagonalArray objects"
    print('Called our custom gemm for DiagonalTensor input')
    if not mat1._N == mat2._N:
        raise ValueError("Dimension mismatch")

    return DiagonalTensor(mat1._N, mat1._i * mat2._i)

t1 = DiagonalTensor(5, 1)
t2 = DiagonalTensor(5, 2)
t3 = DiagonalTensor(5, 1)

print(gemm_diag(t1, t2))
print(torch.gemm(t1, t2))
print(torch.gemm(t1.tensor(), t2.tensor()))
print(torch.mm(t1.tensor(), t2.tensor()))

class TestOverride(unittest.TestCase):
    def test_gemm_equality(self):
        self.assertEqual(t1, t3)

    def test_gemm_diag(self):
        self.assertEqual(torch.gemm(t1, t2), t2)


requires_torch_function = pytest.mark.skipif(
    not TORCH_FUNCTION_ENABLED,
    reason="__torch_function__ dispatch not enabled.")

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


class TestGetImplementingArgs(object):

    def test_tensor(self):
        tensor = torch.tensor(1)

        args = _get_implementing_args([tensor])
        assert_equal(list(args), [tensor])

        args = _get_implementing_args([tensor, tensor])
        assert_equal(list(args), [tensor])

        args = _get_implementing_args([tensor, 1])
        assert_equal(list(args), [tensor])

        args = _get_implementing_args([1, tensor])
        assert_equal(list(args), [tensor])

    def test_tensor_subclasses(self):

        class OverrideSub(torch.Tensor):
            __torch_function__ = _return_not_implemented

        class NoOverrideSub(torch.Tensor):
            pass

        tensor = torch.tensor(1).view(torch.tensor)
        override_sub = torch.tensor(1).view(OverrideSub)
        no_override_sub = torch.tensor(1).view(NoOverrideSub)

        args = _get_implementing_args([tensor, override_sub])
        assert_equal(list(args), [override_sub, tensor])

        args = _get_implementing_args([tensor, no_override_sub])
        assert_equal(list(args), [no_override_sub, tensor])

        args = _get_implementing_args(
            [override_sub, no_override_sub])
        assert_equal(list(args), [override_sub, no_override_sub])

    def test_tensor_and_duck_tensor(self):

        class Other(object):
            __torch_function__ = _return_not_implemented

        tensor = torch.tensor(1)
        other = Other()

        args = _get_implementing_args([other, tensor])
        assert_equal(list(args), [other, tensor])

        args = _get_implementing_args([tensor, other])
        assert_equal(list(args), [tensor, other])

    def test_tensor_subclass_and_duck_tensor(self):

        class OverrideSub(torch.Tensor):
            __torch_function__ = _return_not_implemented

        class Other(object):
            __torch_function__ = _return_not_implemented

        tensor = torch.tensor(1)
        subtensor = torch.tensor(1).view(OverrideSub)
        other = Other()

        assert_equal(_get_implementing_args([tensor, subtensor, other]),
                     [subtensor, tensor, other])
        assert_equal(_get_implementing_args([tensor, other, subtensor]),
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

        assert_equal(_get_implementing_args([1]), [])
        assert_equal(_get_implementing_args([a]), [a])
        assert_equal(_get_implementing_args([a, 1]), [a])
        assert_equal(_get_implementing_args([a, a, a]), [a])
        assert_equal(_get_implementing_args([a, d, a]), [a, d])
        assert_equal(_get_implementing_args([a, b]), [b, a])
        assert_equal(_get_implementing_args([b, a]), [b, a])
        assert_equal(_get_implementing_args([a, b, c]), [b, c, a])
        assert_equal(_get_implementing_args([a, c, b]), [c, b, a])

    def test_too_many_duck_tensors(self):
        namespace = dict(__torch_function__=_return_not_implemented)
        types = [type('A' + str(i), (object,), namespace) for i in range(33)]
        relevant_args = [t() for t in types]

        actual = _get_implementing_args(relevant_args[:32])
        assert_equal(actual, relevant_args[:32])

        with assert_raises_regex(TypeError, 'distinct argument types'):
            _get_implementing_args(relevant_args)


class TestTensorTorchFunction(object):

    @requires_torch_function
    def test_method(self):

        class Other(object):
            __torch_function__ = _return_not_implemented

        class NoOverrideSub(torch.tensor):
            pass

        class OverrideSub(torch.tensor):
            __torch_function__ = _return_not_implemented

        tensor = torch.tensor([1])
        other = Other()
        no_override_sub = tensor.view(NoOverrideSub)
        override_sub = tensor.view(OverrideSub)

        result = tensor.__torch_function__(func=dispatched_two_arg,
                                          types=(torch.tensor,),
                                          args=(tensor, 1.), kwargs={})
        assert_equal(result, 'original')

        result = tensor.__torch_function__(func=dispatched_two_arg,
                                          types=(torch.tensor, Other),
                                          args=(tensor, other), kwargs={})
        assert_(result is NotImplemented)

        result = tensor.__torch_function__(func=dispatched_two_arg,
                                          types=(torch.tensor, NoOverrideSub),
                                          args=(tensor, no_override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        result = tensor.__torch_function__(func=dispatched_two_arg,
                                          types=(torch.tensor, OverrideSub),
                                          args=(tensor, override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        with assert_raises_regex(TypeError, 'no implementation found'):
            torch.concatenate((tensor, other))

        expected =torch.concatenate((tensor, tensor))
        result = torch.concatenate((tensor, no_override_sub))
        assert_equal(result, expected.view(NoOverrideSub))
        result = torch.concatenate((tensor, override_sub))
        assert_equal(result, expected.view(OverrideSub))

    def test_no_wrapper(self):
        # This shouldn't happen unless a user intentionally calls
        # __torch_function__ with invalid arguments, but check that we raise
        # an appropriate error all the same.
        tensor = torch.tensor(1)
        func = lambda x: x
        with assert_raises_regex(AttributeError, '_implementation'):
            tensor.__torch_function__(func=func, types=(torch.Tensor,),
                                     args=(tensor,), kwargs={})


@requires_torch_function
class TestTorchFunctionDispatch(object):

    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            roundtripped = pickle.loads(
                    pickle.dumps(dispatched_one_arg, protocol=proto))
            assert_(roundtripped is dispatched_one_arg)

    def test_name_and_docstring(self):
        assert_equal(dispatched_one_arg.__name__, 'dispatched_one_arg')
        if sys.flags.optimize < 2:
            assert_equal(dispatched_one_arg.__doc__, 'Docstring.')

    def test_interface(self):

        class MyTensor(object):
            def __torch_function__(self, func, types, args, kwargs):
                return (self, func, types, args, kwargs)

        original = MyTensor()
        (obj, func, types, args, kwargs) = dispatched_one_arg(original)
        assert_(obj is original)
        assert_(func is dispatched_one_arg)
        assert_equal(set(types), {MyTensor})
        # assert_equal uses the overloaded torch.iscomplexobj() internally
        assert_(args == (original,))
        assert_equal(kwargs, {})

    def test_not_implemented(self):

        class MyTensor(object):
            def __torch_function__(self, func, types, args, kwargs):
                return NotImplemented

        tensor = MyTensor()
        with assert_raises_regex(TypeError, 'no implementation found'):
            dispatched_one_arg(tensor)


@requires_torch_function
class TestVerifyMatchingSignatures(object):

    def test_verify_matching_signatures(self):

        verify_matching_signatures(lambda x: 0, lambda x: 0)
        verify_matching_signatures(lambda x=None: 0, lambda x=None: 0)
        verify_matching_signatures(lambda x=1: 0, lambda x=None: 0)

        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda a: 0, lambda b: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x: 0, lambda x=None: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x=None: 0, lambda y=None: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x=1: 0, lambda y=1: 0)

    def test_torch_function_dispatch(self):

        with assert_raises(RuntimeError):
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


@requires_torch_function
class TestTensorFunctionImplementation(object):

    def test_one_arg(self):
        MyTensor, implements = _new_duck_type_and_implements()

        @implements(dispatched_one_arg)
        def _(tensor):
            return 'mytensor'

        assert_equal(dispatched_one_arg(1), 'original')
        assert_equal(dispatched_one_arg(MyTensor()), 'mytensor')

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
        assert_equal(func_with_option(1), 'default')
        assert_equal(func_with_option(1, option='extra'), 'extra')
        assert_equal(func_with_option(MyTensor()), 'mytensor')
        with assert_raises(TypeError):
            func_with_option(MyTensor(), option='extra')

        # but new options on implementations can't be used
        result = my_tensor_func_with_option(MyTensor(), new_option='yes')
        assert_equal(result, 'yes')
        with assert_raises(TypeError):
            func_with_option(MyTensor(), new_option='no')

    def test_not_implemented(self):
        MyTensor, implements = _new_duck_type_and_implements()

        @torch_function_dispatch(lambda tensor: (tensor,), module='my')
        def func(tensor):
            return tensor

        tensor = torch.tensor(1)
        assert_(func(tensor) is tensor)
        assert_equal(func.__module__, 'my')

        with assert_raises_regex(
                TypeError, "no implementation found for 'my.func'"):
            func(MyTensor())


class TestTensorMethods(object):

    def test_repr(self):
        # gh-12162: should still be defined even if __torch_function__ doesn't
        # implement torch.tensor_repr()

        class MyTensor(torch.tensor):
            def __torch_function__(*args, **kwargs):
                return NotImplemented

        tensor = torch.tensor(1).view(MyTensor)
        assert_equal(repr(tensor), 'MyTensor(1)')
        assert_equal(str(tensor), '1')


class TestTorchFunctions(object):

    def test_set_module(self):
        assert_equal(torch.sum.__module__, 'torch')
        assert_equal(torch.char.equal.__module__, 'torch.char')
        assert_equal(torch.fft.fft.__module__, 'torch.fft')
        assert_equal(torch.linalg.solve.__module__, 'torch.linalg')

    def test_inspect_sum(self):
        signature = inspect.signature(torch.sum)
        assert_('axis' in signature.parameters)

    @requires_torch_function
    def test_override_sum(self):
        MyTensor, implements = _new_duck_type_and_implements()

        @implements(torch.sum)
        def _(tensor):
            return 'yes'

        assert_equal(torch.sum(MyTensor()), 'yes')

    @requires_torch_function
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
        assert_equal(result, 1)
        proxy.value.__torch_function__.assert_called_once_with(
            torch.sum, (TensorProxy,), (proxy,), {})
        proxy.value.__tensor__.assert_not_called()

    @requires_torch_function
    def test_sum_forwarding_implementation(self):

        class MyTensor(torch.Tensor):

            def sum(self, axis, out):
                return 'summed'

            def __torch_function__(self, func, types, args, kwargs):
                return super().__torch_function__(func, types, args, kwargs)

        # note: the internal implementation of torch.sum() calls the .sum() method
        tensor = torch.tensor(1).view(MyTensor)
        assert_equal(torch.sum(tensor), 'summed')

if __name__ == '__main__':
    unittest.main()
