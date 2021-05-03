from functools import reduce, wraps, partial
from itertools import product
from operator import mul
import collections
import operator
import random

import torch
import json

from typing import List, Sequence, Tuple, Dict, Any, Union

from torch.testing import \
    (make_non_contiguous, floating_types, floating_types_and, complex_types,
     floating_and_complex_types, floating_and_complex_types_and,
     all_types_and_complex_and, all_types_and, all_types_and_complex,
     integral_types_and, all_types)
from .._core import _dispatch_dtypes
from torch.testing._internal.common_device_type import \
    (skipIf, skipCUDAIfNoMagma, skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfNoCusolver,
     skipCPUIfNoLapack, skipCPUIfNoMkl, skipCUDAIfRocm, precisionOverride,)
from torch.testing._internal.common_cuda import CUDA11OrLater, SM53OrLater
from torch.testing._internal.common_utils import \
    (is_iterable_of_tensors,
     random_symmetric_matrix, random_symmetric_psd_matrix,
     make_fullrank_matrices_with_distinct_singular_values,
     random_symmetric_pd_matrix, make_symmetric_matrices,
     make_symmetric_pd_matrices,
     random_fullrank_matrix_distinct_singular_value, set_rng_seed, SEED,
     TEST_WITH_ROCM, IS_WINDOWS, IS_MACOS, make_tensor, TEST_SCIPY,
     torch_to_numpy_dtype_dict, slowTest, TEST_WITH_ASAN, _wrap_warn_once,
     GRADCHECK_NONDET_TOL,)

from setuptools import distutils

    e.g. getattr(torch, 'fft.rfft')
    """
    path = qname.split('.')
    for name in path:
        obj = getattr(obj, name, _END_SENTINEL)
        if obj is _END_SENTINEL:
            return default
    return obj

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests

import pickle

PRECISION = 1e-4


@contextlib.contextmanager
def backward_engine(engine):
    _prev_engine = Variable._execution_engine
    Variable._execution_engine = engine()
    try:
        yield
    finally:
        Variable._execution_engine = _prev_engine


def graph_desc(fn):
    if fn is None:
        return 'None'
    result = type(fn).__name__ + '('
    next_functions = fn.next_functions
    for next_fn, _ in next_functions:
        result += graph_desc(next_fn)
        result += ', '
    if next_functions:
        result = result[:-2]
    return result + ')'


class TestAutograd(TestCase):

    __slots__ = ['input', 'args', 'kwargs', 'output_process_fn_grad', 'broadcasts_input']

    def __init__(self, input, *, args=tuple(), kwargs=None, output_process_fn_grad=None, broadcasts_input=False):
        # input is the first input to the op and must be either a Tensor or TensorList (Sequence[Tensor]).
        # This follows the typical pattern where for Tensor inputs op(t, ...) = t.op(...).
        # op with TensorList inputs do not support method or inplace variants.
        assert isinstance(input, torch.Tensor) or is_iterable_of_tensors(input)
        self.input: Union[torch.Tensor, Sequence[torch.Tensor]] = input
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.output_process_fn_grad = output_process_fn_grad

        # Specifies if `self.input` is broadcasted or not,
        # given that the operator supports broadcasting.
        # This field is used to verify the behavior for inplace variant.
        #
        # If a SampleInput is marked with `broadcasts_input=True`,
        # it is verified that we get a `RuntimerError` with this sample,
        # and inplace variant. Also inplace grad{grad} tests are skipped,
        # for such inputs (as they will error out otherwise).
        self.broadcasts_input = broadcasts_input

    def __repr__(self):
        arguments = [
            'input=Tensor' if isinstance(self.input, torch.Tensor) else f'input=TensorList[{len(self.input)}]',
            f'args={self.args}' if len(self.args) > 0 else None,
            f'kwargs={self.kwargs}' if len(self.kwargs) > 0 else None,
            (f'output_process_fn_grad={self.output_process_fn_grad}'
             if self.output_process_fn_grad is not None else None),
            f'broadcasts_input={self.broadcasts_input}']

            # Accessing .grad on non-leaf that retains gradients
            dummy.retain_grad()
            foo = dummy.grad
            self.assertEqual(len(w), 1)

    def _function_test(self, cls):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5, requires_grad=True)
        result = cls.apply(x, 2, y)
        go = torch.ones((), requires_grad=True)
        result.sum().backward(go, create_graph=True)

        self.assertEqual(x.grad, y + torch.ones(5, 5))
        self.assertEqual(y.grad, x + torch.ones(5, 5) * 2)
        self.assertIsNotNone(x.grad.grad_fn)
        self.assertIsNotNone(y.grad.grad_fn)

        return x, y

    def test_function(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, tensor1, pyscalar, tensor2):
                ctx.pyscalar = pyscalar
                ctx.save_for_backward(tensor1, tensor2)
                return tensor1 + pyscalar * tensor2 + tensor1 * tensor2

            @staticmethod
            def backward(ctx, grad_output):
                var1, var2 = ctx.saved_tensors
                # NOTE: self is the test case here
                self.assertIsInstance(var1, torch.Tensor)
                self.assertIsInstance(var2, torch.Tensor)
                self.assertIsInstance(grad_output, torch.Tensor)
                return (grad_output + grad_output * var2, None,
                        grad_output * ctx.pyscalar + grad_output * var1)

        x, y = self._function_test(MyFunction)

        x_grad_desc = graph_desc(x.grad.grad_fn)
        y_grad_desc = graph_desc(y.grad.grad_fn)
        self.assertExpected(x_grad_desc, "x_grad_desc")
        self.assertExpected(y_grad_desc, "y_grad_desc")

    def test_once_differentiable(self):
        class MyFunction(Function):

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 op=None,  # the function variant of the operation, populated as torch.<name> if None
                 dtypes=floating_types(),  # dtypes this function is expected to work with
                 dtypesIfCPU=None,  # dtypes this function is expected to work with on CPU
                 dtypesIfCUDA=None,  # dtypes this function is expected to work with on CUDA
                 dtypesIfROCM=None,  # dtypes this function is expected to work with on ROCM
                 backward_dtypes=None,  # backward dtypes this function is expected to work with
                 backward_dtypesIfCPU=None,  # backward dtypes this function is expected to work with on CPU
                 backward_dtypesIfCUDA=None,  # backward dtypes this function is expected to work with on CUDA
                 backward_dtypesIfROCM=None,  # backward dtypes this function is expected to work with on ROCM
                 default_test_dtypes=None,  # dtypes to test with by default. Gets intersected
                                            # with the dtypes support on the tested device
                 assert_autodiffed=False,  # if a op's aten::node is expected to be symbolically autodiffed
                 autodiff_nonfusible_nodes=None,  # a list of strings with node names that are expected to be in a
                                                  # DifferentiableGraph when autodiffed. Ex: ['aten::add', 'aten::mm'],
                                                  # default is populated to be ['aten::(name of Python operator)']
                 autodiff_fusible_nodes=None,  # a list of strings with node names that are expected to be in FusionGroups
                                               # inside of DifferentiableGraphs when this operation is autodiffed.
                                               # Ex: ['aten::add', 'aten::mm'], defaults to an empty list
                                               # Note: currently no ops use fusible nodes
                 supports_out=True,  # whether the op supports the out kwarg
                 skips=tuple(),  # information about which tests to skip
                 decorators=None,  # decorators to apply to generated tests
                 safe_casts_outputs=False,  # whether op allows safe casting when writing to out arguments
                 sample_inputs_func=None,  # function to generate sample inputs
                 aten_name=None,  # name of the corresponding aten:: operator
                 aliases=None,  # iterable of aliases, e.g. ("absolute",) for torch.abs
                 variant_test_name='',  # additional string to include in the test name
                 supports_autograd=True,  # support for autograd
                 supports_gradgrad=True,  # support second order gradients (this value is ignored if supports_autograd=False)
                 supports_inplace_autograd=None,  # whether the operation supports inplace autograd
                                                  # defaults to supports_autograd's value
                 supports_sparse=False,  # whether the op supports sparse inputs
                 gradcheck_wrapper=lambda op, *args, **kwargs: op(*args, **kwargs),  # wrapper function for gradcheck
                 check_batched_grad=True,  # check batched grad when doing gradcheck
                 check_batched_gradgrad=True,  # check batched grad grad when doing gradgradcheck
                 gradcheck_nondet_tol=0.0,  # tolerance for nondeterminism while performing gradcheck
                 gradcheck_fast_mode=None,  # Whether to use the fast implmentation for gradcheck/gradgradcheck.
                                            # When set to None, defers to the default value provided by the wrapper
                                            # function around gradcheck (testing._internal.common_utils.gradcheck)
                 ):

            @staticmethod
            @once_differentiable
            def backward(ctx, grad_output):
                self.assertFalse(torch.is_grad_enabled())
                t1, t2 = ctx.saved_tensors
                return (grad_output + grad_output * t2, None,
                        grad_output * ctx.pyscalar + grad_output * t1)

        x, y = self._function_test(MyFunction)
        self.assertEqual(graph_desc(x.grad.grad_fn),
                         'CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))')
        self.assertEqual(graph_desc(y.grad.grad_fn),
                         'CopyBackwards(None, Error(AccumulateGrad(), None, AccumulateGrad()))')

        self.dtypes = set(dtypes)
        self.dtypesIfCPU = set(dtypesIfCPU) if dtypesIfCPU is not None else self.dtypes
        self.dtypesIfCUDA = set(dtypesIfCUDA) if dtypesIfCUDA is not None else self.dtypes
        self.dtypesIfROCM = set(dtypesIfROCM) if dtypesIfROCM is not None else self.dtypesIfCUDA

        self.backward_dtypes = set(backward_dtypes) if backward_dtypes is not None else self.dtypes
        self.backward_dtypesIfCPU = set(backward_dtypesIfCPU) if backward_dtypesIfCPU is not None else (
            self.dtypesIfCPU if dtypesIfCPU is not None else self.backward_dtypes)
        self.backward_dtypesIfCUDA = set(backward_dtypesIfCUDA) if backward_dtypesIfCUDA is not None else (
            self.dtypesIfCUDA if dtypesIfCUDA is not None else self.backward_dtypes)
        self.backward_dtypesIfROCM = set(backward_dtypesIfROCM) if backward_dtypesIfROCM is not None else (
            self.dtypesIfROCM if dtypesIfROCM is not None else self.backward_dtypesIfCUDA)

        self._default_test_dtypes = set(default_test_dtypes) if default_test_dtypes is not None else None

        # NOTE: if the op is unspecified it is assumed to be under the torch namespace
        self.op = op if op else _getattr_qual(torch, self.name)
        method_variant = getattr(torch.Tensor, name, None)
        # attributes like real, imag are not callable
        self.method_variant = method_variant if callable(method_variant) else None
        inplace_name = name + "_"
        self.inplace_variant = getattr(torch.Tensor, inplace_name, None)
        self.operator_variant = getattr(operator, name, None)

        for shape in [(1,), ()]:
            v = torch.ones(shape, requires_grad=True)
            MyFunction.apply(v).backward()
            self.assertEqual(v.grad, torch.full(shape, 2.))

        self.skips = skips
        self.decorators = decorators
        self.sample_inputs_func = sample_inputs_func

    def test_function_returns_undefined_tensor(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

        # autograd support
        self.supports_autograd = supports_autograd
        self.supports_inplace_autograd = supports_inplace_autograd
        if self.supports_inplace_autograd is None:
            self.supports_inplace_autograd = supports_autograd

        self.gradcheck_wrapper = gradcheck_wrapper
        self.supports_gradgrad = supports_gradgrad
        self.check_batched_grad = check_batched_grad
        self.check_batched_gradgrad = check_batched_gradgrad
        self.gradcheck_nondet_tol = gradcheck_nondet_tol
        self.gradcheck_fast_mode = gradcheck_fast_mode

        MyFunction.apply(x).backward()
        self.assertIsNone(x.grad)

        self.aliases = ()
        if aliases is not None:
            self.aliases = tuple(AliasInfo(a) for a in aliases)  # type: ignore[assignment]

        MyFunction.apply(x).sum().backward()
        self.assertIsNone(x.grad)

        self.assertIsNone(torch.autograd.grad(MyFunction.apply(x), x, allow_unused=True)[0])

    def test_materialize_grads(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, grad):
                self.assertEqual(grad, torch.zeros(1))
                return grad

        x = torch.ones(1, requires_grad=True)
        torch._C._functions.UndefinedGrad()(MyFunction.apply(x)).backward()

    def sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        """Returns an iterable of SampleInputs.

        These samples should be sufficient to test the function works correctly
        with autograd, TorchScript, etc.
        """

        # TODO: Remove the try/except once all operators have sample_inputs_func with
        #       **kwargs in their signature.
        try:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad, **kwargs)
        except TypeError:
            samples = self.sample_inputs_func(self, device, dtype, requires_grad)
        return samples

    # Returns True if the test should be skipped and False otherwise
    def should_skip(self, cls_name, test_name, device_type, dtype):
        return any(si.is_active(cls_name, test_name, device_type, dtype)
                   for si in self.skips)

    def test_legacy_function_deprecation_exception(self):
        # Trigger exception
        class MyFunction(Function):
            def forward(self, x):
                return x

    def supported_backward_dtypes(self, device_type):
        if device_type == 'cpu':
            return self.backward_dtypesIfCPU
        if device_type == 'cuda':
            return self.backward_dtypesIfROCM if TEST_WITH_ROCM else self.backward_dtypesIfCUDA
        else:
            return self.backward_dtypes

    def supports_complex_autograd(self, device_type):
        if device_type == 'cpu':
            return any(dtype.is_complex for dtype in self.backward_dtypesIfCPU)
        if device_type == 'cuda':
            if TEST_WITH_ROCM:
                return any(dtype.is_complex for dtype in self.backward_dtypesIfROCM)
            else:
                return any(dtype.is_complex for dtype in self.backward_dtypesIfCUDA)
        else:
            return any(dtype.is_complex for dtype in self.backward_dtypes)

    def supports_dtype(self, dtype, device_type):
        return dtype in self.supported_dtypes(device_type)

    class SimulateBackwardError(Function):

        @staticmethod
        def forward(ctx, input):
            return input.clone()

        @staticmethod
        @once_differentiable
        def backward(ctx, input):
            raise Exception("Simulate error on backward pass")

    def test_custom_function_exception(self):

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

def sample_inputs_unary(op_info, device, dtype, requires_grad, **kwargs):
    low, high = op_info.domain
    low = low if low is None else low + op_info._domain_eps
    high = high if high is None else high - op_info._domain_eps

    return (SampleInput(make_tensor((L,), device=device, dtype=dtype,
                                    low=low, high=high,
                                    requires_grad=requires_grad)),
            SampleInput(make_tensor((), device=device, dtype=dtype,
                                    low=low, high=high,
                                    requires_grad=requires_grad)))

            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale

    def __init__(self,
                 name,  # the string name of the function
                 *,
                 ref,  # a reference function
                 dtypes=floating_types(),
                 dtypesIfCPU=floating_and_complex_types_and(torch.bfloat16),
                 dtypesIfCUDA=floating_and_complex_types_and(torch.half),
                 dtypesIfROCM=None,
                 default_test_dtypes=(
                     torch.uint8, torch.long, torch.half, torch.bfloat16,
                     torch.float32, torch.cfloat),  # dtypes which tests check by default
                 domain=(None, None),  # the [low, high) domain of the function
                 handles_large_floats=True,  # whether the op correctly handles large float values (like 1e20)
                 handles_extremals=True,  # whether the op correctly handles extremal values (like inf)
                 handles_complex_extremals=True,  # whether the op correct handles complex extremals (like inf -infj)
                 supports_complex_to_float=False,  # op supports casting from complex input to real output safely eg. angle
                 sample_inputs_func=sample_inputs_unary,
                 sample_kwargs=lambda device, dtype, input: ({}, {}),
                 supports_sparse=False,
                 **kwargs):
        super(UnaryUfuncInfo, self).__init__(name,
                                             dtypes=dtypes,
                                             dtypesIfCPU=dtypesIfCPU,
                                             dtypesIfCUDA=dtypesIfCUDA,
                                             dtypesIfROCM=dtypesIfROCM,
                                             default_test_dtypes=default_test_dtypes,
                                             sample_inputs_func=sample_inputs_func,
                                             supports_sparse=supports_sparse,
                                             **kwargs)
        self.ref = ref
        self.domain = domain
        self.handles_large_floats = handles_large_floats
        self.handles_extremals = handles_extremals
        self.handles_complex_extremals = handles_complex_extremals
        self.supports_complex_to_float = supports_complex_to_float

        # test_unary_ufuncs.py generates its own inputs to test the consistency
        # of the operator on sliced tensors, non-contig tensors, etc.
        # `sample_kwargs` is a utility function to provide kwargs
        # along with those inputs if required (eg. clamp).
        # It should return two dictionaries, first holding kwarg for
        # torch operator and second one for reference NumPy operator.
        self.sample_kwargs = sample_kwargs

        # Epsilon to ensure grad and gradgrad checks don't test values
        #   outside a function's domain.
        self._domain_eps = 1e-5

def sample_inputs_tensor_split(op_info, device, dtype, requires_grad, **kwargs):
    return (SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(torch.tensor([1, 2, 3]),),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(torch.tensor(1),),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(torch.tensor([1, 2, 3]),),
                        kwargs=dict(dim=1)),)

def sample_inputs_linalg_det(op_info, device, dtype, requires_grad):
    kw = dict(device=device, dtype=dtype)
    inputs = [
        make_tensor((S, S), **kw),
        make_tensor((1, 1), **kw),  # 1x1
        random_symmetric_matrix(S, **kw),  # symmetric
        random_symmetric_psd_matrix(S, **kw),  # symmetric_psd
        random_symmetric_pd_matrix(S, **kw),  # symmetric_pd

        # dim2_null, rank1 and rank2 are disabled because of
        # https://github.com/pytorch/pytorch/issues/53364
        # we should re-enable them once the issue is solved
        # random_square_matrix_of_rank(S, S - 2, **kw),  # dim2_null
        # random_square_matrix_of_rank(S, 1, **kw),  # rank1
        # random_square_matrix_of_rank(S, 2, **kw),  # rank2

        random_fullrank_matrix_distinct_singular_value(S, **kw),  # distinct_singular_value
        make_tensor((3, 3, S, S), **kw),  # batched
        make_tensor((3, 3, 1, 1), **kw),  # batched_1x1
        random_symmetric_matrix(S, 3, **kw),  # batched_symmetric
        random_symmetric_psd_matrix(S, 3, **kw),  # batched_symmetric_psd
        random_symmetric_pd_matrix(S, 3, **kw),  # batched_symmetric_pd
        random_fullrank_matrix_distinct_singular_value(S, 3, 3, **kw),  # batched_distinct_singular_values
        make_tensor((0, 0), **kw),
        make_tensor((0, S, S), **kw),
    ]
    for t in inputs:
        t.requires_grad = requires_grad
    return [SampleInput(t) for t in inputs]

def sample_inputs_linalg_matrix_power(op_info, device, dtype, requires_grad):
    # (<matrix_size>, (<batch_sizes, ...>))
    test_sizes = [
        (1, ()),
        (2, (0,)),
        (2, (2,)),
    ]

        # Validate running backward.
        torch.autograd.backward([res[1].sum(), res[4].sum(), res[6].sum()])
        self.assertIsNotNone(t1.grad)
        self.assertIsNotNone(t2.grad)
        self.assertIsNone(t3.grad)

        # Test gradcheck
        def foo(t1, t2, t3):
            res = MyFunction.apply(t1, t2, scale, t3)
            return res[1], res[4], res[6]

def sample_inputs_hsplit(op_info, device, dtype, requires_grad):
    return (SampleInput(make_tensor((6,), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),)

def sample_inputs_vsplit(op_info, device, dtype, requires_grad):
    return (SampleInput(make_tensor((6, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),
            SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),)

def sample_inputs_dsplit(op_info, device, dtype, requires_grad):
    return (SampleInput(make_tensor((S, S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=([1, 2, 3],),),
            SampleInput(make_tensor((S, S, 6), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(2,),),)

def sample_inputs_linalg_multi_dot(op_info, device, dtype, requires_grad):
    # Each test case consists of the sizes in the chain of multiplications
    # e.g. [2, 3, 4, 5] generates matrices (2, 3) @ (3, 4) @ (4, 5)
    test_cases = [
        [1, 2, 1],
        [2, 0, 2],
        [0, 2, 2],
        [2, 2, 2, 2],
        [2, 3, 4, 5],
        [5, 4, 0, 2],
        [2, 4, 3, 5, 3, 2]
    ]

    result = []
    for sizes in test_cases:
        tensors = []
        for size in zip(sizes[:-1], sizes[1:]):
            t = make_tensor(size, device, dtype, requires_grad=requires_grad)
            tensors.append(t)
        result.append(SampleInput(tensors))

    return result

def sample_inputs_linalg_norm(op_info, device, dtype, requires_grad):
    test_sizes = [
        (S,),
        (0,),
        (S, S),
        (0, 0),
        (S, 0),
        (0, S),
        (S, S, S),
        (0, S, S),
        (S, 0, S),
        (0, 0, 0),
    ]

    def test_custom_function_no_tensors(self):
        class MyFunction(Function):

            @staticmethod
            def forward(ctx, t1, t2, scale, t3):
                t4 = t1 + t2 * t3
                t5 = t1 * t2 + t3
                t4 *= scale
                t5 *= scale
                return scale, t4, None, True, t5, "bar", t1

            @staticmethod
            @once_differentiable
            def backward(ctx, *args):
                return (args[0], args[1], None, args[2])

        t1 = random.random()
        t2 = random.random()
        t3 = random.random()
        scale = random.randint(0, 10)
        res = MyFunction.apply(t1, t2, scale, t3)
        self.assertEqual(scale, res[0])
        self.assertEqual((t1 + t2 * t3) * scale, res[1])
        self.assertEqual(None, res[2])
        self.assertEqual(True, res[3])
        self.assertEqual((t1 * t2 + t3) * scale, res[4])
        self.assertEqual("bar", res[5])
        self.assertEqual(t1, res[6])

    def test_invalid_gradients(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                return torch.randn(10, dtype=torch.float)

        with self.assertRaisesRegex(RuntimeError, 'expected shape'):
            input = torch.randn(5, 5, dtype=torch.float, requires_grad=True)
            MyFunction.apply(input).sum().backward()

    def test_accumulate_grad(self):
        grad_output = torch.ones(5, 5)

        def compute_grad(create_graph):
            x = torch.randn(5, 5, requires_grad=True)
            y = x + 2
            y.backward(grad_output, retain_graph=True)
            x_grad = x.grad
            x_grad_clone = x.grad.clone()
            y.backward(grad_output, create_graph=create_graph)
            return x_grad, x_grad_clone

        # Accumulate in-place when create_graph is False
        x_grad, x_grad_clone = compute_grad(create_graph=False)
        self.assertEqual(x_grad, x_grad_clone * 2)

def sample_inputs_linalg_vector_norm(op_info, device, dtype, requires_grad, **kwargs):
    size_1D = (S,)
    size_2D = (2, 2)

    def test_accumulate_grad_tensor_reference(self):
        def _test_grad_tensor(params_grad_tensor, backward_grad_tensor, should_preserve_reference, create_graph):
            params = torch.tensor([1.5, 1.5]).requires_grad_()
            params.grad = params_grad_tensor
            grad_saved = params.grad
            params.backward(backward_grad_tensor, create_graph=create_graph)
            self.assertEqual(id(grad_saved) == id(params.grad), should_preserve_reference)

        for create_graph in (False, True):
            # Accumulate dense gradient to sparse gradient will change the `params.grad` reference
            _test_grad_tensor(
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                torch.tensor([1.5, 1.5]),
                False,  # never accumulates in-place
                create_graph)

            # Accumulate dense gradient to dense gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.tensor([1.5, 1.5]),
                torch.tensor([1.5, 1.5]),
                not create_graph,
                create_graph)

            # Accumulate sparse gradient to sparse gradient will preserve the `params.grad` reference,
            # but only if create_graph=False.
            _test_grad_tensor(
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                torch.sparse_coo_tensor(torch.tensor([[1, 1]]).long(), torch.tensor([1., 1.])),
                not create_graph,
                create_graph)

# In order to use the kwarg alpha, partials should be used in an OpInfo's sample_inputs_func
# eg. sample_inputs_func=partial(sample_inputs_binary_pwise, alpha=2)
# Then one sample input would also be generated corresponding to the value of alpha provided.
# In the future, kwargs 'alpha_floating', 'alpha_integral' & 'alpha_complex' can be used to
# specify scalars of floating, integral & complex types as values for "alpha".
def sample_inputs_binary_pwise(op_info, device, dtype, requires_grad, **kwargs):
    scalar = 3.14 + 3.14j if dtype.is_complex else (3.14 if dtype.is_floating_point else 3)
    scalar = 1 if dtype is torch.bool else scalar
    tests_list = [
        ((S, S, S), (S, S, S), False),
        ((S, S, S), (S, S), False),
        ((), (), False),
        ((S, S, S), (), False),
        ((S, S, S), scalar, False),
        ((), scalar, False)
    ]
    tests_with_lhs_broadcasting = [
        ((S, S), (S, S, S), True),
        ((), (S, S, S), True),
        ((S, 1, S), (M, S), True),
    ]
    test_cases = tests_list + tests_with_lhs_broadcasting  # type: ignore[operator]
    samples = []
    for first_shape, shape_or_scalar, broadcasts_input in test_cases:
        arg = shape_or_scalar
        if isinstance(shape_or_scalar, tuple):
            arg = make_tensor(shape_or_scalar, device=device, dtype=dtype,
                              requires_grad=requires_grad)
        samples.append(SampleInput(make_tensor(first_shape, device=device, dtype=dtype,
                                               requires_grad=requires_grad),
                                   args=(arg,),
                                   broadcasts_input=broadcasts_input))
    # Adds an extra sample using "alpha" if it's passed in kwargs
    if 'alpha' in kwargs:
        a = make_tensor((S, S, S), device=device, dtype=dtype, requires_grad=requires_grad)
        b = make_tensor((S, S, S), device=device, dtype=dtype, requires_grad=requires_grad)
        sample = SampleInput(a, args=(b,), kwargs={'alpha': kwargs['alpha']})
        samples.append(sample)
    return tuple(samples)

def sample_inputs_mm(op_info, device, dtype, requires_grad, **kwargs):
    args_list = (
        ((S, M), (M, S)),
    )
    inputs = tuple(SampleInput(make_tensor(first_shape, device, dtype,
                                           requires_grad=requires_grad),
                               args=(make_tensor(second_shape, device, dtype,
                                     requires_grad=requires_grad),))
                   for first_shape, second_shape in args_list)
    return inputs

def sample_inputs_addmm(op_info, device, dtype, requires_grad, **kwargs):
    alpha_val = kwargs.get('alpha', 2 + 3j if dtype.is_complex else 0.6)
    beta_val = kwargs.get('beta', 1 + 2j if dtype.is_complex else 0.2)
    tests_list = [
        ((2, 3), (2, 2), (2, 3), False)
    ]
    tests_with_lhs_broadcasting = [
        ((1,), (2, 2), (2, 3), True),
        ((), (2, 2), (2, 3), True)
    ]
    test_cases = tests_list + tests_with_lhs_broadcasting  # type: ignore[operator]
    inputs = tuple(SampleInput(make_tensor(shape_a, device, dtype, requires_grad=requires_grad),
                               args=(make_tensor(shape_b, device, dtype,
                                                 requires_grad=requires_grad),
                                     make_tensor(shape_c, device, dtype,
                                                 requires_grad=requires_grad)),
                               kwargs={'alpha': alpha_val, 'beta': beta_val},
                               broadcasts_input=broadcasts_input)
                   for shape_a, shape_b, shape_c, broadcasts_input in test_cases)
    return inputs

def sample_inputs_mv(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_bmm(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((M, S, M, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((M, M, S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_dot_vdot(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            )
        ),
    )

def sample_inputs_addmv(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (((S,), (S, M), (M,), 1, 1, False),
                  ((S,), (S, M), (M,), 0.2, 0.6, False),
                  )

    test_cases_with_broadcast = (((1,), (S, M), (M,), 1, 1, True),
                                 ((1,), (S, M), (M,), 0.2, 0.6, True),
                                 ((), (S, M), (M,), 1, 1, True),
                                 ((), (S, M), (M,), 0.2, 0.6, True),
                                 )

    cases = test_cases + test_cases_with_broadcast
    sample_inputs = []
    for input_args in cases:
        args = (make_tensor(input_args[0], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(input_args[1], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(input_args[2], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad))
        alpha, beta = input_args[3], input_args[4]
        broadcasts_input = input_args[5]
        sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]), kwargs=dict(beta=beta, alpha=alpha),
                                         broadcasts_input=broadcasts_input))
    return tuple(sample_inputs)

def sample_inputs_addbmm(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [((S, M), (S, S, S), (S, S, M), 1, 1),
                  ((1,), (S, S, S), (S, S, M), 1, 1),
                  ((S, M), (S, S, S), (S, S, M), 0.6, 0.2),
                  ((1,), (S, S, S), (S, S, M), 0.6, 0.2),
                  ((), (S, S, S), (S, S, M), 1, 1),
                  ((), (S, S, S), (S, S, M), 0.6, 0.2),
                  ]
    sample_inputs = []
    for input_args in test_cases:
        args = (make_tensor(input_args[0], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(input_args[1], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(input_args[2], device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad))
        alpha, beta = input_args[3], input_args[4]
        sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]), kwargs=dict(beta=beta, alpha=alpha)))
        if dtype.is_complex:
            sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]),
                                             kwargs=dict(beta=beta * (1 + 2j), alpha=alpha * (2 + 3j))))

    return tuple(sample_inputs)

def sample_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [((S, S), (S, S), (S, S)),
                  ((S, S), (S, 1), (1, S)),
                  ((1,), (S, S, 1), (1, S)),
                  ((), (), ()),
                  ((S, S), (), ()),
                  ((), (S, S, 1), (1, S)),
                  ]

    sample_inputs = []
    for input_args in test_cases:
        args = tuple(make_tensor(arg, device, dtype, requires_grad=requires_grad) if isinstance(arg, tuple) else arg
                     for arg in input_args)
        sample_inputs.append(SampleInput(args[0], args=args[1:]))

        sample_inputs.append(SampleInput(args[0], args=args[1:], kwargs=dict(value=3.14)))

    return tuple(sample_inputs)

def sample_inputs_baddbmm(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = [((S, S, M), (S, S, S), (S, S, M), 1, 1, False),
                  ((1,), (S, S, S), (S, S, M), 1, 1, True),
                  ((S, S, M), (S, S, S), (S, S, M), 0.6, 0.2, False),
                  ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ((), (S, S, S), (S, S, M), 1, 1, True),
                  ((), (S, S, S), (S, S, M), 0.6, 0.2, True),
                  ]
    sample_inputs = []
    for (input_shape, batch1_shape, batch2_shape, alpha, beta, broadcasts_input) in test_cases:
        args = (make_tensor(input_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(batch1_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad),
                make_tensor(batch2_shape, device, dtype,
                            low=None, high=None,
                            requires_grad=requires_grad))
        sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]),
                             kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=broadcasts_input))
        if dtype.is_complex:
            sample_inputs.append(SampleInput(args[0], args=(args[1], args[2]),
                                             kwargs=dict(beta=beta * (1 + 2j), alpha=alpha * (2 + 3j)),
                                             broadcasts_input=broadcasts_input))
    return tuple(sample_inputs)

def sample_inputs_addr(op_info, device, dtype, requires_grad, **kwargs):
    input1 = SampleInput(
        make_tensor((S, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=(
            make_tensor((S, ), device, dtype, low=None, high=None, requires_grad=requires_grad),
            make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad)))

        # test that backward through computation involving sign works
        def sign_mul_logdet(mat):
            s, logdet = mat.slogdet()
            return s * logdet

        u, s, v = a.detach().svd()
        s.abs_().clamp_(0.0001)
        for sign in (-1, 1):
            s[-1] = sign
            mat = torch.linalg.multi_dot([u, s.diag(), v.t()]).requires_grad_()
            gradcheck(sign_mul_logdet, mat)
            gradgradcheck(sign_mul_logdet, mat)

    def test_sum_to_with_empty_dim_grad(self):
        a = torch.rand(4, 0, requires_grad=True)
        b = torch.rand(4, 1, requires_grad=True)
        c = a + b
        assert c.shape == (4, 0)
        c.sum().backward()

        self.assertEqual(b.grad, torch.zeros(4, 1))
        self.assertEqual(a.grad, torch.zeros(4, 0))

    def test_hessian_vector(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)

def sample_inputs_xlogy(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                make_tensor((S, S), device, dtype, low=0, high=None, requires_grad=requires_grad),
            )
        ),
    )


def sample_inputs_xlog1py(self, device, dtype, requires_grad):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def generator():
        # same shape
        yield SampleInput(make_arg((S, S)), args=(make_arg((S, S), low=-1),))
        # rhs broadcast
        yield SampleInput(make_arg((S, S)), args=(make_arg((S,), low=-1),))
        # all zero `x`
        with torch.no_grad():
            x = make_arg((S, S))
            x.fill_(0)
        yield SampleInput(x, args=(make_arg((S, S), low=-1),))

        # randomly zero-masked `x`
        x = make_arg((S, S))
        y = make_arg((S, S), low=-1)
        with torch.no_grad():
            x[torch.rand(x.shape) > 0.5] = 0
        yield SampleInput(x, args=(y,))

        # Scalar x
        # `input` has to be a tensor
        # yield SampleInput(0, args=(make_arg((S, S), low=-1),))
        # yield SampleInput(2.1, args=(make_arg((S, S), low=-1),))

        # Scalar y
        yield SampleInput(make_arg((S, S)), args=(-0.5,))
        yield SampleInput(make_arg((S, S)), args=(1.2,))

    return list(generator())


def sample_inputs_logsumexp(self, device, dtype, requires_grad):
    inputs = (
        ((), (0,), True),
        ((S, S), (1,), True),
        ((S, S), (1,), False)
    )
    samples = []

    for shape, dim, keepdim in inputs:
        t = make_tensor(shape, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, args=(dim, keepdim)))

    return tuple(samples)

def sample_inputs_logcumsumexp(self, device, dtype, requires_grad):
    inputs = (
        ((S, S, S), 0),
        ((S, S, S), 1),
        ((), 0),
    )
    samples = []

    for shape, dim in inputs:
        t = make_tensor(shape, device, dtype,
                        low=None, high=None,
                        requires_grad=requires_grad)
        samples.append(SampleInput(t, args=(dim,)))

    return tuple(samples)

def sample_inputs_trace(self, device, dtype, requires_grad, **kwargs):
    return (SampleInput((make_tensor((S, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad))),)


def sample_inputs_transpose_swapdims(self, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    cases = (((1, 2, 3), (-1, -2)),
             ((1, 2, 3), (-1, 2)),
             ((1, 2, 3), (1, -2)),
             ((1, 2, 3), (1, 2)),
             ((), (0, 0)),
             ((1, ), (0, 0)),
             ((M, M), (0, 1)),
             ((S, S, S), (2, 0)), )

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=args)

    return list(generator())


def sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates always invertible input for linear algebra ops using
    random_fullrank_matrix_distinct_singular_value.
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    def test_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        y = torch.randn(2, 2, requires_grad=True)
        z = x ** 2 + y * x + y ** 2
        z.backward(torch.ones(2, 2), create_graph=True)

        x_grad = 2 * x + y
        y_grad = x + 2 * y
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

def sample_inputs_broadcast_to(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, 1, 1), (S, S, S)),
        ((S, 1, S), (S, S, S)),
        ((S, 1), (S, S, S)),
        ((1,), (S, S, S)),
        ((1, S), (1, 1, S)),
        ((), ()),
        ((), (1, 3, 2)),
    )

        self.assertEqual(x_hv[0], expected_x_hv)
        self.assertEqual(x.grad, x_grad)
        self.assertEqual(y.grad, y_grad)

def sample_inputs_cdist(op_info, device, dtype, requires_grad, **kwargs):
    small_S = 2
    test_cases = (
        ((S, S, 2), (S, S + 1, 2)),
        ((S, S), (S, S)),
        ((S, S, S), (S, S, S)),
        ((3, 5), (3, 5)),
        ((2, 3, 5), (2, 3, 5)),
        ((1, 2, 3), (1, 2, 3)),
        ((1, 1), (S, 1)),
        ((0, 5), (4, 5)),
        ((4, 5), (0, 5)),
        ((0, 4, 5), (3, 5)),
        ((4, 5), (0, 3, 5)),
        ((0, 4, 5), (1, 3, 5)),
        ((1, 4, 5), (0, 3, 5)),
        # Using S here would make this one test take 9s
        ((small_S, small_S, small_S + 1, 2), (small_S, small_S, small_S + 2, 2)),
        ((small_S, 1, 1, small_S), (1, small_S, small_S)),
        ((1, 1, small_S), (small_S, 1, small_S, small_S)),
    )

    samples = []
    for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
        for p in [0, 1, 2, 3, 0.5, 1.5, 2.5, float("inf")]:
            for t1_size, t2_size in test_cases:
                # The args should never be non-contiguous as this is not supported in the backward
                samples.append(SampleInput(
                    make_tensor(t1_size, device, dtype, requires_grad=requires_grad, noncontiguous=False),
                    args=(make_tensor(t2_size, device, dtype, requires_grad=requires_grad, noncontiguous=False), p, cm)))

    return samples

def sample_inputs_comparison_ops(self, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, S, S), (S, S, S), False),
        ((S, S, S), (), False),
        ((S, S, S), (1,), False),
        ((S,), (1,), False),
        ((), (), False),
    )
    test_cases_lhs_broadcasting = (
        ((S, 1, S), (S, S, S), True),
        ((1,), (S, S, S), True),
        ((1, S), (1, 1, S), True),
        ((), (0,), True),
        ((), (S, S, S), True),
    )
    cases = test_cases + test_cases_lhs_broadcasting
    sample_inputs = list(SampleInput(make_tensor(first_shape, device, dtype,
                                                 requires_grad=requires_grad),
                                     args=(make_tensor(second_shape, device, dtype,
                                                       requires_grad=requires_grad),),
                                     broadcasts_input=broadcasts_input)
                         for first_shape, second_shape, broadcasts_input in cases)
    equal_tensors_non_bool = (
        ([[[-8, 6], [9, 0]], [[0, 5], [5, 7]]]),
        ([[[6, 5]], [[1, -5]]]),
        ([[2], [-1]]),
        ([0, -6]),
        ([3],),
    )
    equal_tensors_bool = (
        ([[[1, 0], [0, 0]], [[0, 1], [1, 0]]]),
        ([[[1, 1]], [[1, 0]]]),
        ([[1], [0]]),
        ([0, 1]),
        ([1],),
    )
    more_cases = equal_tensors_bool if dtype is torch.bool else equal_tensors_non_bool
    more_inputs = list(SampleInput(torch.tensor(elements, device=device, dtype=dtype,
                                                requires_grad=requires_grad),
                                   args=(torch.tensor(elements, device=device, dtype=dtype,
                                                      requires_grad=requires_grad),))
                       for elements in more_cases)
    sample_inputs = [*sample_inputs, *more_inputs]
    return tuple(sample_inputs)

def sample_inputs_div(self, device, dtype, requires_grad, rounding_mode=None, **kwargs):
    a = make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    is_integral = not dtype.is_floating_point and not dtype.is_complex
    b = make_tensor((S, S, S), device, dtype, low=1 if is_integral else 0.1, high=None,
                    requires_grad=requires_grad)

    kwargs = None  # type: ignore[assignment]
    if rounding_mode is not None:
        kwargs = dict(rounding_mode=rounding_mode)

        def fn(x):
            return x ** 2 + y * x + y ** 2

def sample_inputs_stack(op_info, device, dtype, requires_grad, **kwargs):
    tensors = [
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
    ]

    return (SampleInput(tensors, args=(0,)),)

def sample_inputs_hstack_dstack_vstack(op_info, device, dtype, requires_grad, **kwargs):
    tensors = [
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
    ]

    return (SampleInput(tensors),)

def sample_inputs_hypot(op_info, device, dtype, requires_grad):
    input = make_tensor((S, S), device, dtype, requires_grad=requires_grad)
    args = make_tensor((S, S), device, dtype, requires_grad=requires_grad)

    return (
        SampleInput(input, args=(args,)),
    )

def sample_inputs_gather(op_info, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, gather_variable((S, S), 1, M, True, device=device))),
        SampleInput(
            make_tensor((M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(1, gather_variable((M, S // 2), 0, S, True, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor([0], dtype=torch.int64, device=device))),
        SampleInput(
            make_tensor((S,), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor(0, dtype=torch.int64, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor(0, dtype=torch.int64, device=device))),
    )


def sample_inputs_take_along_dim(op_info, device, dtype, requires_grad, **kwargs):
    return (SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S), 1, S, True, device=device), 0)),

            # `indices` broadcast
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((1, S // 2), 0, S, True, device=device), 1)),

            # `self` broadcast
            SampleInput(make_tensor((1, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device), 1)),

            # without `dim` arg
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device), )),
            SampleInput(make_tensor((S, S), device, dtype,
                                    low=None, high=None,
                                    requires_grad=requires_grad),
                        args=(gather_variable((S, S // 2), 0, S, True, device=device),)),
            )

def sample_inputs_amax_amin(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((S, S, S), ()),
        ((S, S, S), (1,)),
        ((S, S, S), ((1, 2,),)),
        ((S, S, S), (1, True,)),
        ((), (0,)),
        ((), ()),
        ((), (0, True,)),
    )
    return tuple(SampleInput((make_tensor(size, device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad)),
                             args=args)
                 for size, args in test_cases)

def sample_inputs_argmax_argmin(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((2, 2, 2), ()),
        ((2, 2, 2), (0,)),
        ((2, 2, 2), (1,)),
        ((2, 2, 2), (2,)),
        ((2, 2, 2), (2, True,)),
        ((2, 2, 2), (None,)),
        ((), (0,)),
        ((), ()),
        ((), (None, True,)),
        ((1,), ()),
        ((1,), (0,)),
        ((1,), (0, True)),
        ((2,), ()),
        ((2,), (0,)),
        ((2,), (0, True)),
        ((2, 2, 3), ()),
        ((2, 2, 3), (0,)),
        ((2, 2, 3), (1,)),
        ((2, 2, 3), (None, True)),
    )
    return tuple(SampleInput((make_tensor(size, device, dtype,
                                          requires_grad=requires_grad)),
                             args=args)
                 for size, args in test_cases)

def sample_inputs_diff(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((1,), 0, None, None),
        ((S,), 0, None, None),
        ((S, 1), 0, None, None),
        ((S, 1), 1, None, None),
        ((S, S), 0, None, None),
        ((S, S), 1, None, None),
        ((S, S), 0, (1, S), (2, S)),
        ((S, S), 0, None, (2, S)),
        ((S, S, S), 1, None, None),
        ((S, S, S), 1, (S, 1, S), (S, 1, S)),)

        def hook(*grads):
            hook_called[0] = True
        hook_called = [False]
        x.register_hook(hook)

        go = torch.randn(2, 2)
        grad_a, grad_b = torch.autograd.grad(
            (a + 2 * b), [a, b], grad_outputs=go, create_graph=True)

def sample_inputs_index_select(op_info, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, index_variable(2, S, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor([0], dtype=torch.int64, device=device))),
        SampleInput(
            make_tensor((), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(0, torch.tensor(0, dtype=torch.int64, device=device))),
    )

def sample_inputs_getitem(op_info, device, dtype, requires_grad, **kwargs):
    test_args = [
        (dont_convert([1, 2]),),
        (slice(0, 3),),
        (dont_convert([slice(0, 3), 1]),),
        (dont_convert([[0, 2, 3], [1, 3, 3], [0, 0, 2]]),),
        (dont_convert([[0, 0, 3], [1, 1, 3], [0, 0, 2]]),),
        (dont_convert([slice(None), slice(None), [0, 3]]),),
        (dont_convert([slice(None), [0, 3], slice(None)]),),
        (dont_convert([[0, 3], slice(None), slice(None)]),),
        (dont_convert([[0, 3], [1, 2], slice(None)]),),
        (dont_convert([[0, 3], ]),),
        (dont_convert([[0, 3], slice(None)]),),
        (dont_convert([[0, 3], Ellipsis]),),
        (dont_convert([[0, 2, 3], [1, 3, 3], torch.LongTensor([0, 0, 2])]),),
        (index_variable(2, S, device=device),),
        (mask_not_all_zeros((S,)),),
    ]

    return tuple(SampleInput(
        make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
        args=args)
        for args in test_args)

def sample_inputs_index_put(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    for accumulate in [False, True]:
        # Test with indices arg
        inputs.append(SampleInput(
            make_tensor((S, S,), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                (index_variable(2, S, device=device), ),
                make_tensor((2, S), device, dtype, low=None, high=None)),
            kwargs=dict(accumulate=accumulate)))

        # Test with mask arg
        mask = torch.zeros(S, dtype=torch.bool) if accumulate else mask_not_all_zeros((S,))
        inputs.append(SampleInput(
            make_tensor((S, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=(
                (mask, ),
                make_tensor((S,), device, dtype, low=None, high=None),),
            kwargs=dict(accumulate=accumulate)))

    return inputs

# Missing to test the nondeterminism of the operation
# https://github.com/pytorch/pytorch/issues/53352
def sample_inputs_index_add(op_info, device, dtype, requires_grad, **kwargs):
    # These testa are pretty much the same as those from index_copy.
    # Perhaps merge?
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

        x0 = x_list[0]
        hook_results = [None]

    idx = make_arg((S,), dtype=torch.int64, low=0, high=S)
    idx_nonctg = make_arg((S,), dtype=torch.int64, low=0, high=S, noncontiguous=True)
    samples = [SampleInput(tensor, args=(1, idx, source))
               for tensor, idx, source in product([t, t_nonctg], [idx, idx_nonctg], [s, s_nonctg])]
    samples.extend(SampleInput(tensor, args=(1, idx, source), kwargs=dict(alpha=a))
                   for tensor, idx, source, a in product([t, t_nonctg], [idx, idx_nonctg], [s, s_nonctg], [-1, 0, 2]))

        x_list[0].backward()
        self.assertEqual(hook_results[0], torch.tensor(1.))
        expected_grad = torch.tensor([1., 0, 0, 0, 0])
        self.assertEqual(x.grad, expected_grad)
        self.assertIsNone(x_list[0].grad)

    samples.extend(SampleInput(t, args=(0, idx, s)) for t, idx, s in product(ts, idxs, ss))
    samples.extend(SampleInput(t, args=(0, idx, s), kwargs=dict(alpha=a)) for t, idx, s, a in product(ts, idxs, ss, [-1, 0, 2]))
    return samples

def sample_inputs_sort(op_info, device, dtype, requires_grad, **kwargs):
    def apply_grad(t):
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            t.requires_grad_(requires_grad)

        x = torch.randn(5, requires_grad=True).clone()
        x.register_hook(MyHookClass())
        x.sum().backward()
        # Should run fine

    def large_1d_unique(dtype, device):
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype)
        apply_grad(res)
        return res

    samples = []
    # Test case for large tensor.
    largesample = SampleInput(large_1d_unique(dtype, device))
    samples.append(largesample)

        # define a helper for dividing intermediates into groups
        def group(l, group_size):
            return (l[i:i + group_size] for i in range(0, len(l), group_size))

    # Test cases for scalar tensor
    scalar = torch.tensor(1, dtype=dtype, device=device)
    apply_grad(scalar)
    samples.append(SampleInput(scalar))
    samples.append(SampleInput(scalar, args=(0,)))
    samples.append(SampleInput(scalar, args=(0, True)))
    # no CUDA support for stable sort yet
    if not device.startswith('cuda'):
        samples.append(SampleInput(scalar, kwargs=dict(stable=True)))
        samples.append(SampleInput(scalar, kwargs=dict(dim=0, stable=True)))
        samples.append(SampleInput(scalar, kwargs=dict(dim=0, descending=True, stable=True)))
    return samples

def sample_inputs_index_fill(op_info, device, dtype, requires_grad, **kwargs):
    samples = []
    t = make_tensor((S, S, S), device, dtype,
                    low=None, high=None,
                    requires_grad=requires_grad)
    fill_val = torch.tensor(-1 + 1j if t.is_complex() else -1)
    # non-contiguous input
    t01 = t.transpose(0, 1)
    t02 = t.transpose(0, 2)
    t12 = t.transpose(1, 2)
    idx = index_variable(1, S, device=device)
    # non-contiguous index
    idx_nonctg = torch.empty_strided((S,), (2,), device=device, dtype=torch.int64)
    idx_nonctg.copy_(idx)
    for d in range(t.dim()):
        for tensor in [t, t01, t02, t12]:
            samples.append(SampleInput(tensor, args=(d, idx, fill_val)))
            samples.append(SampleInput(tensor, args=(d, -idx - 1, fill_val)))
            samples.append(SampleInput(tensor, args=(d, idx_nonctg, fill_val)))

    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    index_tensor = partial(torch.tensor, device=device, dtype=torch.long)

    def unique_idx(numel, max_idx):
        # Generate unique random indices vector of `numel`
        # elements in range [0, max_idx).
        indices = random.sample(range(max_idx), numel)
        return index_tensor(indices)

    samples.append(SampleInput(make_arg((S, S)), args=(0, unique_idx(2, S), 2)))
    samples.append(SampleInput(make_arg((S, S)), args=(0, unique_idx(2, S), make_arg(()))))
    samples.append(SampleInput(make_arg((S, S)), args=(0, index_tensor(0), 2)))
    samples.append(SampleInput(make_arg(()), args=(0, index_tensor([0]), 2)))
    samples.append(SampleInput(make_arg(()), args=(0, index_tensor(0), 2)))

    # Duplicate indices
    samples.append(SampleInput(make_arg((S, S)), args=(0, index_tensor([0, 0]), 2)))
    samples.append(SampleInput(make_arg((S, S)), args=(0, index_tensor([0, 0, 2]), make_arg(()))))

    return samples

def sample_inputs_max_min_binary(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    args_for_binary_op = (
        ((S, S, S), (S, S, S),),
        ((S, S, S), (S,),),
        ((S,), (S, S, S),),
        ((S, 1, S), (S, S),),
        ((S, S), (S, S),),
        ((), (),),
        ((S, S, S), (),),
        ((), (S, S, S),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=(make_tensor(other_tensor, device, dtype,
                                                 low=None, high=None,
                                                 requires_grad=requires_grad),),))
                  for input_tensor, other_tensor in args_for_binary_op)
    return inputs

def sample_inputs_max_min_reduction_with_dim(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    args_for_reduction_with_dim = (
        ((S, S, S), (1,),),
        ((S, S, S), (1, True, ),),
        ((), (0,),),
        ((), (0, True,),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=args,))
                  for input_tensor, args in args_for_reduction_with_dim)
    return inputs

def sample_inputs_max_min_reduction_no_dim(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    inputs.append(SampleInput(make_tensor((S, S, S), device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad),))
    inputs.append(SampleInput(make_tensor((), device, dtype,
                                          low=None, high=None,
                                          requires_grad=requires_grad),))
    return inputs

# Generates input tensors for testing reduction ops
def _generate_reduction_inputs(device, dtype, requires_grad):
    yield make_tensor((), device, dtype, requires_grad=requires_grad)
    yield make_tensor((2,), device, dtype, requires_grad=requires_grad)
    yield make_tensor((2, 3), device, dtype, requires_grad=requires_grad, noncontiguous=True)
    yield make_tensor((3, 2, 1, 2, 2), device, dtype, requires_grad=requires_grad)

# Generates a subset of possible dim and keepdim kwargs for a tensor
# with ndim dims appropriate for testing. If supports_multiple_dims
# is True (default) then dim kwarg can be a list of dims.
def _generate_reduction_kwargs(ndim, supports_multiple_dims=True):
    for keepdim in [True, False]:
        # Always test reducing inner and outer most dimensions
        yield {'dim': 0, 'keepdim': keepdim}
        yield {'dim': -1, 'keepdim': keepdim}

        # Also reduce middle dimension
        if ndim > 2:
            yield {'dim': ndim // 2, 'keepdim': keepdim}

        if supports_multiple_dims:
            # Always test reducing all dims
            yield {'dim': tuple(range(ndim)), 'keepdim': keepdim}

            # Test reducing both first and last dimensions
            if ndim > 1:
                yield {'dim': (0, ndim - 1), 'keepdim': keepdim}

            # Test reducing every other dimension starting with the second
            if ndim > 3:
                yield {'dim': tuple(range(1, ndim, 2)), 'keepdim': keepdim}

# Wraps sample_inputs_reduction function to provide the additional supports_multiple_dims args
def sample_inputs_reduction_wrapper(supports_multiple_dims):
    # Generates sample inputs for reduction ops that contain the input tensor
    # and dim and keepdim kwargs. If a reduction op needs to test additional
    # args/kwargs then create a separate sample_inputs function
    def fn(op_info, device, dtype, requires_grad):
        inputs = []

        for t in _generate_reduction_inputs(device, dtype, requires_grad):
            # Add case without dim and keepdim kwargs
            inputs.append(SampleInput(t))
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims):
                inputs.append(SampleInput(t, kwargs=kwargs))

        return inputs

    return fn

def sample_inputs_reduction_quantile(op_info, device, dtype, requires_grad):
    test_quantiles = (0.5, make_tensor((2,), device, dtype, low=0, high=1))
    test_interpolations = ['linear', 'midpoint']

    inputs = []
    for quantiles in test_quantiles:
        for t in _generate_reduction_inputs(device, dtype, requires_grad):
            # Add case without dim and keepdim kwargs
            inputs.append(SampleInput(t, args=(quantiles,)))
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims=False):
                # Interpolation kwarg for now is only supported when providing both dim and keepdim
                for interpolation in test_interpolations:
                    kwargs['interpolation'] = interpolation
                    inputs.append(SampleInput(t, args=(quantiles,), kwargs=kwargs))

    return inputs

def sample_inputs_topk(op_info, device, dtype, requires_grad, **kwargs):
    def get_tensor_input(size):
        return make_tensor(size, device, dtype, requires_grad=requires_grad)

    inputs = []
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3,)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, 1, True, True)))
    inputs.append(SampleInput(get_tensor_input((S, M, S)), args=(3, -2, True, True)))

    inputs.append(SampleInput(get_tensor_input(()), args=(1,)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, 0, True, True)))
    inputs.append(SampleInput(get_tensor_input(()), args=(1, -1, True, True)))

    return inputs

def sample_inputs_outer(op_info, device, dtype, requires_grad, **kwargs):
    inputs = []
    arg_a = make_tensor((S,), device, dtype, requires_grad=requires_grad)
    arg_b = make_tensor((M,), device, dtype, requires_grad=requires_grad)
    inputs.append(SampleInput(arg_a, args=(arg_b,)))
    return inputs

def sample_inputs_dist(op_info, device, dtype, requires_grad):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    sizes = ((S, S, S), (S,), (S, 1, S), (), (S, S))
    ps = (2, 4)

    def generate_samples():
        for size_x, size_y, p in product(sizes, sizes, ps):
            yield SampleInput(make_arg(size_x), args=(make_arg(size_y), p))

    return list(generate_samples())

# Missing to test the nondeterminism of the operation
# https://github.com/pytorch/pytorch/issues/53352
def sample_inputs_index_copy(op_info, device, dtype, requires_grad, **kwargs):
    def make_arg(shape, low=None, high=None, dtype=dtype):
        return make_tensor(shape, device=device, dtype=dtype,
                           low=low, high=high,
                           requires_grad=requires_grad)

    def test_grad_unreachable(self):
        x = torch.ones(1, requires_grad=True)
        y = torch.ones(1, requires_grad=True)
        # Make sure x and y have grad accumulators allocated
        z = x * 2
        w = y * 2

        grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_y)

        # This is slightly different than the case above, because z doesn't even
        # have a grad accumulator allocated.
        z = torch.ones(1, requires_grad=True)
        grad_x, grad_z = torch.autograd.grad(x * 2, [x, z], allow_unused=True)
        self.assertEqual(grad_x, x * 2)
        self.assertIsNone(grad_z)

        # allow_unused=False, but grads contains None inside, should throw
        with self.assertRaisesRegex(RuntimeError,
                                    "Set allow_unused=True"):
            grad_x, grad_y = torch.autograd.grad(x * 2, [x, y], allow_unused=False)

def sample_inputs_mode(op_info, device, dtype, requires_grad):
    inputs = []
    args = (
        ((S, S, S), (),),
        ((S, S, S), (1, ),),
        ((S, S, S), (1, True, ),),
        ((), (),),
        ((), (0,),),
        ((), (0, True,),),
    )
    inputs = list((SampleInput(make_tensor(input_tensor, device, dtype,
                                           low=None, high=None,
                                           requires_grad=requires_grad),
                               args=args,))
                  for input_tensor, args in args)
    return inputs

# Missing to test the nondeterminism of the operation
# https://github.com/pytorch/pytorch/issues/53352
def sample_inputs_put(op_info, device, dtype, requires_grad):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)

    S = 3

    def gen_inputs():
        # Generic inputs
        tgt_gen = (make_arg((S, S), noncontiguous=not ctg) for ctg in (True, False))
        src_gen = (make_arg((S,), noncontiguous=not ctg) for ctg in (True, False))
        idx = torch.randperm(S * S, device=device, dtype=torch.int64)[:S]
        idx_nonctg = torch.repeat_interleave(idx, 2, dim=-1)[::2]
        idx_neg = -idx - 1
        idx_list = [idx, idx_nonctg, idx_neg]
        for tgt, idx, src, acc in product(tgt_gen, idx_list, src_gen, (True, False)):
            yield SampleInput(input=tgt, args=(idx, src, acc))

        # Scalar cases
        scalar_sizes = [(), (1,)]
        tgt_gen = (make_arg(size) for size in scalar_sizes)
        idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
        src_gen = (make_arg(size) for size in scalar_sizes)
        for tgt, idx, src, acc in product(tgt_gen, idx_gen, src_gen, (True, False)):
            yield SampleInput(input=tgt, args=(idx, src, acc))

        # Empty cases
        tgt_sizes = [(0,), (), (1,), (3, 2)]
        tgt_gen = (make_arg(size) for size in tgt_sizes)
        idx = make_idx((0,), high=1)
        src = make_arg((0,))
        for tgt, acc in product(tgt, (True, False)):
            yield SampleInput(input=tgt, args=(idx, src, acc))

    return list(gen_inputs())

def sample_inputs_take(op_info, device, dtype, requires_grad):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)

    S = 3

    def gen_inputs():
        # Generic inputs: take S elements out of S * S
        src_gen = (make_arg((S, S), noncontiguous=not ctg) for ctg in (True, False))
        idx = make_idx((S,), high=S * S)
        idx_nonctg = make_idx((S,), high=S * S, noncontiguous=True)
        idx_neg = -idx - 1
        idx_list = [idx, idx_nonctg, idx_neg]
        for src, idx in product(src_gen, idx_list):
            yield SampleInput(input=src, args=(idx,))

        # Scalar cases
        scalar_sizes = [(), (1,)]
        src_gen = (make_arg(size) for size in scalar_sizes)
        idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
        for src, idx in product(src_gen, idx_gen):
            yield SampleInput(input=src, args=(idx,))

        # Empty cases
        src_sizes = [(0,), (), (1,), (3, 2)]
        src_gen = (make_arg(size) for size in src_sizes)
        idx = make_idx((0,), high=1)
        for src in src_gen:
            yield SampleInput(input=src, args=(idx,))

    return list(gen_inputs())

def sample_movedim_moveaxis(op_info, device, dtype, requires_grad):
    return (
        SampleInput(
            make_tensor((4, 3, 2, 1), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=((0, 1, 2, 3), (3, 2, 1, 0))),
        SampleInput(
            make_tensor((4, 3, 2, 1), device, dtype, low=None, high=None, requires_grad=requires_grad),
            args=((0, -1, -2, -3), (-3, -2, -1, -0)))
    )

            @staticmethod
            def backward(ctx, x):
                self.fail("This node should not be executed!")

def sample_repeat_tile(op_info, device, dtype, requires_grad, **kwargs):
    rep_dims = ((), (0, ), (1, ), (0, 2), (1, 1), (2, 3), (2, 3, 2), (0, 2, 3), (2, 1, 1, 1),)
    shapes = ((), (0,), (2,), (3, 0), (3, 2), (3, 0, 1))

    if requires_grad:
        # Tests for variant_consistency_jit, grad, gradgrad
        # are slower. Use smaller bags of `rep_dims` and `shapes`
        # in this case.
        rep_dims = ((), (0, ), (0, 2), (1, 1), (2, 3), (1, 3, 2), (3, 1, 1))  # type: ignore[assignment]
        shapes = ((), (0,), (2,), (3, 2))  # type: ignore[assignment]

        x = MyFunc.apply(torch.randn(1, requires_grad=True) * 2)
        y = torch.randn(1, requires_grad=True)
        torch.autograd.backward(x, inputs=(y, ))  # allow_unused is implicitly True!
        self.assertIsNone(y.grad)

    def test_hooks(self):
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5) * 4
        y.requires_grad_(True)

        counter = [0]

def sample_unsqueeze(op_info, device, dtype, requires_grad, **kwargs):
    shapes_and_axes = [
        ((3, 4, 5), 0),
        ((3, 4, 5), 1),
        ((3, 4, 5), 3),
        ((3, 4, 5), -1),
        ((3, 4, 5), -3),
        ((), 0)
    ]

    samples = []
    for shape, axis in shapes_and_axes:
        tensor = make_tensor(shape, device, dtype, low=None, high=None,
                             requires_grad=requires_grad)
        samples.append(SampleInput(tensor, args=(axis,),))

    return samples

# TODO: reconcile with torch.linalg.det and torch.linalg.slogdet
# Creates matrices with a positive nonzero determinant
def sample_inputs_logdet(op_info, device, dtype, requires_grad, **kwargs):
    def make_nonzero_det(A, *, sign=1, min_singular_value=0.1, **kwargs):
        u, s, v = A.svd()
        s.clamp_(min=min_singular_value)
        A = torch.matmul(u, torch.matmul(torch.diag_embed(s), v.transpose(-2, -1)))
        det = A.det()
        if sign is not None:
            if A.dim() == 2:
                det = det.item()
                if (det < 0) ^ (sign < 0):
                    A[0, :].neg_()
            else:
                cond = ((det < 0) ^ (sign < 0)).nonzero()
                if cond.size(0) > 0:
                    for i in range(cond.size(0)):
                        A[list(cond[i])][0, :].neg_()
        return A

    samples = []

    # cases constructed using make_tensor()
    tensor_shapes = (
        (S, S),
        (1, 1),
        (3, 3, S, S),
        (3, 3, 1, 1)
    )

    for shape in tensor_shapes:
        t = make_tensor(shape, device=device, dtype=dtype)
        d = make_nonzero_det(t).requires_grad_(requires_grad)
        samples.append(SampleInput(d))

    # cases constructed using:
    #  1) make_symmetric_matrices
    #  2) make_symmetric_pd_matrices
    #  3) make_fullrank_matrices_with_distinct_singular_values
    symmetric_shapes = (
        (S, S),
        (3, S, S),
    )


    def _helper(constructor, *shape, **kwargs):
        t = constructor(*shape, device=device, dtype=dtype)
        d = make_nonzero_det(t, **kwargs).requires_grad_(requires_grad)
        samples.append(SampleInput(d))

    for shape in symmetric_shapes:
        _helper(make_symmetric_matrices, *shape)
        _helper(make_symmetric_pd_matrices, *shape)
        _helper(make_fullrank_matrices_with_distinct_singular_values, *shape, min_singular_value=0)

    return tuple(samples)

def np_unary_ufunc_integer_promotion_wrapper(fn):
    # Wrapper that passes PyTorch's default scalar
    #   type as an argument to the wrapped NumPy
    #   unary ufunc when given an integer input.
    #   This mimicks PyTorch's integer->floating point
    #   type promotion.
    #
    # This is necessary when NumPy promotes
    #   integer types to double, since PyTorch promotes
    #   integer types to the default scalar type.

        z = x ** 2 + x * 2 + x * y + y
        x.register_hook(lambda *args: bw_hook(0, *args))
        test = z.register_hook(lambda *args: bw_hook(1, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 1)

        test2 = z.register_hook(lambda *args: bw_hook(2, *args))
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 4)

        test2.remove()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(counter[0], 5)

        def bw_hook_modify(grad):
            return grad.mul(2)

        test.remove()
        z.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5), retain_graph=True)
        self.assertEqual(y.grad, (x + 1) * 2)

        y.register_hook(bw_hook_modify)
        with torch.no_grad():
            y.grad.zero_()
        z.backward(torch.ones(5, 5))
        self.assertEqual(y.grad, (x + 1) * 4)

    def test_hooks_cpp(self):
        # Tests hooks for autograd function implemented in C++
        bn = torch.nn.BatchNorm1d(5, affine=False)
        bn.double()
        bn.eval()

        counter = [0]

        def bw_hook(grad):
            counter[0] += 1
            return grad * 2

    def sample_inputs(self, device, dtype, requires_grad=False, **kwargs):
        nd_tensor = make_tensor((S, S + 1, S + 2), device, dtype, low=None, high=None,
                                requires_grad=requires_grad)
        tensor = make_tensor((31,), device, dtype, low=None, high=None,
                             requires_grad=requires_grad)

        self.assertEqual(counter[0], 1, msg='bw_hook not called')
        self.assertEqual(x.grad, torch.ones(5, 5, dtype=torch.double) * 2, atol=1e-5, rtol=0)

    def test_hook_none(self):
        # WARNING: this is a test for autograd internals.
        # You should never have to use such things in your code.
        class NoneGradientFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                assert ctx.needs_input_grad[0]
                assert not ctx.needs_input_grad[1]
                return x, y

            @staticmethod
            def backward(ctx, grad_x, grad_y):
                return grad_x, None

        was_called = [False]

        def hook(grad):
            self.assertIsNotNone(grad)
            was_called[0] = True

        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5, 5)
        rx, ry = NoneGradientFunction.apply(x, y)
        rx.register_hook(hook)
        ry.register_hook(hook)
        sum(rx, ry).sum().backward()
        self.assertTrue(was_called[0])

    def test_retain_grad(self):
        input = torch.rand(1, 3, requires_grad=True)
        h1 = input * 3
        out = (h1 * h1).sum()

        # It should be possible to call retain_grad() multiple times
        h1.retain_grad()
        h1.retain_grad()

        # Gradient should be accumulated
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 2, h1.grad)
        out.backward(retain_graph=True)
        self.assertEqual(h1 * 4, h1.grad)

        with torch.no_grad():
            input.grad.zero_()
        # It should be a no-op for leaves
        input.retain_grad()
        input.retain_grad()
        out.backward()
        self.assertEqual(input * 18, input.grad)

    def test_retain_grad_cycle(self):
        x = torch.ones(5, 5, requires_grad=True)

        def run_test():
            y = x * 2
            y.retain_grad()

            return y / 2, torch._C._WeakTensorRef(y)


def sample_inputs_linalg_cholesky_inverse(op_info, device, dtype, requires_grad=False):
    # Generate Cholesky factors of positive-definite (non-singular) Hermitian (symmetric) matrices
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix
    inputs = (
        torch.zeros(0, 0, dtype=dtype, device=device),  # 0x0 matrix
        torch.zeros(0, 2, 2, dtype=dtype, device=device),  # zero batch of matrices
        random_hermitian_pd_matrix(S, dtype=dtype, device=device),  # single matrix
        random_hermitian_pd_matrix(S, 2, dtype=dtype, device=device),  # batch of matrices
    )
    test_cases = (torch.linalg.cholesky(a) for a in inputs)
    out = []
    for a in test_cases:
        a.requires_grad = requires_grad
        out.append(SampleInput(a))
        out.append(SampleInput(a, kwargs=dict(upper=True)))
    return out

def sample_inputs_linalg_lstsq(op_info, device, dtype, requires_grad=False, **kwargs):
    from torch.testing._internal.common_utils import random_well_conditioned_matrix
    out = []
    for batch in ((), (3,), (3, 3)):
        shape = batch + (3, 3)
        # NOTE: inputs are not marked with `requires_grad` since
        # linalg_lstsq is not differentiable
        a = random_well_conditioned_matrix(*shape, dtype=dtype, device=device)
        b = make_tensor(shape, device, dtype, low=None, high=None)
        out.append(SampleInput(a, args=(b,)))
    return out

def sample_inputs_householder_product(op_info, device, dtype, requires_grad, **kwargs):
    """
    This function generates input for torch.linalg.householder_product (torch.orgqr).
    The first argument should be a square matrix or batch of square matrices, the second argument is a vector or batch of vectors.
    Empty, square, rectangular, batched square and batched rectangular input is generated.
    """
    # Each column of the matrix is getting multiplied many times leading to very large values for
    # the Jacobian matrix entries and making the finite-difference result of grad check less accurate.
    # That's why gradcheck with the default range [-9, 9] fails and [-2, 2] is used here.
    samples = (
        SampleInput(make_tensor((S, S), device, dtype, low=-2, high=2, requires_grad=requires_grad),
                    args=(make_tensor((S,), device, dtype, low=-2, high=2, requires_grad=requires_grad),)),

        for _ in range(10):
            CollectOnDelete().forward(torch.randn(1, requires_grad=True)).backward()

    # Delete this test when legacy custom autograd functions are deleted.
    def test_naughty_legacy_variable_grad_fn(self):
        class Id(Function):
            def forward(self, x):
                return x

            def backward(self, grad_x):
                return grad_x

        self.assertRaises(RuntimeError, lambda: Variable(torch.zeros(1), _grad_fn=Id()))

    # Delete this test when legacy custom autograd functions are deleted.
    def test_naughty_legacy_function_backward_before_forward(self):
        class Id(Function):
            def forward(self, x):
                return x

            def backward(self, grad_x):
                return grad_x

def sample_inputs_linalg_cholesky(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates always positive-definite input for torch.linalg.cholesky using
    random_hermitian_pd_matrix.
    The input is generated as the itertools.product of 'batches' and 'ns'.
    In total this function generates 8 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices,
        (1, 1) - 1x1 batch of matrices
    'ns' gives 0x0 and 5x5 matrices.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    """
    from torch.testing._internal.common_utils import random_hermitian_pd_matrix

    # Delete this test when legacy custom autograd functions are deleted.
    def test_naughty_legacy_function_early_access(self):
        class Id(Function):
            def forward(self, x):
                return x

def sample_inputs_symeig(op_info, device, dtype, requires_grad=False):
    out = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)

    for o in out:
        o.kwargs = {"upper": bool(np.random.choice([True, False])),
                    "eigenvectors": True}
        # A gauge-invariant function
        o.output_process_fn_grad = lambda output: (output[0], abs(output[1]))
    return out


def sample_inputs_linalg_eigh(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.eigh with UPLO="U" or "L" keyword argument.
    """
    def out_fn(output):
        return output[0], abs(output[1])

    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.kwargs = {"UPLO": np.random.choice(["L", "U"])}
        sample.output_process_fn_grad = out_fn

    return samples


def sample_inputs_linalg_slogdet(op_info, device, dtype, requires_grad=False):
    def out_fn(output):
        return output[1]

    samples = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad)
    for sample in samples:
        sample.output_process_fn_grad = out_fn

    return samples


def sample_inputs_linalg_pinv_hermitian(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.pinv with hermitian=True keyword argument.
    """
    out = sample_inputs_linalg_invertible(op_info, device, dtype, requires_grad, **kwargs)
    for o in out:
        o.kwargs = {"hermitian": True}
    return out

def sample_inputs_linalg_solve(op_info, device, dtype, requires_grad=False, vector_rhs_allowed=True, **kwargs):
    """
    This function generates always solvable input for torch.linalg.solve
    Using random_fullrank_matrix_distinct_singular_value gives a non-singular (=invertible, =solvable) matrices 'a'.
    The first input to torch.linalg.solve is generated as the itertools.product of 'batches' and 'ns'.
    The second input is generated as the product of 'batches', 'ns' and 'nrhs'.
    In total this function generates 18 SampleInputs
    'batches' cases include:
        () - single input,
        (0,) - zero batched dimension,
        (2,) - batch of two matrices.
    'ns' gives 0x0 and 5x5 matrices.
    and 'nrhs' controls the number of vectors to solve for:
        () - using 1 as the number of vectors implicitly
        (1,) - same as () but explicit
        (3,) - solve for 3 vectors.
    Zeros in dimensions are edge cases in the implementation and important to test for in order to avoid unexpected crashes.
    'vector_rhs_allowed' controls whether to include nrhs = () to the list of SampleInputs.
    torch.solve / triangular_solve / cholesky_solve (opposed to torch.linalg.solve) do not allow
    1D tensors (vectors) as the right-hand-side.
    Once torch.solve / triangular_solve / cholesky_solve and its testing are removed,
    'vector_rhs_allowed' may be removed here as well.
    """
    from torch.testing._internal.common_utils import random_fullrank_matrix_distinct_singular_value

    def test_naughty_autograd_function_stashing_ctx(self):
        saved_ctx = []

        class Id(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return x

def sample_inputs_legacy_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates always solvable input for legacy solve functions
    (the ones that are not in torch.linalg module).
    The difference from sample_inputs_linalg_solve is that here the right-hand-side of A x = b equation
    should have b.ndim >= 2, vectors are not allowed.
    Also the arguments order is swapped.
    """
    out = sample_inputs_linalg_solve(
        op_info, device, dtype, requires_grad=requires_grad, vector_rhs_allowed=False
    )

        p = torch.zeros(1, requires_grad=True)
        loss = Id.apply(p)
        loss.backward(retain_graph=True)
        del loss
        # At this point in time, it complains that the graph has been freed
        # (which indeed true, although a somewhat indirect way of stating the
        # problem).
        self.assertRaises(RuntimeError, lambda: saved_ctx[0].saved_tensors)

    def test_custom_autograd_repeated_grad_grad(self):
        # This test failed the equality check in PR #22983; it's an interesting
        # and different test case worth enshrining.  mult1 is not testing
        # anything that interesting, but mult2 is the interesting case.

        def mult1(x):
            return x.prod(dim=-1).prod(dim=-1)

def sample_inputs_lu(op_info, device, dtype, requires_grad=False, **kwargs):
    # not needed once OpInfo tests support Iterables
    def generate_samples():
        batch_shapes = ((), (3,), (3, 3))
        for batch_shape, get_infos in product(batch_shapes, (True, False)):
            shape = batch_shape + (S, S)
            input = make_tensor(shape, device, dtype, requires_grad=requires_grad, low=None, high=None)
            yield SampleInput(input, args=(True, get_infos))

    return list(generate_samples())


def sample_inputs_roll(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    args = ((0, 0), (1, 2), (0, 2), (2, 0), (-1, 0), (10000, 1), (2,), ((1, 2, -1), (0, 1, 2)))

    def generator():
        for arg in args:
            yield SampleInput(make_arg((S, S, S)), args=arg)

    return list(generator())


def sample_inputs_rot90(op_info, device, dtype, requires_grad=False, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    args = ((1, (0, 1),),
            (1, (1, 2),),
            (1, (1, -1),),
            ())

    def generator():
        for arg in args:
            yield SampleInput(make_arg((S, S, S)), args=arg)

    return list(generator())


def sample_inputs_std_var(op_info, device, dtype, requires_grad, **kwargs):
    tensor_nd = make_tensor((S, S, S), device=device, dtype=dtype,
                            low=None, high=None, requires_grad=requires_grad)
    tensor_1d = make_tensor((S,), device=device, dtype=dtype,
                            low=None, high=None, requires_grad=requires_grad)

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return (grad_output * y)[:, None, None] / x

        mult2 = Mult.apply

        def check_gradgrad_repeated(x, y):
            gy, = torch.autograd.grad(y[0], x, create_graph=True)
            ggy_1, = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            gy, = torch.autograd.grad(y[0], x, create_graph=True)
            ggy_2, = torch.autograd.grad(gy[0, 0, 0], x, retain_graph=True)
            self.assertEqual(ggy_1[0, 0, 1], ggy_2[0, 0, 1])

        x = torch.ones(2, 4, 4).requires_grad_()
        check_gradgrad_repeated(x, mult1(x))
        check_gradgrad_repeated(x, mult2(x))

    def test_custom_autograd_no_early_free(self):
        # This test failed complaining that buffers had already been freed
        # prior to #22983.  Also pretty interesting test case.
        class Double(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                y = x ** 2
                ctx.save_for_backward(x, y)
                return y

            @staticmethod
            def backward(ctx, grad_output):
                x, _ = ctx.saved_tensors
                return grad_output * 2 * x

        # this is equivalent, but uses the output of .forward() in .backward()
        class Double2(Double):
            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return grad_output * 2 * y / x

        double = Double.apply
        double2 = Double2.apply

        x = torch.tensor(2).double().requires_grad_()

        self.assertTrue(gradcheck(double, x))
        self.assertTrue(gradgradcheck(double, x))
        self.assertTrue(gradcheck(double2, x))
        self.assertTrue(gradgradcheck(double2, x))


def sample_inputs_permute(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    cases = [((1, 2, 3, 4), (0, 2, 3, 1)),
             ((1, 2, 3, 4), (0, -2, -1, 1)),
             ((), ()),
             ((1, 2, 3, 4), (2, 1, 3, 0))]

    def generator():
        for shape, args in cases:
            yield SampleInput(make_arg(shape), args=(args,))

    return list(generator())


# Based on erstwhile method_tests tests & some tensor_op_tests for pow
def sample_inputs_pow(op_info, device, dtype, requires_grad, **kwargs):
    samples = []

    if dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
        test_cases = (
            ((2, 2), 0, 5, 1e-3, requires_grad, (2, 2), 0, 1, 0.1, requires_grad, False),
            ((2, 2), 0, 5, 1e-3, requires_grad, (1,), 0, 1, 0.1, requires_grad, False),
            ((), 1e-3, 1e-3 + 1, 0, True, (), 0.1, 1.1, 0, False, False),
            ((2, 2), 0, 5, 1e-3, requires_grad, (), 0.1, 1.1, 1, False, False),
        )
        tests_require_resizing = (
            ((1,), 0, 5, 1e-3, requires_grad, (2, 2), 0, 1, 0.1, requires_grad, True),
            ((2, 1, 2), 0, 5, 1e-3, requires_grad, (1, 2, 1), 0, 1, 0.1, requires_grad, True),
            ((), 1e-3, 1e-3 + 1, 0, True, (1, S, 1), 0, 1, 0.1, requires_grad, True),
        )
        cases = test_cases + tests_require_resizing
        samples = list(SampleInput(make_tensor(shape_b, low=low_b, high=high_b,
                                               requires_grad=b_grad, device=device,
                                               dtype=dtype) + additive_b,
                                   args=(make_tensor(shape_e, low=low_e, high=high_e,
                                                     requires_grad=e_grad, device=device,
                                                     dtype=dtype) + additive_e,),
                                   broadcasts_input=broadcasts_input)
                       for shape_b, low_b, high_b, additive_b, b_grad, shape_e, low_e,
                       high_e, additive_e, e_grad, broadcasts_input in cases)
        tensor_scalar_inputs = (
            ((2, 2), 0, 5, 1e-3, requires_grad, (3.14,)),
            ((), 1e-3, 1e-3 + 1, 0, True, (3.14,))
        )
        more_samples = list(SampleInput(make_tensor(shape, dtype=dtype, device=device,
                                                    high=high, low=low,
                                                    requires_grad=b_grad) + additive,
                                        args=exp)
                            for shape, low, high, additive, b_grad, exp in tensor_scalar_inputs)
        samples = [*samples, *more_samples]
    elif dtype in [torch.complex64, torch.complex128]:
        args_tuple = (
            ((2, 2), 0, 5, requires_grad, (3.14,)),
            ((), 0, 1, True, (3.14,)),
            ((), 0, 1, True, (3.14j,))
        )
        samples = list(SampleInput(make_tensor(shape, dtype=dtype, device=device,
                                               high=high, low=low,
                                               requires_grad=b_grad) + 1e-3 * (1 + 1j),
                                   args=arg)
                       for shape, low, high, b_grad, arg in args_tuple)
    elif dtype == torch.bool:
        arg_tuple = (0, 1, 1., 2.3)
        samples = list(SampleInput(make_tensor((2, 2), device=device, dtype=dtype,
                                               requires_grad=requires_grad),
                                   args=(arg,))
                       for arg in arg_tuple)
        dtypes_list = [torch.float64, torch.float32, torch.int64, torch.int32]
        more_samples = list(SampleInput(make_tensor((2, 2), device, dtype=torch.bool,
                                                    requires_grad=requires_grad),
                                        args=(make_tensor((2, 2), device, dtype=dtype,
                                                          requires_grad=requires_grad),))
                            for dtype in dtypes_list)
        samples = [*samples, *more_samples]
        samples.append(SampleInput(make_tensor((2, 2, 2), device, dtype=torch.bool,
                                               requires_grad=requires_grad),
                                   args=(make_tensor((2, 1), device, dtype=torch.float64,
                                                     requires_grad=requires_grad),)))
    else:
        exp_tuple = (1, 2, 3)
        samples = list(SampleInput(make_tensor((2, 2), device, dtype,
                                               requires_grad=requires_grad),
                                   args=(arg,))
                       for arg in exp_tuple)
        samples.append(SampleInput(make_tensor((2, 2), device, dtype,
                                               requires_grad=requires_grad),
                                   args=(make_tensor((2, 2), device, dtype,
                                                     requires_grad=requires_grad),)))
    return tuple(samples)

def sample_inputs_svd(op_info, device, dtype, requires_grad=False, **kwargs):
    return _sample_inputs_svd(op_info, device, dtype, requires_grad, is_linalg_svd=False)

def sample_inputs_linalg_svd(op_info, device, dtype, requires_grad=False, **kwargs):
    return _sample_inputs_svd(op_info, device, dtype, requires_grad, is_linalg_svd=True)

def sample_inputs_linalg_svdvals(op_info, device, dtype, requires_grad=False, **kwargs):
    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 2, 0]
    samples = []
    for batch, (m, n) in product(batches, product(ns, ns)):
        a = make_tensor((*batch, m, n), device, dtype, low=None, high=None, requires_grad=requires_grad)
        samples.append(SampleInput(a))
    return samples

def sample_inputs_eig(op_info, device, dtype, requires_grad=False, **kwargs):
    eigvecs = make_tensor((S, S), device=device, dtype=dtype,
                          low=None, high=None)
    eigvals = make_tensor((S,), device=device, dtype=dtype,
                          low=None, high=None)
    # we produce only diagonazible inputs which do not have
    # complex eigenvalues for real inputs, as there is no
    # backward implementation for real inputs with complex
    # eigenvalues yet.
    input = (eigvecs * eigvals.unsqueeze(-2)) @ eigvecs.inverse()
    input.requires_grad_(requires_grad)

        x = torch.randn(10, 10, requires_grad=True)
        y = x * 2
        y = y.detach()
        self.assertFalse(y.requires_grad)
        self.assertIsNone(y.grad_fn)
        z = x + y
        z.sum().backward()
        # This is an incorrect gradient, but we assume that's what the user
        # wanted. detach() is an advanced option.
        self.assertEqual(x.grad, torch.ones(10, 10))

        # in-place detach
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=True)
        a = x * 2
        (y + a).sum().backward(retain_graph=True)
        a.detach_()
        self.assertFalse(a.requires_grad)
        (y + a).sum().backward()  # this won't backprop to x
        self.assertEqual(x.grad, torch.ones(10, 10) * 2)
        self.assertEqual(y.grad, torch.ones(10, 10) * 2)


def sample_inputs_einsum(op_info, device, dtype, requires_grad=False, **kwargs):
    x = make_tensor((3,), device, dtype, requires_grad=requires_grad)
    y = make_tensor((4,), device, dtype, requires_grad=requires_grad)
    A = make_tensor((2, 3,), device, dtype, requires_grad=requires_grad, noncontiguous=True)
    B = make_tensor((1, 3,), device, dtype, requires_grad=requires_grad)
    C = make_tensor((1, 2, 3,), device, dtype, requires_grad=requires_grad)
    D = make_tensor((1, 3, 4,), device, dtype, requires_grad=requires_grad, noncontiguous=True)
    E = make_tensor((4, 4,), device, dtype, requires_grad=requires_grad)
    H = make_tensor((3, 3,), device, dtype, requires_grad=requires_grad, noncontiguous=True)
    I = make_tensor((1, 3, 1,), device, dtype, requires_grad=requires_grad)

    inputs = []

    # Vector operations
    inputs.append(SampleInput([x], args=('i->',)))                      # sum
    inputs.append(SampleInput([x, y], args=('i,j->ij',)))               # outer

    # Matrix operations
    inputs.append(SampleInput([A], args=("ij->i",)))                    # col sum
    inputs.append(SampleInput([A, B], args=("ij,kj->ik",)))             # matmul
    inputs.append(SampleInput([A, E], args=("ij,Ab->ijAb",)))           # matrix outer product

    # Tensor operations
    inputs.append(SampleInput([C, D], args=("aij,ajk->aik",)))          # batch matmul
    inputs.append(SampleInput([D, E], args=("aij,jk->aik",)))           # tensor matrix contraction
    inputs.append(SampleInput([C, B], args=("ijk,ik->j",)))             # non contiguous

    # Test diagonals
    inputs.append(SampleInput([I], args=('iji->j',)))                   # non-contiguous trace

    # Test ellipsis
    inputs.append(SampleInput([H], args=("i...->...",)))
    inputs.append(SampleInput([C, x], args=('...ik, ...j -> ij',)))

    return inputs


def sample_inputs_linalg_qr(op_info, device, dtype, requires_grad=False, **kwargs):
    """
    This function generates input for torch.linalg.qr
    The input is generated as the itertools.product of 'batches' and 'ns'.
    """
    batches = [(), (0,), (2, ), (1, 1)]
    ns = [5, 2, 0]
    out = []
    for batch, (m, n) in product(batches, product(ns, ns)):
        a = torch.randn(*batch, m, n, dtype=dtype, device=device, requires_grad=requires_grad)
        out.append(SampleInput(a))
    return out

def sample_inputs_geqrf(op_info, device, dtype, requires_grad=False):
    batches = [(), (0, ), (2, ), (1, 1)]
    ns = [5, 2, 0]
    samples = []
    for batch, (m, n) in product(batches, product(ns, ns)):
        # TODO: CUDA path doesn't work with batched or empty inputs
        if torch.device(device).type == 'cuda' and (batch != () or m == 0 or n == 0):
            continue
        a = make_tensor((*batch, m, n), device, dtype, low=None, high=None, requires_grad=requires_grad)
        samples.append(SampleInput(a))
    return samples

def sample_inputs_flip(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((S, 0, M), device, dtype, low=None, high=None, requires_grad=requires_grad)
    )

    def _test_type_conversion_backward(self, t, ):
        fvar = Variable(t(torch.randn(5, 5).float()), requires_grad=True)
        fvar.double().sum().backward()
        self.assertEqual(fvar.grad, torch.ones_like(fvar))
        self.assertEqual(type(fvar.grad), type(fvar))
        dvar = Variable(t(torch.randn(5, 5).double()), requires_grad=True)
        dvar.float().sum().backward()
        self.assertEqual(dvar.grad, torch.ones_like(dvar))
        self.assertEqual(type(dvar.grad), type(dvar))

    def test_type_conversions(self):
        x = torch.randn(5, 5)
        self.assertIsInstance(x.float(), torch.FloatTensor)
        self.assertIsInstance(x.int(), torch.IntTensor)
        if torch.cuda.is_available():
            self.assertIsInstance(x.float().cuda(), torch.cuda.FloatTensor)
            self.assertIsInstance(x.int().cuda(), torch.cuda.IntTensor)
            self.assertIsInstance(x.int().cuda().cpu(), torch.IntTensor)
            if torch.cuda.device_count() >= 2:
                x2 = x.float().cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                x2 = x.float().cuda()
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 0)
                x2 = x2.cuda(1)
                self.assertIsInstance(x2, torch.cuda.FloatTensor)
                self.assertIs(x2.get_device(), 1)
                y = Variable(torch.randn(5).cuda(1), requires_grad=True)
                y.cpu().sum().backward()
                self.assertIs(y.grad.get_device(), 1)
                self.assertIs(y.long().get_device(), 1)

        for t in [torch.DoubleTensor, torch.FloatTensor, torch.IntTensor, torch.ByteTensor]:
            for y_var in (True, False):
                y = torch.randint(5, (5, 5), dtype=t.dtype)
                y = Variable(y) if y_var else y
                self.assertIsInstance(x.type(t), t)
                self.assertIsInstance(x.type_as(y), t)
                # TODO: t.dtype should work
                t_dtype = t().dtype
                self.assertIsInstance(x.type(t_dtype), t)
                self.assertIs(t_dtype, x.type(t_dtype).dtype)
                self.assertEqual(y.data_ptr(), y.type(t).data_ptr())
                if torch.cuda.is_available():
                    for x_cuda in (True, False):
                        for y_cuda in (True, False):
                            x_c = x.cuda() if x_cuda else x
                            y_c = y.cuda() if y_cuda else y
                            _, y_type = y_c.type().rsplit('.', 1)
                            y_typestr = ('torch.cuda.' if y_cuda else 'torch.') + y_type
                            self.assertEqual(y_c.type(), x_c.type(y_typestr).type())
                            self.assertIs(y_c.dtype, x_c.type(y_c.dtype).dtype)
                            self.assertEqual(y_c.data_ptr(), y_c.cuda().data_ptr() if y_cuda else y_c.data_ptr())

def sample_inputs_fliplr_flipud(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((S, M, S), device, dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((S, 0, M), device, dtype, low=None, high=None, requires_grad=requires_grad)
    )
    return [SampleInput(tensor) for tensor in tensors]

# TODO: clamp shares tensors among its sample inputs --- we should prohibit this!
def sample_inputs_clamp(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((2, 3, 2), device=device, dtype=dtype, low=None, high=None, requires_grad=requires_grad),
        make_tensor((2, 0, 3), device=device, dtype=dtype, low=None, high=None, requires_grad=requires_grad),
    )
    if dtype is torch.uint8:
        min_max_vals = ((2, 5), (3, 7))
    else:
        min_max_vals = ((0, 1), (-1, 1))
    output = [SampleInput(tensor, args=vals) for tensor, vals in product(tensors, min_max_vals)]
    output += [SampleInput(tensors[0], args=(0.5, None)), SampleInput(tensors[0], args=(None, 0.5))]
    empty_tensor = make_tensor((), device=device, dtype=dtype, low=None, high=None, requires_grad=requires_grad)
    output += [SampleInput(empty_tensor, args=(0.0, 1.0)), ]
    return output

def sample_kwargs_clamp(device, dtype, input):
    if dtype is torch.uint8:
        min_val, max_val = (random.randint(1, 3), random.randint(4, 8))
    elif dtype.is_floating_point:
        min_val, max_val = (random.uniform(-8, 0), random.uniform(1, 8))  # type: ignore[assignment]
    else:
        min_val, max_val = (random.randint(-8, 0), random.randint(1, 8))
    return {'min': min_val, 'max': max_val}, {'a_min': min_val, 'a_max': max_val}

def sample_inputs_cumprod(op_info, device, dtype, requires_grad, **kwargs):
    def make_arg(shape):
        # shrink values to be in the interval [-1, +1] for better precision in gradgradcheck
        return make_tensor(shape, device, dtype, low=-1, high=+1, requires_grad=requires_grad)

    def test_shape(self):
        x = torch.randn(3, 4)
        self.assertEqual(2, len(x.shape))
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[1], 4)

    def test_numpy_requires_grad(self):
        x = torch.randn(2, 2, requires_grad=True)
        err_msg_outputs = r"Can't call numpy\(\) on Tensor that requires grad. Use tensor.detach\(\).numpy\(\) instead."
        with self.assertRaisesRegex(RuntimeError, err_msg_outputs):
            x.numpy()

        with torch.no_grad():
            x.numpy()

        x = torch.randn(2, 2)
        x.numpy()

        with torch.no_grad():
            x.numpy()

    def test_return_leaf(self):
        class Identity(Function):
            @staticmethod
            def forward(ctx, a, b):
                return a, a + b

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a + grad_b, grad_b

def sample_inputs_view_as_complex(op_info, device, dtype, requires_grad, **kwargs):
    return [SampleInput(make_tensor((S, 2), device, dtype, requires_grad=requires_grad),)]

def sample_inputs_view_as_real(op_info, device, dtype, requires_grad, **kwargs):
    tensors = (
        make_tensor((S, S), device, dtype, requires_grad=requires_grad),
        make_tensor((), device, dtype, requires_grad=requires_grad)
    )
    return [SampleInput(tensor) for tensor in tensors]

def sample_inputs_copysign(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor(*shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    cases = [
        # no broadcast
        ((S, S, S), (S, S, S), False),
        # broadcast rhs
        ((S, S, S), (S, S), False),

        # scalar
        ((S, S), 3.14, False),
        # scalar positive zero
        ((S, S), 0.0, False),
        # scalar negative zero
        ((S, S), -0.0, False),
    ]

    # broadcast lhs
    cases.append(((S, S), (S, S, S), True))
    # broadcast all
    cases.append(((S, 1, S), (M, S), True))

    def generator():
        for input_shape, arg_val, broadcasts_input in cases:
            if isinstance(arg_val, tuple):
                arg = _make_tensor(*arg_val)
            else:
                # arg_val is scalar
                arg = arg_val

            yield SampleInput(_make_tensor(*input_shape), args=(arg, ), broadcasts_input=broadcasts_input)

    return list(generator())

def sample_inputs_prod(op_info, device, dtype, requires_grad):
    def make_arg(shape):
        # shrink values to be in the interval [-1, +1] for better precision in gradgradcheck
        return make_tensor(shape, device, dtype, low=-1, high=+1, requires_grad=requires_grad)

        q, p = Identity.apply(x, y)

        # Make sure hooks only receive grad from usage of q, not x.
        def hook(grad):
            hook_called[0] = True
            self.assertEqual(grad, torch.ones(5, 5))

        q.register_hook(hook)
        (q + p + x).sum().backward()
        self.assertEqual(x.grad, torch.ones(5, 5) * 3)
        self.assertEqual(y.grad, torch.ones(5, 5))
        self.assertTrue(hook_called[0])

    def test_return_leaf_inplace(self):
        class Inplace(InplaceFunction):
            @staticmethod
            def forward(ctx, a, b):
                ctx.mark_dirty(a)
                return a.add_(b), b + 2

            @staticmethod
            def backward(ctx, grad_a, grad_b):
                return grad_a, grad_a + grad_b

        x = torch.randn(5, 5)
        y = torch.randn(5, 5, requires_grad=True)

        fn = Inplace(True)
        q, p = fn.apply(x, y)
        self.assertIs(q, x)
        self.assertIs(q.grad_fn.__class__, fn._backward_cls)
        self.assertTrue(q.requires_grad)
        q.sum().backward()
        self.assertEqual(y.grad, torch.ones(5, 5))

    def test_leaf_assignment(self):
        x = torch.randn(5, 5)
        y = torch.randn(5, requires_grad=True)
        z = torch.randn(5, requires_grad=True)

        x[0] = y
        x[1] = 2 * z
        self.assertTrue(x.requires_grad)
        self.assertIsNot(x.grad_fn, None)
        x.sum().backward()
        self.assertEqual(y.grad, torch.ones(5))
        self.assertEqual(z.grad, torch.ones(5) * 2)

    def test_no_grad_assignment(self):
        x = torch.randn(5, 5, requires_grad=True)
        y = torch.randn(5)
        with torch.no_grad():
            x[0] = y

        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad_fn)

    def test_no_grad_modifies_version(self):
        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        z = (x * y).sum()
        with torch.no_grad():
            x *= 2
        self.assertRaisesRegex(RuntimeError, 'modified by an inplace operation',
                               lambda: z.backward())

    def test_no_grad_input(self):
        class MyFunction(Function):
            @staticmethod
            def forward(self, x):
                return x

def sample_inputs_diag(op_info, device, dtype, requires_grad, **kwargs):
    vec_sample = SampleInput(make_tensor((M, ), device, dtype, low=None, high=None, requires_grad=requires_grad))

        x = torch.randn(5, requires_grad=True)
        with torch.no_grad():
            y = MyFunction.apply(x)

        self.assertTrue(x.requires_grad)
        self.assertIsNone(y.grad_fn)

    def test_backward_copy(self):
        # This tests checks backward engine for a very subtle bug that appreared
        # in one of the initial versions of autograd. Gradients tensors were
        # simply stored in lists while the function waited for all its gradients
        # to be computed. However, sometimes an output was used multiple times,
        # so the gradients needed to be summed. Engine used to keep a need_copy
        # set of tensors that will need a clone upon next addition and removed
        # them from the set as soon as the clone was performed. However, this
        # could lead to incorrect results if the same gradient tensor was
        # buffered in three places in the graph:
        # 1. When accumulating gradients in one of these places it was cloned
        #    and removed from need_copy set.
        # 2. When accumulating in second place, it wasn't in the need_copy set,
        #    so the gradients were simply accumulated in-place (which already
        #    modified the grad in 3rd place)
        # 3. When accumulating in the third place, it wasn't in the need_copy set
        #    as well, so the incoming gradient was summed in-place, yielding
        #    incorrect results in all functions, except the first one.
        x = torch.ones(5, 5, requires_grad=True)
        y = torch.ones(5, 5, requires_grad=True)
        # Simulate that we're in the middle of the graph
        a = x + 2
        b = y + 2
        c = x + 2
        # This op will just return grad_output two times in backward
        add1 = a + b
        add2 = add1 + c
        # Simulate a long branch, so grad_output will get buffered.
        for _ in range(4):
            a = a * 2
            b = b * 2
            c = c * 2
        branch = a + b + c
        out = add2 + branch
        # expected gradients are:
        # for x: 34 (16 from final a, 16 from final c, 2 from add2)
        # for y: 17 (16 from final b, 1 from add2)
        grad_output = torch.ones(5, 5)
        out.backward(grad_output)
        self.assertEqual(x.grad, torch.ones(5, 5) * 34)
        self.assertEqual(y.grad, torch.ones(5, 5) * 17)

    def test_save_none_for_backward(self):
        test_case = self

def sample_inputs_logit(op_info, device, dtype, requires_grad, **kwargs):
    low, high = op_info.domain

            @staticmethod
            def backward(ctx, grad_output):
                n1, input, n2 = ctx.saved_tensors
                test_case.assertIsNone(n1)
                test_case.assertIsNone(n2)
                return 2 * input * grad_output

        x = torch.randn(5, 5, requires_grad=True)
        y = MyFn.apply(x)
        y.sum().backward()
        self.assertEqual(x.grad, 2 * x)

    def test_too_many_grads(self):
        class MyFn(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output, None, None

def sample_inputs_floor_divide(op_info, device, dtype, requires_grad, **kwargs):
    lhs = make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    rhs = make_tensor((S, S, S), device, dtype, low=None, high=None, requires_grad=requires_grad)
    # Avoid integer divide by 0
    if not (dtype.is_floating_point or dtype.is_complex):
        rhs[rhs == 0] = 1

    def test_pickle(self):
        x = torch.randn(10, 10, requires_grad=True)
        y = torch.randn(10, 10, requires_grad=False)

        def assert_strict_equal(var1, var2):
            self.assertEqual(var1, var2)
            self.assertEqual(var1.requires_grad, var2.requires_grad)

def sample_inputs_masked_scatter(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def samples_generator():
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, make_arg((S, S))))
        yield SampleInput(make_arg((S, S)), args=(torch.randn((S,), device=device) > 0, make_arg((S, S))))
        yield SampleInput(make_arg((S, S)), args=(bernoulli_scalar().to(device), make_arg((S, S))))
        yield SampleInput(make_arg((S,)),
                          args=(torch.randn(S, S, device=device) > 0, make_arg((S, S))),
                          broadcasts_input=True)

    samples = tuple(samples_generator())
    return samples


def sample_inputs_masked_fill(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def sample_generator():
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, 10))
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, make_arg(())))
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, device=device) > 0, 10))
        yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, 10))
        yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, make_arg(())))
        yield SampleInput(make_arg((S, S)), args=(torch.randn((), device=device) > 0, 10))

        yield SampleInput(make_arg((S,)),
                          args=(torch.randn(S, S, device=device) > 0, make_arg(())),
                          broadcasts_input=True)
        yield SampleInput(make_arg((S,)),
                          args=(torch.randn(S, S, device=device) > 0, 10),
                          broadcasts_input=True)

    samples = tuple(sample_generator())
    return samples

def sample_inputs_masked_select(op_info, device, dtype, requires_grad, **kwargs):
    samples = (
        SampleInput(make_tensor((M, M), device, dtype, low=None, high=None, requires_grad=requires_grad),
                    args=(torch.randn(M, M, device=device) > 0,)),

        x = torch.randn(5, requires_grad=True)
        a, b = F1.apply(x)
        b = b + 1  # separate F1 from F2 by another op
        self.assertTrue(a.requires_grad)
        self.assertFalse(b.requires_grad)
        c = F2.apply(a, b)
        c.backward(torch.ones(c.size()))
        self.assertEqual(x.grad, torch.ones(x.size()))

    def test_set_grad_enabled(self):
        x = torch.tensor([1.], requires_grad=True)
        with torch.set_grad_enabled(False):
            y = x * 2
        self.assertFalse(y.requires_grad)
        with torch.set_grad_enabled(True):
            y = x * 2
        self.assertTrue(y.requires_grad)
        with torch.set_grad_enabled(False):
            torch.set_grad_enabled(True)
            y = x * 2
        self.assertTrue(y.requires_grad)

    def test_simple_reentrant(self):
        y_data = torch.randn(2, 2)

        class Reenter(Function):
            @staticmethod
            def forward(ctx, x):
                with torch.enable_grad():
                    ctx.x = Variable(x, requires_grad=True)
                    ctx.y = Variable(y_data, requires_grad=True)
                    ctx.output_var = ctx.x * ctx.y
                return ctx.output_var.detach()

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    ctx.output_var.sum().backward()
                return ctx.x.grad * grad_output

        # Reentrant starts on CPU thread, finishs on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        out = Reenter.apply(x)
        out.sum().backward()
        self.assertEqual(x.grad, y_data)

    def test_reentrant_child_error(self):
        # Parent graph.
        a = torch.rand(3, 3, requires_grad=True)
        c = a * a

def sample_inputs_matrix_exp(op_info, device, dtype, requires_grad, **kwargs):
    samples = (
        SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad)),
        SampleInput(make_tensor((S, S, S), device, dtype, requires_grad=requires_grad)),
    )

    return samples

def sample_inputs_matmul(op_info, device, dtype, requires_grad):
    test_cases = (((L,), (L,)),
                  ((S, M), (M,)),
                  ((M,), (M, S)),
                  ((S, M), (M, S)),
                  ((S, S, M), (M,)),
                  ((S, S, M), (M, S)),
                  ((M,), (S, M, S)),
                  ((S, M), (S, M, S)),
                  ((S, S, M, M), (S, S, M, S)),
                  ((S, S, M, M), (M,)),
                  ((M,), (S, S, M, S)))
    sample_inputs = []
    for lhs_shape, rhs_shape in test_cases:
        lhs = make_tensor(lhs_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        rhs = make_tensor(rhs_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        sample_inputs.append(SampleInput(lhs, args=(rhs,)))
    return tuple(sample_inputs)


def sample_inputs_polar(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will throw an error.
                reentrant_root.backward()
                return grad

def sample_inputs_complex(op_info, device, dtype, requires_grad, **kwargs):
    def _make_tensor_helper(shape):
        return make_tensor(shape, device, dtype, requires_grad=requires_grad)

    samples = (
        SampleInput(_make_tensor_helper((S, S)), args=(_make_tensor_helper((S, S)),)),
        SampleInput(_make_tensor_helper(()), args=(_make_tensor_helper(()),)),
    )

    return samples


def sample_inputs_polygamma(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)

    def generator():
        for shape, n in product(tensor_shapes, ns):
            yield SampleInput(make_arg(shape), args=(n,))

    return list(generator())


def sample_inputs_entr(op_info, device, dtype, requires_grad, **kwargs):
    low, _ = op_info.domain

    if requires_grad:
        low = 0 + op_info._domain_eps

    return (SampleInput(make_tensor((L,), device, dtype,
                                    low=low,
                                    requires_grad=requires_grad)),
            SampleInput(make_tensor((), device, dtype,
                                    low=low,
                                    requires_grad=requires_grad)))

def sample_inputs_rsub(op_info, device, dtype, requires_grad, variant='tensor', **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _samples_with_alpha_helper(args, alphas, filter_fn=lambda arg_alpha: True):
        filtered_product = filter(filter_fn, product(args, alphas))  # type: ignore[var-annotated]
        return (SampleInput(input, args=(arg,), kwargs=dict(alpha=alpha))
                for (input, arg), alpha in filtered_product)

    def test_cat(self):
        f_args_variable = (torch.randn(1, S, S, dtype=torch.double, requires_grad=True),
                           torch.randn(2, S, S, dtype=torch.double, requires_grad=True),
                           torch.randn(3, S, S, dtype=torch.double, requires_grad=True),
                           0)
        f_args_tensor = deepcopy(unpack_variables(f_args_variable))
        run_functional_checks(self, "test_cat", "cat",
                              lambda a, b, c, dim: torch.cat((a, b, c), dim),
                              True, f_args_variable, f_args_tensor)

    if variant == 'tensor':
        samples = (
            SampleInput(_make_tensor_helper((S, S)), args=(_make_tensor_helper((S, S)),)),
            SampleInput(_make_tensor_helper((S, S)), args=(_make_tensor_helper((S,)),)),
            SampleInput(_make_tensor_helper((S,)), args=(_make_tensor_helper((S, S)),)),
            SampleInput(_make_tensor_helper(()), args=(_make_tensor_helper(()),)),
            SampleInput(_make_tensor_helper(()), args=(_make_tensor_helper((S,)),)),
            SampleInput(_make_tensor_helper((S,)), args=(_make_tensor_helper(()),)),
        )
        print(
            "CPU time measurement python side overhead: {:.2f}%".format(
                (total_time_us / prof.self_cpu_time_total - 1.0) * 100.0
            )
        )

        if sys.platform != "win32":
            with tempfile.NamedTemporaryFile() as trace_file:
                prof.export_chrome_trace(trace_file.name)

        args = ((_make_tensor_helper((S, S)), _make_tensor_helper((S, S))),
                (_make_tensor_helper((S, S)), _make_tensor_helper((S,))),
                (_make_tensor_helper(()), _make_tensor_helper(())))
        samples += tuple(_samples_with_alpha_helper(args, alphas))  # type: ignore[assignment]
    elif variant == 'scalar':
        # Scalar Other
        samples = (SampleInput(_make_tensor_helper((S, S)), args=(0.5,)),
                   SampleInput(_make_tensor_helper(()), args=(0.5,)),
                   SampleInput(_make_tensor_helper((S, S)), args=(1.5j,)),
                   SampleInput(_make_tensor_helper(()), args=(1.5j,)),
                   SampleInput(_make_tensor_helper((S, S)), args=(0.4 + 1.2j,)),
                   SampleInput(_make_tensor_helper(()), args=(1.2 + 1.76j,)))

        scalar_args = [(_make_tensor_helper((S, S)), 0.5), (_make_tensor_helper(()), 0.5),
                       (_make_tensor_helper((S, S)), 2.7j), (_make_tensor_helper(()), 2.7j),
                       (_make_tensor_helper((S, S)), 1 - 2.7j), (_make_tensor_helper(()), 1 + 2.7j)]

        forward(x)

        with profile(use_kineto=kineto_available()) as p:
            forward(x)

        events = p.function_events
        important_events = [
            'outer',
            'aten::mul',
            'aten::add',
            'inner',
            'aten::sub',
            'aten::div'
        ]
        idx = 0
        for info in events:
            if info.name == important_events[idx]:
                idx = idx + 1
            if idx == len(important_events):
                break
        self.assertEqual(idx, len(important_events))

        # We can also use record_function to decorate arbitrary function
        @record_function('my_func')
        def f(x, y):
            return x + y

        with profile(use_kineto=kineto_available()) as p:
            f(1, 2)

        self.assertTrue('my_func' in str(p))

    def test_record_function_multithreaded(self):
        rf = record_function("outer")
        rf.__enter__()
        with record_function("inner"):
            # test that exiting the record function after starting another one
            # doesn't throw.
            rf.__exit__(None, None, None)

        with record_function("inner"):
            rf.__enter__()
        # test that exiting the record function after ending another one
        # doesn't throw.
        rf.__exit__(None, None, None)


    def test_dir(self):
        x = torch.randn(10, 10)
        keys = dir(x)
        self.assertIn('shape', keys)

        # real and imag are only implemented for complex tensors.
        y = torch.randn(10, 10, dtype=torch.cfloat)
        for key in ['real', 'imag']:
            self.assertRaises(RuntimeError, lambda: hasattr(x, key))
            self.assertTrue(hasattr(y, key))
            keys.remove(key)

        for key in keys:
            self.assertTrue(hasattr(x, key))

    def test_as_strided(self):

        def test(x, prepro_fn, size, strides, offset=None):
            x = x.to(torch.double).detach().requires_grad_()

            # Check that forward will **not** resize storage because it may
            # cause NaN in output and fail numerical Jacobian check consequently
            with torch.no_grad():
                y = prepro_fn(x) if prepro_fn is not None else x
                max_offset = sum((si - 1) * st for si, st in zip(size, strides))
                max_offset += offset if offset is not None else y.storage_offset()
                assert max_offset < len(y.storage()), "test case resizes storage"

            def closure(x):
                if prepro_fn is not None:
                    x = prepro_fn(x)
                return x.as_strided(size, strides, offset)

            gradcheck(closure, [x])
            gradgradcheck(closure, [x])

        # test
        test(torch.arange(0, 25), lambda x: x.view(5, 5), [3, 3], [6, 2], 2)

        # test crazy stride at dim with size 1 case
        test(torch.randn(12), None, [1, 2, 1, 5], [0, 5, 100, 1], 2)

        # test expand case
        test(torch.randn(5), None, [3, 3, 3], [0, 1, 0], 2)
        test(torch.randn(5), None, [3, 3, 3], [0, 0, 0], 4)
        test(torch.randn(5), lambda x: x.expand(5, 5), [5, 5], [0, 1], 0)

        # test non-expand overlapping case
        test(torch.randn(35), None, [6, 6], [5, 1], 2)
        test(torch.randn(15), None, [3, 2], [3, 6], 2)

        # test transpose case
        test(torch.randn(3, 4), None, [4, 3], [1, 4])

        # test "getting things outside the input" case
        x = torch.randn(6, 2)
        test(x[3:], None, [3, 2], [2, 1], 0)  # should be all zeros
        self.assertEqual(x[3:].as_strided([3, 2], [2, 1], 0), x[:3])

        # test select on expanded input case
        test(torch.randn(2, 3), lambda x: x.expand(10, 2, 3), [2, 3], [3, 1], 0)

    def _test_lerp_tensor_weights(self, cast):
        def construct_inputs(*shapes):
            start = cast(torch.randn(shapes[0], dtype=torch.double)).requires_grad_()
            end = cast(torch.randn(shapes[1], dtype=torch.double)).requires_grad_()
            weight = cast(torch.randn(shapes[2], dtype=torch.double)).requires_grad_()
            return [start, end, weight]

        all_test_shapes = [((3, 3, 3), (3, 3, 3), (3, 3, 3)),  # no broadcasting
                           ((3,), (3, 3, 3), (3, 3, 3)),  # start broadcasting - 1
                           ((3, 3, 3), (3,), (3, 3, 3)),  # end broadcasting - 1
                           ((3, 3, 3), (3, 3, 3), (3,)),  # weight broadcasting - 1
                           ((), (3, 3, 3), (3, 3, 3)),  # start broadcasting - 2
                           ((3, 3, 3), (), (3, 3, 3)),  # end broadcasting - 2
                           ((3, 3, 3), (3, 3, 3), ()),  # weight broadcasting - 2
                           ((3, 3), (3, 3, 3), (3,))]  # all broadcasting

        for shapes in all_test_shapes:
            cur_inputs = construct_inputs(*shapes)
            gradcheck(torch.lerp, cur_inputs)
            gradgradcheck(torch.lerp, cur_inputs)

    def test_lerp_tensor_weights(self):
        self._test_lerp_tensor_weights(lambda t: t)

    def test_reduce_dtype(self):
        def test_reduction(op, has_no_dim, takes_dtype=True):
            x = torch.randn(3, 3, dtype=torch.float, requires_grad=True)

            if has_no_dim:
                grad1, = torch.autograd.grad([op(x)], [x])
                grad2, = torch.autograd.grad([op(x, dtype=torch.double)], [x])
                self.assertEqual(grad1, grad2)
                self.assertEqual(grad2.dtype, torch.float)

            gi = torch.randn(op(x, dim=0).shape, dtype=torch.float)
            grad1, = torch.autograd.grad([op(x, dim=0)], [x], gi)
            if takes_dtype:
                grad2, = torch.autograd.grad([op(x, dim=0, dtype=torch.double)], [x], gi.double())
            else:
                grad2, = torch.autograd.grad([op(x.double(), dim=0)], [x], gi.double())
            self.assertEqual(grad1, grad2)
            self.assertEqual(grad2.dtype, torch.float)

        test_reduction(torch.sum, True)
        test_reduction(torch.prod, True)
        test_reduction(torch.cumsum, False)
        test_reduction(torch.cumprod, False)
        test_reduction(torch.logcumsumexp, False, takes_dtype=False)

    def test_inplace_view_saved_output(self):
        # Test an in-place operation on a view in which the in-place op saves
        # its output. Previously, this created a reference cycle.
        dealloc = [0]

        class IncrementOnDelete(object):
            def __del__(self):
                dealloc[0] += 1

        def test():
            root = torch.randn(3, 3, requires_grad=True)
            copy = root.clone()
            copy.grad_fn.register_hook(IncrementOnDelete())
            view = copy.view(9)
            torch.nn.functional.relu(view, inplace=True)

        test()
        self.assertEqual(dealloc[0], 1)

    def test_inplace_view_leaf_errors(self):
        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        x = torch.zeros(1, requires_grad=True)
        y = x.view_as(x)
        with self.assertRaisesRegex(RuntimeError,
                                    "a view of a leaf Variable that "
                                    "requires grad is being used in "
                                    "an in-place operation."):
            y.add_(1)

    def test_inplace_view_backward(self):
        # Issue #10532: Make sure that this does not raise RuntimeError.
        net = nn.Sequential(
            nn.InstanceNorm2d(2),
            nn.ReLU(True)
        )

        x = torch.tensor([[[[1.0, 1.0]]]], requires_grad=True)
        g, = torch.autograd.grad(net(x).pow(2), [x], grad_outputs=x.new_ones(x.shape) , create_graph=True)
        torch.autograd.grad(g.sum(), [x])
        self.assertEqual(x, torch.tensor([[[[1.0, 1.0]]]]))

        # https://discuss.pytorch.org/t/freeing-buffer-strange-behavior/31955/8
        inputs = torch.ones((1, 3, 256, 256), requires_grad=True)

        tmp1 = (inputs + 1).view_as(inputs)
        tmp2 = torch.nn.functional.threshold(tmp1, 0., 0., True)
        prob_interpolated = torch.sigmoid(tmp2)

        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=inputs,
                                        grad_outputs=torch.ones(prob_interpolated.size()),
                                        create_graph=True, retain_graph=True)[0]

        gradient_penalty = gradients.sum()
        gradient_penalty.backward()

        fn = gradient_penalty.grad_fn.next_functions[0][0].next_functions[1][0]
        self.assertEqual(fn.name(), "ThresholdBackwardBackward")

    def test_inplace_view_weak_grad_fn(self):
        # Issue 23502: Test that b's grad_fn is preserved.
        a = torch.arange(10.0, requires_grad=True)

        b = a.narrow(0, 0, 2).clone().view(-1)
        b.relu_()

        c = b.clone()
        del b
        gc.collect()

        s = c.sum()
        s.backward()
        self.assertEqual(s, torch.tensor(1.0))

        # Issue #21875: Fail faster (when we try to modify the view vs. in backward())
        a = torch.rand(10, requires_grad=True).narrow(0, 0, 10)
        with self.assertRaises(RuntimeError):
            b = a.relu_()

    def test_mul_out(self):
        a = torch.randn(2, 2, requires_grad=True)
        b = torch.randn(2, 2, requires_grad=True)
        x = torch.zeros_like(a)

        # out=... functions don't support automatic differentiation currently
        self.assertRaisesRegex(RuntimeError, 'out=', lambda: torch.mul(a, b, out=x))

        # the inputs can require grad if we're in no_grad() mode
        with torch.no_grad():
            torch.mul(a, b, out=x)
            self.assertEqual(x, a * b)

    def test_mul_out_result_requires_grad(self):
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        x = torch.zeros(2, 2, requires_grad=True)
        # we should throw an exception if the output requires grad
        self.assertRaisesRegex(RuntimeError, 'out=', lambda: torch.mul(a, b, out=x))

    def test_diagonal_derivative_requires_grad(self):
        # test that the backward requires grad
        # we do this is because diagonal_backward uses inplace
        # operations and gradgradcheck does not catch whether
        # they works as expected (it will succeed even if
        # the gradient has requires_grad == False
        a = torch.randn(5, 6, requires_grad=True)
        b = torch.diagonal(a)**2
        c = b.sum()
        d, = torch.autograd.grad(c, a, retain_graph=True, create_graph=True)
        self.assertTrue(d.requires_grad)

    def test_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, inp2, fail_0th):
                ctx.fail_0th = fail_0th
                return inp1.sum(0, keepdim=True)

            @staticmethod
            def backward(ctx, gO):
                gI = gO.clone().expand(size)
                gI[0] = 0
                gI[0] /= 0  # Generate a nan
                if ctx.fail_0th:
                    return gI, None, None
                else:
                    return None, gI, None

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        out.backward()  # Should not fail

        inp = torch.rand(size, requires_grad=True)
        out = MyFunc.apply(inp, inp, True)
        with self.assertRaisesRegex(RuntimeError, "Function 'MyFuncBackward' returned nan values in its 0th output."):
            with warnings.catch_warnings(record=True) as w:
                with detect_anomaly():
                    out.backward()
            self.assertIn('No forward pass information', str(w[0].message))

        samples += tuple(_samples_with_alpha_helper(scalar_args, alphas, filter_fn=filter_fn))  # type: ignore[assignment]
    else:
        raise Exception("Invalid variant!")

    def test_nested_anomaly_detect_nan(self):
        size = 10

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, inp1, fail_0th):
                ctx.fail_0th = fail_0th
                ctx.save_for_backward(inp1)
                return inp1.sum(0, keepdim=True)

def sample_inputs_cumulative_ops(op_info, device, dtype, requires_grad, supports_dtype_kwargs=True, **kwargs):
    def _make_tensor_helper(shape, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    samples = [
        SampleInput(_make_tensor_helper((S, S, S)), args=(0,)),
        SampleInput(_make_tensor_helper((S, S, S)), args=(1,)),
        SampleInput(_make_tensor_helper(()), args=(0,)),
    ]

    if supports_dtype_kwargs:
        # NOTE: if `dtype` is not same as input, then inplace variants fail with
        # `provided dtype must match the dtype of self tensor in cumsum`
        samples.append(SampleInput(_make_tensor_helper((S, S, S)), args=(1,), kwargs={'dtype': dtype}))

            @staticmethod
            def backward(ctx, gO):
                fail_0th = ctx.fail_0th
                g1 = gO.clone()
                g2 = gO.clone()
                g1[0] = 0
                g2[0] = 0
                # generate a nan
                if fail_0th:
                    g1[0] /= 0
                else:
                    g2[0] /= 0
                return g1, g2, None


def sample_inputs_unfold(op_info, device, dtype, requires_grad, **kwargs):
    test_cases = (
        ((), (0, 1, 1)),
        ((S, S, S, S), (0, 3, 1)),
        ((S, S, S, S), (1, 3, 1)),
        ((S, S, S, S), (2, 3, 1)),
        ((S, S, S, S), (3, 3, 1)),
        ((S, S, S, S), (0, 3, 2)),
        ((S, S, S, S), (1, 3, 2)),
        ((S, S, S, S), (2, 3, 2)),
        ((S, S, S, S), (3, 3, 2)),
        ((S, S, S, S), (0, 4, 1)),
        ((S, S, S, S), (1, 4, 1)),
        ((S, S, S, S), (2, 4, 1)),
        ((S, S, S, S), (3, 4, 1)),
        ((M,), (0, 3, 1)),
        ((M,), (0, 3, 2)),
        ((M,), (0, 3, 3)),
        ((1000,), (0, 3, 11)),
        ((1000,), (0, 2, 27)),
        ((10, 10), (0, 1, 2)),
        ((10, 10), (1, 2, 3)),
        ((10, 10), (1, 2, 2)),
        ((S, S, S), (2, 3, 2)),
    )

    sample_inputs = []
    for shape, arguments in test_cases:
        sample_inputs += [SampleInput(make_tensor(shape, device, dtype,
                                      low=None, high=None,
                                      requires_grad=requires_grad),
                                      args=arguments)]
    return sample_inputs


def sample_inputs_atan2(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (
        ((S, S, S), (S, S, S)),
        ((), ()),
        ((S, S, S), (S,)),
        # Enable the cases below once gh-53014 is in
        # ((S,), (S, S, S)),
        # ((S, 1, S), (S, S)),
    )

    def generator():
        for x_shape, y_shape in cases:
            yield SampleInput(make_arg(x_shape), args=(make_arg(y_shape),))

    return list(generator())

def sample_inputs_msort(op_info, device, dtype, requires_grad):
    def apply_grad(t):
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            t.requires_grad_(requires_grad)

    def large_1d_unique(dtype, device):
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype)
        apply_grad(res)
        return res

    samples = []
    # Test case for large tensor.
    largesample = SampleInput(large_1d_unique(dtype, device))

    sample = SampleInput(make_tensor((S, M, S), device, dtype,
                                     low=None, high=None,
                                     requires_grad=requires_grad))

    return [largesample, sample]

def sample_inputs_lerp(op_info, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    samples = (
        # no broadcast
        SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 0.4)),
        # broadcast rhs
        SampleInput(make_arg((S, S)), args=(make_arg((S,)), 0.4)),
        # scalar tensor
        SampleInput(make_arg(()), args=(make_arg(()), 0.4)),
        # broadcast rhs scalar-tensor
        SampleInput(make_arg((S, S)), args=(make_arg(()), 0.4)),
        # broadcast rhs with weight tensor
        SampleInput(make_arg((S, S)), args=(make_arg((S,)), make_arg((S, S)))),
        # broadcast rhs and weight tensor
        SampleInput(make_arg((S, S)), args=(make_arg((S, 1)), make_arg((S,)))),
        # broadcast_lhs
        SampleInput(make_arg((S,)), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # scalar broadcast_lhs
        SampleInput(make_arg(()), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # broadcast all
        SampleInput(make_arg((S, 1)), args=(make_arg((S, S)), 0.4), broadcasts_input=True),
        # tensor broadcast all
        SampleInput(make_arg((S, 1)), args=(make_arg((S, S)), make_arg((S, 1))),
                    broadcasts_input=True),
    )

    if dtype.is_complex:
        samples = samples + (  # type: ignore[assignment]
            # no broadcast
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 1.2 + 0.1j)),
            # broadcast rhs
            SampleInput(make_arg((S, S)), args=(make_arg((S,)), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg((S, S)), 5.4 + 9j)),
            # scalar tensor
            SampleInput(make_arg(()), args=(make_arg(()), 0.4j)),
            SampleInput(make_arg(()), args=(make_arg(()), 6.1 + 0.004j)),
            # broadcast rhs scalar-tensor
            SampleInput(make_arg((S, S)), args=(make_arg(()), 0.4j)),
            SampleInput(make_arg((S, S)), args=(make_arg(()), 1 + 2j)),
        )

        feat_combined = []
        for r in range(num_inp):
            data_r = torch.empty(1, nz_inp)
            data_r.uniform_()
            data_r.requires_grad = True
            feat_r = checkpoint(module, data_r)
            feat_combined.append(feat_r)

def sample_inputs_tensordot(self, device, dtype, requires_grad, **kwargs):
    cases = (
        ((2, 2, 2), (2, 2, 2), (2)),
        ((2, 2, 1), (2, 1, 2), ([0, 1], [2, 0])),
    )
    samples = []
    for first_shape, second_shape, dims in cases:
        samples.append(SampleInput(make_tensor(first_shape, device, dtype,
                                   requires_grad=requires_grad),
                       args=(make_tensor(second_shape, device, dtype,
                             requires_grad=requires_grad),),
                       kwargs=dict(dims=dims,)))
    return tuple(samples)

def sample_inputs_kron(op_info, device, dtype, requires_grad):
    test_cases = (
        ((S, S), (M, L)),
    )

    sample_inputs = []
    for input_shape, other_shape in test_cases:
        input = make_tensor(input_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        other = make_tensor(other_shape, device, dtype, low=None, high=None, requires_grad=requires_grad)
        sample = SampleInput(input, args=(other,))
        sample_inputs.append(sample)
    return tuple(sample_inputs)

def sample_inputs_inner(self, device, dtype, requires_grad, **kwargs):
    return (
        SampleInput(
            make_tensor((S, ), device, dtype, requires_grad=requires_grad),
            args=(
                make_tensor((S, ), device, dtype, requires_grad=requires_grad),
            )
        ),
        SampleInput(
            make_tensor((), device, dtype, requires_grad=requires_grad),
            args=(
                make_tensor((S, S), device, dtype, requires_grad=requires_grad),
            )
        ),
    )

# Tests for scatter when passing the reduce argument are missing
# Reference: https://github.com/pytorch/pytorch/issues/56464
def sample_inputs_scatter(op_info, device, dtype, requires_grad):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        return gather_variable(shape, index_dim, max_indices, device=device)

    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (
        (_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))),
        (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))),
        (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor(()), (0, zero.clone().detach(), _tensor(()))),
        (_tensor(()), (0, zero.clone().detach(), 2.5)),
    )

    return [SampleInput(tensor, args=args) for tensor, args in test_cases]

def sample_inputs_scatter_add(op_info, device, dtype, requires_grad):
    def _tensor(shape, dtype=dtype, low=None, high=None):
        return make_tensor(shape, device, dtype, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        return gather_variable(shape, index_dim, max_indices, device=device)

    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (
        (_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))),
        (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))),
        (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))),
        (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))),
        (_tensor(()), (0, zero.clone().detach(), _tensor(()))),
    )

    return [SampleInput(tensor, args=args) for tensor, args in test_cases]

foreach_unary_op_db: List[OpInfo] = [
    ForeachUnaryFuncInfo('exp'),
    ForeachUnaryFuncInfo('acos'),
    ForeachUnaryFuncInfo('asin'),
    ForeachUnaryFuncInfo('atan'),
    ForeachUnaryFuncInfo('cos'),
    ForeachUnaryFuncInfo('cosh'),
    ForeachUnaryFuncInfo('log'),
    ForeachUnaryFuncInfo('log10'),
    ForeachUnaryFuncInfo('log2'),
    ForeachUnaryFuncInfo('tan'),
    ForeachUnaryFuncInfo('tanh'),
    ForeachUnaryFuncInfo('sin'),
    ForeachUnaryFuncInfo('sinh'),

    def test_checkpoint_valid_reset_on_error(self):
        a = torch.randn(2, 2, requires_grad=True)

        with self.assertRaisesRegex(Exception, "Checkpointing is not compatible with .grad()"):
            b = checkpoint(torch.exp, a).sum()
            torch.autograd.grad(b, (a,))

        c = checkpoint(torch.exp, a).sum()
        c.backward()

    def _test_reentrant_with_callbacks(self, install_callbacks_in_depths):
        counter = {}
        counter["inner"] = 0
        counter["outer"] = 0

        def inc_inner_counter():
            counter["inner"] += 1

        def inc_outer_counter():
            counter["outer"] += 1

        class MyFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if 1 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_inner_counter)

                return input

        class MyReentrantFunc(Function):
            @staticmethod
            def forward(ctx, input):
                return input

            @staticmethod
            @once_differentiable
            def backward(ctx, input):
                if 0 in install_callbacks_in_depths:
                    # Add a callback to execute.
                    Variable._execution_engine.queue_callback(inc_outer_counter)
                # Reentrant backward call.
                tmp_inp = input.detach().requires_grad_()
                with torch.enable_grad():
                    tmp_out = (MyFunc.apply(tmp_inp)).sum()
                tmp_out.backward()
                return input

        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = MyReentrantFunc.apply(t1)
        t3 = t2.sum()
        torch.autograd.backward([t3])

        return counter

    def test_reentrant_with_callbacks_depth_0(self):
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([0])
        self.assertEqual(1, ret["outer"])
        self.assertEqual(0, ret["inner"])

    def test_reentrant_with_callbacks_depth_1(self):
        # Verify callback is called only once.
        ret = self._test_reentrant_with_callbacks([1])
        self.assertEqual(0, ret["outer"])
        self.assertEqual(1, ret["inner"])

    def test_reentrant_with_callbacks_both_depths(self):
        # Verify callback is called twice.
        ret = self._test_reentrant_with_callbacks([0, 1])
        self.assertEqual(1, ret["outer"])
        self.assertEqual(1, ret["inner"])

    def test_reentrant_with_leaf_variable_hook(self):
        handle = None
        param = torch.rand(10, requires_grad=True)

        def add_gradient_penalty_to_grad(grad):
            handle.remove()
            old_param_grad = grad
            param.grad = None
            # Add some sort of gradient penalty by directly updating the gradients
            with torch.enable_grad():
                g = grad.detach().requires_grad_()
                new_param = param.detach().requires_grad_()
                out = ((g * 2) + new_param).sum()
                out.backward()
            res = g.grad + grad
            param.grad = old_param_grad
            return res

        handle = param.register_hook(add_gradient_penalty_to_grad)
        # Forward pass
        tmp = (param * param)
        loss = tmp.sum()
        # Compute the gradients
        loss.backward()

    def test_reentrant_with_non_leaf_variable_hook(self):
        handle = None
        param = torch.rand(10, requires_grad=True)

def reference_sigmoid(x):
    # 'scipy.special.expit' not supported for the input types
    if x.dtype in [np.complex64, np.complex128]:
        return (1 / (1 + np.exp(-x)))
    return scipy.special.expit(x)


def reference_lgamma(x):
    # scipy.special.gammaln returns `-inf` when input is `-inf`.
    # While Pytorch, C and C++, all return `inf` when input is `-inf`.
    # Reference:
    # https://en.cppreference.com/w/cpp/numeric/math/lgamma
    # https://en.cppreference.com/w/c/numeric/math/lgamma

    # To handle the above discrepancy,
    # we replace -inf with inf so values
    # that were originally -inf map to inf as expected
    if x.dtype.kind == 'f':
        x = np.where(x == float('-inf'), np.array(float('inf'), dtype=x.dtype), x)

    out = scipy.special.gammaln(x)

    if x.dtype == np.float16:
        # `scipy.special.gammaln` returns output of float32 when input is float16,
        # while `torch.lgamma` preserves `float16`. But due to smaller range of float16,
        # Pytorch version outputs `inf` while SciPy returns finite values.
        out = out.astype(np.float16)

    return out

def reference_polygamma(x, n):
    # WEIRD `scipy.special.polygamma` behavior
    # >>> scipy.special.polygamma(0, np.array(501, dtype=np.float32)).dtype
    # dtype('float64')
    # >>> scipy.special.polygamma(0, np.array([501], dtype=np.float32)).dtype
    # dtype('float32')
    #
    # Thus we cast output to the default torch dtype.
    np_dtype = torch_to_numpy_dtype_dict[torch.get_default_dtype()]
    return scipy.special.polygamma(n, x).astype(np_dtype)

def gradcheck_wrapper_hermitian_input(op, input, *args, **kwargs):
    """Gradcheck wrapper for functions that take Hermitian matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the Hermitian property of the input.
    """
    return op(input + input.conj().transpose(-2, -1), *args, **kwargs)


def gradcheck_wrapper_triangular_input(op, input, *args, upper=False, **kwargs):
    """Gradcheck wrpper for functions that take lower or upper triangular matrices as input.

    They require a modified function because the finite-difference algorithm
    for calculating derivatives does not preserve the triangular property of the input.
    """
    return op(input.triu() if upper else input.tril(), upper)


# Operator database (sorted alphabetically)
op_db: List[OpInfo] = [
    UnaryUfuncInfo('abs',
                   aliases=('absolute', ),
                   ref=np.abs,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat]),
                       # Reference: https://github.com/pytorch/pytorch/issues/49224
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.int8], active_if=TEST_WITH_ASAN),
                       # TODO: Fix test_out_arg_all_dtypes as torch.empty_like(expected_output) where expected_output=op(input)
                       # We can break the logic of the loop over all possible types but it is OK.
                       # https://github.com/pytorch/pytorch/blob/master/test/test_unary_ufuncs.py#L440-L449
                       SkipInfo('TestUnaryUfuncs', 'test_out_arg_all_dtypes',
                                dtypes=[torch.cfloat, torch.cdouble]),
                   ),
                   supports_inplace_autograd=False,
                   assert_autodiffed=True),
    # NOTE: CPU complex acos produces incorrect outputs (https://github.com/pytorch/pytorch/issues/42952)
    UnaryUfuncInfo('acos',
                   aliases=('arccos', ),
                   ref=np.arccos,
                   domain=(-1, 1),
                   handles_complex_extremals=False,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   # "rsqrt_cpu" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   assert_autodiffed=True,
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-1,
                                                  torch.complex64: 1e-2}),),
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestGradients', 'test_fn_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_method_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_inplace_grad',
                                dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                   )),
    # NOTE: the derivative for inplace acosh is not implemented
    UnaryUfuncInfo('acosh',
                   aliases=('arccosh', ),
                   ref=np.arccosh,
                   domain=(1, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "rsqrt_cuda" not implemented for 'BFloat16'
                   backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   supports_inplace_autograd=False,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       # Reference: https://github.com/pytorch/pytorch/issues/50692
                       SkipInfo('TestGradients', 'test_fn_grad',
                                device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestGradients', 'test_method_grad',
                                device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS),
                   )),
    OpInfo('add',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           assert_autodiffed=True,
           sample_inputs_func=partial(sample_inputs_binary_pwise, alpha=2),
           supports_inplace_autograd=False),
    OpInfo('mul',
           aliases=('multiply',),
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_binary_pwise),
    OpInfo('sub',
           aliases=('subtract',),
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16),
           assert_autodiffed=True,
           sample_inputs_func=partial(sample_inputs_binary_pwise, alpha=2),
           supports_inplace_autograd=False),
    OpInfo('addmm',
           # This addmm OpInfo is for when alpha and beta are not both equal to 1.
           # alpha=beta=1 is tested in the following opinfo, because that special case will
           # trigger addmm being decomposed by a jit pass.
           dtypes=floating_and_complex_types_and(torch.float16),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_inplace_autograd=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_addmm),
    OpInfo('addmm',
           # When alpha=beta=1 as compile-time constants, JIT will decompose addmm into mm and add.
           variant_test_name='decomposed',
           dtypes=floating_and_complex_types_and(torch.float16),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           supports_inplace_autograd=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           autodiff_nonfusible_nodes=['aten::add', 'aten::mm'],
           sample_inputs_func=partial(sample_inputs_addmm, alpha=1, beta=1)),
    OpInfo('addmv',
           dtypes=floating_types(),
           dtypesIfCPU=all_types_and_complex_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128,
                                           *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half),
           supports_inplace_autograd=False,
           skips=(
               # issue may fix: https://github.com/pytorch/pytorch/issues/55589
               # AssertionError: UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               SkipInfo('TestCommon', 'test_out', dtypes=(torch.float32,)),
               # Reference: https://github.com/pytorch/pytorch/issues/55589
               SkipInfo('TestCommon', 'test_variant_consistency_eager'),
           ),
           sample_inputs_func=sample_inputs_addmv),
    OpInfo('addbmm',
           dtypes=floating_types(),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_types_and(torch.half),
           skips=(
               # addbmm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),
               # https://github.com/pytorch/pytorch/issues/55907
               SkipInfo('TestCommon', 'test_variant_consistency_eager'),
               SkipInfo('TestOpInfo', 'test_supported_backward', dtypes=(torch.bfloat16, ),
                        device_type='cuda', active_if=not SM53OrLater)),
           sample_inputs_func=sample_inputs_addbmm),
    OpInfo('baddbmm',
           dtypes=floating_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128,
                                           *[torch.bfloat16] if CUDA11OrLater else []),
           skips=(
               # baddbmm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),
               SkipInfo('TestOpInfo', 'test_supported_backward', dtypes=(torch.bfloat16, ),
                        device_type='cuda', active_if=not SM53OrLater)),
           sample_inputs_func=sample_inputs_baddbmm),
    OpInfo('dot',
           dtypes=all_types_and_complex_and(torch.float16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_dot_vdot),
    OpInfo('vdot',
           dtypes=all_types_and_complex_and(torch.float16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16),
           sample_inputs_func=sample_inputs_dot_vdot),
    OpInfo('bmm',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           skips=(
               # bmm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),
               SkipInfo('TestOpInfo', 'test_supported_backward', dtypes=(torch.bfloat16, ),
                        device_type='cuda', active_if=not SM53OrLater)),
           sample_inputs_func=sample_inputs_bmm),
    OpInfo('mv',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           skips=(
               # bmm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
               SkipInfo('TestOpInfo', 'test_supported_backward', dtypes=(torch.float16,)),
               # mv calls into addmv which doesn't fully support float16
               # RuntimeError: "addmv_impl_cpu" not implemented for 'Half'
               SkipInfo('TestOpInfo', 'test_supported_dtypes', dtypes=(torch.float16,)),),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_mv),
    OpInfo('addr',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           backward_dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
           backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool),
           # Reference: https://github.com/pytorch/pytorch/issues/50747
           supports_inplace_autograd=False,
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/50747
               SkipInfo('TestCommon', 'test_variant_consistency_eager',
                        dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16)),),
           sample_inputs_func=sample_inputs_addr,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('addcmul',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.float16, torch.bfloat16),
           assert_autodiffed=True,
           supports_inplace_autograd=False,
           skips=(
               # TODO: update sample inputs with for_inplace_variant kwarg to support this test
               SkipInfo('TestCommon', 'test_variant_consistency_eager'),),
           sample_inputs_func=sample_inputs_addcmul_addcdiv),
    OpInfo('addcdiv',
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           skips=(
               # TODO: update sample inputs with for_inplace_variant kwarg to support this test
               SkipInfo('TestCommon', 'test_variant_consistency_eager'),),
           sample_inputs_func=sample_inputs_addcmul_addcdiv),
    OpInfo('amax',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_amax_amin,),
    OpInfo('amin',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_amax_amin),
    OpInfo('argmax',
           dtypes=all_types_and(torch.float16, torch.bfloat16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_argmax_argmin,),
    OpInfo('argmin',
           dtypes=all_types_and(torch.float16, torch.bfloat16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_argmax_argmin,),
    UnaryUfuncInfo('asin',
                   aliases=('arcsin', ),
                   ref=np.arcsin,
                   domain=(-1, 1),
                   supports_sparse=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   # "rsqrt_cpu" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   assert_autodiffed=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS)
                   )),
    # NOTE: derivative for inplace asinh is not implemented
    UnaryUfuncInfo('asinh',
                   aliases=('arcsinh', ),
                   ref=np.arcsinh,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "rsqrt_cuda" not implemented for 'BFloat16'
                   backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   supports_inplace_autograd=False,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('atan',
                   aliases=('arctan', ),
                   ref=np.arctan,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    OpInfo('atan2',
           dtypes=all_types_and(torch.bool),
           dtypesIfCPU=all_types_and(torch.bool),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half),
           sample_inputs_func=sample_inputs_atan2,
           ),
    UnaryUfuncInfo('atanh',
                   aliases=('arctanh', ),
                   ref=np.arctanh,
                   domain=(-1, 1),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   supports_inplace_autograd=False,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.cfloat],
                                active_if=IS_WINDOWS),
                   )),
    OpInfo('broadcast_to',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_broadcast_to),
    UnaryUfuncInfo('bitwise_not',
                   ref=np.bitwise_not,
                   dtypes=integral_types_and(torch.bool),
                   dtypesIfCPU=None,
                   dtypesIfCUDA=None,
                   dtypesIfROCM=None,
                   supports_autograd=False),
    OpInfo('cdist',
           dtypes=floating_types(),
           supports_out=False,
           supports_gradgrad=False,
           sample_inputs_func=sample_inputs_cdist),
    UnaryUfuncInfo('ceil',
                   ref=np.ceil,
                   dtypes=floating_types_and(torch.half),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   assert_autodiffed=True),
    OpInfo('cholesky',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('cholesky_inverse',
           dtypes=floating_and_complex_types(),
           backward_dtypes=floating_types(),
           # TODO: RuntimeError: cholesky_inverse does not support automatic differentiation for outputs
           # with complex dtype.
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky_inverse,
           gradcheck_wrapper=gradcheck_wrapper_triangular_input,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           skips=(
               # cholesky_inverse does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),)),
    OpInfo('symeig',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_symeig,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)
           ),
    UnaryUfuncInfo('clamp',
                   aliases=('clip', ),
                   decorators=(precisionOverride({torch.bfloat16: 7e-2, torch.float16: 1e-2}),),
                   ref=np.clip,
                   dtypes=all_types_and(torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and(torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/54841
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                   ),
                   sample_kwargs=sample_kwargs_clamp,
                   sample_inputs_func=sample_inputs_clamp),
    UnaryUfuncInfo('positive',
                   ref=np.positive,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16),
                   supports_out=False,
                   ),
    UnaryUfuncInfo('conj',
                   ref=np.conj,
                   dtypes=all_types_and_complex_and(torch.bool,
                                                    torch.bfloat16, torch.half),
                   dtypesIfCPU=None,
                   dtypesIfCUDA=None,
                   dtypesIfROCM=None,
                   skips=(
                       # File "test_unary_ufuncs.py", line 289, in test_reference_numerics
                       #  if not torch.can_cast(numpy_to_torch_dtype_dict[expected.dtype.type], dtype):
                       # KeyError: <class 'numpy.intc'>
                       # Following error in Windows CI
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.int],
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.int],
                                active_if=IS_WINDOWS),
                   )),
    OpInfo('view_as_real',
           dtypes=complex_types(),
           sample_inputs_func=sample_inputs_view_as_real,
           ),
    OpInfo('view_as_complex',
           dtypes=floating_types_and(torch.half),
           supports_out=False,
           skips=(
               # "sum_cpu/sum_cuda" not implemented for 'ComplexHalf'
               SkipInfo('TestOpInfo', 'test_supported_backward', dtypes=(torch.half,)),
           ),
           sample_inputs_func=sample_inputs_view_as_complex),
    OpInfo('complex',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_complex,
           ),
    OpInfo('copysign',
           dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_copysign,
           supports_inplace_autograd=False,
           ),
    UnaryUfuncInfo('cos',
                   ref=np.cos,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "sin_cuda" not implemented for 'BFloat16'
                   backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   handles_large_floats=False,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                   )),
    UnaryUfuncInfo('cosh',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.cosh),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/48641
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.int8]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard', device_type='cpu',
                                dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS),
                   )),
    OpInfo('cumsum',
           dtypesIfCPU=all_types_and_complex_and(torch.bool),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
           skips=(
               # "cumsum_out_{cpu,cuda}" not implemented for 'Bool'
               SkipInfo('TestOpInfo', 'test_supported_dtypes',
                        dtypes=(torch.bool,)),
               # cumsum does not handle correctly out= dtypes
               SkipInfo('TestCommon', 'test_out'),
           ),
           sample_inputs_func=sample_inputs_cumulative_ops),
    OpInfo('cumprod',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16),
           skips=(
               # "cumprod_out_{cpu, cuda}" not implemented for 'Bool'
               SkipInfo('TestOpInfo', 'test_supported_dtypes',
                        dtypes=(torch.bool,)),
               # cumprod does not handle correctly out= dtypes
               SkipInfo('TestCommon', 'test_out',
                        dtypes=[torch.float32]),
           ),
           # gradgradcheck fails in fast_mode=True: #56275
           sample_inputs_func=sample_inputs_cumprod,
           gradcheck_fast_mode=False),
    OpInfo('cummax',
           dtypesIfCPU=all_types_and(torch.bool),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half),
           sample_inputs_func=partial(sample_inputs_cumulative_ops, supports_dtype_kwargs=False),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('cummin',
           dtypesIfCPU=all_types_and(torch.bool),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half),
           sample_inputs_func=partial(sample_inputs_cumulative_ops, supports_dtype_kwargs=False),
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    UnaryUfuncInfo('deg2rad',
                   ref=np.radians,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/51283#issuecomment-770614273
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16]),
                   ),
                   safe_casts_outputs=True),
    OpInfo('diff',
           op=torch.diff,
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_diff),
    OpInfo('div',
           variant_test_name='no_rounding_mode',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_div,
           skips=(SkipInfo('TestOpInfo', 'test_duplicate_method_tests'),),
           assert_autodiffed=True),
    OpInfo('div',
           variant_test_name='true_rounding',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_div, rounding_mode=None),
           skips=(SkipInfo('TestOpInfo', 'test_duplicate_method_tests'),),
           assert_autodiffed=True),
    OpInfo('div',
           variant_test_name='trunc_rounding',
           dtypes=all_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_div, rounding_mode='trunc'),
           skips=(SkipInfo('TestOpInfo', 'test_duplicate_method_tests'),),
           assert_autodiffed=True),
    OpInfo('div',
           variant_test_name='floor_rounding',
           dtypes=all_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=partial(sample_inputs_div, rounding_mode='floor'),
           skips=(SkipInfo('TestOpInfo', 'test_duplicate_method_tests'),),
           assert_autodiffed=True),
    UnaryUfuncInfo('exp',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.exp),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/50093#pullrequestreview-561791547
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal', dtypes=[torch.bfloat16]),
                       # Reference: https://github.com/pytorch/pytorch/issues/48010
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                   ),
                   assert_autodiffed=True,
                   safe_casts_outputs=True),
    OpInfo('diag',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCPU=all_types_and_complex_and(torch.bool),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
           sample_inputs_func=sample_inputs_diag),
    OpInfo('eq',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('fmax',
           op=torch.fmax,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_binary,),
    OpInfo('fmin',
           op=torch.fmin,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_binary,),
    UnaryUfuncInfo('frac',
                   ref=lambda x: np.modf(x)[0],
                   dtypes=floating_types_and(torch.bfloat16, torch.float16),
                   dtypesIfCPU=floating_types_and(torch.bfloat16, torch.float16),
                   dtypesIfCUDA=floating_types_and(torch.float16),
                   assert_autodiffed=True,
                   # Reference for disabling extremals
                   # https://github.com/pytorch/pytorch/issues/51948
                   handles_extremals=False),
    SpectralFuncInfo('fft.fft',
                     aten_name='fft_fft',
                     ref=np.fft.fft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types()),
    SpectralFuncInfo('fft.fftn',
                     aten_name='fft_fftn',
                     ref=np.fft.fftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     decorators=[precisionOverride(
                         {torch.float: 1e-4, torch.cfloat: 1e-4})],),
    SpectralFuncInfo('fft.hfft',
                     aten_name='fft_hfft',
                     ref=np.fft.hfft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.rfft',
                     aten_name='fft_rfft',
                     ref=np.fft.rfft,
                     ndimensional=False,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.rfftn',
                     aten_name='fft_rfftn',
                     ref=np.fft.rfftn,
                     ndimensional=True,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_grad=False,
                     check_batched_gradgrad=False,
                     decorators=[precisionOverride({torch.float: 1e-4})],),
    SpectralFuncInfo('fft.ifft',
                     aten_name='fft_ifft',
                     ref=np.fft.ifft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types()),
    SpectralFuncInfo('fft.ifftn',
                     aten_name='fft_ifftn',
                     ref=np.fft.ifftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types()),
    SpectralFuncInfo('fft.ihfft',
                     aten_name='fft_ihfft',
                     ref=np.fft.ihfft,
                     ndimensional=False,
                     dtypes=all_types_and(torch.bool),
                     default_test_dtypes=floating_types(),
                     check_batched_grad=False),
    SpectralFuncInfo('fft.irfft',
                     aten_name='fft_irfft',
                     ref=np.fft.irfft,
                     ndimensional=False,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False),
    SpectralFuncInfo('fft.irfftn',
                     aten_name='fft_irfftn',
                     ref=np.fft.irfftn,
                     ndimensional=True,
                     dtypes=all_types_and_complex_and(torch.bool),
                     default_test_dtypes=floating_and_complex_types(),
                     check_batched_gradgrad=False),
    UnaryUfuncInfo('floor',
                   ref=np.floor,
                   dtypes=floating_types_and(torch.half),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   assert_autodiffed=True),
    OpInfo('flip',
           op=torch.flip,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_flip,
           supports_out=False),
    OpInfo('fliplr',
           op=torch.fliplr,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_fliplr_flipud,
           supports_out=False),
    OpInfo('flipud',
           op=torch.flipud,
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_fliplr_flipud,
           supports_out=False),
    UnaryUfuncInfo('i0',
                   ref=np.i0,
                   decorators=(precisionOverride({torch.bfloat16: 3e-1,
                                                  torch.float16: 5e-1}),),
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
                   supports_autograd=False),
    UnaryUfuncInfo('special.i0e',
                   aten_name='special_i0e',
                   ref=scipy.special.i0e if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.bfloat16: 3e-1,
                                                  torch.float16: 3e-1}),),
                   dtypes=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   supports_autograd=False,
                   safe_casts_outputs=True),
    OpInfo('floor_divide',
           dtypes=all_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_floor_divide,
           decorators=[_wrap_warn_once("floor_divide is deprecated, and will be removed")],
           skips=(
               # `test_duplicate_method_tests` doesn't raise any warning, as it doesn't actually
               # call the operator.
               SkipInfo('TestOpInfo', 'test_duplicate_method_tests'),),
           supports_autograd=False,
           ),
    UnaryUfuncInfo('frexp',
                   op=torch.frexp,
                   ref=np.frexp,
                   dtypesIfCPU=floating_types_and(torch.half),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   # skip testing torch.frexp as it is not supported by ROCm platform yet
                   decorators=[skipCUDAIfRocm],
                   supports_out=False,
                   skips=(
                       # skips below tests as torch.frexp returns tuple-like (mantissa, exponent) as outputs,
                       # while theses tests currently requires output to a single tensor.
                       SkipInfo('TestUnaryUfuncs', 'test_batch_vs_slicing'),
                       SkipInfo('TestUnaryUfuncs', 'test_contig_vs_every_other'),
                       SkipInfo('TestUnaryUfuncs', 'test_contig_vs_transposed'),
                       SkipInfo('TestUnaryUfuncs', 'test_non_contig_expand'),
                       SkipInfo('TestUnaryUfuncs', 'test_variant_consistency'),

                       # skips test_reference_numerics due to error in Windows CI.
                       # The np.frexp returns exponent as np.intc dtype on Windows platform,
                       # and np.intc does not have the correspond torch dtype
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                active_if=IS_WINDOWS),
                   )),
    OpInfo('ge',
           aliases=('greater_equal',),
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('geqrf',
           dtypes=floating_and_complex_types(),
           dtypesIfCPU=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_geqrf,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],),
    OpInfo('gt',
           aliases=('greater',),
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    UnaryUfuncInfo('imag',
                   ref=np.imag,
                   dtypes=complex_types(),
                   dtypesIfCPU=complex_types(),
                   dtypesIfCUDA=complex_types(),
                   dtypesIfROCM=complex_types(),
                   supports_out=False,
                   supports_autograd=False,
                   skips=(
                       # Skip since real and imag don't have out variants.
                       SkipInfo('TestUnaryUfuncs', 'test_out_arg_all_dtypes'),
                   )),
    OpInfo('inverse',
           op=torch.inverse,
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('le',
           aliases=('less_equal',),
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('linalg.det',
           op=torch.linalg.det,
           aliases=('det', ),
           dtypes=floating_and_complex_types(),
           # det doesn't support complex autograd, https://github.com/pytorch/pytorch/issues/57358
           backward_dtypes=floating_types(),
           aten_name='linalg_det',
           sample_inputs_func=sample_inputs_linalg_det,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           supports_inplace_autograd=False,
           skips=(
               # The following tests fail only on ROCm. This is probably
               # related to the fact that the current linalg.det backward is
               # unstable if the matrix has repeated singular values, see
               # https://github.com/pytorch/pytorch/issues/53364
               SkipInfo('TestGradients', 'test_fn_grad', device_type='cuda',
                        dtypes=(torch.float64,), active_if=TEST_WITH_ROCM),
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda',
                        dtypes=(torch.float64,), active_if=TEST_WITH_ROCM),
               SkipInfo('TestCommon', 'test_variant_consistency_jit', device_type='cuda',
                        dtypes=(torch.float64, torch.float32), active_if=TEST_WITH_ROCM),
           )),
    OpInfo('linalg.cholesky',
           aten_name='linalg_cholesky',
           dtypes=floating_and_complex_types(),
           # TODO: RuntimeError: While computing batched gradients,
           # got: vmap: Calling Tensor.as_strided is not supported
           # unless the batch dims being vmapped over are at the front of the tensor (in memory layout).
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_cholesky,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)
           ),
    OpInfo('linalg.eig',
           aten_name='linalg_eig',
           op=torch.linalg.eig,
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.eigvals',
           aten_name='linalg_eigvals',
           op=torch.linalg.eigvals,
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.eigh',
           aten_name='linalg_eigh',
           dtypes=floating_and_complex_types(),
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_eigh,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)
           ),
    OpInfo('linalg.householder_product',
           aten_name='linalg_householder_product',
           op=torch.linalg.householder_product,
           aliases=('orgqr', ),
           dtypes=floating_and_complex_types(),
           # TODO: backward uses in-place operations that vmap doesn't like
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_householder_product,
           decorators=[skipCUDAIfNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack,
                       # gradgrad checks are slow
                       DecorateInfo(slowTest, 'TestGradients', 'test_fn_gradgrad'), ]),
    OpInfo('linalg.lstsq',
           aten_name='linalg_lstsq',
           op=torch.linalg.lstsq,
           dtypes=floating_and_complex_types(),
           supports_out=True,
           sample_inputs_func=sample_inputs_linalg_lstsq,
           check_batched_grad=False,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           skips=(
               # skip because `linalg_lstsq` is not differentiable
               SkipInfo('TestGradients', 'test_fn_grad'),
               SkipInfo('TestCommon', 'test_variant_consistency_jit'),
           )),
    OpInfo('linalg.matrix_power',
           aliases=('matrix_power',),
           aten_name='linalg_matrix_power',
           dtypes=floating_and_complex_types(),
           supports_inplace_autograd=False,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, skipCUDAIfRocm],
           sample_inputs_func=sample_inputs_linalg_matrix_power,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('linalg.multi_dot',
           # Need this lambda because gradcheck does not work with TensorList inputs
           aten_name='linalg_multi_dot',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, *[torch.bfloat16] if CUDA11OrLater else []),
           supports_inplace_autograd=False,
           # Batched grad checks fail for empty input tensors (see https://github.com/pytorch/pytorch/issues/53407)
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_multi_dot,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('linalg.norm',
           op=torch.linalg.norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_norm,
           aten_name='linalg_norm',
           skips=(
               # linalg.norm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('linalg.qr',
           aten_name='linalg_qr',
           op=torch.linalg.qr,
           dtypes=floating_and_complex_types(),
           # batched gradients do not work for empty inputs
           # https://github.com/pytorch/pytorch/issues/50743#issuecomment-767376085
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_qr,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('linalg.slogdet',
           aten_name='linalg_slogdet',
           op=torch.linalg.slogdet,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_slogdet,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.vector_norm',
           op=torch.linalg.vector_norm,
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
           sample_inputs_func=sample_inputs_linalg_vector_norm,
           aten_name='linalg_vector_norm',
           skips=(
               # linalg.vector_norm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    UnaryUfuncInfo('log',
                   ref=np.log,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('log10',
                   ref=np.log10,
                   domain=(0, float('inf')),
                   decorators=(precisionOverride({torch.bfloat16: 5e-2}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   assert_autodiffed=True,
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_WINDOWS),
                   )),
    UnaryUfuncInfo('log1p',
                   ref=np.log1p,
                   domain=(-1, float('inf')),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   safe_casts_outputs=True,
                   assert_autodiffed=True),
    UnaryUfuncInfo('log2',
                   ref=np.log2,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-1}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.cfloat, torch.cdouble]),
                   )),
    OpInfo('logaddexp',
           dtypes=floating_types(),
           sample_inputs_func=lambda op_info, device, dtype, requires_grad=False, **kwargs:
           (SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad),
                        args=(make_tensor((S, S), device, dtype, requires_grad=requires_grad),)),)),
    OpInfo('logaddexp2',
           dtypes=floating_types(),
           sample_inputs_func=lambda op_info, device, dtype, requires_grad=False, **kwargs:
           (SampleInput(make_tensor((S, S), device, dtype, requires_grad=requires_grad),
                        args=(make_tensor((S, S), device, dtype, requires_grad=requires_grad),)),)),
    UnaryUfuncInfo('logical_not',
                   ref=np.logical_not,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 5e-1}),),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   safe_casts_outputs=True,
                   supports_autograd=False,
                   skips=(
                       # The function variant always returns BoolTensor
                       # while the inplace variant preserves the input dtype.
                       # >>> t = torch.randn(3)
                       # >>> torch.logical_not(t)
                       # tensor([False, False, False])
                       # >>> torch.logical_not(t).dtype
                       # torch.bool
                       # >>> t.logical_not_().dtype
                       # torch.float32
                       SkipInfo('TestUnaryUfuncs', 'test_variant_consistency',
                                dtypes=all_types_and_complex_and(torch.half, torch.bfloat16)),
                       SkipInfo('TestCommon', 'test_variant_consistency_eager',
                                dtypes=all_types_and_complex_and(torch.half, torch.bfloat16)),
                   )),
    OpInfo('lt',
           aliases=('less',),
           dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    OpInfo('lu',
           op=torch.lu,
           dtypes=floating_and_complex_types(),
           supports_inplace_autograd=False,
           check_batched_gradgrad=False,
           supports_out=False,
           sample_inputs_func=sample_inputs_lu,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),
               # we skip jit tests because lu_backward is impelemented as autograd.Function,
               # which does not support autograd with scripting
               SkipInfo('TestCommon', 'test_variant_consistency_jit'),
               # Skip operator schema test because this is a functional and not an operator
               SkipInfo('TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )),
    OpInfo('masked_fill',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_fill,
           supports_out=False),
    OpInfo('masked_scatter',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_scatter,
           supports_out=False),
    OpInfo('masked_select',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_masked_select),
    OpInfo('matrix_exp',
           dtypesIfCPU=floating_and_complex_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16),
           sample_inputs_func=sample_inputs_matrix_exp,
           supports_out=False),
    OpInfo('matmul',
           dtypes=floating_types(),
           dtypesIfCPU=all_types_and_complex(),
           dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128),
           dtypesIfROCM=floating_types_and(torch.half),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_matmul,
           skips=(
               # matmul does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
               # https://github.com/pytorch/pytorch/issues/55754
               SkipInfo('TestGradients', 'test_fn_grad',
                        device_type='cpu', dtypes=(torch.complex128,)),
               # https://github.com/pytorch/pytorch/issues/55755
               SkipInfo('TestOpInfo', 'test_unsupported_dtypes',
                        device_type='cpu', dtypes=(torch.float16,)),)),
    OpInfo('max',
           op=torch.max,
           variant_test_name='binary',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_binary,
           assert_autodiffed=True,),
    OpInfo('max',
           op=torch.max,
           variant_test_name='reduction_with_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_reduction_with_dim,
           skips=(
               # max does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),)),
    OpInfo('max',
           op=torch.max,
           variant_test_name='reduction_no_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_out=False,
           sample_inputs_func=sample_inputs_max_min_reduction_no_dim,),
    OpInfo('min',
           op=torch.min,
           variant_test_name='binary',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_binary,
           assert_autodiffed=True,),
    OpInfo('min',
           op=torch.min,
           variant_test_name='reduction_with_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_reduction_with_dim,
           skips=(
               # min does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('min',
           op=torch.min,
           variant_test_name='reduction_no_dim',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           supports_out=False,
           sample_inputs_func=sample_inputs_max_min_reduction_no_dim,),
    OpInfo('sum',
           dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool),
           supports_out=False,
           sample_inputs_func=sample_inputs_reduction_wrapper(supports_multiple_dims=True)),
    OpInfo('nansum',
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           dtypesIfCPU=all_types_and(torch.float16, torch.bool),
           supports_out=False,
           sample_inputs_func=sample_inputs_reduction_wrapper(supports_multiple_dims=True)),
    # TODO(@heitorschueroff) Add test for dtype kwarg
    OpInfo('mean',
           dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_reduction_wrapper(supports_multiple_dims=True),
           # Need to skip out test because one of the overload for mean does not support it
           # TODO(@heitorschueroff) fix this when implementing ReductionInfo
           skips=(SkipInfo('TestCommon', 'test_out'),)),
    OpInfo('quantile',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_reduction_quantile),
    OpInfo('nanquantile',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_reduction_quantile),
    OpInfo('maximum',
           op=torch.maximum,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_binary,),
    OpInfo('minimum',
           op=torch.minimum,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_max_min_binary,),
    OpInfo('topk',
           dtypes=all_types(),
           dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16),
           sample_inputs_func=sample_inputs_topk,
           skips=(
               # Topk is not raising a warning when the out is resized
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('mm',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.float16, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_mm,
           skips=(
               # mm does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('mode',
           op=torch.mode,
           dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_mode,),
    OpInfo('ne',
           aliases=('not_equal',),
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_comparison_ops),
    UnaryUfuncInfo('neg',
                   aliases=('negative', ),
                   ref=np.negative,
                   dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16),
                   assert_autodiffed=True,),
    OpInfo('dist',
           op=torch.dist,
           dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
           # "pow" not implemented for 'BFloat16' or 'half'
           backward_dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_dist,
           skips=(
               # dist does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('outer',
           op=torch.outer,
           aliases=('ger', ),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_outer,),
    OpInfo('permute',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_permute),
    OpInfo('pow',
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool),
           # Due to AVX2 curently not being fully supported for Float16, log_vml_cpu can't be enabled
           # for Float16, causing this test to fail. pow's autograd for Float16 is thus currently
           # unsupported on CPU.
           backward_dtypes=all_types_and_complex_and(torch.bfloat16, torch.bool),
           sample_inputs_func=sample_inputs_pow,
           supports_inplace_autograd=False,
           assert_autodiffed=True),
    OpInfo('prod',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           # "cumprod_cuda" not implemented for 'BFloat16'
           backward_dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16),
           skips=(
               # prod does not support the (Tensor, *, out) overload
               SkipInfo('TestCommon', 'test_out',
                        dtypes=[torch.float32]),
           ),
           sample_inputs_func=sample_inputs_prod,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('qr',
           op=torch.qr,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_qr,
           # batched gradients do not work for empty inputs
           # https://github.com/pytorch/pytorch/issues/50743#issuecomment-767376085
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    UnaryUfuncInfo('rad2deg',
                   ref=np.degrees,
                   decorators=(precisionOverride({torch.bfloat16: 7e-1,
                                                  torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCPU=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/51283#issuecomment-770614273
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16]),
                   ),
                   safe_casts_outputs=True),
    UnaryUfuncInfo('real',
                   ref=np.real,
                   dtypes=complex_types(),
                   dtypesIfCPU=complex_types(),
                   dtypesIfCUDA=complex_types(),
                   dtypesIfROCM=complex_types(),
                   supports_out=False,
                   supports_autograd=False,
                   skips=(
                       # Skip since real and imag don't have out variants.
                       SkipInfo('TestUnaryUfuncs', 'test_out_arg_all_dtypes'),
                   )),
    OpInfo('roll',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
           dtypesIfROCM=all_types_and_complex_and(torch.bool, torch.half),
           supports_out=False,
           sample_inputs_func=sample_inputs_roll),
    OpInfo('rot90',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           sample_inputs_func=sample_inputs_rot90),
    UnaryUfuncInfo('round',
                   ref=np.round,
                   dtypes=floating_types_and(torch.half),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   assert_autodiffed=True,),
    UnaryUfuncInfo('sin',
                   ref=np.sin,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),)),
    UnaryUfuncInfo('sinc',
                   ref=np_sinc_with_fp16_as_fp32,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   handles_large_floats=False,
                   handles_complex_extremals=False,
                   safe_casts_outputs=True,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2,
                                                  torch.float16: 1e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/49133
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.cfloat]),
                   )),
    UnaryUfuncInfo('sinh',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.sinh),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   decorators=(precisionOverride({torch.float16: 1e-2}),),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       # Reference: https://github.com/pytorch/pytorch/issues/48641
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.int8]),
                   )),
    UnaryUfuncInfo('sign',
                   ref=reference_sign,
                   dtypes=all_types_and(torch.bfloat16, torch.half),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/41245
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),
                   )),
    UnaryUfuncInfo('sgn',
                   ref=reference_sgn,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/41245
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),
                       # Reference: https://github.com/pytorch/pytorch/issues/53958
                       # Test fails in comparison on Nan as the `equal_nan` is True for
                       # comparing the CPU tensors.
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.complex64, torch.complex128]),
                       # Reference: https://github.com/pytorch/pytorch/issues/48486
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.complex64])
                   )),
    OpInfo('rsub',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
           variant_test_name='rsub_tensor',
           supports_out=False,
           supports_inplace_autograd=False,
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/53797
               # JIT doesn't understand complex literals
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=[torch.cfloat, torch.cdouble]),
           ),
           sample_inputs_func=partial(sample_inputs_rsub, variant='tensor'),),
    OpInfo('rsub',
           dtypes=all_types_and_complex_and(torch.bfloat16, torch.half),
           variant_test_name='rsub_scalar',
           supports_out=False,
           supports_inplace_autograd=False,
           sample_inputs_func=partial(sample_inputs_rsub, variant='scalar'),
           skips=(
               # Reference: https://github.com/pytorch/pytorch/issues/53797
               # JIT doesn't understand complex literals
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=all_types_and_complex_and(torch.bfloat16, torch.half)),),
           assert_autodiffed=True,),
    UnaryUfuncInfo('signbit',
                   ref=np.signbit,
                   dtypes=all_types_and(torch.bfloat16, torch.half),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.half),
                   supports_autograd=False,),
    OpInfo('solve',
           op=torch.solve,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_legacy_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           # cuda gradchecks are slow
           # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
           skips=(SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('std',
           dtypes=floating_types_and(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           # std doesn't support complex autograd, https://github.com/pytorch/pytorch/issues/57358
           backward_dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_std_var,
           # TODO: std does support out in some signatures
           supports_out=False,
           # std has only partial support for complex and half (#51127)
           skips=(SkipInfo('TestOpInfo', 'test_unsupported_dtypes',
                           dtypes=[torch.half, torch.complex64, torch.complex128]),),
           assert_autodiffed=True,
           ),
    UnaryUfuncInfo('tan',
                   ref=np.tan,
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.float64],
                                active_if=TEST_WITH_ROCM),
                   )),
    UnaryUfuncInfo('tanh',
                   ref=np.tanh,
                   decorators=(precisionOverride({torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # "tanh_backward_cpu" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=(IS_MACOS or IS_WINDOWS)),
                   )),
    OpInfo('tensor_split',
           dtypes=all_types_and_complex_and(torch.bool),
           dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           skips=(SkipInfo('TestOpInfo', 'test_duplicate_method_tests'),),
           sample_inputs_func=sample_inputs_tensor_split,),
    OpInfo('hsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           sample_inputs_func=sample_inputs_hsplit,),
    OpInfo('vsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           sample_inputs_func=sample_inputs_vsplit,),
    OpInfo('dsplit',
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
           supports_out=False,
           sample_inputs_func=sample_inputs_dsplit,),
    OpInfo('triangular_solve',
           op=torch.triangular_solve,
           dtypes=floating_and_complex_types(),
           supports_out=False,
           sample_inputs_func=sample_inputs_legacy_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           # CUDA gradchecks are slow and triangular solve backward is a composite operation
           # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
           skips=(SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    UnaryUfuncInfo('trunc',
                   aliases=('fix', ),
                   ref=np.trunc,
                   dtypes=floating_types_and(torch.bfloat16),
                   dtypesIfCPU=floating_types_and(torch.bfloat16),
                   dtypesIfCUDA=floating_types_and(torch.float16),
                   assert_autodiffed=True),
    UnaryUfuncInfo('exp2',
                   aliases=('special.exp2', ),
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.exp2),
                   dtypes=all_types_and(torch.bool, torch.half),
                   dtypesIfCPU=all_types_and(torch.bool, torch.half),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True),
    UnaryUfuncInfo('expm1',
                   aliases=('special.expm1', ),
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.expm1),
                   dtypes=all_types_and(torch.bool, torch.half),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/48926#issuecomment-739734774
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                   )),
    UnaryUfuncInfo('nan_to_num',
                   ref=np.nan_to_num,
                   dtypes=all_types_and(torch.half, torch.bool),
                   dtypesIfCPU=None,
                   dtypesIfCUDA=None),
    UnaryUfuncInfo('reciprocal',
                   ref=np_unary_ufunc_integer_promotion_wrapper(np.reciprocal),
                   dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   dtypesIfCPU=None,
                   dtypesIfCUDA=None,
                   assert_autodiffed=True,
                   safe_casts_outputs=True,
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/45690
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.cfloat, torch.cdouble]),
                       # Reference: https://github.com/pytorch/pytorch/pull/49102#issuecomment-744604601
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.bfloat16]),
                   )),
    UnaryUfuncInfo('rsqrt',
                   ref=lambda x: np.reciprocal(np.sqrt(x)),
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
                   decorators=(precisionOverride({torch.half: 5e-2}),),
                   safe_casts_outputs=True,
                   assert_autodiffed=True,
                   handles_complex_extremals=False),
    UnaryUfuncInfo('sqrt',
                   ref=np.sqrt,
                   supports_sparse=True,
                   domain=(0, float('inf')),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   decorators=(precisionOverride({torch.bfloat16: 7e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/47358
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble],
                                active_if=IS_MACOS),
                       # Reference: https://github.com/pytorch/pytorch/pull/47293#issuecomment-721774436
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16])),
                   safe_casts_outputs=True,
                   handles_complex_extremals=False),
    UnaryUfuncInfo('square',
                   ref=np.square,
                   dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   decorators=(precisionOverride({torch.complex64: 3e-4, torch.bfloat16: 3e-1}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/52549
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.cfloat, torch.cdouble]),
                       # >>> t = torch.tensor(complex(-0.01, float("inf")))
                       # >>> np.square(t.numpy())
                       # (-inf-infj)
                       # >>> t.square()
                       # tensor(-inf-infj)
                       # >>> t.cuda().square()
                       # tensor(inf+nanj, device='cuda:0')
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]),
                       # Reference: https://github.com/pytorch/pytorch/pull/52551#issuecomment-782596181
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16]),
                   ),),
    OpInfo('lerp',
           dtypes=floating_and_complex_types(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half),
           dtypesIfROCM=floating_and_complex_types_and(torch.half),
           sample_inputs_func=sample_inputs_lerp,
           assert_autodiffed=True),
    OpInfo('linalg.inv',
           aten_name='linalg_inv',
           op=torch.linalg.inv,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_invertible,
           check_batched_gradgrad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # linalg_inv does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    UnaryUfuncInfo('angle',
                   ref=np.angle,
                   dtypes=all_types_and_complex_and(torch.bool),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool),
                   dtypesIfROCM=all_types_and_complex_and(torch.bool),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   safe_casts_outputs=True,
                   supports_complex_to_float=True),
    OpInfo('linalg.solve',
           aten_name='linalg_solve',
           op=torch.linalg.solve,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_solve,
           check_batched_gradgrad=False,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('linalg.matrix_rank',
           aten_name='linalg_matrix_rank',
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.matrix_rank',
           aten_name='linalg_matrix_rank',
           variant_test_name='hermitian',
           dtypes=floating_and_complex_types(),
           supports_autograd=False,
           sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack]),
    OpInfo('linalg.pinv',
           aten_name='linalg_pinv',
           op=torch.linalg.pinv,
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('linalg.pinv',
           aten_name='linalg_pinv',
           variant_test_name='hermitian',
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
           gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
           decorators=[skipCUDAIfNoMagma, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('eig',
           op=torch.eig,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_eig,
           decorators=[
               skipCUDAIfNoMagma,
               skipCPUIfNoLapack,
               skipCUDAIfRocm
           ],),
    OpInfo('einsum',
           # we need this lambda because SampleInput expects tensor input as the first argument
           # TODO(@heitorschueroff) update SampleInput to handle such cases
           op=lambda tensors, equation: torch.einsum(equation, tensors),
           dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half),
           supports_out=False,
           sample_inputs_func=sample_inputs_einsum,
           skips=(
               # test does not work with passing lambda for op
               # there's a test `test_einsum` in `test_jit.py` to handle this case
               SkipInfo('TestCommon', 'test_variant_consistency_jit'),
               # The following dtypes are only supported for some inputs, ideally we should have
               # checked this in the einsum code but to keep BC we'll just skip the tests for now.
               SkipInfo('TestOpInfo', 'test_unsupported_dtypes',
                        dtypes=[torch.bool]),
               SkipInfo('TestOpInfo', 'test_unsupported_dtypes',
                        device_type='cuda', dtypes=integral_types_and(torch.bfloat16)))),
    OpInfo('svd',
           op=torch.svd,
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_svd,
           decorators=[
               skipCUDAIfNoMagmaAndNoCusolver,
               skipCUDAIfRocm,
               skipCPUIfNoLapack,
               # gradgrad checks are slow
               DecorateInfo(slowTest, 'TestGradients', 'test_fn_gradgrad'),
           ],
           skips=(
               # cuda gradchecks are very slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('linalg.svd',
           op=torch.linalg.svd,
           aten_name='linalg_svd',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_svd,
           decorators=[
               skipCUDAIfNoMagmaAndNoCusolver,
               skipCUDAIfRocm,
               skipCPUIfNoLapack,
               # gradgrad checks are slow
               DecorateInfo(slowTest, 'TestGradients', 'test_fn_gradgrad'),
           ],
           skips=(
               # cuda gradchecks are very slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('linalg.svdvals',
           op=torch.linalg.svdvals,
           aten_name='linalg_svdvals',
           dtypes=floating_and_complex_types(),
           sample_inputs_func=sample_inputs_linalg_svdvals,
           supports_autograd=False,
           decorators=[
               skipCUDAIfNoMagmaAndNoCusolver,
               skipCPUIfNoLapack]),
    OpInfo('polar',
           dtypes=floating_types(),
           sample_inputs_func=sample_inputs_polar),
    # To test reference numerics against multiple values of argument `n`,
    # we make multiple OpInfo entries with each entry corresponding to different value of n (currently 0 to 4).
    # We run the op tests from test_ops.py only for `n=0` to avoid redundancy in testing.
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_0',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=floating_types(),
                   dtypesIfCPU=floating_types(),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Probably related to the way the function is
                       # scripted for JIT tests (or maybe not).
                       # RuntimeError:
                       # Arguments for call are not valid.
                       # The following variants are available:
                       #   aten::polygamma(int n, Tensor self) -> (Tensor):
                       #   Expected a value of type 'Tensor' for argument 'self' but instead found type 'int'.
                       #   aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> (Tensor(a!)):
                       #   Expected a value of type 'Tensor' for argument 'self' but instead found type 'int'.
                       # The original call is:
                       #   File "<string>", line 3
                       # def the_method(i0):
                       #     return torch.polygamma(i0, 1)
                       #            ~~~~~~~~~~~~~~~ <--- HERE
                       SkipInfo('TestCommon', 'test_variant_consistency_jit'),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 0}, {'n': 0})),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_1',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=floating_types(),
                   dtypesIfCPU=floating_types(),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       SkipInfo('TestGradients'),
                       SkipInfo('TestOpInfo'),
                       SkipInfo('TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal'),
                   ),
                   sample_kwargs=lambda device, dtype, input: ({'n': 1}, {'n': 1})),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_2',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=floating_types(),
                   dtypesIfCPU=floating_types(),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       SkipInfo('TestGradients'),
                       SkipInfo('TestOpInfo'),
                       SkipInfo('TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=TEST_WITH_ROCM),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 2}, {'n': 2})),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_3',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   dtypes=floating_types(),
                   dtypesIfCPU=floating_types(),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       SkipInfo('TestGradients'),
                       SkipInfo('TestOpInfo'),
                       SkipInfo('TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=TEST_WITH_ROCM),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 3}, {'n': 3})),
    UnaryUfuncInfo('polygamma',
                   op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs),
                   variant_test_name='polygamma_n_4',
                   ref=reference_polygamma if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.float16: 5e-4, torch.float32: 5e-4}),),
                   dtypes=floating_types(),
                   dtypesIfCPU=floating_types(),
                   dtypesIfCUDA=floating_types_and(torch.half),
                   sample_inputs_func=sample_inputs_polygamma,
                   skips=(
                       # Redundant tests
                       SkipInfo('TestGradients'),
                       SkipInfo('TestOpInfo'),
                       SkipInfo('TestCommon'),
                       # Mismatch: https://github.com/pytorch/pytorch/issues/55357
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal'),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=TEST_WITH_ROCM),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=TEST_WITH_ROCM),),
                   sample_kwargs=lambda device, dtype, input: ({'n': 4}, {'n': 4})),
    OpInfo('pinverse',
           op=torch.pinverse,
           dtypes=floating_and_complex_types(),
           check_batched_grad=False,
           check_batched_gradgrad=False,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           supports_out=False,
           sample_inputs_func=sample_inputs_linalg_invertible,
           decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfRocm, skipCPUIfNoLapack],
           skips=(
               # cuda gradchecks are slow
               # see discussion https://github.com/pytorch/pytorch/pull/47761#issuecomment-747316775
               SkipInfo('TestGradients', 'test_fn_gradgrad', device_type='cuda'),)),
    OpInfo('gather',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_gather,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
           ),
    OpInfo('index_fill',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           skips=(SkipInfo('TestOpInfo', 'test_duplicate_method_tests'),),
           supports_out=False,
           sample_inputs_func=sample_inputs_index_fill),
    OpInfo('index_copy',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           supports_out=False,
           sample_inputs_func=sample_inputs_index_copy,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('index_select',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_index_select,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('index_add',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_inputs_index_add,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    OpInfo('__getitem__',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_inplace_autograd=False,
           op=torch.Tensor.__getitem__,
           sample_inputs_func=sample_inputs_getitem,
           skips=(SkipInfo('TestCommon', 'test_variant_consistency_jit'),)),
    OpInfo('index_put',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           supports_inplace_autograd=True,
           sample_inputs_func=sample_inputs_index_put,
           skips=(
               SkipInfo('TestCommon', 'test_variant_consistency_jit'),
           )),
    OpInfo('sort',
           dtypes=all_types_and(torch.bool, torch.float16),
           dtypesIfCUDA=all_types_and(torch.float16),
           dtypesIfROCM=all_types_and(torch.float16),
           sample_inputs_func=sample_inputs_sort,
           skips=(
               # sort does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),
           )),
    OpInfo('put',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           check_batched_gradgrad=False,  # vmap complains of the sizes
           sample_inputs_func=sample_inputs_put),
    OpInfo('take',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           check_batched_grad=False,  # vmap complains of the sizes
           sample_inputs_func=sample_inputs_take),
    OpInfo('scatter',
           dtypes=all_types_and_complex_and(torch.bool, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_scatter,
           supports_out=False),
    OpInfo('scatter_add',
           dtypes=all_types_and_complex_and(torch.bool, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_scatter_add,
           supports_out=False),
    OpInfo('stack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_stack,
           assert_autodiffed=True,
           skips=(
               # stack does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),),),
    OpInfo('hstack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           skips=(
               # hstack does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),),),
    OpInfo('hypot',
           dtypes=floating_types(),
           dtypesIfCPU=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.half),
           sample_inputs_func=sample_inputs_hypot,
           ),
    OpInfo('vstack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           skips=(
               # vstack does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),),),
    OpInfo('dstack',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           sample_inputs_func=sample_inputs_hstack_dstack_vstack,
           skips=(
               # dstack does not correctly warn when resizing out= inputs
               SkipInfo('TestCommon', 'test_out'),),),
    OpInfo('unfold',
           op=lambda x, *args: x.unfold(*args),
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           check_batched_gradgrad=False,
           skips=(
               # torch.unfold does not exist so we get a RuntimeError.
               SkipInfo('TestCommon', 'test_variant_consistency_jit',
                        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
               # Skip operator schema test because this is a functional and not an operator
               SkipInfo('TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           ),
           sample_inputs_func=sample_inputs_unfold),
    OpInfo('msort',
           dtypes=all_types_and(torch.float16),
           check_batched_gradgrad=False,
           skips=(
               #  msort does not correctly warn when resizing out= inputs.
               SkipInfo('TestCommon', 'test_out',
                        dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
               #  msort does not raise expected Runtime Error.
               SkipInfo('TestOpInfo', 'test_unsupported_dtypes', dtypes=[torch.bool]),
           ),
           sample_inputs_func=sample_inputs_msort),
    OpInfo('movedim',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_movedim_moveaxis),
    OpInfo('moveaxis',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           sample_inputs_func=sample_movedim_moveaxis),
    ShapeFuncInfo('repeat',
                  op=lambda x, dims: x.repeat(dims),
                  ref=np.tile,
                  dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                  supports_out=False,
                  skips=(
                      # torch.repeat does not exist so we get a RuntimeError.
                      SkipInfo('TestCommon', 'test_variant_consistency_jit',
                               dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16)),
                  ),
                  sample_inputs_func=sample_repeat_tile),
    OpInfo('take_along_dim',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_inplace_autograd=False,
           sample_inputs_func=sample_inputs_take_along_dim,
           gradcheck_nondet_tol=GRADCHECK_NONDET_TOL),
    ShapeFuncInfo('tile',
                  ref=np.tile,
                  dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
                  supports_out=False,
                  sample_inputs_func=sample_repeat_tile),
    OpInfo('unsqueeze',
           dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16),
           supports_out=False,
           assert_autodiffed=True,
           sample_inputs_func=sample_unsqueeze),
    OpInfo('var',
           dtypes=floating_types_and(),
           dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
           # var doesn't support complex autograd, https://github.com/pytorch/pytorch/issues/57358
           backward_dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_std_var,
           # TODO: revisit, some var signatures do support out (see std, too)
           supports_out=False,
           # var has only partial support for complex and half (#51127)
           skips=(SkipInfo('TestOpInfo', 'test_unsupported_dtypes',
                           dtypes=[torch.half, torch.complex64, torch.complex128]),),
           assert_autodiffed=True,
           ),
    OpInfo('xlogy',
           dtypes=all_types_and(torch.bool),
           dtypesIfCPU=all_types_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
           supports_inplace_autograd=True,
           safe_casts_outputs=True,
           sample_inputs_func=sample_inputs_xlogy),
    OpInfo('special.xlog1py',
           aten_name='special_xlog1py',
           dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16),
           safe_casts_outputs=True,
           skips=(
               SkipInfo('TestOpInfo', 'test_supported_backward',
                        device_type='cpu', dtypes=[torch.float16]),
           ),
           sample_inputs_func=sample_inputs_xlog1py),
    OpInfo('logsumexp',
           dtypes=floating_types_and(torch.bfloat16),
           dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.half),
           assert_autodiffed=True,
           sample_inputs_func=sample_inputs_logsumexp),
    OpInfo('trace',
           dtypes=all_types_and_complex(),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half),
           supports_inplace_autograd=False,
           supports_out=False,
           sample_inputs_func=sample_inputs_trace),
    OpInfo('transpose',
           aliases=('swapdims', 'swapaxes'),
           dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half),
           supports_out=False,
           sample_inputs_func=sample_inputs_transpose_swapdims),
    OpInfo('kron',
           dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
           supports_inplace_autograd=False,
           sample_inputs_func=sample_inputs_kron),
    OpInfo('inner',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16),
           sample_inputs_func=sample_inputs_inner),
    OpInfo('tensordot',
           dtypes=floating_and_complex_types_and(torch.half),
           dtypesIfCPU=all_types_and_complex_and(torch.half, torch.bfloat16),
           dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *[torch.bfloat16] if CUDA11OrLater else []),
           dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16),
           safe_casts_outputs=True,
           sample_inputs_func=sample_inputs_tensordot,
           skips=(
               # Currently failing due to an INTERNAL_ASSERT_FAILED error.
               # Reference: https://github.com/pytorch/pytorch/issues/56314
               SkipInfo("TestCommon", "test_variant_consistency_jit", dtypes=[torch.float32]),
               # Skip operator schema test because this is a functional and not an operator.
               # Reference: https://github.com/pytorch/pytorch/issues/54574
               SkipInfo('TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),
           )
           ),
    OpInfo('logcumsumexp',
           dtypes=floating_types_and(),
           dtypesIfCUDA=floating_types_and(torch.half),
           skips=(
               # AssertionError: UserWarning not triggered : Resized a non-empty tensor but did not warn about it.
               SkipInfo('TestCommon', 'test_out', dtypes=(torch.float32,), device_type='cuda'),
               # logcumsumexp_backward not implemented for 'Half
               SkipInfo('TestOpInfo', 'test_supported_backward', dtypes=(torch.float16,), device_type='cuda'),
           ),
           sample_inputs_func=sample_inputs_logcumsumexp),
    UnaryUfuncInfo('sigmoid',
                   aliases=('special.expit', ),
                   ref=reference_sigmoid if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.complex64: 1e-1,
                                                  torch.bfloat16: 1e-2}),),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/issues/56012
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cuda', dtypes=[torch.complex64]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cuda', dtypes=[torch.complex64]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.cfloat, torch.cdouble])),
                   dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCPU=all_types_and_complex_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16),
                   # sigmoid doesn't support complex autograd, https://github.com/pytorch/pytorch/issues/48552
                   backward_dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   backward_dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16),
                   safe_casts_outputs=True,
                   assert_autodiffed=True),
    UnaryUfuncInfo('digamma',
                   ref=scipy.special.digamma if TEST_SCIPY else _NOTHING,
                   decorators=(precisionOverride({torch.float16: 5e-1}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True),
    UnaryUfuncInfo('special.entr',
                   ref=scipy.special.entr if TEST_SCIPY else _NOTHING,
                   aten_name='special_entr',
                   decorators=(precisionOverride({torch.float16: 1e-1,
                                                  torch.bfloat16: 1e-1}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   skips=(
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.bfloat16, torch.float16]),
                   ),
                   supports_inplace_autograd=False,
                   safe_casts_outputs=True,
                   sample_inputs_func=sample_inputs_entr),
    UnaryUfuncInfo('erf',
                   ref=scipy.special.erf if TEST_SCIPY else _NOTHING,
                   aliases=('special.erf', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   assert_autodiffed=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('erfc',
                   ref=scipy.special.erfc if TEST_SCIPY else _NOTHING,
                   aliases=('special.erfc', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   assert_autodiffed=True,
                   safe_casts_outputs=True),
    UnaryUfuncInfo('erfinv',
                   ref=scipy.special.erfinv if TEST_SCIPY else _NOTHING,
                   aliases=('special.erfinv', ),
                   decorators=(precisionOverride({torch.float16: 1e-2,
                                                  torch.bfloat16: 1e-2,
                                                  torch.float32: 1e-4}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   safe_casts_outputs=True,
                   domain=(-1, 1),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/49155#issuecomment-742664611
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                active_if=TEST_SCIPY and distutils.version.LooseVersion(scipy.__version__) < "1.4.0"),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                active_if=TEST_SCIPY and distutils.version.LooseVersion(scipy.__version__) < "1.4.0"),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                active_if=TEST_SCIPY and distutils.version.LooseVersion(scipy.__version__) < "1.4.0"),
                   )),
    UnaryUfuncInfo('lgamma',
                   ref=reference_lgamma if TEST_SCIPY else _NOTHING,
                   aliases=('special.gammaln', ),
                   decorators=(precisionOverride({torch.float16: 7e-1}),),
                   dtypes=all_types_and(torch.bool),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half),
                   # "digamma" not implemented for 'BFloat16'
                   backward_dtypesIfCPU=all_types_and(torch.bool),
                   skips=(
                       # Reference: https://github.com/pytorch/pytorch/pull/50140#discussion_r552615345
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                device_type='cpu', dtypes=[torch.bfloat16]),
                       # Reference: https://github.com/pytorch/pytorch/pull/50140#issuecomment-756150214
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_extremal',
                                dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_hard',
                                dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                       SkipInfo('TestUnaryUfuncs', 'test_reference_numerics_normal',
                                dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS),
                   ),
                   safe_casts_outputs=True),
    OpInfo(
        'logdet',
        supports_out=False,
        sample_inputs_func=sample_inputs_logdet,
        decorators=(skipCPUIfNoLapack, skipCUDAIfNoMagma, skipCUDAIfRocm)),
    UnaryUfuncInfo('logit',
                   ref=scipy.special.logit if TEST_SCIPY else _NOTHING,
                   domain=(0, 1),
                   aliases=('special.logit', ),
                   decorators=(precisionOverride({torch.bfloat16: 5e-1,
                                                  torch.float16: 5e-1}),),
                   dtypes=all_types_and(torch.half),
                   dtypesIfCPU=all_types_and(torch.bool, torch.bfloat16),
                   dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16),
                   sample_inputs_func=sample_inputs_logit,
                   safe_casts_outputs=True),
]

# Common operator groupings
unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo)]
spectral_funcs = [op for op in op_db if isinstance(op, SpectralFuncInfo)]
sparse_unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo) and op.supports_sparse is True]
shape_funcs = [op for op in op_db if isinstance(op, ShapeFuncInfo)]

def index_variable(shape, max_indices, device=torch.device('cpu')):
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape, dtype=torch.double, device=device).mul_(max_indices).floor_().long()
    return index

        a = torch.ones(2, 2, requires_grad=True)
        out = a.as_strided((3,), (1,), 1)
        self.assertEqual(out.grad_fn._saved_storage_offset, 1)            # c10:optional<int64_t> -> int?
        self.assertIsInstance(out.grad_fn._saved_storage_offset, int)
        out = a.as_strided((3,), (1,))
        self.assertIsNone(out.grad_fn._saved_storage_offset)

        a = torch.ones(2, requires_grad=True)
        out = torch.tanh(a)
        self.assertEqual(out, out.grad_fn._saved_result)                  # saved variable when output

        a = torch.randn(3, 5, requires_grad=True)
        b = torch.tensor([1, 0, 4])
        loss = nn.NLLLoss()
        out = loss(a, b)
        self.assertIsNone(out.grad_fn._saved_weight)
        loss = nn.NLLLoss(weight=torch.ones((5,)))
        out = loss(a, b)
        self.assertEqual(out.grad_fn._saved_weight, torch.ones((5,)))     # c10:optional<Tensor> -> Tensor?

        out.sum().backward()
        with self.assertRaisesRegex(RuntimeError, "after they have already been freed"):
            out.grad_fn._saved_weight

    def test_autograd_views_codegen(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks the behavior of two codegen functions (view_as and unbind)
        # with respect to view tracking and inplace operation on the output.

        def run_test(grad_mode, requires_grad, is_view, should_raise_tuple):
            def maybe_check_raise(fn, should_raise):
                self.assertTrue(should_raise is None or isinstance(should_raise, str))
                if should_raise is not None:
                    with self.assertRaisesRegex(RuntimeError, should_raise):
                        fn()
                else:
                    fn()

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.view_as(inp)
            # Are they differentiable views?
            self.assertTrue(out._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out.add_(1), should_raise_tuple[0])

            inp = torch.rand(2, requires_grad=requires_grad).clone()
            with torch.set_grad_enabled(grad_mode):
                out = inp.unbind()
            # Are they differentiable views?
            self.assertTrue(out[0]._is_view() == is_view)
            self.assertTrue(out[1]._is_view() == is_view)
            # Are inplace allowed?
            maybe_check_raise(lambda: out[0].add_(1), should_raise_tuple[1])
            maybe_check_raise(lambda: out[1].add_(1), should_raise_tuple[2])

        # should_raise contains None if it should not raise
        # should_raise contains a string of the error if it should raise
        # The 3 elements are for view_as, first output of unbind and second output of unbind
        run_test(grad_mode=True, requires_grad=False, is_view=True,
                 should_raise_tuple=(None, None, None))
        inp_change_err = "Output {} of UnbindBackward is a view and is being modified inplace."
        run_test(grad_mode=True, requires_grad=True, is_view=True,
                 should_raise_tuple=(None, inp_change_err.format("0"), inp_change_err.format("1")))
        grad_mode_err = "A view was created in no_grad mode and is being modified inplace"
        leaf_grad_err = "a leaf Variable that requires grad is being used in an in-place operation."
        run_test(grad_mode=False, requires_grad=True, is_view=True,
                 should_raise_tuple=(leaf_grad_err, grad_mode_err, grad_mode_err))
        run_test(grad_mode=False, requires_grad=False, is_view=True,
                 should_raise_tuple=(None, None, None))

    def test_inplace_not_requires_grad(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx, grad):
                return grad

        # Original Tensor does not require grad
        a = torch.rand(1, 2)

        # Tensor being written does require grad
        b = torch.rand(1, requires_grad=True)

        # Take an invalid view on 'a' that should raise an error (warns during deprecation)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(RuntimeError, "This view was created inside a custom Function"):
            view_a += b

        # Extra test for copy_ that is a manual implementation and could be easily
        # forgotten when the codegen is updated (warns during deprecation)
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = MyFn.apply(a)

        with self.assertRaisesRegex(RuntimeError, "This view was created inside a custom Function"):
            view_a.copy_(b)

        # Functions that should throw must properly throw
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        view_a = a.unbind()[0]
        with self.assertRaisesRegex(RuntimeError, "This view is the output of a function that returns "
                                                  "multiple views."):
            view_a.copy_(b)

        # Sanity check that views that should work still work
        a = torch.rand(1, 2)
        b = torch.rand(1, requires_grad=True)
        a.select(1, 0).copy_(b)

    def _do_test_autograd_simple_views_python(self, dtype):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks the autograd.Function behavior when we return one or multiple outputs
        # while one of these is an input, a view of an input or of a temporary tensor.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]
        # This indicator is used to check if the argument `ga` contains non-zero values
        ga_nz = [False]

        class IdOneOutput(Function):
            @staticmethod
            def forward(ctx, a, b, make_view):
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a

            @staticmethod
            def backward(ctx, ga):
                bw_called[0] += 1
                return ga, None, None

        class IdTwoOutput(Function):
            @staticmethod
            def forward(ctx, a, b, make_view):
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                if ga.eq(0).all():
                    ga_nz[0] = False
                else:
                    ga_nz[0] = True
                return ga + gab, gab, None

        class ViewOfTemp(Function):
            @staticmethod
            def forward(ctx, a, make_view):
                ctx.save_for_backward(a)
                if make_view:
                    a = a.narrow(0, 0, 2)
                else:
                    a = a.clone()
                b = a.clone()
                return b.select(0, 0)

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                a, = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, 0).copy_(grad)
                return res, None

        fn_id_to_inplace_view_err_msg = {
            "one_output": ("Output 0 of IdOneOutputBackward is a view and is being "
                           "modified inplace. This view was created inside a custom Function"),
            "two_output": ("Output 0 of IdTwoOutputBackward is a view and is being modified inplace."
                           " This view is the output of a function that returns multiple views."),
            "view_of_temp": ("Output 0 of ViewOfTempBackward is a view and is being "
                             "modified inplace. This view was created inside a custom Function")
        }

        for fn_id in ["one_output", "two_output", "view_of_temp"]:
            for inplace in [True, False]:
                for make_view in [True, False]:
                    # Used for special casing the tests below
                    output_is_a_view = (make_view or fn_id == "view_of_temp")

                    def fn(a, b):
                        # never modify a, b inplace for gracheck
                        a = a.clone()
                        b = b.clone()
                        if fn_id == "two_output":
                            tmp1, tmp2 = IdTwoOutput.apply(a, b, make_view)
                            if inplace:
                                tmp1 += 3
                                tmp2 += 3
                            else:
                                tmp1 = tmp1 + 3
                                tmp2 = tmp2 + 3
                            tmp = tmp1 * tmp2
                        else:
                            if fn_id == "one_output":
                                tmp = IdOneOutput.apply(a, b, make_view)
                            else:
                                tmp = ViewOfTemp.apply(a + b, make_view)
                            if inplace:
                                tmp += 3
                            else:
                                tmp = tmp + 3

                        return tmp.sum()

                    a = torch.ones(2, dtype=dtype, requires_grad=True)
                    b = torch.ones(2, dtype=dtype, requires_grad=True)

                    err_msg = fn_id_to_inplace_view_err_msg[fn_id]

                    if not inplace or not output_is_a_view:
                        gradcheck(fn, (a, b), check_batched_grad=False)

                    # Was the custom backward called properly
                    bw_called[0] = 0
                    ga_nz[0] = True  # For the case where the backward is called

                    if inplace and output_is_a_view:
                        with self.assertRaisesRegex(RuntimeError, err_msg):
                            fn(a, b)
                    else:
                        fn(a, b).backward()

                    expected_called = 1
                    expected_ga_nz = True

                    if output_is_a_view and inplace:
                        expected_called = 0

                    self.assertTrue(bw_called[0] == expected_called)
                    self.assertTrue(ga_nz[0] == expected_ga_nz)

    def test_autograd_simple_views_python(self):
        self._do_test_autograd_simple_views_python(torch.double)
        self._do_test_autograd_simple_views_python(torch.cdouble)

    def test_autograd_complex_views_python(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This checks that multiples views in the forward are properly traced and how they
        # behave with respect to inplace operations.

        # This indicator is used to track how many times the backward function was called
        bw_called = [0]

        class ComplexView(Function):
            @staticmethod
            def forward(ctx, a, idx):
                res = a.narrow(0, idx, 1)
                res = a.select(0, idx)
                ctx.save_for_backward(a)
                ctx.idx = idx
                return res

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                a, = ctx.saved_tensors
                res = torch.zeros_like(a)
                res.select(0, ctx.idx).copy_(grad)
                return res, None

        a = torch.ones(2, requires_grad=True)
        idx = 1

        bw_called[0] = 0
        out = ComplexView.apply(a.clone(), idx)
        out.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        out = ComplexView.apply(a.clone(), idx)
        with self.assertRaisesRegex(RuntimeError,
                                    "Output 0 of ComplexViewBackward is a view and is being modified inplace"):
            out += 1

    def test_autograd_inplace_views_python(self):
        # This is not necessarily the absolute correct behavior, but this is the current
        # one. This test is here to make sure that any change to this behavior is detected
        # and not silent. The TODOs below mark the places with unexpected behavior.
        # Note that any change in these test will be BC-breaking and should be done carefully.

        # This test checks custom autograd.Function that perform inplace operations

        bw_called = [0]

        # I) Single output
        class MyAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                return grad, grad


        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        # No extra inplace
        c = MyAdder.apply(a.clone(), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c = MyAdder.apply(a.clone(), b)
        c += 2
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        bw_called[0] = 0
        c = MyAdder.apply(a.clone().view_as(a), b)
        c.sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # Should not give non-inputs to mark_dirty
        class MyAdderBad(Function):
            @staticmethod
            def forward(ctx, a, b):
                c = 3 * a
                c.add_(b)
                ctx.mark_dirty(c)
                return c

            @staticmethod
            def backward(ctx, grad):
                bw_called[0] += 1
                grad = 3 * grad
                return grad, grad

        a = torch.ones(2, requires_grad=True)
        b = torch.ones(2, requires_grad=True)

        with warnings.catch_warnings(record=True) as w:
            MyAdderBad.apply(a.clone(), b)
        self.assertEqual(len(w), 1)

        # II) Multiple outputs
        class MyBadAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a, a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                return ga + gab, ga + gab

        # No extra inplace
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # With extra inplace on the output
        bw_called[0] = 0
        c, d = MyBadAdder.apply(a.clone(), b)
        c += 2
        (c * d).sum().backward()
        self.assertTrue(bw_called[0] == 1)

        # The input is a view
        inplace_on_view_err = "your Function modifies inplace an input that is a view of another Tensor"
        with self.assertRaisesRegex(RuntimeError, inplace_on_view_err):
            c, d = MyBadAdder.apply(a.clone().view_as(a), b)

        # III) Inplace + other op
        class MyOutPlaceAdder(Function):
            @staticmethod
            def forward(ctx, a, b):
                a.add_(b)
                ctx.mark_dirty(a)
                return a.clone(), a + b

            @staticmethod
            def backward(ctx, ga, gab):
                bw_called[0] += 1
                return ga + gab, ga + 2 * gab

        # We don't reuse the input
        def fn(a, b):
            orig_a = a.clone().view_as(a)
            c, d = MyOutPlaceAdder.apply(orig_a, b)
            return (c * d).sum()

        bad_mark_dirty_err = "Some elements marked as dirty during the forward method were not returned as output."
        with self.assertRaisesRegex(RuntimeError, bad_mark_dirty_err):
            fn(a, b)

    def test_named_tensor_for_complex_views(self):
        names = ["batch", "height", "width", "complex"]
        z = torch.ones((5, 12, 14, 2), requires_grad=True)
        z_named = z.refine_names(*names)
        z_complex = torch.view_as_complex(z_named.rename(None)).refine_names(*names[:-1])
        z_complex.sum().backward()
        self.assertEqual(z.grad, torch.view_as_real(torch.ones_like(z_complex).rename(None)))

    def test_custom_function_return_view_in_nograd(self):
        class Alias(Function):
            @staticmethod
            def forward(ctx, x):
                return x[:]

            @staticmethod
            def backward(ctx, gx):
                return gx

        inp = torch.rand(2, requires_grad=True)

        with torch.no_grad():
            output = Alias.apply(inp)

        with torch.no_grad():
            expected_output = inp[:]

        # Calling the custom function should operate as if we called an equivalent op
        self.assertEqual(output.requires_grad, expected_output.requires_grad)

        # Check that in-place modification on view throws
        leaf_grad_err = "A view was created in no_grad mode and is being modified inplace"
        with self.assertRaisesRegex(RuntimeError, leaf_grad_err):
            output.zero_()

    def test_grad_mode_restored_reentrant(self):
        class MyFunction(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, go):
                original = torch._C.is_grad_enabled()
                with torch.enable_grad():
                    self.assertTrue(torch._C.is_grad_enabled())
                    foo = torch.rand(go.size(), requires_grad=True)
                    grad, = torch.autograd.grad(
                        foo ** 3, foo, grad_outputs=go
                    )
                    self.assertTrue(torch._C.is_grad_enabled())
                self.assertTrue(torch._C.is_grad_enabled() == original)
                return grad

        inp = torch.rand(3, requires_grad=True)

        # Case where original==False
        MyFunction.apply(inp).sum().backward()
        # Case where original==True
        MyFunction.apply(inp).sum().backward(create_graph=True)

    def test_power_function(self):
        a = torch.tensor([0., 0., 0.])
        b = torch.tensor([-1., 0., 1.], requires_grad=True)
        c = torch.sum(a**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0., 0.]))

        s = 0
        b = torch.tensor([-1., 0., 1.], requires_grad=True)
        c = torch.sum(s**b)
        c.backward()
        self.assertEqual(b.grad, torch.tensor([-inf, 0., 0.]))

    def test_nansum_with_nans(self):
        a = torch.randn(2, 2, 2, 2, dtype=torch.double)
        with torch.no_grad():
            a[a < 0.2] = float('nan')
        a.requires_grad = True

        # No args
        gradcheck(lambda x: x.nansum(), a)
        gradgradcheck(lambda x: x.nansum(), a)

        # Single dim
        gradcheck(lambda x: x.nansum((0)), a)
        gradgradcheck(lambda x: x.nansum((0)), a)

        # Multi dim
        gradcheck(lambda x: x.nansum((0, 2)), a)
        gradgradcheck(lambda x: x.nansum((0, 2)), a)

        gradcheck(lambda x: x.nansum((0, -1)), a)
        gradgradcheck(lambda x: x.nansum((0, -1)), a)

        # With keep-dim
        gradcheck(lambda x: x.nansum((0, -1), True), a)
        gradgradcheck(lambda x: x.nansum((0, -1), True), a)

    def test_nansum_dtype(self):
        inp = torch.randn(2, 2, 2, 2)
        with torch.no_grad():
            inp[inp < 0.2] = float('nan')

        def test(inp, inp_dtype, out_dtype):
            with torch.no_grad():
                a = inp.to(inp_dtype)
            a.requires_grad = True
            b = torch.sum(a, dtype=out_dtype)
            b.backward()
            self.assertEqual(a.dtype, a.grad.dtype)

        test(inp, torch.float, torch.double)
        test(inp, torch.double, torch.float)

    def test_nan_to_num(self):
        a = torch.randn(3, 3, 3, 3, dtype=torch.double)
        with torch.no_grad():
            a[torch.rand_like(a) < 0.2] = float('nan')
            a[torch.rand_like(a) < 0.2] = float('inf')
            a[torch.rand_like(a) < 0.2] = -float('inf')

        a.requires_grad = True

        gradcheck(lambda x: x.nan_to_num(), a)
        gradgradcheck(lambda x: x.nan_to_num(), a)

        gradcheck(lambda x: x.nan_to_num(nan=1.2), a)
        gradgradcheck(lambda x: x.nan_to_num(nan=1.2), a)

        gradcheck(lambda x: x.nan_to_num(nan=1.2, posinf=2.0), a)
        gradgradcheck(lambda x: x.nan_to_num(nan=1.2, posinf=2.0), a)

        gradcheck(lambda x: x.nan_to_num(nan=1.2, posinf=2.0, neginf=-2.0), a)
        gradgradcheck(lambda x: x.nan_to_num(nan=1.2, posinf=2.0, neginf=-2.0), a)

        gradcheck(lambda x: x.nan_to_num(posinf=2.0, neginf=-2.0), a)
        gradgradcheck(lambda x: x.nan_to_num(posinf=2.0, neginf=-2.0), a)

        gradcheck(lambda x: x.nan_to_num(neginf=-2.0), a)
        gradgradcheck(lambda x: x.nan_to_num(neginf=-2.0), a)

    def test_custom_function_error(self):
        class BadFw(Function):
            @staticmethod
            def backward(ctx, foo):
                return foo

        class BadBw(Function):
            @staticmethod
            def forward(ctx, foo):
                return foo.clone()

        inp = torch.rand(1, requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "must implement the forward"):
            BadFw.apply(inp)

        with self.assertRaisesRegex(RuntimeError, "must implement the backward"):
            BadBw.apply(inp).sum().backward()

    def test_custom_function_local_inplace(self):
        class MyFn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, inp, inplace):
                view = inp.clone()[:3]
                if inplace:
                    view += 2
                return view

            @staticmethod
            def backward(ctx, grad):
                return grad, None

        base = torch.rand(10, requires_grad=True)

        foo = MyFn.apply(base, False)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

        foo = MyFn.apply(base, True)
        self.assertEqual(foo.grad_fn.__class__.__name__, "MyFnBackward")

    def test_integer_outputs(self):
        inp = torch.rand(4, requires_grad=True)

        out = inp.argmax()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argmin()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        out = inp.argsort()
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.rand((), requires_grad=True)

        out = torch.searchsorted(inp, val)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        bins = torch.linspace(0, 1.0, steps=100, requires_grad=True)
        vals = torch.rand(5, 5, requires_grad=True)
        out = torch.bucketize(vals, bins)
        self.assertFalse(out.dtype.is_floating_point)
        self.assertFalse(out.requires_grad)

        val = torch.empty(5).requires_grad_()
        out = val.count_nonzero()
        self.assertFalse(out.requires_grad)

        def assert_only_first_requires_grad(res):
            if not isinstance(res, tuple):
                res = (res,)
            self.assertTrue(res[0].requires_grad)
            for out in res[1:]:
                if out is not None:
                    self.assertFalse(out.requires_grad)

        for sort in [True, False]:
            for return_inverse in [True, False]:
                for return_counts in [True, False]:
                    res = torch.unique(inp, sorted=sort, return_inverse=return_inverse,
                                       return_counts=return_counts)
                    assert_only_first_requires_grad(res)

                    res = torch.unique(inp, sorted=sort, return_inverse=return_inverse,
                                       return_counts=return_counts, dim=0)
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(inp, return_inverse=return_inverse,
                                                   return_counts=return_counts)
                    assert_only_first_requires_grad(res)

                    res = torch.unique_consecutive(inp, return_inverse=return_inverse,
                                                   return_counts=return_counts, dim=0)
                    assert_only_first_requires_grad(res)

                    # Here we test the internal functions to make sure all of them are
                    # covered on top of the public API
                    res = torch._unique(inp, sorted=sort, return_inverse=return_inverse)
                    assert_only_first_requires_grad(res)

                    # This looks public but is actually manually deleted from the
                    # torch namespace in torch/functional.py
                    res = torch._VF.unique_dim(inp, dim=0, sorted=sort, return_inverse=return_inverse,
                                               return_counts=return_counts)
                    assert_only_first_requires_grad(res)

                    # We don't test `unique_dim_consecutive` here.
                    # It looks public but the python binding is actually manually disabled in
                    # tools/autograd/gen_python_functions.py

                    res = torch._unique2(inp, sorted=sort, return_inverse=return_inverse,
                                         return_counts=return_counts)
                    assert_only_first_requires_grad(res)


def index_perm_variable(shape, max_indices):
    if not isinstance(shape, tuple):
        shape = (shape,)

    index = torch.randperm(max_indices).narrow(0, 0, reduce(mul, shape)).view(shape)
    return index

def bernoulli_scalar():
    return torch.tensor(0, dtype=torch.uint8).bernoulli_()


def gradgradcheck_method_precision_override(test_name):
    # these are just empirical observations, we should improve
    gradgradcheck_precision_override = {
        'test_norm': {'atol': 2e-2, 'rtol': 1e-2},
        'test_norm_1_5': {'atol': 1.5e-2, 'rtol': 1e-2},
        'test_norm_3': {'atol': 5e-2, 'rtol': 1e-2},
        'test_dist': {'atol': 5e-2, 'rtol': 1e-2},
        'test_dist_4': {'atol': 8e-2, 'rtol': 1e-2},
    }
    non_broadcasted_test_name = test_name.split("_broadcast")[0]
    override = gradgradcheck_precision_override.get(non_broadcasted_test_name)
    if override:
        if 'broadcast_lhs' in test_name or 'broadcast_rhs' in test_name:
            # errors accumulated across 1 dimension
            override = {'atol': override['atol'] * S, 'rtol': override['atol'] * S}
        elif 'broadcast_all' in test_name:
            # errors accumulated across multiple dimensions
            override = {'atol': override['atol'] * S * S, 'rtol': override['atol'] * S * S}
    return override

def run_grad_and_gradgrad_checks(test_case, name, test_name, apply_method, output_variable,
                                 input_variables, run_gradgradcheck=True, check_batched_grad=True):
    test_case.assertTrue(gradcheck(apply_method, input_variables, eps=1e-6, atol=PRECISION,
                                   check_batched_grad=check_batched_grad))
    if name in EXCLUDE_GRADGRADCHECK or test_name in EXCLUDE_GRADGRADCHECK_BY_TEST_NAME:
        return
    gradgradcheck_precision_override = gradgradcheck_method_precision_override(test_name)
    if gradgradcheck_precision_override is not None:
        atol = gradgradcheck_precision_override['atol']
        rtol = gradgradcheck_precision_override['rtol']
        test_case.assertTrue(gradgradcheck(apply_method, input_variables, None, atol=atol, rtol=rtol,
                                           gen_non_contig_grad_outputs=True,
                                           check_batched_grad=check_batched_grad))
    else:
        test_case.assertTrue(gradgradcheck(apply_method, input_variables,
                                           gen_non_contig_grad_outputs=True,
                                           check_batched_grad=check_batched_grad))


def run_functional_checks(test_case, test_name, name, apply_fn, run_grad_checks,
                          f_args_variable, f_args_tensor):
    output_variable = apply_fn(*f_args_variable)

    if run_grad_checks:
        run_grad_and_gradgrad_checks(test_case, name, test_name, apply_fn,
                                     output_variable, f_args_variable)

    self_variable = f_args_variable[0]
    if isinstance(output_variable, torch.Tensor) and output_variable.requires_grad and self_variable is not None:
        output_variable.backward(randn_like(output_variable))
        test_case.assertEqualTypeString(self_variable, self_variable.grad)
        test_case.assertEqual(self_variable.size(), self_variable.grad.size())

# this list corresponds to ops which have separate tests defined for complex dtypes in
# common_methods_invocations.py
# test for these ops with 'complex' in variant should only run for complex and
# the tests for these ops which do not have 'complex' in variant should not run for complex
# and only run for floating point

separate_complex_tests = ['div', '__rdiv__', 'sub']

# allow list for complex
complex_list = ['t', 'view', 'reshape', 'reshape_as', 'view_as', 'roll', 'clone',
                'expand', 'rot90', 'transpose',
                'permute', 'squeeze', 'unsqueeze', 'resize', 'resize_as', 'tril', 'triu',
                'chunk', 'split', 'split_with_sizes', 'zero_',
                '__radd__', 'mul', '__rmul__', 'diagonal', 'fill_', 'sub', 'narrow',
                'swapaxes', 'swapdims', 'tensor_split'] + separate_complex_tests

# deny list for batched grad computation
EXCLUDE_BATCHED_GRAD_TESTS = set([
    'test_to_sparse',
])

def add_test(
        name,
        self_size,
        args,
        variant_name='',
        check_ad=(),  # only used in test_jit
        dim_args_idx=(),
        skipTestIf=(),
        output_process_fn=lambda x: x,
        kwargs=None):
    kwargs = kwargs if kwargs else {}
    basic_test_name = 'test_' + name
    if variant_name != '':
        basic_test_name += '_' + variant_name

    if name in separate_complex_tests and 'complex' in variant_name:
        run_only_complex = True
    else:
        run_only_complex = False

    for dtype in [torch.double, torch.cdouble]:
        for dim_perm in product([-1, 1], repeat=len(dim_args_idx)):
            test_name = basic_test_name
            new_args = [arg * dim_perm[dim_args_idx.index(i)] if i in dim_args_idx else arg for i, arg in enumerate(args)]
            test_name = basic_test_name + ''.join('_neg' + str(i) for i, idx in enumerate(dim_perm) if idx < 0)

            if dtype.is_complex:
                # TODO: remove this. this is temporary while we ramp up the complex support.
                if name in complex_list:
                    if name in separate_complex_tests and 'complex' not in variant_name:
                        continue
                    if not run_only_complex:
                        test_name = test_name + '_complex'
                else:
                    continue
            elif run_only_complex:
                continue

            new_args = tuple(new_args)

            # for-loop bodies don't define scopes, so we have to save the variables
            # we want to close over in some way
            def do_test(self, device, dtype=dtype, name=name, self_size=self_size, args=new_args, test_name=test_name,
                        output_process_fn=output_process_fn):
                def check(name):
                    is_magic_method = name[:2] == '__' and name[-2:] == '__'
                    is_inplace = name[-1] == "_" and not is_magic_method
                    self_variable = create_input((self_size,), dtype=dtype, device=device)[0][0]
                    # FixMe: run grad checks on inplace self
                    if is_inplace:
                        self_variable.requires_grad = False
                    # need to record this because methods can change the size (e.g. unsqueeze)
                    args_variable, kwargs_variable = create_input(args, requires_grad=not is_inplace,
                                                                  call_kwargs=kwargs, dtype=dtype, device=device)
                    self_tensor = deepcopy(self_variable)
                    args_tensor = deepcopy(unpack_variables(args_variable))
                    if not exclude_tensor_method(name, test_name):
                        output_variable = getattr(self_variable, name)(*args_variable, **kwargs_variable)
                        output_tensor = getattr(self_tensor, name)(*args_tensor, **kwargs_variable)
                        if not isinstance(output_tensor, torch.Tensor) and not isinstance(output_tensor, tuple):
                            if dtype.is_complex:
                                output_tensor = torch.tensor((output_tensor, ), dtype=torch.cfloat, device=device)
                            else:
                                output_tensor = torch.tensor((output_tensor, ), dtype=torch.float, device=device)
                        self.assertEqual(unpack_variables(output_variable), output_tensor)
                        # TODO: check that both have changed after adding all inplace ops

                        def fn(*inputs):
                            output = getattr(inputs[0], name)(*inputs[1:], **kwargs)
                            return output_process_fn(output)

                        if not is_inplace and name not in EXCLUDE_GRADCHECK:
                            check_batched_grad = test_name not in EXCLUDE_BATCHED_GRAD_TESTS
                            run_grad_and_gradgrad_checks(self, name, test_name, fn,
                                                         output_variable, (self_variable,) + args_variable,
                                                         check_batched_grad=check_batched_grad)

                    # functional interface tests
                    torch_fn = getattr_qualified(torch, name)
                    if torch_fn is not None and name not in EXCLUDE_FUNCTIONAL:
                        def fn(*inputs):
                            output = torch_fn(*inputs, **kwargs)
                            return output_process_fn(output)

                        f_args_variable = (self_variable,) + args_variable
                        f_args_tensor = (self_tensor,) + args_tensor
                        # could run the gradchecks again, but skip since we did it for the methods above.
                        run_gradcheck = exclude_tensor_method(name, test_name) and not is_inplace and name not in EXCLUDE_GRADCHECK
                        run_functional_checks(self, test_name, name, fn,
                                              run_gradcheck, f_args_variable, f_args_tensor)

# Do NOT add to this list. Method tests are being DEPRECATED and replaced by OpInfos.
# See https://github.com/pytorch/pytorch/wiki/Writing-tests-in-PyTorch-1.8
#
# (
#   method name,
#   input size/constructing fn,
#   args (tuple represents shape of a tensor arg),
#   test variant name (will be used at test name suffix),    // optional
#   (should_autodiff_node[bool], nonfusible_nodes, fusible_nodes) for autodiff, // optional
#   indices for possible dim arg,                            // optional
#   fn mapping output to part that should be gradcheck'ed,   // optional
#   kwargs                                                   // optional
# )
# Note: some functions have separate schema for (Tensor other) and (Scalar other),
#       and it's possible that we only support AD for Scalar version but not Tensor
#       version, and vice versa.
#       When writing tests, only scalar(float/int) input triggers the Scalar schema.
#       uniform_scalar produces a scalar **Tensor** which won't match Scalar input.
def method_tests():
    set_rng_seed(SEED)
    return [
        ('__radd__', (S, S, S), (3.14,), 'constant', (True, 'aten::add')),
        ('__radd__', (), (3.14,), 'scalar_constant', (True, 'aten::add')),
        ('__rsub__', (S, S, S), (3.14,), 'constant', (True, 'aten::rsub')),
        ('__rsub__', (), (3.14,), 'scalar_constant', (True, 'aten::rsub')),
        ('__rmul__', (S, S, S), (3.14,), 'constant', (True, 'aten::mul')),
        ('__rmul__', (), (3.14,), 'scalar_constant', (True, 'aten::mul')),
        ('div', (S, S, S), (torch.rand(S, S, S) + 0.1,), '', (True,)),
        ('div', (S, S, S), (torch.rand(S, S) + 0.1,), 'broadcast_rhs', (True,)),
        ('div', (S, S), (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs', (True,)),
        ('div', (S, 1, S), (torch.rand(M, S) + 0.1,), 'broadcast_all', (True,)),
        ('div', (), (uniform_scalar(0.1),), 'scalar', (True,)),
        ('div', (S, S, S), (uniform_scalar(0.1),), 'scalar_broadcast_rhs', (True,)),
        ('div', (), (uniform_scalar(0.1),), 'scalar_broadcast_lhs', (True,)),
        ('div', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant', (True,)),
        ('div', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant', (True,)),
        ('true_divide', (S, S, S), (torch.rand(S, S, S) + 0.1,), '', (True,)),
        ('true_divide', (S, S, S), (torch.rand(S, S) + 0.1,), 'broadcast_rhs', (True,)),
        ('true_divide', (S, S), (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs', (True,)),
        ('true_divide', (S, 1, S), (torch.rand(M, S) + 0.1,), 'broadcast_all', (True,)),
        ('true_divide', (), (uniform_scalar(0.1),), 'scalar', (True,)),
        ('true_divide', (S, S, S), (uniform_scalar(0.1),), 'scalar_broadcast_rhs', (True,)),
        ('true_divide', (), (uniform_scalar(0.1),), 'scalar_broadcast_lhs', (True,)),
        ('true_divide', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant', (True,)),
        ('true_divide', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant', (True,)),
        ('__rdiv__', torch.rand(S, S, S) + 1e-1, (3.14,), 'constant',
            (True, [], ['aten::mul', 'aten::reciprocal'])),
        ('__rdiv__', uniform_scalar(1e-1, requires_grad=True), (3.14,), 'scalar_constant',
            (True, [], ['aten::mul', 'aten::reciprocal'])),
        ('__rdiv__', torch.rand(S, S, S, dtype=torch.cdouble) + 1e-1, (3.14j,), 'complex_constant',
            (True, [], ['aten::mul', 'aten::reciprocal'])),
        ('__rdiv__', uniform_scalar(1e-1 * (1 + 1j), requires_grad=True), (3.14j,), 'complex_scalar_constant',
            (True, [], ['aten::mul', 'aten::reciprocal'])),
        ('div', (S, S, S), (torch.rand(S, S, S, dtype=torch.cdouble) + 0.1,), 'complex', (True,)),
        ('div', (S, S, S), (torch.rand(S, S, dtype=torch.cdouble) + 0.1,), 'complex_broadcast_rhs', (True,)),
        ('div', (S, S), (torch.rand(S, S, S, dtype=torch.cdouble) + 0.1,), 'complex_broadcast_lhs', (True,)),
        ('div', (S, 1, S), (torch.rand(M, S, dtype=torch.cdouble) + 0.1,), 'complex_broadcast_all', (True,)),
        ('div', (), (uniform_scalar(0.1j),), 'complex_scalar', (True,)),
        ('div', (S, S, S), (uniform_scalar(0.1j),), 'complex_scalar_broadcast_rhs', (True,)),
        ('div', (), (uniform_scalar(0.1j),), 'complex_scalar_broadcast_lhs', (True,)),
        ('div', torch.rand(S, S, S, dtype=torch.cdouble) + 1e-1, (3.14j,), 'complex_constant', (True,)),
        ('div', uniform_scalar(1e-1j, requires_grad=True), (3.14j,), 'complex_scalar_constant', (True,)),
        ('__rpow__', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant', (True, 'aten::pow')),
        ('__rpow__', uniform_scalar(1e-3, requires_grad=True), (3.14,), 'scalar_constant', (True, 'aten::pow')),
        ('float_power', torch.rand(S, S, S) + 1e-3, (torch.rand(S, S, S) + 0.1,), ''),
        ('float_power', torch.rand(S, S, S) + 1e-3, (torch.rand(1,) + 0.1,), 'broadcast_rhs'),
        ('float_power', torch.rand(1,) + 1e-3, (torch.rand(S, S, S) + 0.1,), 'broadcast_lhs'),
        ('float_power', torch.rand(S, 1, S) + 1e-3, (torch.rand(1, S, 1) + 0.1,), 'broadcast_all'),
        ('float_power', uniform_scalar(1e-3, requires_grad=True), (uniform_scalar(0.1),), 'scalar'),
        ('float_power', torch.rand(S, S, S) + 1e-3, (uniform_scalar(0.1),), 'scalar_broadcast_rhs'),
        ('float_power', uniform_scalar(1e-3, requires_grad=True), (torch.rand(S, S, S) + 0.1,), 'scalar_broadcast_lhs'),
        ('float_power', torch.rand(S, S, S) + 1e-3, (3.14,), 'constant'),
        ('t', (1, 2), NO_ARGS, '', (False,)),
        ('view', (S, S, S), (S * S, S), '', (False,)),
        ('view', (torch.Size([S * S, S]),), (S, S, S), 'size', (False,)),
        ('view', (S,), (S,), '1d', (False,)),
        ('view', (), (dont_convert(()),), 'scalar_to_scalar', (False,)),
        ('view', (), (1,), 'scalar_to_1d', (False,)),
        ('ravel', (S, S, S), NO_ARGS, '', (False,)),
        ('reshape', (S, S, S), (S * S, S), '', (False,)),
        ('reshape', (torch.Size([S * S, S]),), (S, S, S), 'size', (False,)),
        ('reshape', (S,), (S,), '1d', (False,)),
        ('reshape', (), (dont_convert(()),), 'scalar_to_scalar', (False,)),
        ('reshape', (), (1,), 'scalar_to_1d', (False,)),
        ('reshape_as', (S, S, S), (non_differentiable(torch.rand(S * S, S)),)),
        ('reshape_as', (), (non_differentiable(torch.tensor(42.)),), 'scalar'),
        ('reshape_as', (), (non_differentiable(torch.rand(1, 1)),), 'scalar_to_dims'),
        ('view_as', (S, S, S), (non_differentiable(torch.rand(S * S, S)),)),
        ('view_as', (), (non_differentiable(torch.tensor(5.5)),), 'scalar'),
        ('view_as', (), (non_differentiable(torch.rand(1, 1)),), 'scalar_to_dims'),
        ('expand', (S, 1, 1), (S, S, S), '', (False,)),
        ('expand', (torch.Size([S, 1, S]),), (S, S, S), 'size', (False,)),
        ('expand', (S, 1), (S, S, S), 'new_dim', (False,)),
        ('expand', (1,), (S, S, S), '1_element', (False,)),
        ('expand', (1, S), (1, 1, S), 'new_dim_front_old_front_1', (False,)),
        ('expand', (), (dont_convert(()),), 'scalar_to_scalar'),
        ('expand', (), (1, 3, 2), 'scalar_to_dims', (False,)),
        ('expand_as', (S, 1, 1), (torch.rand(S, S, S),), '', (False,)),
        ('fmod', (S, S, S), (1.5,), '', (True,)),
        ('fmod', (), (1.5,), 'scalar', (True,)),
        ('fmod', (S, S, S), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor'),
        ('fmod', (S,), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor_broadcast_lhs'),
        ('fmod', (S, S, S), (non_differentiable(torch.rand(S) + 1.5),), 'tensor_broadcast_rhs'),
        ('fmod', (S, 1, S), (non_differentiable(torch.rand(S, S) + 1.5),), 'tensor_broadcast_all'),
        ('fmod', (), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor'),
        ('fmod', (), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'scalar_tensor_broadcast_lhs'),
        ('fmod', (S, S, S), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor_broadcast_rhs'),
        ('remainder', (S, S, S), (1.5,), '', (True,)),
        ('remainder', (), (1.5,), 'scalar', (True,)),
        ('remainder', (S, S, S), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor'),
        ('remainder', (S,), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'tensor_broadcast_lhs'),
        ('remainder', (S, 1, S), (non_differentiable(torch.rand(S, S) + 1.5),), 'tensor_broadcast_all'),
        ('remainder', (), (non_differentiable(uniform_scalar(1.5)),), 'scalar_tensor'),
        ('remainder', (), (non_differentiable(torch.rand(S, S, S) + 1.5),), 'scalar_tensor_broadcast_lhs'),
        ('kthvalue', (S, S, S), (2,)),
        ('kthvalue', (S, S, S), (2, 1,), 'dim', (), [1]),
        ('kthvalue', (S, S, S), (2, 1, True,), 'keepdim_dim', (), [1]),
        ('kthvalue', (S,), (2, 0,), 'dim_1d', (), [1]),
        ('kthvalue', (S,), (2, 0, True,), 'keepdim_dim_1d', (), [1]),
        ('kthvalue', (), (1,), 'scalar', (), ()),
        ('kthvalue', (), (1, 0,), 'scalar_dim', (), [1]),
        ('kthvalue', (), (1, 0, True), 'scalar_keepdim_dim', (), [1]),
        ('median', (S, S, S), NO_ARGS),
        ('median', (S, S, S), (1,), 'dim', (), [0]),
        ('median', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('median', (), NO_ARGS, 'scalar'),
        ('median', (), (0,), 'scalar_dim', (), [0]),
        ('median', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('nanmedian', (S, S, S), NO_ARGS),
        ('nanmedian', (S, S, S), (1,), 'dim', (), [0]),
        ('nanmedian', (S, S, S), (1, True,), 'keepdim_dim', (), [0]),
        ('nanmedian', (), NO_ARGS, 'scalar'),
        ('nanmedian', (), (0,), 'scalar_dim', (), [0]),
        ('nanmedian', (), (0, True,), 'scalar_keepdim_dim', (), [0]),
        ('var_mean', (S, S, S), NO_ARGS, ''),
        ('var_mean', (S, S, S), (1,), 'dim', [0]),
        ('var_mean', (S, S, S), (1, True, True), 'keepdim_dim', [0]),
        ('var_mean', (S,), (0,), 'dim_1d', [0]),
        ('var_mean', (S,), (0, True, True), 'keepdim_dim_1d', [0]),
        ('std_mean', (S, S, S), NO_ARGS, ''),
        ('std_mean', (S, S, S), (1,), 'dim', [0]),
        ('std_mean', (S, S, S), (1, True, True), 'keepdim_dim', [0]),
        ('std_mean', (S,), (0,), 'dim_1d', [0]),
        ('std_mean', (S,), (0, True, True), 'keepdim_dim_1d', [0]),
        ('renorm', (S, S, S), (2, 1, 0.5), 'dim', (), [1]),
        ('renorm', (S, S, S), (1, 2, 3), 'norm_1'),
        ('renorm', (S, S, S), (inf, 2, 0.5), 'norm_inf'),
        ('log_softmax', (S, S, S), (1, torch.float64,), 'kwarg_dtype_would_break_jit_loader', (True,)),
        ('mvlgamma', torch.empty(S,).uniform_(0.5, 1), [1], "p=1"),
        ('mvlgamma', torch.empty(S,).uniform_(1, 2), [2], "p=2"),
        ('mvlgamma', torch.empty(S, S).uniform_(1.5, 3), [3], "p=3"),
        ('mvlgamma', torch.empty(S, S).uniform_(2.5, 5), [5], "p=5"),
        ('zero_', (S, S, S), NO_ARGS),
        ('zero_', (), NO_ARGS, 'scalar'),
        ('norm', (S, S), (), 'default'),
        ('norm', (S, S), (2,), '2'),
        ('norm', (S, S), (0,), '0'),
        ('norm', (S, S), (0.5,), '0_5'),
        ('norm', (S, S), (1,), '1'),
        ('norm', (S, S), (3,), '3'),
        ('norm', (S, S), (inf,), 'inf'),
        ('norm', (S, S), (-inf,), '-inf'),
        ('norm', (S, S), ('fro',), 'fro_default'),
        ('norm', (S, S), ('fro', [0, 1],), 'fro'),
        ('norm', (S, S), ('nuc',), 'nuc', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('norm', (S, S, S), ('nuc', [1, 2]), 'nuc_batched', (), NO_ARGS, [skipCPUIfNoLapack, skipCUDAIfNoMagma]),
        ('norm', (S, S), (-1,), 'neg_1'),
        ('norm', (S, S), (-2,), 'neg_2'),
        ('norm', (S, S), (-0.5,), 'neg_0_5'),
        ('norm', (S, S), (-1.5,), 'neg_1_5'),
        ('norm', (S, S), (-2, 1,), 'neg_2_2_dim', (), [1]),
        ('norm', (S, S), (-1, 1,), 'neg_1_2_dim', (), [1]),
        ('norm', (S, S), (0, 1,), '0_2_dim', (), [1]),
        ('norm', (S, S), (1, 1,), '1_2_dim', (), [1]),
        ('norm', (S, S), (2, 1,), '2_2_dim', (), [1]),
        ('norm', (S, S), (3, 1,), '3_2_dim', (), [1]),
        ('norm', (S, S), (inf, 1,), 'inf_2_dim'),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5,), '1_5_default'),
        ('norm', (S, S, S), (2, 1), '2_dim', (), [1]),
        ('norm', (S, S, S), (3, 1), '3_dim', (), [1]),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5, 1), '1_5_dim', (), [1]),
        ('norm', (S, S, S), (2, 1, True), 'keepdim_2_dim', (), [1]),
        ('norm', (S, S, S), (3, 1, True), 'keepdim_3_dim', (), [1]),
        ('norm', torch.rand(S, S, S) + 5e-2, (1.5, 1, True), 'keepdim_1_5_dim', (), [1]),
        ('norm', (), (2, 0), '2_dim_scalar', (), [1]),
        ('norm', (), (3, 0), '3_dim_scalar', (), [1]),
        ('norm', (), (2, 0, True), 'keepdim_2_dim_scalar', (), [1]),
        ('norm', (), (3, 0, True), 'keepdim_3_dim_scalar', (), [1]),
        ('clone', (S, M, S), NO_ARGS),
        ('clone', (), NO_ARGS, 'scalar'),
        ('contiguous', (S, S), NO_ARGS, '', (True,)),
        ('contiguous', torch.randn(S, S).transpose(0, 1), NO_ARGS, 'not_contiguous', (True,)),
        ('diag_embed', (S, S), NO_ARGS),
        ('diagonal', (M, M), NO_ARGS, '2d'),
        ('diagonal', (3, 5), NO_ARGS, '2d_wide'),
        ('diagonal', (3, 5), (2,), '2d_wide_pos'),
        ('diagonal', (3, 5), (-2,), '2d_wide_neg'),
        ('diagonal', (5, 3), NO_ARGS, '2d_tall'),
        ('diagonal', (5, 3), (2,), '2d_tall_pos'),
        ('diagonal', (5, 3), (-2,), '2d_tall_neg'),
        ('diagonal', (M, M), (1,), '2d_1'),
        ('diagonal', (M, M), (2,), '2d_2'),
        ('diagonal', (M, M, M), (1, 1, 2), '3d_1'),
        ('diagonal', (M, M, M), (2, 0, 1), '3d_2'),
        ('diagonal', (M, M, M), (-2, 0, 1), '3d_3'),
        ('tril', (M, M), NO_ARGS),
        ('tril', (M, M), (2,), 'idx'),
        ('tril', (S, M, M), NO_ARGS, 'batched'),
        ('tril', (S, M, M), (2,), 'batched_idx'),
        ('tril', (3, 3, S, S), NO_ARGS, 'more_batched'),
        ('triu', (M, M), NO_ARGS),
        ('triu', (M, M), (2,), 'idx'),
        ('triu', (S, M, M), NO_ARGS, 'batched'),
        ('triu', (S, M, M), (2,), 'batched_idx'),
        ('triu', (3, 3, S, S), NO_ARGS, 'more_batched'),
        ('cross', (S, 3), ((S, 3),)),
        ('cross', (S, 3, S), ((S, 3, S), 1), 'dim'),
        ('fill_', (S, S, S), (1,), 'number'),
        ('fill_', (), (1,), 'number_scalar'),
        ('fill_', (S, S, S), ((),), 'variable'),
        ('select', (S, S, S), (1, 2), 'dim', (), [0]),
        ('select', (S, S, S), (1, -1), 'wrap_dim', (), [0]),
        ('select', (S,), (0, 2), '1d'),
        ('narrow', (S, S, S), (1, 2, 2), 'dim', (), [0]),
        ('narrow', (S, S, S), (1, 0, 0), 'empty_dim', (), [0]),
        ('squeeze', (S, 1, S, 1), NO_ARGS, '', (True,)),
        ('squeeze', (1, 1, 1, 1), NO_ARGS, 'input_sizes_are_ones', (True,)),
        ('squeeze', (S, 1, S, 1), (1,), '1_dim', (True,), [0]),
        ('squeeze', (S, 1, S, 1), (2,), 'not_1_dim', (True,), [0]),
        ('squeeze', (), (0,), 'scalar', (True,), [0]),
        ('chunk', (S, S, S), (2,), '', (True, 'prim::ConstantChunk')),
        ('chunk', (S, S, S), (S, 1), 'dim', (True, 'prim::ConstantChunk'), [1]),
        ('split', (S, S, S), (2,), '', (True,)),
        ('split', (S, S, S), (S, 1), 'dim', (True,), [1]),
        ('split', (S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],), 'size_list',
            (True, 'aten::split_with_sizes')),
        ('split', (S, S, S), ([int(S / 2), S - int(S / 2) * 2, int(S / 2)], 2), 'size_list_dim',
            (True, 'aten::split_with_sizes'), [1]),
        ('split_with_sizes', (S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],), '', (True,)),
        ('split_with_sizes', (S, S, S), ([int(S / 3), S - int(S / 3), 0],), 'size_0', (True, )),
        ('split_with_sizes', (S, S, S), ([int(S / 3), S - int(S / 3) * 2, int(S / 3)],), 'dim', (True, ), [1]),
        ('tensor_split', (S, S, S), (3,), 'sections', (False,)),
        ('tensor_split', (S, S, S), (3, 1), 'sections_dim', (False,), [1]),
        ('tensor_split', (S, S, S), ([2, 4],), 'indices', (False,)),
        ('tensor_split', (S, S, S), ([2, 4], 1), 'indices_dim', (False,), [1]),
        ('resize_', (S, S, S), (torch.Size([S * S, S])), 'fewer_dims'),
        ('resize_', (), (dont_convert(()),), 'scalar'),
        ('resize_', (), (torch.Size([1, 1, 1])), 'scalar_to_dims'),
        ('resize_as_', (), (non_differentiable(torch.tensor(5.)),), 'scalar'),
        ('resize_as_', (), (non_differentiable(torch.randn((1, 1, 1))),), 'scalar_to_dims'),
        ('resize_as_', (S, S, S), (non_differentiable(torch.randn(S * S, S)),)),
        ('where', (M, M), (mask_not_all_zeros((M, M)), (M, M)), '', (True,)),
        ('where', (M, 1, M), (mask_not_all_zeros((M, M)), (M, M, 1)), 'broadcast_all', (True,)),
        ('where', (), (bernoulli_scalar(), ()), 'scalar', (True,)),
        ('where', (M, 1, M), (bernoulli_scalar(), (M, M, 1)), 'scalar_broadcast_mask', (True,)),
        ('where', (), (mask_not_all_zeros((M, M)), ()), 'scalar_broadcast_non_mask', (True,)),
        ('to_sparse', (S, S), (), '', (), (), [], lambda x: x.to_dense())
    ]

                        # compare grads to inplace grads
                        inplace_name = name + '_'
                        # can't broadcast inplace to left hand side
                        skip_inplace = ('broadcast_lhs' in test_name or
                                        'broadcast_all' in test_name)
                        if hasattr(torch.ones(1), inplace_name) and not skip_inplace:
                            output_variable = getattr(self_variable, name)(*args_variable, **kwargs_variable)
                            if not isinstance(output_variable, tuple):
                                output_variable = (output_variable,)
                            inplace_self_variable = deepcopy(self_variable)
                            inplace_self_variable_copy = tuple(i.clone() if isinstance(i, torch.Tensor) else i
                                                               for i in (inplace_self_variable,))
                            inplace_args_variable = deepcopy(args_variable)
                            inplace_args_variable_copy = tuple(i.clone() if isinstance(i, torch.Tensor) else i
                                                               for i in inplace_args_variable)

                            inplace_output_variable = (
                                getattr(inplace_self_variable_copy[0], inplace_name)(*inplace_args_variable_copy,
                                                                                     **kwargs_variable))
                            if not isinstance(inplace_output_variable, tuple):
                                inplace_output_variable = (inplace_output_variable,)
                            self.assertEqual(inplace_output_variable, output_variable)
                            # Check that gradient is the same
                            for inp_i, i in zip((inplace_self_variable,) + inplace_args_variable,
                                                (self_variable,) + args_variable):
                                if not isinstance(inp_i, torch.Tensor):
                                    assert not isinstance(i, torch.Tensor)
                                    continue
                                if inp_i.grad is not None:
                                    with torch.no_grad():
                                        inp_i.grad.zero_()
                                if i.grad is not None:
                                    with torch.no_grad():
                                        i.grad.zero_()
                            for i_o, o in zip(inplace_output_variable, output_variable):
                                if dtype.is_complex:
                                    grad = randn_like(i_o).to(torch.cdouble)
                                else:
                                    grad = randn_like(i_o).double()
                                i_o.backward(grad)
                                o.backward(grad)
                            for inp_i, i in zip((inplace_self_variable,) + inplace_args_variable,
                                                (self_variable,) + args_variable):
                                if not isinstance(inp_i, torch.Tensor):
                                    continue
                                self.assertEqual(inp_i.grad, i.grad)

        if isinstance(arg, torch.Size) or isinstance(arg, dont_convert):
            return arg
        elif isinstance(arg, tuple) and len(arg) == 0:
            var = torch.randn((), dtype=dtype, device=device)
            var.requires_grad = requires_grad
            return var
        elif isinstance(arg, tuple) and not isinstance(arg[0], torch.Tensor):
            return Variable(maybe_non_contig(torch.randn(*arg, dtype=dtype, device=device)), requires_grad=requires_grad)
        # double check casting
        elif isinstance(arg, non_differentiable):
            if isinstance(arg.tensor, torch.Tensor):
                if arg.tensor.dtype == torch.float:
                    return maybe_non_contig(arg.tensor.to(dtype=torch.double, device=device))
                if arg.tensor.dtype == torch.cfloat:
                    return maybe_non_contig(arg.tensor.to(dtype=torch.cdouble, device=device))
                return maybe_non_contig(arg.tensor.to(device=device))
            return maybe_non_contig(arg.tensor.to(device=device))
        elif isinstance(arg, torch.Tensor):
            if arg.dtype == torch.float:
                arg = arg.double()
            if arg.dtype == torch.cfloat:
                arg = arg.to(torch.cdouble)
            if arg.is_complex() != dtype.is_complex:
                raise RuntimeError("User provided tensor is real for a test that runs with complex dtype, ",
                                   "which is not supported for now")
            # NOTE: We do clone() after detach() here because we need to be able to change size/storage of v afterwards
            v = maybe_non_contig(arg).detach().to(device=device).clone()
            v.requires_grad = requires_grad and (v.is_floating_point() or v.is_complex())
            return v
        elif callable(arg):
            return map_arg(arg(dtype=dtype, device=device))
        else:
            # Wrong base
            raise RuntimeError("The base given to `_assert_same_struct` doesn't have"
                               " the right structure.")

    def _assert_interleaved_struct(self, res, base1, base2):
        # base1 and base2 can be Tensors or tuples of Tensors.
        # If they are tuples, res should be a tuple as well.
        # The indexing works as follows for base1, base2 being
        # - tuple, tuple: res[i][j][k][l] = (base1[i][k], base2[j][l])
        # - tuple, Tensor: res[i][k][l] = (base1[i][k], base2[l])
        # - Tensor, tuple: res[i][j][l] = (base1[i], base2[j][l])
        # - Tensor, Tensor: res[k][l] = (base1[k], base2[l])
        if isinstance(base1, torch.Tensor) and isinstance(base2, torch.Tensor):
            self.assertTrue(isinstance(res, torch.Tensor))
            self.assertEqual(res.size(), base1.size() + base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, torch.Tensor):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base1, torch.Tensor))
                self.assertEqual(el_res.size(), el_base1.size() + base2.size())
        elif isinstance(base1, torch.Tensor) and isinstance(base2, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base2))
            for el_res, el_base2 in zip(res, base2):
                self.assertTrue(isinstance(el_res, torch.Tensor))
                self.assertTrue(isinstance(el_base2, torch.Tensor))
                self.assertEqual(el_res.size(), base1.size() + el_base2.size())
        elif isinstance(base1, tuple) and isinstance(base2, tuple):
            self.assertTrue(isinstance(res, tuple))
            self.assertEqual(len(res), len(base1))
            for el_res, el_base1 in zip(res, base1):
                self.assertTrue(isinstance(el_res, tuple))
                self.assertEqual(len(res), len(base2))
                for el_el_res, el_base2 in zip(el_res, base2):
                    self.assertTrue(isinstance(el_el_res, torch.Tensor))
                    self.assertTrue(isinstance(el_base2, torch.Tensor))
                    self.assertEqual(el_el_res.size(), el_base1.size() + el_base2.size())
        else:
            # Wrong bases
            raise RuntimeError("The bases given to `_assert_interleaved_struct` don't have"
                               " the right structure.")

    def test_vjp_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = torch.rand(4)
        v = torch.ones(3)
        with self.assertRaisesRegex(TypeError, "The inputs given to vjp must be either a Tensor"):
            res = autogradF.vjp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to vjp must"):
            res = autogradF.vjp(bar, inp, v)

        with self.assertRaisesRegex(RuntimeError, "The vector v can only be None if the user-provided function returns"):
            res = autogradF.vjp(foo, inp)

        with self.assertRaisesRegex(RuntimeError, "The given v should contain a single Tensor."):
            res = autogradF.vjp(foo, inp, (torch.ones_like(inp), torch.ones_like(inp)))

        with self.assertRaisesRegex(RuntimeError, "v has invalid size: should be torch.Size"):
            res = autogradF.vjp(foo, inp, v[:2])

        res = autogradF.vjp(foo, inp, v)[1]
        self._assert_same_struct(res, inp)

    def test_vjp_err_check_strict(self):
        def foo(a):
            return a.detach()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.vjp(foo, inp, v, strict=True)
        res = autogradF.vjp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.vjp(bar, inp, v, strict=True)
        res = autogradF.vjp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        # The Jacobian does not depend on the input
        def foo(a):
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function is independent of input 0."):
            res = autogradF.vjp(foo, inp, v, create_graph=True, strict=True)
        res = autogradF.vjp(foo, inp, v, create_graph=True, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1], v)

    def test_vjp_no_grad(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(4, 4)
        v = torch.ones(4)
        with torch.no_grad():
            res = autogradF.vjp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

        inputs.requires_grad_()
        v.requires_grad_()
        with torch.no_grad():
            res = autogradF.vjp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

    def test_vjp_output(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(4, 4)
        v = torch.ones(4)
        res = autogradF.vjp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y

        inputs = (torch.rand(2), torch.rand(2))
        v = torch.ones(2)
        out, vjp_val = autogradF.vjp(adder, inputs, v)
        self._assert_same_struct(vjp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(vjp_val[0].grad_fn)
        self.assertIsNone(vjp_val[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y, x + y

        inputs = (torch.rand(2), torch.rand(2))
        v = (torch.tensor([1., 0.]), torch.tensor([1., 0.]))
        out, vjp_val = autogradF.vjp(adder, inputs, v)
        self._assert_same_struct(vjp_val, inputs)
        self.assertIsNone(out[0].grad_fn)
        self.assertIsNone(out[1].grad_fn)
        self.assertIsNone(vjp_val[0].grad_fn)
        self.assertIsNone(vjp_val[1].grad_fn)

    def test_vjp_scalar(self):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        v = torch.ones([])
        res = autogradF.vjp(reducer, inputs, v)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.vjp(reducer, inputs)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

        def expander(x):
            return x.unsqueeze(0).repeat(4)
        inputs = torch.rand([])
        v = torch.ones(4)
        res = autogradF.vjp(expander, inputs, v)
        self._assert_same_struct(res[0], v)
        self._assert_same_struct(res[1], inputs)

    def test_vjp_create_graph(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(2, 2, dtype=torch.double)
        v = torch.ones(2, dtype=torch.double)

        inputs.requires_grad_()
        v.requires_grad_()
        res = autogradF.vjp(reducer, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.vjp(reducer, inputs, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.vjp(reducer, inputs, v, create_graph=True), (inputs, v))

        def adder(x, y):
            return 2 * x + 3 * y, x * y

        inputs = (torch.rand(2, dtype=torch.double, requires_grad=True),
                  torch.rand(2, dtype=torch.double, requires_grad=True))
        v = (torch.tensor([1., 0.], dtype=torch.double, requires_grad=True),
             torch.tensor([1., 0.], dtype=torch.double, requires_grad=True))

        gradcheck(lambda *args: autogradF.vjp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.vjp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.vjp(adder, (x, y), v, create_graph=True)

            return val[0].exp() + val[1].exp() + grad[0].exp() + grad[1].exp() + x.exp() + y.exp()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def test_jvp_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to jvp must be either a Tensor"):
            res = autogradF.jvp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to jvp must"):
            res = autogradF.jvp(bar, inp, v)

        with self.assertRaisesRegex(RuntimeError, "The vector v can only be None if the input to the user-provided function"):
            res = autogradF.jvp(foo, inp)

        with self.assertRaisesRegex(RuntimeError, "The given v should contain a single Tensor."):
            res = autogradF.jvp(foo, inp, (v, v))

        with self.assertRaisesRegex(RuntimeError, "v has invalid size: should be torch.Size"):
            res = autogradF.jvp(foo, inp, v[:2])

        res = autogradF.jvp(foo, inp, v)[1]
        self._assert_same_struct(res, foo(inp))

    def test_jvp_err_check_strict(self):
        def foo(a):
            return a.detach()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.jvp(foo, inp, v, strict=True)
        res = autogradF.jvp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], res[0])
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.jvp(bar, inp, v, strict=True)
        res = autogradF.jvp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], res[0])
        self.assertEqual(res[1].abs().sum(), 0.)

        # The Jacobian does not depend on the input
        def foo(a):
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function is independent of input 0."):
            res = autogradF.jvp(foo, inp, v, create_graph=True, strict=True)
        res = autogradF.jvp(foo, inp, v, create_graph=True, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1], v)

    def test_jvp_no_grad(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        with torch.no_grad():
            res = autogradF.jvp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

        inputs.requires_grad_()
        v.requires_grad_()
        with torch.no_grad():
            res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

    def test_jvp_output(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.jvp(reducer, inputs, v)
        self._assert_same_struct(res[1], res[0])
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y

        inputs = (torch.rand(2), torch.rand(2))
        v = (torch.ones(2), torch.ones(2))
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        self._assert_same_struct(jvp_val, out)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(jvp_val[0].grad_fn)
        self.assertIsNone(jvp_val[1].grad_fn)

        def adder(x, y):
            return 2 * x + 3 * y, x + y

        inputs = (torch.rand(2), torch.rand(2))
        v = (torch.tensor([1., 0.]), torch.tensor([1., 0.]))
        out, jvp_val = autogradF.jvp(adder, inputs, v)
        self._assert_same_struct(jvp_val, out)
        self.assertIsNone(out[0].grad_fn)
        self.assertIsNone(out[1].grad_fn)
        self.assertIsNone(jvp_val[0].grad_fn)
        self.assertIsNone(jvp_val[1].grad_fn)

    def test_jvp_scalar(self):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.jvp(reducer, inputs, v)
        self._assert_same_struct(res[0], torch.zeros([]))
        self._assert_same_struct(res[1], res[0])

        def expander(x):
            return x.unsqueeze(0).repeat(4)
        inputs = torch.rand([])
        v = torch.ones([])
        res = autogradF.jvp(expander, inputs, v)
        self._assert_same_struct(res[0], torch.zeros(4))
        self._assert_same_struct(res[1], res[0])

        res = autogradF.jvp(expander, inputs)
        self._assert_same_struct(res[0], torch.zeros(4))
        self._assert_same_struct(res[1], res[0])

    def test_jvp_create_graph(self):
        def reducer(x):
            return x.sum(dim=1)
        inputs = torch.rand(2, 2, dtype=torch.double)
        v = torch.ones(2, 2, dtype=torch.double)

        inputs.requires_grad_()
        v.requires_grad_()
        res = autogradF.jvp(reducer, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], res[0])
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.jvp(reducer, inp, v, create_graph=True), (inputs, v))

        def adder(x, y):
            return 2 * x + 3 * y, x * y

        inputs = (torch.rand(2, dtype=torch.double, requires_grad=True),
                  torch.rand(2, dtype=torch.double, requires_grad=True))
        v = (torch.tensor([1., 0.], dtype=torch.double, requires_grad=True),
             torch.tensor([1., 0.], dtype=torch.double, requires_grad=True))

        gradcheck(lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.jvp(adder, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.jvp(adder, (x, y), v, create_graph=True)

            return val[0].exp() + val[1].exp() + grad[0].exp() + grad[1].exp() + x.exp() + y.exp()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def _test_construct_standard_basis_for(self, inputs):
        numels = tuple(tensor.numel() for tensor in inputs)
        results = autogradF._construct_standard_basis_for(inputs, numels)
        for result, inp in zip(results, inputs):
            self.assertEqual(result.dtype, inp.dtype)
            self.assertEqual(result.device, inp.device)
        results = torch.cat([result.to(device='cpu', dtype=torch.float)
                             for result in results], dim=1)
        expected = torch.eye(results[0].shape[0], dtype=torch.float)
        self.assertEqual(results, expected)

    def test_construct_standard_basis_for(self):
        test_cases = [
            (torch.randn(2, 3),),
            (torch.randn(1),),
            (torch.randn([]),),
            (torch.randn(1), torch.randn([]), torch.randn([])),
            (torch.randn(2), torch.randn(3), torch.randn([])),
            (torch.randn(2), torch.randn([]), torch.randn(3)),
            (torch.randn(2, 3), torch.randn(3), torch.randn(3, 4, 2)),
            (torch.randn(2, dtype=torch.float64), torch.randn(3, dtype=torch.float32)),
        ]

        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_construct_standard_basis_for_cuda(self):
        test_cases = [
            (torch.randn(2), torch.randn(3, device='cuda')),
            (torch.randn(3, device='cuda'), torch.randn(2)),
        ]

        for inputs in test_cases:
            self._test_construct_standard_basis_for(inputs)

    def _test_vectorize_raises_no_warnings(self, api):
        # vmap is an experimental prototype. When someone calls torch.vmap,
        # it raises a python warning. This test checks that
        # autogradF.{jacobian, hessian} don't raise that experimental prototype
        # warning; it is not nice for a public-facing API to raise a warning
        # no matter how it is called.
        def foo(a):
            return (a ** 2).sum()

        x = torch.randn(3)
        with warnings.catch_warnings(record=True) as wa:
            result = api(foo, x, vectorize=True)
        self.assertEqual(len(wa), 0)

    def test_jacobian_vectorize_raises_no_warnings(self):
        return self._test_vectorize_raises_no_warnings(autogradF.jacobian)

    def test_hessian_vectorize_raises_no_warnings(self):
        return self._test_vectorize_raises_no_warnings(autogradF.hessian)

    def _test_jacobian_err_check(self, vectorize):
        def foo(a):
            return 3 * a.narrow(0, 0, 3)

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        inp = torch.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to jacobian must be either a Tensor"):
            res = autogradF.jacobian(foo, (inp, 2), vectorize=vectorize)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to jacobian must"):
            res = autogradF.jacobian(bar, inp, vectorize=vectorize)

        res = autogradF.jacobian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, foo(inp), inp)

        def foo(a, b):
            return b, 3 * a.narrow(0, 0, 3)

        inp = (torch.rand(4), torch.rand(5))

        res = autogradF.jacobian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, foo(*inp), inp)

    def test_jacobian_err_check(self):
        return self._test_jacobian_err_check(vectorize=False)

    def test_jacobian_err_check_vectorize(self):
        return self._test_jacobian_err_check(vectorize=True)

    def test_jacobian_err_check_strict(self):
        def foo(a):
            return a.detach()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone()

        inp = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.jacobian(foo, inp, strict=True)
        res = autogradF.jacobian(foo, inp, strict=False)
        self._assert_interleaved_struct(res, foo(inp), inp)
        self.assertEqual(res.abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function is independent of input 0."):
            res = autogradF.jacobian(bar, inp, strict=True)
        res = autogradF.jacobian(bar, inp, strict=False)
        self._assert_interleaved_struct(res, foo(inp), inp)
        self.assertEqual(res.abs().sum(), 0.)

        # The Jacobian does not depend on the input
        def foo(a):
            return a.clone()

        inp.requires_grad_()
        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function is independent of input 0."):
            res = autogradF.jacobian(foo, inp, create_graph=True, strict=True)
        res = autogradF.jacobian(foo, inp, create_graph=True, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res, torch.eye(4))

    def test_jacobian_err_check_strict_vectorize(self):
        def foo(x):
            return x

        inp = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            res = autogradF.jacobian(foo, inp, strict=True, vectorize=True)

    def test_jacobian_no_grad(self):
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        inputs = torch.rand(4, 4)
        with torch.no_grad():
            res = autogradF.jacobian(exp_reducer, inputs)
        self.assertIsNone(res.grad_fn)
        self.assertNotEqual(res, torch.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.jacobian(exp_reducer, inputs, create_graph=True)
        self.assertIsNotNone(res.grad_fn)
        self.assertNotEqual(res, torch.zeros(4, 4))

    def _test_jacobian_output(self, vectorize):
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        inputs = torch.rand(4, 4)
        res = autogradF.jacobian(exp_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, exp_reducer(inputs), inputs)
        self.assertIsNone(res.grad_fn)

        def identity(x):
            return x.clone()

        inputs = torch.rand(4)
        res = autogradF.jacobian(identity, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, identity(inputs), inputs)
        self.assertIsNone(res.grad_fn)
        self.assertEqual(res, torch.eye(4))

        def add_exp_reducer(x, y):
            return (x + y.exp()).sum(dim=1)

        inputs = (torch.rand(4, 4), torch.rand(4, 4))
        res = autogradF.jacobian(add_exp_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, add_exp_reducer(*inputs), inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

    def test_jacobian_output(self):
        self._test_jacobian_output(vectorize=False)

    def test_jacobian_output_vectorize(self):
        self._test_jacobian_output(vectorize=True)

    def _test_jacobian_scalar(self, vectorize):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        res = autogradF.jacobian(reducer, inputs, vectorize=vectorize)
        self._assert_same_struct(res, inputs)

        def expander(x):
            return x.unsqueeze(0).repeat(4)
        inputs = torch.rand([])
        res = autogradF.jacobian(expander, inputs, vectorize=vectorize)
        self._assert_same_struct(res, torch.zeros(4))

    def test_jacobian_scalar(self):
        self._test_jacobian_scalar(vectorize=False)

    def test_jacobian_scalar_vectorize(self):
        self._test_jacobian_scalar(vectorize=True)

    def _test_jacobian_create_graph(self, vectorize):
        def exp_reducer(x):
            return x.exp().sum(dim=1)

        inputs = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.jacobian(exp_reducer, inputs, create_graph=True, vectorize=vectorize)
        self._assert_interleaved_struct(res, exp_reducer(inputs), inputs)
        self.assertIsNotNone(res.grad_fn)

        gradcheck(lambda inp: autogradF.jacobian(exp_reducer, inp, create_graph=True, vectorize=vectorize), inputs)
        gradgradcheck(lambda inp: autogradF.jacobian(exp_reducer, inp, create_graph=True, vectorize=vectorize), inputs)

        def add_exp_reducer(x, y):
            return (x + y).exp().sum(dim=1)

        inputs = (torch.rand(4, 4, dtype=torch.double, requires_grad=True),
                  torch.rand(4, 4, dtype=torch.double, requires_grad=True))
        res = autogradF.jacobian(add_exp_reducer, inputs, create_graph=True, vectorize=vectorize)
        self._assert_interleaved_struct(res, add_exp_reducer(*inputs), inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda *inp: autogradF.jacobian(add_exp_reducer, inp, create_graph=True, vectorize=vectorize), inputs)
        gradgradcheck(lambda *inp: autogradF.jacobian(add_exp_reducer, inp, create_graph=True, vectorize=vectorize), inputs)

        def foo(x, y):
            x = x.cos()
            val, jac = autogradF.jacobian(add_exp_reducer, (x, y), create_graph=True, vectorize=vectorize)

            res = val[0].exp().sum() + val[1].exp().sum() + jac[0].exp().sum()
            res = res + jac[1].exp().sum() + x.exp().sum() + y.exp().sum()
            return res

        gradcheck(foo, inputs)
        gradgradcheck(foo, inputs)

    def test_jacobian_create_graph(self):
        self._test_jacobian_create_graph(vectorize=False)

    def test_jacobian_create_graph_vectorize(self):
        self._test_jacobian_create_graph(vectorize=True)

    def _check_jacobian_vectorize_correctness(self, f, inputs):
        expected = autogradF.jacobian(f, inputs, vectorize=False)
        result = autogradF.jacobian(f, inputs, vectorize=True)
        self.assertEqual(result, expected)

    def test_jacobian_vectorize_correctness_simple(self):
        def f(x):
            return 3 * x ** 2

        x = torch.randn(2, 3, 5)
        self._check_jacobian_vectorize_correctness(f, x)

    def test_jacobian_vectorize_correctness_multi_input(self):
        def f(x, y):
            return (x.cos() * x) @ y.sin()

        x = torch.randn(2, 3)
        y = torch.randn(3, 5)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def test_jacobian_vectorize_correctness_multi_input_multi_output(self):
        def f(x, y):
            return (x * x) @ y, x @ (x.sum(1) * y), y.sum()

        x = torch.randn(5, 3)
        y = torch.randn(3, 5)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def test_jacobian_vectorize_correctness_unrelated_outputs(self):
        def f(x, y):
            return x, y, x, y

        x = torch.randn(2)
        y = torch.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def test_jacobian_vectorize_correctness_zero_dim(self):
        # zero-dim output
        def f(x, y):
            return x.sum(), y.sum(), x * y

        x = torch.randn(3)
        y = torch.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

        # zero-dim input
        def g(x):
            return torch.stack([x, x, x])

        x = torch.randn([])
        self._check_jacobian_vectorize_correctness(g, x)

        # Mixed zero-dim input / zero-dim output
        def h(x, y):
            return y.sum(), x * y

        x = torch.randn([])
        y = torch.randn(1)
        self._check_jacobian_vectorize_correctness(h, (x, y))

    @unittest.skipIf(not TEST_CUDA, "test requires CUDA")
    def test_jacobian_vectorize_correctness_different_devices(self):
        def f(x, y):
            return x * y, (x * y).cuda()

        x = torch.randn(3)
        y = torch.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def test_jacobian_vectorize_correctness_different_dtype(self):
        def f(x, y):
            return (x * y).float(), (x * y).double()

        x = torch.randn(3)
        y = torch.randn(3)
        self._check_jacobian_vectorize_correctness(f, (x, y))

    def _check_hessian_vectorize_correctness(self, f, inputs):
        expected = autogradF.hessian(f, inputs, vectorize=False)
        result = autogradF.hessian(f, inputs, vectorize=True)
        self.assertEqual(result, expected)

    def test_hessian_vectorize_correctness_simple(self):
        def f(x):
            return (3 * x ** 2).sum()

        x = torch.randn(2, 3, 5)
        self._check_hessian_vectorize_correctness(f, x)

    def test_hessian_vectorize_correctness_multi_input(self):
        def f(x, y, z):
            return ((x.relu() * x) @ y.sin() @ z).sum()

        x = torch.randn(2, 3)
        y = torch.randn(3, 5)
        z = torch.randn(5, 5)
        self._check_hessian_vectorize_correctness(f, (x, y, z))

    def test_hessian_vectorize_correctness_unrelated_outputs(self):
        # output unrelated to one input
        def f(x, y):
            return (x ** 2).sum()

        x = torch.randn(2)
        y = torch.randn(3)
        self._check_hessian_vectorize_correctness(f, (x, y))

        # output unrelated to all inputs
        def f(x, y):
            return torch.randn([])

        x = torch.randn(2)
        y = torch.randn(3)
        self._check_hessian_vectorize_correctness(f, (x, y))

    def _test_hessian_err_check(self, vectorize):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            return 3 * a.narrow(0, 0, 3)

        def bar3(a):
            return 3 * a.narrow(0, 0, 3), 3 * a.narrow(0, 0, 3)

        inp = torch.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to hessian must be either a Tensor"):
            res = autogradF.hessian(foo, (inp, 2), vectorize=vectorize)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to hessian must"):
            res = autogradF.hessian(bar, inp, vectorize=vectorize)

        err_msg_out = "The Tensor returned by the function given to hessian should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.hessian(bar2, inp, vectorize=vectorize)

        with self.assertRaisesRegex(RuntimeError, "The function given to hessian should return a single Tensor"):
            res = autogradF.hessian(bar3, inp, vectorize=vectorize)

        res = autogradF.hessian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, inp, inp)

        def foo(a, b):
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (torch.rand(4), torch.rand(5))

        res = autogradF.hessian(foo, inp, vectorize=vectorize)
        self._assert_interleaved_struct(res, inp, inp)

    def test_hessian_err_check(self):
        self._test_hessian_err_check(vectorize=False)

    def test_hessian_err_check_vectorize(self):
        self._test_hessian_err_check(vectorize=True)

    def test_hessian_err_check_strict(self):
        def foo(a):
            return a.detach().sum()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.hessian(foo, inp, strict=True)
        res = autogradF.hessian(foo, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function with respect to input 0"):
            res = autogradF.hessian(bar, inp, strict=True)
        res = autogradF.hessian(bar, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function with respect to input 0 is"):
            res = autogradF.hessian(bar2, inp, strict=True)
        res = autogradF.hessian(bar2, inp, strict=False)
        self._assert_interleaved_struct(res, inp, inp)
        self.assertEqual(res.abs().sum(), 0.)

    def test_hessian_err_check_strict_vectorize(self):
        def foo(x):
            return (x ** 3).sum()

        inp = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "not supported together"):
            res = autogradF.hessian(foo, inp, strict=True, vectorize=True)

    def test_hessian_no_grad(self):
        def pow_reducer(x):
            return x.pow(3).sum()

        inputs = torch.rand(2, 2)
        with torch.no_grad():
            res = autogradF.hessian(pow_reducer, inputs)
        self.assertIsNone(res[0][0].grad_fn)
        self.assertIsNone(res[0][1].grad_fn)
        self.assertIsNone(res[1][0].grad_fn)
        self.assertIsNone(res[1][1].grad_fn)
        self.assertNotEqual(res, torch.zeros(2, 2, 2))

        with torch.no_grad():
            res = autogradF.hessian(pow_reducer, inputs, create_graph=True)
        self.assertIsNotNone(res[0][0].grad_fn)
        self.assertIsNotNone(res[0][1].grad_fn)
        self.assertIsNotNone(res[1][0].grad_fn)
        self.assertIsNotNone(res[1][1].grad_fn)
        self.assertNotEqual(res, torch.zeros(2, 2, 2))


    def _test_hessian_output(self, vectorize):
        def pow_reducer(x):
            return x.pow(3).sum()

        inputs = torch.rand(2, 2)
        res = autogradF.hessian(pow_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNone(res.grad_fn)

        def add_pow_reducer(x, y):
            return (x + y).pow(3).sum()

        inputs = (torch.rand(2, 2), torch.rand(2, 2))
        res = autogradF.hessian(add_pow_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNone(res[0][0].grad_fn)
        self.assertIsNone(res[0][1].grad_fn)
        self.assertIsNone(res[1][0].grad_fn)
        self.assertIsNone(res[1][1].grad_fn)

    def test_hessian_output(self):
        self._test_hessian_output(vectorize=False)

    def test_hessian_output_vectorize(self):
        self._test_hessian_output(vectorize=True)

    def _test_hessian_scalar(self, vectorize):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        res = autogradF.hessian(reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)

        inputs = torch.rand([])
        res = autogradF.hessian(reducer, inputs, vectorize=vectorize)
        self._assert_same_struct(res, inputs)

        def bad_reducer(x):
            return x.sum().view(1, 1, 1)
        inputs = torch.rand(4, 4)
        res = autogradF.hessian(bad_reducer, inputs, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)

    def test_hessian_scalar(self):
        return self._test_hessian_scalar(vectorize=False)

    def test_hessian_scalar_vectorize(self):
        return self._test_hessian_scalar(vectorize=True)

    def _test_hessian_create_graph(self, vectorize):
        def pow_reducer(x):
            return x.pow(3).sum()

        inputs = torch.rand(2, 2, dtype=torch.double, requires_grad=True)
        res = autogradF.hessian(pow_reducer, inputs, create_graph=True, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNotNone(res.grad_fn)

        gradcheck(lambda inp: autogradF.hessian(pow_reducer, inp, create_graph=True, vectorize=vectorize), inputs)
        gradgradcheck(lambda inp: autogradF.hessian(pow_reducer, inp, create_graph=True, vectorize=vectorize), inputs)

        def add_pow_reducer(x, y):
            return (x + y).pow(3).sum()

        inputs = (torch.rand(2, 2, dtype=torch.double, requires_grad=True),
                  torch.rand(2, 2, dtype=torch.double, requires_grad=True))
        res = autogradF.hessian(add_pow_reducer, inputs, create_graph=True, vectorize=vectorize)
        self._assert_interleaved_struct(res, inputs, inputs)
        self.assertIsNotNone(res[0][0].grad_fn)
        self.assertIsNotNone(res[0][1].grad_fn)
        self.assertIsNotNone(res[1][0].grad_fn)
        self.assertIsNotNone(res[1][1].grad_fn)

        def flatten(inp):
            return tuple(el_lvl2 for el_lvl1 in inp for el_lvl2 in el_lvl1)

        gradcheck(lambda *inp: flatten(autogradF.hessian(add_pow_reducer, inp, create_graph=True, vectorize=vectorize)), inputs)
        gradgradcheck(lambda *inp: flatten(autogradF.hessian(add_pow_reducer, inp, create_graph=True, vectorize=vectorize)), inputs)

        def foo(x, y):
            x = x.cos()
            val, hess = autogradF.hessian(add_pow_reducer, (x, y), create_graph=True, vectorize=vectorize)

            res = val[0].cos().sum() + val[1].cos().sum() + hess[0].cos().sum()
            res = res + hess[1].cos().sum() + x.cos().sum() + y.cos().sum()
            return res

        gradcheck(foo, inputs)
        gradgradcheck(foo, inputs)

    def test_hessian_create_graph(self):
        self._test_hessian_create_graph(vectorize=False)

    def test_hessian_create_graph_vectorize(self):
        self._test_hessian_create_graph(vectorize=True)

    def test_vhp_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            return 3 * a.narrow(0, 0, 3)

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(TypeError, "The inputs given to vhp must be either a Tensor"):
            res = autogradF.vhp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to vhp must"):
            res = autogradF.vhp(bar, inp, v)

        err_msg_out = "The Tensor returned by the function given to vhp should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.vhp(bar2, inp, v)

        with self.assertRaisesRegex(RuntimeError, "v has invalid size:"):
            res = autogradF.vhp(foo, inp, torch.rand(5))

        with self.assertRaisesRegex(TypeError, "The v given to vhp must be either a Tensor or a tuple of Tensors"):
            res = autogradF.vhp(foo, inp, (v, 2))

        res = autogradF.vhp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

        def foo(a, b):
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (torch.rand(4), torch.rand(5))
        v = (torch.rand(4), torch.rand(5))

        res = autogradF.vhp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

    def test_vhp_err_check_strict(self):
        def foo(a):
            return a.detach().sum()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.vhp(foo, inp, v, strict=True)
        res = autogradF.vhp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.vhp(bar, inp, v, strict=True)
        res = autogradF.vhp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function with respect to input 0 is"):
            res = autogradF.vhp(bar2, inp, v, strict=True)
        res = autogradF.vhp(bar2, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

    def test_vhp_no_grad(self):
        def reducer(x):
            return x.exp().sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        with torch.no_grad():
            res = autogradF.vhp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.vhp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

    def test_vhp_output(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.vhp(foo, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (torch.rand(3), torch.rand(4))
        v = (torch.ones(3), torch.ones(4))
        out, vhp_val = autogradF.vhp(bar, inputs, v)
        self._assert_same_struct(vhp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(vhp_val[0].grad_fn)
        self.assertIsNone(vhp_val[1].grad_fn)

    def test_vhp_scalar(self):
        def reducer(x):
            return x.sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.vhp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        inputs = torch.rand([])
        v = torch.rand([])
        res = autogradF.vhp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.vhp(reducer, inputs)
        self._assert_same_struct(res[1], inputs)

        def bad_reducer(x):
            return x.sum().view(1, 1, 1)
        inputs = torch.rand(4, 4)
        v = torch.rand(4, 4)
        res = autogradF.vhp(bad_reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

    def test_vhp_create_graph(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        v = torch.ones(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.vhp(foo, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.vhp(foo, inp, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.vhp(foo, inp, v, create_graph=True), (inputs, v))

        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (torch.rand(3, dtype=torch.double, requires_grad=True),
                  torch.rand(4, dtype=torch.double, requires_grad=True))
        v = (torch.ones(3, dtype=torch.double, requires_grad=True),
             torch.ones(4, dtype=torch.double, requires_grad=True))
        out, vhp_val = autogradF.vhp(bar, inputs, v, create_graph=True)
        self._assert_same_struct(vhp_val, inputs)
        self.assertIsNotNone(out.grad_fn)
        self.assertIsNotNone(vhp_val[0].grad_fn)
        self.assertIsNotNone(vhp_val[1].grad_fn)

        gradcheck(lambda *args: autogradF.vhp(bar, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.vhp(bar, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.vhp(bar, (x, y), v, create_graph=True)

            return val.cos() + grad[0].cos().sum() + grad[1].cos() + x.cos().sum() + y.cos()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def test_hvp_err_check(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        def bar(a):
            return 3 * a.narrow(0, 0, 3), "bar"

        def bar2(a):
            return 3 * a.narrow(0, 0, 3)

        inp = torch.rand(4)
        v = torch.rand(4)
        res = autogradF.hvp(foo, inp, v)
        with self.assertRaisesRegex(TypeError, "The inputs given to hvp must be either a Tensor"):
            res = autogradF.hvp(foo, (inp, 2), v)

        with self.assertRaisesRegex(TypeError, "The outputs of the user-provided function given to hvp must"):
            res = autogradF.hvp(bar, inp, v)

        err_msg_out = "The Tensor returned by the function given to hvp should contain a single element"
        with self.assertRaisesRegex(RuntimeError, err_msg_out):
            res = autogradF.hvp(bar2, inp, v)

        with self.assertRaisesRegex(RuntimeError, "v has invalid size:"):
            res = autogradF.hvp(foo, inp, torch.rand(5))

        with self.assertRaisesRegex(TypeError, "The v given to hvp must be either a Tensor or a tuple of Tensors"):
            res = autogradF.hvp(foo, inp, (v, 2))

        res = autogradF.hvp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

        def foo(a, b):
            return (3 * b.narrow(0, 0, 3) * a.narrow(0, 0, 3)).sum()

        inp = (torch.rand(4), torch.rand(5))
        v = (torch.rand(4), torch.rand(5))

        res = autogradF.hvp(foo, inp, v)
        self._assert_same_struct(res[1], inp)

    def test_hvp_err_check_strict(self):
        def foo(a):
            return a.detach().sum()

        def bar(a):
            # Make a non-leaf Tensor that requires_grad but that is not connected to the input
            return a.long().float().requires_grad_().clone().sum()

        def bar2(a):
            # A Linear function for which the jacobian is independent of the input
            return (3 * a).sum()

        inp = torch.rand(4)
        v = torch.rand(4)
        with self.assertRaisesRegex(RuntimeError, "Output 0 of the user-provided function does not require gradients."):
            res = autogradF.hvp(foo, inp, v, strict=True)
        res = autogradF.hvp(foo, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "The output of the user-provided function is independent of input 0"):
            res = autogradF.hvp(bar, inp, v, strict=True)
        res = autogradF.hvp(bar, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

        with self.assertRaisesRegex(RuntimeError, "jacobian of the user-provided function with respect to input 0 is"):
            res = autogradF.hvp(bar2, inp, v, strict=True)
        res = autogradF.hvp(bar2, inp, v, strict=False)
        self._assert_same_struct(res[1], inp)
        self.assertEqual(res[1].abs().sum(), 0.)

    def test_hvp_no_grad(self):
        def reducer(x):
            return x.exp().sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        with torch.no_grad():
            res = autogradF.hvp(reducer, inputs, v)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

        with torch.no_grad():
            res = autogradF.hvp(reducer, inputs, v, create_graph=True)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)
        self.assertNotEqual(res[1], torch.zeros(4, 4))

    def test_hvp_output(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.hvp(foo, inputs, v)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNone(res[0].grad_fn)
        self.assertIsNone(res[1].grad_fn)

        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (torch.rand(3), torch.rand(4))
        v = (torch.ones(3), torch.ones(4))
        out, hvp_val = autogradF.hvp(bar, inputs, v)
        self._assert_same_struct(hvp_val, inputs)
        self.assertIsNone(out.grad_fn)
        self.assertIsNone(hvp_val[0].grad_fn)
        self.assertIsNone(hvp_val[1].grad_fn)

    def test_hvp_scalar(self):
        def reducer(x):
            return x.exp().sum()
        inputs = torch.rand(4, 4)
        v = torch.ones(4, 4)
        res = autogradF.hvp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        inputs = torch.rand([])
        v = torch.rand([])
        res = autogradF.hvp(reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

        res = autogradF.hvp(reducer, inputs)
        self._assert_same_struct(res[1], inputs)

        def bad_reducer(x):
            return x.exp().sum().view(1, 1, 1)
        inputs = torch.rand(4, 4)
        v = torch.rand(4, 4)
        res = autogradF.hvp(bad_reducer, inputs, v)
        self._assert_same_struct(res[1], inputs)

    def test_hvp_create_graph(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4, 4, dtype=torch.double, requires_grad=True)
        v = torch.ones(4, 4, dtype=torch.double, requires_grad=True)
        res = autogradF.hvp(foo, inputs, v, create_graph=True)
        self._assert_same_struct(res[1], inputs)
        self.assertIsNotNone(res[0].grad_fn)
        self.assertIsNotNone(res[1].grad_fn)

        gradcheck(lambda inp, v: autogradF.hvp(foo, inp, v, create_graph=True), (inputs, v))
        gradgradcheck(lambda inp, v: autogradF.hvp(foo, inp, v, create_graph=True), (inputs, v))

        def bar(a, b):
            return (a + 3 * b.narrow(0, 0, 3)).exp().sum()

        inputs = (torch.rand(3, dtype=torch.double, requires_grad=True),
                  torch.rand(4, dtype=torch.double, requires_grad=True))
        v = (torch.ones(3, dtype=torch.double, requires_grad=True),
             torch.ones(4, dtype=torch.double, requires_grad=True))
        out, hvp_val = autogradF.hvp(bar, inputs, v, create_graph=True)
        self._assert_same_struct(hvp_val, inputs)
        self.assertIsNotNone(out.grad_fn)
        self.assertIsNotNone(hvp_val[0].grad_fn)
        self.assertIsNotNone(hvp_val[1].grad_fn)

        gradcheck(lambda *args: autogradF.hvp(bar, args[:2], args[2:], create_graph=True)[1], inputs + v)
        gradgradcheck(lambda *args: autogradF.hvp(bar, args[:2], args[2:], create_graph=True)[1], inputs + v)

        def foo(*args):
            x, y = args[:2]
            v = args[2:]

            x = x.cos()
            val, grad = autogradF.hvp(bar, (x, y), v, create_graph=True)

            return val.cos() + grad[0].cos().sum() + grad[1].cos() + x.cos().sum() + y.cos()

        gradcheck(foo, inputs + v)
        gradgradcheck(foo, inputs + v)

    def test_jacobian_match_vjp_jvp(self):
        def foo(x):
            return x ** 3 + x.sum()

        inputs = torch.rand(4)
        v = torch.rand(4)

        jac = autogradF.jacobian(foo, inputs)
        jvp = autogradF.jvp(foo, inputs, v)[1]
        vjp = autogradF.vjp(foo, inputs, v)[1]

        self.assertEqual(jvp, torch.mm(jac, v.unsqueeze(1)).squeeze(1))
        self.assertEqual(vjp, torch.mm(v.unsqueeze(0), jac).squeeze(0))

    def test_hessian_match_vhp_hvp(self):
        def foo(a):
            return 3 * a.narrow(0, 0, 3).exp().sum()

        inputs = torch.rand(4)
        v = torch.rand(4)

        hes = autogradF.hessian(foo, inputs)
        hvp = autogradF.hvp(foo, inputs, v)[1]
        vhp = autogradF.vhp(foo, inputs, v)[1]

        self.assertEqual(hvp, torch.mm(hes, v.unsqueeze(1)).squeeze(1))
        self.assertEqual(vhp, torch.mm(v.unsqueeze(0), hes).squeeze(0))

class TestAutogradForwardMode(TestCase):
    def tearDown(self):
        # Ensure that a failing test won't make others fail
        while fwAD._current_level >= 0:
            fwAD.exit_dual_level()

        super().tearDown()

    def test_forward_level_cleanup(self):
        def get_tensor_and_weak_ref():
            # Create a new Tensor and weak reference
            t = torch.rand(2, requires_grad=True)
            return t, torch._C._WeakTensorRef(t)

        # Sanity check that the helper function works as expected
        t, t_ref = get_tensor_and_weak_ref()
        self.assertFalse(t_ref.expired())

        del t
        self.assertTrue(t_ref.expired())

        # Main test code
        foo = torch.rand(2)

        with fwAD.dual_level():
            tangent, tangent_ref = get_tensor_and_weak_ref()
            self.assertFalse(tangent_ref.expired())

            dual = fwAD.make_dual(foo, tangent)
            self.assertFalse(tangent_ref.expired())

            # Make sure that the tangent we provided has been re-used as is
            self.assertTrue(fwAD.unpack_dual(dual)[1] is tangent)

            # Make sure that dual is keeping the tangent alive
            del tangent
            self.assertFalse(tangent_ref.expired())

            # Make sure that the dual level does not keep the c++
            # version of the tangent alive
            del dual
            self.assertTrue(tangent_ref.expired())

    def test_size_check(self):
        foo = torch.rand(2)
        tangent = torch.rand(3)

        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, "Trying to set a forward gradient that has a different size"):
                dual = fwAD.make_dual(foo, tangent)

            dual = fwAD.make_dual(foo, tangent[1:])

    # The following test functions want to ensure all the following behaviors:
    #   - Ensure that default level system in the python binding works
    #   - Ensure that only level 0 exists and nesting is properly disabled
    #   - Ensure that printing works fine
    #   - Ensure that basic packing/unpacking works
    #   - Ensure that advanced packing/unpacking works
    #     - For memory / version counter share
    #     - For backward AD (regular ops)
    #   - Ensure that view + inplace for both modes work fine
    #   - Ensure we do proper cleanup on exit of a level

    def test_default_level(self):
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        # We don't actually need to enforce that these two are the exact same python
        # object, feel free to relax in the future
        self.assertIs(baz_tangent, bar)

        baz_primal, baz_tangent = fwAD.unpack_dual(baz)
        self.assertEqual(baz_primal, foo)
        self.assertEqual(baz_tangent, None)

    def test_nested_level(self):
        with fwAD.dual_level() as level:
            # For now only level 0 exists
            self.assertEqual(level, 0)

        with fwAD.dual_level():
            with self.assertRaisesRegex(RuntimeError, "Nested forward mode AD is not supported at the moment"):
                nest_level = fwAD.enter_dual_level()

    def test_print(self):
        with fwAD.dual_level() as level:
            a = torch.rand(3)
            self.assertFalse("tangent=" in str(a))

            b = fwAD.make_dual(a, torch.rand(3))
            self.assertFalse("tangent=" in str(a))
            self.assertTrue("tangent=" in str(b))

            b_primal, b_tangent = fwAD.unpack_dual(b)
            self.assertFalse("tangent=" in str(b_primal))
            self.assertFalse("tangent=" in str(b_tangent))

    def test_basic_packing_unpacking(self):
        foo = torch.rand(2)
        bar = torch.rand(2)

        with fwAD.dual_level():
            baz = fwAD.make_dual(foo, bar)
            baz_primal, baz_tangent = fwAD.unpack_dual(baz)
            self.assertEqual(baz_primal, foo)
            self.assertIs(baz_tangent, bar)

            # Check that packing/unpacking did not change the input
            foo_primal, foo_tangent = fwAD.unpack_dual(foo)
            self.assertEqual(foo_primal, foo)
            self.assertIsNone(foo_tangent)

    def test_advanced_packing_unpacking(self):
        foo = torch.rand(2)
        bar = torch.ones(2)

        # Memory and version counter check
        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)

            # Ensure that they are sharing memory and version counter
            self.assertEqual(dual.storage().data_ptr(), foo.storage().data_ptr())

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual._version)

            # Unpacking should only create aliases as well
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertEqual(dual_primal.storage().data_ptr(), foo.storage().data_ptr())
            self.assertEqual(dual_tangent.storage().data_ptr(), bar.storage().data_ptr())
            # And the tangent is actually re-used as-is so it is still the same Tensor
            self.assertIs(dual_tangent, bar)

            # Ensure we properly share the version counter
            self.assertEqual(foo._version, dual_primal._version)
            foo.add_(1)
            self.assertEqual(foo._version, dual_primal._version)
            self.assertEqual(bar._version, dual_tangent._version)
            bar.add_(1)
            self.assertEqual(bar._version, dual_tangent._version)

        # backward mode check
        with fwAD.dual_level():
            foo.requires_grad_()
            bar.requires_grad_()

            # Check that backward gradients properly propagates through packing/unpacking
            dual = fwAD.make_dual(foo, bar)
            p, t = fwAD.unpack_dual(dual)

            gfoo, gbar = torch.autograd.grad(p.sum(), (foo, bar), retain_graph=True, allow_unused=True)
            self.assertEqual(gfoo, torch.ones_like(foo))
            self.assertIsNone(gbar)

            gfoo, gbar = torch.autograd.grad(t.sum(), (foo, bar), retain_graph=True, allow_unused=True)
            self.assertIsNone(gfoo)
            self.assertEqual(gbar, torch.ones_like(bar))

            # Check that forward gradients are not impacted by detach
            detached_dual = dual.detach()
            out = detached_dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertFalse(t.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertEqual(t, bar * 2)

            # Check that forward gradients are not impacted by no_grad
            with torch.no_grad():
                out = dual * 3
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertFalse(t.requires_grad)
            self.assertEqual(p, foo * 3)
            self.assertEqual(t, bar * 3)

            # Check that forward gradients are not impacted by inplace detach
            dual = dual.clone()
            dual.detach_()
            out = dual * 2
            p, t = fwAD.unpack_dual(out)
            self.assertFalse(p.requires_grad)
            self.assertFalse(t.requires_grad)
            self.assertEqual(p, foo * 2)
            self.assertEqual(t, bar * 2)

    def test_view_inplace_non_differentiable_views(self):
        original_foo = torch.rand(2, dtype=torch.double)
        original_bar = torch.ones(2, dtype=torch.double)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Note that in this test, we use "update" to mean computing the right tangent for the dual
            # All the inplace operations here are expected to update the primal value of the Tensors but
            # not always their tangents.
            # Also all mentions of "non differentiable view" here means non forward differentiable view
            # unless specified otherwise.
            # See note [Forward Grad View/inplace] for more details on how these views work.

            # Check that inplace ops do not update non-differentiable views
            # Non differentiable view
            dual = fwAD.make_dual(foo, bar)
            dual *= 2
            # Check that non differentiable view's tangent was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that the computed result is correct
            self.assertEqual(bar, original_bar * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            self.assertEqual(foo, original_foo * 2)
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 2)
            # Other non differentiable view
            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            self.assertIsNone(fwAD.unpack_dual(dual_primal)[1])
            self.assertIsNone(fwAD.unpack_dual(dual_tangent)[1])
            dual_primal *= 2
            # Ensure dual's tangent did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 2)
            dual_tangent *= 2
            # Ensure dual's primal did not change
            self.assertEqual(fwAD.unpack_dual(dual)[0], original_foo * 4)
            self.assertEqual(fwAD.unpack_dual(dual)[1], original_bar * 4)


    def test_view_inplace_differentiable_views(self):
        original_foo = torch.rand(2)
        original_bar = torch.ones(2)

        # Do clones to be able to compare the values updated inplace
        # with the original content of these Tensors
        foo = original_foo.clone()
        bar = original_bar.clone()

        with fwAD.dual_level():
            # Check that inplace ops do update differentiable view but stop at non differentiable ones
            # A non differentiable view
            dual = fwAD.make_dual(foo, bar)
            # A differentiable view
            view = dual.narrow(0, 0, 1)
            view *= 2
            # Check that non differentiable view was not updated
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            # Check that differentiable view was updated
            self.assertEqual(fwAD.unpack_dual(dual)[1], torch.tensor([2., 1.]))
            self.assertEqual(fwAD.unpack_dual(view)[1], torch.tensor([2.]))

            # Check that we track differentiable view even for Tensors that are not dual
            baz = torch.rand(2)
            baz += dual
            self.assertEqual(fwAD.unpack_dual(baz)[1], fwAD.unpack_dual(dual)[1])
            # Updates on view should as well
            baz = torch.rand(2)
            baz[0] = dual[0]
            self.assertEqual(fwAD.unpack_dual(baz)[1][0], fwAD.unpack_dual(dual)[1][0])
            # Unused values get a gradient of 0
            self.assertEqual(fwAD.unpack_dual(baz)[1][1], 0.)

            # Check that backward non-differentiable views don't prevent gradient update
            baz = torch.rand(2)
            view = baz.detach()
            view += dual
            self.assertEqual(fwAD.unpack_dual(baz)[1], fwAD.unpack_dual(dual)[1])

    def test_grad_cleanup(self):
        foo = torch.rand(2)
        bar = torch.rand(2)
        baz = torch.rand(2)

        with fwAD.dual_level():
            dual = fwAD.make_dual(foo, bar)
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            self.assertIs(fwAD.unpack_dual(dual)[1], bar)

        self.assertIsNone(fwAD.unpack_dual(dual)[1])

        with fwAD.dual_level():
            self.assertIsNone(fwAD.unpack_dual(foo)[1])
            new_dual = fwAD.make_dual(foo, baz)

            dual_primal, dual_tangent = fwAD.unpack_dual(dual)
            new_dual_primal, new_dual_tangent = fwAD.unpack_dual(new_dual)
            self.assertEqual(dual_primal, new_dual_primal)
            self.assertIsNone(dual_tangent)
            self.assertEqual(new_dual_tangent, baz)


# Generic device type autograd tests.
class TestAutogradDeviceType(TestCase):

    def test_min_max_median_backprops_to_all_values(self, device):
        for f in [torch.min, torch.max, torch.median, torch.nanmedian]:
            x1 = torch.tensor([1., 0., 1., 0., 1., 0.], device=device, requires_grad=True)
            x2 = torch.tensor([float('nan'), float('nan'), float('nan')], requires_grad=True)
            for x in [x1, x2]:
                y = f(x)
                y.backward()
                self.assertEqual(x.grad.sum(), 1.)
                self.assertEqual((x.grad == 1 / 3).sum(), 3)

    def test_cdist(self, device):
        def _test_euclidean_large_cdist(sizex, sizey=None):
            if sizey is None:
                sizey = sizex
            x = torch.randn(sizex, device=device, dtype=torch.float)
            y = torch.randn(sizey, device=device, dtype=torch.float)
            eps = 1e-6
            # to avoid extremum
            x = x - (((x - y) < eps).float() * 2 * eps)
            x.requires_grad = True
            y.requires_grad = True
            dist = torch.cdist(x, y, p=2)
            # Do a backward pass to check that it is valid for large
            # matrices
            loss = dist.sum()
            loss.backward()

        _test_euclidean_large_cdist((2000, 5))

    # Ensure that cdist backward with p<1 does not produce NaNs
    def test_cdist_grad_p_lt_1_no_nan(self, device):
        for p in [0.99, 0.7, 0.5, 0.1, 0.01]:
            x = torch.randn(1, 2, device=device)
            y = x.clone().detach() + torch.tensor([[1., 0.]], device=device)
            x.requires_grad = True
            y.requires_grad = True
            result = torch.cdist(x, y, p=p)
            result.backward(torch.ones_like(result))
            self.assertFalse(torch.isnan(x.grad).any())
            self.assertFalse(torch.isnan(y.grad).any())

    def test_cdist_same_inputs(self, device):
        # Test to detect issues in cdist gradient calculation
        # When the distances are 0
        sizex = (1, 27, 32)
        for p in [0, 1, 2, 3, 1.5, 2.5, float('inf')]:
            x = torch.randn(sizex, device=device, dtype=torch.float)
            dist_grad = torch.randn((1, 27, 27), device=device, dtype=torch.float)
            y = x.clone()
            eps = 1e-6
            x.requires_grad = True
            d = torch.cdist(x, y)
            d.backward(dist_grad)
            # Check that the backward passs does not contain invalid
            # values such as nan or inf
            assert torch.isfinite(x.grad).all()

    def test_parameter_resize(self, device):
        asd = torch.nn.Parameter(torch.ones(16, dtype=torch.double, device=device))

        for i in range(2):
            with torch.no_grad():
                asd.set_(asd[1:])
                asd.grad = None

            m = torch.cat((asd, asd))
            m.sum().backward()

    def test_sparse_ctor_getter_backward(self, device):
        # See NOTE [ Sparse: autograd and API ] on the expected behavior of this test
        def _test(size, sparse_dim, nnz, device):
            v_size = [nnz] + list(size[sparse_dim:])
            i = torch.rand(sparse_dim, nnz)
            i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
            i = i.to(torch.long)

            inp = torch.randn(v_size, dtype=torch.double, requires_grad=True)
            other = self.genSparseTensor(size, sparse_dim, nnz, is_uncoalesced=True, device=device,
                                         dtype=torch.double)[0]

            def fn(v):
                x = torch.sparse_coo_tensor(i, v, size, device=device)
                y = (x + other).coalesce()
                yv = y.values()
                new_v = yv.tanh()
                z = torch.sparse_coo_tensor(y.indices(), new_v, y.size())
                return z.coalesce().values()

            gradcheck(fn, (inp,), check_batched_grad=False)
            # FIXME: make gradgradcheck work.
            # gradgradcheck(fn, (inp,), check_batched_grad=False)

            # assert that _values is non-differentiable
            with self.assertRaisesRegex(RuntimeError, "does not have a grad_fn"):
                other.detach().requires_grad_()._values().backward(torch.ones_like(other._values()))

        for empty_i, empty_v, empty_nnz in product([True, False], repeat=3):
            sparse_size = [] if empty_i else [2, 1]
            dense_size = [1, 0, 2] if empty_v else [1, 2]
            nnz = 0 if empty_nnz else 5
            _test(sparse_size + dense_size, len(sparse_size), nnz, device)

    # autograd tests via common_method_invocations don't allow input tensors to
    # be sparse (RuntimeError: gradcheck expects all tensor inputs are dense when
    # check_sparse_nnz is set to False.)
    def test_sparse_mask_autograd(self, device):
        tensor = torch.randn(3, requires_grad=True, device=device)
        mask = torch.ones(3, device=device)
        mask[1] = 0
        mask = mask.to_sparse()
        converted = tensor.sparse_mask(mask).to_dense()
        converted.sum().backward()
        self.assertEqual(tensor.grad, mask.to_dense())

    def test_pyscalar_conversions(self, device):
        def _test_pyscalar_conversions(t, integral_conv):
            # integral -> integral
            l = t(torch.zeros(1, 1, 1, dtype=torch.long))
            pyscalar = -12345
            l[0] = pyscalar
            self.assertEqual(integral_conv(l), pyscalar)

            # floating point -> floating point
            f = Variable(t(torch.randn(1, 1, dtype=torch.double)))
            pyscalar = -12345.1
            f[0] = pyscalar
            self.assertEqual(float(f), pyscalar)
            f[0] = nan
            self.assertTrue(math.isnan(float(f)))
            f[0] = inf
            self.assertEqual(float(f), inf)
            f[0] = -inf
            self.assertEqual(float(f), -inf)

            # integral -> floating point
            # check we can convert something that loses precision
            pyscalar = 1234567890123456789
            self.assertNotEqual(pyscalar, integral_conv(float(pyscalar)))
            l[0] = pyscalar
            self.assertEqual(float(l), float(pyscalar))

            # floating point -> integral
            f[0] = nan
            self.assertRaises(ValueError, lambda: integral_conv(f[0]))
            f[0] = inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = -inf
            self.assertRaises(OverflowError, lambda: integral_conv(f[0]))
            f[0] = sys.float_info.max
            self.assertEqual(integral_conv(f), sys.float_info.max)

            # bool, nonzero
            def test_nonzero(tensor, value, expected):
                tensor[0] = value
                self.assertEqual(expected, bool(tensor))
                self.assertEqual(expected, True if tensor else False)

            test_nonzero(l, 0, False)
            test_nonzero(l, -2, True)
            test_nonzero(f, 0.0, False)
            test_nonzero(f, sys.float_info.min, True)
            test_nonzero(f, nan, bool(nan))
            test_nonzero(f, inf, bool(inf))
            test_nonzero(f, -inf, bool(-inf))


        _test_pyscalar_conversions(lambda x: x.to(device), lambda x: int(x))

    @dtypesIfCUDA(torch.half, torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    @dtypes(torch.float, torch.double, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_set_requires_grad_only_for_floats(self, device, dtype):
        def f1():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad_()

        def f2():
            a = torch.ones(1, dtype=dtype, device=device)
            a.requires_grad = True

        def f3():
            torch.ones(1, dtype=dtype, device=device, requires_grad=True)

        a = torch.ones(1, dtype=dtype, device=device)
        a.requires_grad = False  # should always work
        a.requires_grad_(False)

        for f in [f1, f2, f3]:
            if dtype.is_floating_point:
                f()
            else:
                with self.assertRaisesRegex(RuntimeError, 'floating point', msg="dt: {} device: {}".format(a.dtype, a.device)):
                    f()

    @onlyCUDA
    def test_advanced_indexing_backwards_large(self, device):
        # See https://github.com/pytorch/pytorch/issues/22843
        n = (1 << 16)
        x = torch.rand(n, 1, device=device, requires_grad=True)
        a = x[:, [0]]
        a.sum().backward()
        self.assertEqual(x.grad, torch.ones(n, 1, device=device))

    def test_advanced_indexing_backwards_memory_format(self, device):
        # See https://github.com/pytorch/pytorch/issues/36956
        shape = (2, 8, 1, 2)
        i = torch.randint(1, shape, device=device).contiguous(memory_format=torch.channels_last)
        x = torch.randn(shape, requires_grad=True, device=device)
        x[i].sum().backward()

    def _test_reentrant_parent_error_on_cpu(self, device):
        t1 = torch.rand([3, 3], requires_grad=True)
        t2 = torch.rand([3, 3], device=device, requires_grad=True)
        t3 = torch.rand([3, 3], device=device, requires_grad=True)

        # Parent graph cpu graph.
        t4 = t1 * t1
        t5 = TestAutograd.SimulateBackwardError.apply(t4)

        # Child gpu graph (much longer than parent graph).
        prev = t2 * t2
        for i in range(10):
            prev = prev * t2
        reentrant_root = prev

        class ReentrantFunc(Function):
            @staticmethod
            def forward(ctx, inp):
                return inp.clone()

            @staticmethod
            def backward(ctx, grad):
                # Reentrant backward in child will take much longer.
                reentrant_root.backward()
                return grad

        # Parent gpu graph.
        t6 = ReentrantFunc.apply(t3)
        t7 = t6 * t6

        # Parent graph will error out first, while child graph will continue executing.
        with self.assertRaisesRegex(Exception, "Simulate error"):
            torch.autograd.backward([t5.sum(), t7.sum()])

        # No grads should be accumulated since child graph will stop execution
        # after parent receives error.
        self.assertIsNone(t2.grad)
        self.assertIsNone(t1.grad)
        self.assertIsNone(t3.grad)

    @onlyCUDA
    def test_reentrant_parent_error_on_cpu(self, device):
        before = CudaMemoryLeakCheck.get_cuda_memory_usage()

        # Run as separate function so that gc can clean up everything when we
        # check for memory usage.
        self._test_reentrant_parent_error_on_cpu(device)

        # Wait for autograd thread to cleanup failed tasks.
        after = CudaMemoryLeakCheck.get_cuda_memory_usage()
        start = time.time()
        while before != after and time.time() - start < 30:
            time.sleep(0.1)
            after = CudaMemoryLeakCheck.get_cuda_memory_usage()

        self.assertEqual(before, after)

    # test for backward in https://github.com/pytorch/pytorch/issues/15511
    def test_pdist_large(self, device):
        def func(x):
            return torch.pdist(x, p=2)

        # shape[0] should be able to be (roughly) arbitrarily large, but the kernel
        # is currently limited to smaller sizes (see issue above); this is just testing
        # a floor.
        shape = (1000, 1)
        x = torch.randn(shape, device=device).requires_grad_()
        output = torch.pdist(x, p=2)
        # just run a single backward, as gradcheck/gradgradcheck is expensive here
        output.sum().backward()

    def test_where_functional(self, device):
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where(cond, x, y):
            return torch.where(cond, x, y)

        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, device=device)])

        x = torch.randn(5, 1, 5, dtype=torch.double, device=device, requires_grad=True)
        y = torch.randn(5, 5, 1, dtype=torch.double, device=device, requires_grad=True)
        gradcheck(where, [cond, x, y], raise_exception=True)
        gradgradcheck(where, [cond, x, y], [torch.randn(5, 5, 5, device=device)])

    def test_where_scalar(self, device):
        x = torch.randn(5, 5, dtype=torch.double, device=device, requires_grad=True)
        scalar = 4.
        cond = mask_not_all_zeros((5, 5)).to(device=device)

        def where_scalar_first(cond, x):
            return torch.where(cond, scalar, x)

        def where_scalar_second(cond, x):
            return torch.where(cond, x, scalar)

        gradcheck(where_scalar_first, (cond, x))
        gradgradcheck(where_scalar_first, (cond, x))

        gradcheck(where_scalar_second, (cond, x))
        gradgradcheck(where_scalar_second, (cond, x))

    @skipCUDAIf(True, """Test is flaky on Linux and Windows, typical error message:
            https://github.com/pytorch/pytorch/issues/34870""")
    def test_ctc_loss(self, device):
        batch_size = 64
        num_labels = 101
        target_length = 15
        gradcheck_input_size = 10

        ZERO_NONE = 0
        ZERO_SOME = 1
        ZERO_ALL = 2

        # input_length, vary_lengths, zero_lengths
        tests = [(150, False, ZERO_NONE),
                 (150, True, ZERO_NONE),
                 (50, True, ZERO_SOME),
                 (50, True, ZERO_ALL)]

        if 'cuda' in device:
            tests += [(50, False, ZERO_NONE),
                      (50, True, ZERO_NONE),
                      (150, True, ZERO_SOME),
                      (150, True, ZERO_ALL)]

        for input_length, vary_lengths, zero_mode in tests:
            targets = torch.randint(1, num_labels, (batch_size, target_length),
                                    device=device, dtype=torch.long)
            x = torch.randn(gradcheck_input_size, dtype=torch.double, device=device, requires_grad=True)
            tile_factors = torch.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                       device=device)
            input_lengths = [(torch.randint(input_length // 2, input_length + 1, ()).item()
                              if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
            if zero_mode == ZERO_ALL:
                target_lengths = [0 for _ in range(batch_size)]
            else:
                target_lengths = [(torch.randint(target_length // 2, target_length + 1, ()).item()
                                   if vary_lengths else target_length) for _ in range(batch_size)]
                if zero_mode == ZERO_SOME:
                    idxes = torch.randint(0, batch_size, (10,))
                    for i in idxes:
                        target_lengths[i] = 0

            def ctc_after_softmax(x):
                x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                          .view(input_length, batch_size, num_labels))
                log_probs = torch.log_softmax(x_full, 2)
                return torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths)

            gradcheck(ctc_after_softmax, [x])

    @onlyCUDA
    @skipCUDAIfRocm
    @skipCUDAIfCudnnVersionLessThan(7600)
    def test_ctc_loss_cudnn(self, device):
        batch_size = 16
        input_length = 30
        num_labels = 101
        target_length = 15
        targets = torch.randint(1, num_labels, (batch_size * target_length,),
                                device='cuda', dtype=torch.long)
        log_probs = torch.log_softmax(torch.randn(input_length, batch_size, num_labels, device='cuda', dtype=torch.float), 2)
        log_probs.requires_grad_()

        input_lengths = batch_size * [input_length]
        target_lengths = batch_size * [target_length]
        grad_out = torch.randn(batch_size, device='cuda', dtype=torch.float)
        with torch.backends.cudnn.flags(enabled=False):
            loss_native = torch.nn.functional.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
            grad_native, = torch.autograd.grad(loss_native, log_probs, grad_out)
        loss_cudnn = torch.nn.functional.ctc_loss(log_probs, targets.to('cpu', torch.int32),
                                                  input_lengths, target_lengths, reduction='none')
        self.assertTrue("Cudnn" in str(loss_cudnn.grad_fn))
        grad_cudnn, = torch.autograd.grad(loss_cudnn, log_probs, grad_out)
        self.assertEqual(grad_cudnn, grad_native, atol=1e-4, rtol=0)

    def test_leaky_relu_inplace_with_neg_slope(self, device):
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.leaky_relu_(a.clone(), -2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.rrelu_(a.clone(), -5.0, 1.0)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    def test_leaky_relu_inplace_with_zero_slope(self, device):
        a = torch.tensor([-2., 0., 2.], device=device, requires_grad=True)
        b = torch.nn.functional.leaky_relu_(a.clone(), 0.0)
        b.backward(torch.ones(3, device=device))
        expected = torch.tensor([0., 0., 1.], device=device)
        self.assertEqual(a.grad, expected)

    @onlyOnCPUAndCUDA
    def test_elu_inplace_with_neg_alpha(self, device):
        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.elu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

        a = torch.tensor([-1., 1.], device=device, requires_grad=True)
        b = torch.nn.functional.celu_(a.clone(), alpha=-2)
        with self.assertRaisesRegex(RuntimeError, "call out-of-place version"):
            b.backward(torch.ones(2, device=device))

    @onlyCUDA
    def test_free_unneeded_tensor(self, device):
        x = torch.randn(2, 3, 10, 10, device=device, requires_grad=True)
        m = torch.randn(1, 3, 1, 1, device=device)

        z = x.sum()
        base_mem = torch.cuda.memory_allocated()
        z = ((x + 2) * m).sum()
        end_mem = torch.cuda.memory_allocated()

        # In the end the memory usage should remain equal, because neither of
        # (x + 2) and ((x + 2) * m) should be kept alive for backward, while the
        # previous allocation of z had the same size as the current one.
        self.assertEqual(base_mem, end_mem)

    @onlyCUDA
    def test_pin_memory(self, device):
        x = torch.randn(2, 2, dtype=torch.double, requires_grad=True)
        self.assertEqual(x, x.pin_memory())
        self.assertIsNot(x, x.pin_memory())
        self.assertTrue(x.pin_memory().requires_grad)
        gradcheck(lambda x: x.pin_memory(), [x])
        gradgradcheck(lambda x: x.pin_memory(), [x])

    @skipCUDAIfRocm
    @onlyCUDA
    def test_profiler_emit_nvtx(self, device):
        # This test is not intended to ensure correctness of nvtx ranges.
        # That would require something a great deal more complex (you'd have to create a
        # profile in a subprocess, open it, and parse the sql somehow).
        # This test is merely intended to catch if emit_nvtx breaks on construction.
        a = torch.tensor([1, 2, 3], dtype=torch.float32, device=device)
        with torch.cuda.profiler.profile():
            with emit_nvtx():
                a.add(1.0)

    @onlyCUDA
    def test_rnn_backward_to_input_but_not_parameters(self, device):
        # this checks whether it is possible to not require
        # weight parameters, but require inputs, see #7722
        l = torch.nn.LSTM(2, 3).to(device)
        for p in l.parameters():
            p.requires_grad = False
        s = torch.randn(1, 1, 2, requires_grad=True, device=device)
        out, _ = l(s)
        out.sum().backward()
        self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    @onlyCUDA
    def test_lstmcell_backward_only_one_output_grad(self, device):
        # checks that undefined gradients doen't hamper the backward
        # see #11872
        l = torch.nn.LSTMCell(2, 3).to(device).double()
        s = torch.randn(1, 2, device=device, dtype=torch.double, requires_grad=True)
        for i in range(2):
            out = l(s)[i]
            out.sum().backward()
            self.assertFalse(s.grad is None or s.grad.abs().sum().item() == 0)

    def _test_rnn_mod(self, mod, inp):
        def flatten_out(mod, inp):
            out = mod(inp)
            return tuple([t if isinstance(t, torch.Tensor) else tt for t in out for tt in t])
        gradcheckfunc = partial(flatten_out, mod)
        with torch.backends.cudnn.flags(enabled=False):
            gradcheck(gradcheckfunc, inp, check_batched_grad=False)
            gradgradcheck(gradcheckfunc, inp, check_batched_grad=False)

        if inp.is_cuda and not TEST_WITH_ROCM:
            # Assert that we have good error message around unsupported CuDNN double backward
            # NB: we trigger double backward using .backward() instead of autograd.grad due to
            # https://github.com/pytorch/pytorch/issues/37874
            with torch.backends.cudnn.flags(enabled=True):
                result = gradcheckfunc(inp)
                result[0].sum().backward(create_graph=True)
                grad0 = next(mod.parameters()).grad
                with self.assertRaisesRegex(RuntimeError,
                                            "please disable the CuDNN backend temporarily"):
                    grad0.sum().backward()

                # Here we avoid the backward(create_graph=True) memory leak
                # described in https://github.com/pytorch/pytorch/issues/7343
                for param in mod.parameters():
                    param.grad = None
                inp.grad = None

    def test_LSTM_grad_and_gradgrad(self, device):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=torch.float64, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.LSTM(hsize, hsize, bias=bias).to(device).to(torch.float64)
            self._test_rnn_mod(mod, inp)

    def test_GRU_grad_and_gradgrad(self, device):
        hsize = 4
        inp = torch.rand(1, 3, hsize, device=device, dtype=torch.float64, requires_grad=True)
        for bias in [True, False]:
            mod = torch.nn.GRU(hsize, hsize, bias=bias).to(device).to(torch.float64)
            self._test_rnn_mod(mod, inp)

    def test_copysign_subgradient(self, device):
        # Input is 0.0
        x = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Input is -0.0
        x = torch.tensor([-0.0, -0.0, -0.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is 0.0
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [-1.0, 0.0, 1.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

        # Other is -0.0
        x = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float, device=device, requires_grad=True)
        y = torch.tensor([-0.0, -0.0, -0.0], dtype=torch.float, device=device, requires_grad=True)
        out = torch.copysign(x, y)
        out.sum().backward()
        self.assertEqual(x.grad.tolist(), [1.0, 0.0, -1.0])
        self.assertEqual(y.grad.tolist(), [0.0] * 3)

    @deviceCountAtLeast(1)
    def test_grad_assignment(self, devices):
        x = torch.randn(5, 5, device=devices[0])

        # Tests that the wrong shape raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(2, 2, device=devices[0])

        # Tests that the wrong dtype raises
        with self.assertRaises(RuntimeError):
            x.grad = torch.randn(5, 5, dtype=torch.long, device=devices[0])

        # Tests that self-assignment raises
        with self.assertRaises(RuntimeError):
            x.grad = x

        # Tests device -> cpu grad assignment raises
        if self.device_type != 'cpu':
            with self.assertRaises(RuntimeError):
                t_cpu = torch.rand(5, 5)
                t_cpu.grad = torch.randn(5, 5, device=devices[0])

        # Tests half type on CUDA
        if self.device_type == 'cuda':
            x = x.to(dtype=torch.half, device=devices[0])
            x.grad = torch.zeros_like(x)

        # Tests cross-device assignment raises
        if len(devices) > 1:
            x = torch.randn(5, 5, device=devices[0])
            with self.assertRaises(RuntimeError):
                x.grad = torch.randn(5, 5, device=devices[1])

    @deviceCountAtLeast(1)
    @dtypes(torch.float, torch.double)
    def test_requires_grad_factory(self, devices, dtype):
        fns = [torch.ones_like, torch.testing.randn_like]
        x = torch.randn(2, 3, dtype=dtype, device=devices[0])

        for fn in fns:
            for requires_grad in [True, False]:
                output = fn(x, dtype=dtype, device=devices[0], requires_grad=requires_grad)
                self.assertEqual(requires_grad, output.requires_grad)
                self.assertIs(dtype, output.dtype)
                self.assertEqual(devices[0], str(x.device))

    @deviceCountAtLeast(2)
    def test_unused_output_device(self, devices):
        from torch.nn.parallel._functions import Broadcast
        x = torch.randn(5, 5, dtype=torch.float, device=devices[0], requires_grad=True)
        outputs = Broadcast.apply(list(range(len(devices))), x)
        y = outputs[-1] * 2
        y.sum().backward()
        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
        self.assertEqualIgnoreType(x.grad, torch.ones(5, 5) * 2)

    @deviceCountAtLeast(2)
    def test_backward_device(self, devices):
        # check that current device matches the variable's device
        device = [None]

        class Identity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x):
                return x.clone()

            @staticmethod
            def backward(ctx, grad_output):
                device[0] = grad_output.device
                return grad_output.clone()

        v = torch.randn(1, device=devices[1], requires_grad=True)
        Identity.apply(v).backward()
        self.assertEqual(str(device[0]), devices[1])

    @deviceCountAtLeast(2)
    def test_inputbuffer_add_multidevice(self, devices):
        input = torch.randn(1, device=devices[0], requires_grad=True)
        output = input.to(device=devices[1]) + input.to(device=devices[1])
        output.backward()

    @onlyCPU
    def test_copy_(self, device):
        # At the time of writing this test, copy_ is not generated from native_functions.yaml
        # there was a bug that bfloat16 was not recognized as floating.
        x = torch.randn(10, device=device, requires_grad=True)
        floating_dt = [dt for dt in torch.testing.get_all_dtypes() if dt.is_floating_point]
        for dt in floating_dt:
            y = torch.empty(10, device=device, dtype=dt)
            y.copy_(x)
            self.assertTrue(y.requires_grad)
            z = x.to(torch.bfloat16)
            self.assertTrue(z.requires_grad)

    @onlyCUDA
    def test_simple_reentrant_cross_device(self, device):
        class ReentrantFunc(Function):
            _cpu_mode = True

            @staticmethod
            def forward(ctx, x):
                return x * (x + 2)

            @staticmethod
            def backward(ctx, grad_output):
                with torch.enable_grad():
                    if ReentrantFunc._cpu_mode:
                        new_param = torch.randn(2, 2, requires_grad=True)
                        (new_param ** 2).sum().backward()
                    else:
                        new_param = torch.randn(2, 2, device=device, requires_grad=True)
                        (new_param ** 2).sum().backward()
                return grad_output

        # Reentrant starts on GPU thread, finishs on GPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on CPU thread, finishs on GPU thread
        x = torch.randn(2, 2, requires_grad=True)
        # set ReentrantFunc node to GPU to emit tasks to GPU queue
        ReentrantFunc._cpu_mode = False
        out = ReentrantFunc.apply(x)
        out.sum().backward()

        # Reentrant starts on GPU thread, finishs on CPU thread
        x = torch.randn(2, 2, device=device, requires_grad=True)
        # set ReentrantFunc node to CPU to emit tasks to CPU queue
        ReentrantFunc._cpu_mode = True
        out = ReentrantFunc.apply(x)
        out.sum().backward()

    @onlyCUDA
    def test_cross_device_reentrant_autograd(self, device):
        # Output on gpu so that this task will be associated with the gpu thread
        def fn_on_gpu(inp):
            # Artificially increase the priority of the next op to make sure it runs
            # as soon as we reach it before the ops of branch1.
            dummy = inp * 2 * 2 * 2 * 2
            return inp.to(device=device)

        def parent_on_cpu(inp):
            # Slow branch of ops on gpu so that the work queue for the gpu thread
            # won't empty too quickly. They also have smaller priorities than the
            # ones created by fn_on_gpu
            branch1 = inp.to(device=device)
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            branch1 = branch1 / branch1
            # Perform checkpoint on cpu tensors. So the last op performed in the reentrant
            # autograd is an AccumulateGrad that runs on the cpu thread for the gpu thread.
            # So the cpu thread will notify the gpu thread with an empty NodeTask.
            branch2 = checkpoint(fn_on_gpu, inp)
            out = branch2 + branch1
            return out

        inp = torch.rand(2, requires_grad=True)
        out = parent_on_cpu(inp)
        # This will segfault if the empty NodeTask is not handled properly in the
        # gpu thread ReadyQueue
        out.sum().backward()

EXCLUDE_FUNCTIONAL = {
    'addmm',
    'addmm_',
    'reshape',
    'where'  # argument order
}
EXCLUDE_GRADCHECK: Dict[str, Any] = {
}
EXCLUDE_GRADGRADCHECK: Dict[str, Any] = {
}
EXCLUDE_GRADGRADCHECK_BY_TEST_NAME = {
    # `other` expand_as(self, other) is not used in autograd.
    'test_expand_as',
    'test_cdist',
}

    def test_inplace_view_backprop_view_of_view(self, device):
        # modify view and backprop through view-of-view
        root = torch.randn(2, 2, device=device, requires_grad=True)
        x = root.clone()
        v1 = x.narrow(0, 0, 1)
        v2 = x.narrow(0, 0, 1)
        v1.mul_(2)
        v2.sum().backward()
        self.assertEqual(root.grad.tolist(), [[2, 2], [0, 0]])

def exclude_tensor_method(name, test_name):
    # there are no tensor equivalents for these (inplace or out)
    exclude_all_tensor_method_by_test_name = {
        'test_slice',
        'test_where',
        'test_where_broadcast_all',
        'test_where_scalar',
        'test_where_scalar_broadcast_mask',
        'test_where_scalar_broadcast_non_mask',
        'test_var_mean_keepdim_dim_1d',
        'test_var_mean_keepdim_dim',
        'test_var_mean_dim_1d',
        'test_var_mean_dim',
        'test_var_mean',
        'test_std_mean_keepdim_dim_1d',
        'test_std_mean_keepdim_dim',
        'test_std_mean_dim_1d',
        'test_std_mean_dim',
        'test_std_mean',
    }
    # there are no out-of-place tensor equivalents for these
    exclude_outplace_tensor_method = {
        'index_fill',
        'scatter',
        'scatter_add',
    }
    if test_name in exclude_all_tensor_method_by_test_name:
        return True
    is_magic_method = name[:2] == '__' and name[-2:] == '__'
    is_inplace = name[-1] == "_" and not is_magic_method
    if not is_inplace and name in exclude_outplace_tensor_method:
        return True
    return False