# Owner(s): ["module: mta"]

import itertools
from numbers import Number
import random
import re
import torch
import unittest

from torch.testing import make_tensor
from torch.testing._comparison import default_tolerances
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, TEST_WITH_SLOW, skipIfTorchDynamo
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, onlyCUDA, skipMeta, ops)
from torch.testing._internal.common_methods_invocations import (
    foreach_unary_op_db, foreach_binary_op_db, foreach_pointwise_op_db, foreach_minmax_op_db,
    foreach_reduce_op_db)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, all_types_and, integral_types, complex_types,
    floating_types_and, floating_types, integral_types_and,
)

# Includes some values such that N * N won't be a multiple of 4,
# which should ensure we test the vectorized and non-vectorized
# kernel code paths.
N_values = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]
Scalars = (
    random.randint(1, 10),
    1.0 - random.random(),
    True,
    complex(1.0 - random.random(), 1.0 - random.random()),
)


def getScalarLists(N):
    return (
        ("int", [random.randint(0, 9) + 1 for _ in range(N)]),
        ("float", [1.0 - random.random() for _ in range(N)]),
        ("complex", [complex(1.0 - random.random(), 1.0 - random.random()) for _ in range(N)]),
        ("bool", [True for _ in range(N)]),
        ("mixed", [1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(N - 3)]),
        ("mixed", [True, 1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(N - 4)]),
    )


_BOOL_SUB_ERR_MSG = "Subtraction, the `-` operator"


class RegularFuncWrapper:

    def __init__(self, func):
        self.func = func

    def __call__(self, inputs, values=None, **kwargs):
        if values is not None:
            assert len(inputs) == 3
            if isinstance(values, Number):
                values = [values for _ in range(len(inputs[0]))]
            return [self.func(*i, value=values[idx], **kwargs) for idx, i in enumerate(zip(*inputs))]
        if len(inputs) == 2 and isinstance(inputs[1], Number):
            # binary op with tensorlist and scalar.
            inputs[1] = [inputs[1] for _ in range(len(inputs[0]))]
        return [self.func(*i, **kwargs) for i in zip(*inputs)]


class ForeachFuncWrapper:

    def __init__(self, func, n_expected_cudaLaunchKernels):
        self.func = func
        self.n_expected_cudaLaunchKernels = n_expected_cudaLaunchKernels
        # Some foreach functions don't have in-place implementations.
        self._is_inplace = False if func is None else func.__name__.endswith('_')

    def __call__(self, inputs, is_cuda, is_fastpath, **kwargs):
        actual = None
        if (
            is_cuda and
            torch.autograd.kineto_available() and
            torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
        ):
            with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU,)) as p:
                actual = self.func(*inputs, **kwargs)
            for e in p.key_averages():
                if e.key == 'cudaLaunchKernel':
                    if is_fastpath:
                        assert e.count == self.n_expected_cudaLaunchKernels
                    else:
                        assert e.count > self.n_expected_cudaLaunchKernels
        else:
            actual = self.func(*inputs, **kwargs)
        # note(mkozuki): inplace foreach functions are void functions.
        return inputs[0] if self._is_inplace else actual


class TestForeach(TestCase):

    @property
    def is_cuda(self):
        return self.device_type == 'cuda'

    # note(mkozuki): It might be the case that the expected number of `cudaLaunchKernel`s
    # is greater than 1 once foreach functions internally separate their input `TensorList`s by
    # devices & dtypes into vectors of tensors.
    def _get_funcs(self, op, n_expected_cudaLaunchKernels: int):
        return (
            ForeachFuncWrapper(op.method_variant, n_expected_cudaLaunchKernels),
            RegularFuncWrapper(op.ref),
            ForeachFuncWrapper(op.inplace_variant, n_expected_cudaLaunchKernels),
            RegularFuncWrapper(op.ref_inplace),
        )

    def _binary_test(self, dtype, op, ref, inputs, is_fastpath, is_inplace, *, alpha=None):
        ref_inputs = [[t.clone().detach() for t in inputs[0]], inputs[1]] if is_inplace else inputs
        try:
            actual = op(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref(ref_inputs)
        else:
            expected = ref(ref_inputs)
            self.assertEqual(actual, expected)
        if alpha is not None:
            kwargs = {'alpha': alpha}
            ref_inputs = inputs
            try:
                actual = op(inputs, self.is_cuda, is_fastpath, **kwargs)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    ref(ref_inputs, **kwargs)
            else:
                expected = ref(ref_inputs, **kwargs)
                if dtype in (torch.float16, torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(expected, actual, atol=1.e-3, rtol=default_tolerances(dtype)[0])
                else:
                    self.assertEqual(expected, actual)

    def _test_binary_op_tensorlists(self, device, dtype, opinfo, N, is_fastpath, disable_fastpath):
        n_expected_cudaLaunchKernels = N if disable_fastpath else 1
        op, ref, inplace_op, inplace_ref = self._get_funcs(opinfo, n_expected_cudaLaunchKernels)
        inputs = [
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
        ]
        self._binary_test(dtype, op, ref, inputs, is_fastpath, is_inplace=False)
        self._binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath, is_inplace=True)
        if opinfo.supports_alpha_param:
            alpha = None
            if dtype in integral_types():
                alpha = 3
            elif dtype.is_complex:
                alpha = complex(3, 3)
            else:
                alpha = 3.14
            self._binary_test(dtype, op, ref, inputs, is_fastpath, is_inplace=False, alpha=alpha)
            self._binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath, is_inplace=True, alpha=alpha)

        # Tests of implicit broadcasting
        # When sizes of tensors don't match, foreach functions are supposed to choose slow path
        # even if this methods's argument `is_fastpath` is True.
        # `cudaLaunchKernel` will be equal to `N`. For assert in `ForeachFuncWrapper` to pass,
        # we pass `is_fastpath and disable_fastpath` to `_binary_test`'s argument of is_fastpath.
        # as n_expected_cudaLaunchKernels is N if disable_fastpath.
        inputs = [
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
            [
                make_tensor((N - i, 1), device=device, dtype=dtype, noncontiguous=not is_fastpath) for i in range(N)
            ],
        ]
        self._binary_test(dtype, op, ref, inputs, is_fastpath and disable_fastpath, is_inplace=False)
        self._binary_test(
            dtype, inplace_op, inplace_ref, inputs, is_fastpath and disable_fastpath, is_inplace=True)

    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_op_tensorlists_fastpath(self, device, dtype, op):
        for N in N_values:
            disable_fastpath = op.ref == torch.div and dtype in integral_types_and(torch.bool)
            if op.ref == torch.add and dtype == torch.bool:
                disable_fastpath = True
            self._test_binary_op_tensorlists(device, dtype, op, N, True, disable_fastpath)

    @ops(foreach_binary_op_db)
    def test_binary_op_tensorlists_slowpath(self, device, dtype, op):
        for N in N_values:
            self._test_binary_op_tensorlists(device, dtype, op, N, False, False)

    def _test_binary_op_scalar(self, device, dtype, opinfo, N, scalar, is_fastpath, disable_fastpath):
        n_expected_cudaLaunchKernels = N if disable_fastpath else 1
        op, ref, inplace_op, inplace_ref = self._get_funcs(opinfo, n_expected_cudaLaunchKernels)
        inputs = [opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath), scalar]
        self._binary_test(dtype, op, ref, inputs, is_fastpath, is_inplace=False)
        self._binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath, is_inplace=True)

    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_op_scalar_fastpath(self, device, dtype, op):
        for N, scalar in itertools.product(N_values, Scalars):
            disable_fastpath = op.ref == torch.div and dtype in integral_types_and(torch.bool)
            if isinstance(scalar, int):
                disable_fastpath |= dtype == torch.bool
            if isinstance(scalar, float):
                disable_fastpath |= dtype in integral_types_and(torch.bool)
            if isinstance(scalar, bool):
                disable_fastpath |= dtype == torch.bool
                if op.ref in (torch.add, torch.mul):
                    disable_fastpath = False
            if isinstance(scalar, complex):
                disable_fastpath |= dtype not in complex_types()
            self._test_binary_op_scalar(device, dtype, op, N, scalar, True, disable_fastpath)

    @ops(foreach_binary_op_db)
    def test_binary_op_scalar_slowpath(self, device, dtype, op):
        for N, scalar in itertools.product(N_values, Scalars):
            self._test_binary_op_scalar(device, dtype, op, N, scalar, False, False)

    def _test_binary_op_scalarlist(self, device, dtype, opinfo, N, scalarlist, is_fastpath, disable_fastpath):
        n_expected_cudaLaunchKernels = N if disable_fastpath else 1
        op, ref, inplace_op, inplace_ref = self._get_funcs(opinfo, n_expected_cudaLaunchKernels)
        inputs = [opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath), scalarlist]
        self._binary_test(dtype, op, ref, inputs, is_fastpath, is_inplace=False)
        self._binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath, is_inplace=True)

    # note(mkozuki): Why two functions depending on with/without bool?
    # `foreach_sub` & `foreach_sub_` do `sub_check(tensors[i], scalars[i])` from i=1...N.
    # So, if scalarlist has one or more bool values, `foreach_sub` and `foreach_sub_`
    # raise bool subtraction error before doing any math.
    # While regular `sub` and `sub_` do some math until they encounter bool.
    # So, foreach sub's throw bool sub error first. However, regular sub's throw different
    # errors depending on the order of scalarlist. To keep actual unit test impl simple,
    # separating mixed scalarlist tests. By setting the first element of scalarlist to bool,
    # they are expected to throw bool sub error even in inplace test.
    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_op_scalarlist_fastpath(self, device, dtype, op):
        for N in N_values:
            for type_str, scalarlist in getScalarLists(N):
                bool_int_div = op.ref == torch.div and dtype in integral_types_and(torch.bool)
                disable_fastpath = bool_int_div
                if type_str == "int":
                    disable_fastpath |= dtype == torch.bool
                if type_str == "float":
                    disable_fastpath |= dtype in integral_types_and(torch.bool)
                if type_str == "complex":
                    disable_fastpath |= dtype not in complex_types()
                if type_str == "mixed":
                    disable_fastpath |= True and dtype not in complex_types()
                self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, True, disable_fastpath)

    @ops(foreach_binary_op_db)
    def test_binary_op_scalarlist_slowpath(self, device, dtype, op):
        for N in N_values:
            for _, scalarlist in getScalarLists(N):
                self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, False, False)

    def _pointwise_test(self, dtype, op, ref, inputs, is_fastpath, is_inplace, *, values=None, custom_values_err=None):
        ref_inputs = [[t.clone().detach() for t in inputs[0]], inputs[1], inputs[2]] if is_inplace else inputs
        try:
            actual = op(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref(ref_inputs)
        else:
            expected = ref(ref_inputs)
            self.assertEqual(expected, actual)
        if values is not None:
            try:
                actual = op(inputs + [values], self.is_cuda, is_fastpath)
            except RuntimeError as e:
                # Match with error messages from regular non-foreach reference if no
                # custom error message was provided.
                if custom_values_err is None:
                    with self.assertRaisesRegex(type(e), re.escape(str(e))):
                        ref(ref_inputs, values=values)
                else:
                    self.assertEqual(re.escape(str(e)), re.escape(custom_values_err))
            else:
                expected = ref(ref_inputs, values=values)
                self.assertEqual(expected, actual)

    def _test_pointwise_op(self, device, dtype, opinfo, N, is_fastpath, disable_fastpath, *, values=None, custom_values_err=None):
        n_expected_cudaLaunchKernels = N if disable_fastpath else 1
        op, ref, inplace_op, inplace_ref = self._get_funcs(opinfo, n_expected_cudaLaunchKernels)
        inputs = [
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
        ]
        self._pointwise_test(dtype, op, ref, inputs, is_fastpath, is_inplace=False,
                             values=values, custom_values_err=custom_values_err)
        self._pointwise_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath,
                             is_inplace=True, values=values, custom_values_err=custom_values_err)

        # Tests of implicit broadcasting
        inputs = [
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath, same_size=True),
            [
                make_tensor((N - i, 1), device=device, dtype=dtype, noncontiguous=not is_fastpath) for i in range(N)
            ],
            [
                make_tensor((1, N - i), device=device, dtype=dtype, noncontiguous=not is_fastpath) for i in range(N)
            ],
        ]
        self._pointwise_test(dtype, op, ref, inputs, is_fastpath and disable_fastpath,
                             is_inplace=False, values=values, custom_values_err=custom_values_err)
        self._pointwise_test(
            dtype, inplace_op, inplace_ref, inputs, is_fastpath and disable_fastpath,
            is_inplace=True, values=values, custom_values_err=custom_values_err)

    @skipMeta
    @ops(foreach_pointwise_op_db)
    def test_pointwise_op_fastpath(self, device, dtype, op):
        disable_fastpath = dtype in integral_types_and(torch.bool)
        # for N, scalar in itertools.product(N_values, Scalars):
        for N in N_values:
            self._test_pointwise_op(device, dtype, op, N, True, disable_fastpath)
            for scalar in Scalars:
                self._test_pointwise_op(device, dtype, op, N, True, disable_fastpath, values=scalar)
            for case, scalarlist in getScalarLists(N):
                self._test_pointwise_op(
                    device, dtype, op, N, True, disable_fastpath, values=scalarlist)
                self._test_pointwise_op(
                    device, dtype, op, N, True, disable_fastpath, values=torch.tensor(scalarlist))
                self._test_pointwise_op(
                    device, dtype, op, N, True, disable_fastpath, values=torch.tensor(scalarlist)[0],
                    custom_values_err="Expected packed scalar Tensor to be of dimension 1. Got 0 instead.")
                if device == "cuda":
                    self._test_pointwise_op(
                        device, dtype, op, N, True, disable_fastpath, values=torch.tensor(scalarlist, device="cuda"),
                        custom_values_err="Expected scalars to be on CPU, got cuda:0 instead.")
                self._test_pointwise_op(
                    device, dtype, op, N, True, disable_fastpath, values=torch.tensor(scalarlist)[:2],
                    custom_values_err=f"Expected length of scalars to match input of length {len(scalarlist)} but got 2 instead.")
                self._test_pointwise_op(
                    device, dtype, op, N, True, disable_fastpath, values=torch.tensor([[0, 1], [2, 3]])[:, 1],
                    custom_values_err="Expected scalars to be contiguous.")

    @ops(foreach_pointwise_op_db)
    def test_pointwise_op_slowpath(self, device, dtype, op):
        # for N, scalar in itertools.product(N_values, Scalars):
        for N in N_values:
            self._test_pointwise_op(device, dtype, op, N, False, False)
            for scalar in Scalars:
                self._test_pointwise_op(device, dtype, op, N, False, False, values=scalar)
            for case, scalarlist in getScalarLists(N):
                self._test_pointwise_op(
                    device, dtype, op, N, False, False, values=scalarlist)
                self._test_pointwise_op(
                    device, dtype, op, N, False, False, values=torch.tensor(scalarlist))

    # note(mkozuki): fastpath test uses dtypes which fastpath implementation supports.
    # To confirm the dtypes of `OpInfo` cover the dtypes that the function support,
    # this test does not use `try-except` for fastpath.
    def _regular_unary_test(self, dtype, op, ref, inputs, is_fastpath):
        if is_fastpath:
            self.assertEqual(ref(inputs), op(inputs, self.is_cuda, is_fastpath))
            return
        try:
            actual = op(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                ref(inputs)
        else:
            expected = ref(inputs)
            self.assertEqual(actual, expected)

    # note(mkozuki): why `try-except` for both fastpath?
    # - inputs for fastpath can be integer tensors.
    #    - this is becase opinfo dtypes are configured for outpulace implementation
    # - for integer inputs, trigonometric functions and exponential function returns float outputs,
    #   which causes "result type Float can't be case to the desired type" error.
    # Thus, `try-except` is used even if `is_fastpath` is `True`.
    def _inplace_unary_test(self, dtype, inplace, inplace_ref, inputs, is_fastpath):
        copied_inputs = [[t.clone().detach() for t in tensors] for tensors in inputs]
        try:
            inplace(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                inplace_ref(copied_inputs)
        else:
            inplace_ref(copied_inputs),
            self.assertEqual(copied_inputs, inputs)

    def _test_unary(self, device, dtype, opinfo, N, is_fastpath):
        op, ref, inplace_op, inplace_ref = self._get_funcs(opinfo, 1)
        inputs = opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
        # note(mkozuki): Complex inputs for `_foreach_abs` go through slowpath.
        if opinfo.name == "_foreach_abs" and dtype in complex_types():
            is_fastpath = False
        self._regular_unary_test(dtype, op, ref, inputs, is_fastpath)
        self._inplace_unary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath)

    @skipMeta
    @ops(foreach_unary_op_db)
    def test_unary_fastpath(self, device, dtype, op):
        for N in N_values:
            self._test_unary(device, dtype, op, N, is_fastpath=True)

    @ops(foreach_unary_op_db, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_unary_slowpath(self, device, dtype, op):
        for N in N_values:
            self._test_unary(device, dtype, op, N, is_fastpath=False)

    # note(crcrpar): `torch.maximum` and `torch.minimum` support `out` arg but there seem to be no inplace versions.
    # So, compare `inplace_op` results with `ref`'s outputs.
    def _minmax_test(self, opinfo, inputs, is_fastpath, n_expected_cudaLaunchKernels):
        op, ref, inplace_op, _ = self._get_funcs(opinfo, n_expected_cudaLaunchKernels)
        expected = ref(inputs)
        self.assertEqual(expected, op(inputs, self.is_cuda, is_fastpath))

        inplace_inputs = [[t.clone() for t in inputs[0]], inputs[1]]
        inplace_op(inplace_inputs, self.is_cuda, is_fastpath)
        self.assertEqual(expected, inplace_inputs[0])

    @ops(foreach_minmax_op_db)
    def test_minmax_fastpath(self, device, dtype, op):
        for N in N_values:
            inputs = tuple(op.sample_inputs(device, dtype, N) for _ in range(2))
            self._minmax_test(op, inputs, True, N if dtype == torch.bool else 1)

    @ops(foreach_minmax_op_db,
         dtypes=all_types_and(torch.half, torch.bfloat16, torch.bool))
    def test_minmax_slowpath(self, device, dtype, op):
        for N in N_values:
            inputs = tuple(op.sample_inputs(device, dtype, N, noncontiguous=True) for _ in range(2))
            self._minmax_test(op, inputs, False, 1)

    # note(mkozuki): ForeachFuncInfo's of both `_foreach_maximum` and `_foreach_minimum` include integer types.
    # so, manually limit dtypes to fp types for inf&nan tests.
    @ops(foreach_minmax_op_db, dtypes=floating_types_and(torch.half, torch.bfloat16))
    def test_minmax_float_inf_nan(self, device, dtype, op):
        inputs = (
            [
                torch.tensor([float('inf')], device=device, dtype=dtype),
                torch.tensor([-float('inf')], device=device, dtype=dtype),
                torch.tensor([float('nan')], device=device, dtype=dtype),
                torch.tensor([float('nan')], device=device, dtype=dtype)
            ],
            [
                torch.tensor([-float('inf')], device=device, dtype=dtype),
                torch.tensor([float('inf')], device=device, dtype=dtype),
                torch.tensor([float('inf')], device=device, dtype=dtype),
                torch.tensor([float('nan')], device=device, dtype=dtype)
            ],
        )
        self._minmax_test(op, inputs, True, 1)

    def _reduce_test(self, opinfo, inputs, ord, is_fastpath, n_expected_cudaLaunchKernels):
        op, ref, _, _ = self._get_funcs(opinfo, n_expected_cudaLaunchKernels)
        self.assertEqual(ref(inputs, ord=ord), op(inputs, self.is_cuda, is_fastpath, ord=ord))

    @ops(foreach_reduce_op_db)
    def test_reduce_fastpath(self, device, dtype, op):
        for N, ord in itertools.product(N_values, (0, 1, 2, -1, -2)):
            if ord in (1, 2) and dtype in floating_types_and(torch.half, torch.bfloat16):
                n_expected_cudaLaunchKernels = 3
            else:
                n_expected_cudaLaunchKernels = N
            inputs = op.sample_inputs(device, dtype, N, noncontiguous=False),
            self._reduce_test(op, inputs, ord, True, n_expected_cudaLaunchKernels)

    @ops(foreach_reduce_op_db)
    def test_reduce_slowpath(self, device, dtype, op):
        for N, ord in itertools.product(N_values, (0, 1, 2, -1, -2)):
            inputs = op.sample_inputs(device, dtype, N, noncontiguous=True),
            self._reduce_test(op, inputs, ord, False, 1)

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_add_scalar_with_empty_list_and_empty_tensor(self, device, dtype):
        # TODO: enable empty list case
        for tensors in [[torch.randn([0])]]:
            res = torch._foreach_add(tensors, 1)
            self.assertEqual(res, tensors)

            torch._foreach_add_(tensors, 1)
            self.assertEqual(res, tensors)

    @ops(foreach_binary_op_db, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_binary_op_scalar_with_overlapping_tensors(self, device, dtype, op):
        foreach_op, ref = op.method_variant, op.ref
        tensors = [torch.ones(1, 1, device=device, dtype=dtype).expand(2, 1, 3)]

        if ref == torch.sub and dtype == torch.bool:
            with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                [ref(t, 1) for t in tensors]
            with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                foreach_op(tensors, 1)
            return

        expected = [ref(t, 1) for t in tensors]
        res = foreach_op(tensors, 1)
        self.assertEqual(res, expected)

    # note(mkozuki): this test case fails with Meta at least in my local environment.
    # The message was
    # `AssertionError: NotImplementedError("Could not run 'aten::_foreach_add.Scalar' with arguments from the 'Meta' backend.`
    @skipMeta
    @ops(foreach_binary_op_db, allowed_dtypes=[torch.float])
    def test_binary_op_scalar_with_different_tensor_dtypes(self, device, dtype, op):
        foreach_op = op.method_variant
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device),
                   torch.tensor([1], dtype=torch.long, device=device)]
        runtime_error = None
        try:
            foreach_op(tensors, 1)
        except RuntimeError as e:
            runtime_error = e
        self.assertIsNone(runtime_error)

    @skipIfTorchDynamo("Different error msgs, TODO")
    @ops(foreach_binary_op_db, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_binary_op_list_error_cases(self, device, dtype, op):
        foreach_op, foreach_op_, ref, ref_ = op.method_variant, op.inplace_variant, op.ref, op.ref_inplace
        tensors1 = []
        tensors2 = []

        # Empty lists
        with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
            foreach_op(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "There were no tensor arguments to this function"):
            foreach_op_(tensors1, tensors2)

        # One empty list
        tensors1.append(torch.tensor([1], device=device, dtype=dtype))
        with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
            foreach_op(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
            foreach_op_(tensors1, tensors2)

        # Lists have different amount of tensors
        tensors2.append(torch.tensor([1], device=device))
        tensors2.append(torch.tensor([1], device=device))
        with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
            foreach_op(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, "Tensor lists must have the same number of tensors, got 1 and 2"):
            foreach_op_(tensors1, tensors2)

        # Corresponding tensors with different sizes that aren't compatible with broadcast
        # If sizes are different then foreach chooses slow path, thus error messages are expected
        # to be the same as torch regular function.
        tensors1 = [torch.zeros(10, 10, device=device, dtype=dtype) for _ in range(10)]
        tensors2 = [torch.ones(11, 11, device=device, dtype=dtype) for _ in range(10)]
        try:
            foreach_op(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [ref(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
        try:
            foreach_op_(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [ref_(t1, t2) for t1, t2 in zip(tensors1, tensors2)]

        # different devices
        if self.device_type == "cuda" and torch.cuda.device_count() > 1:
            tensor1 = torch.zeros(10, 10, device="cuda:0", dtype=dtype)
            tensor2 = torch.ones(10, 10, device="cuda:1", dtype=dtype)
            if dtype == torch.bool and foreach_op == torch._foreach_sub:
                with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                    foreach_op([tensor1], [tensor2])
                with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                    foreach_op_([tensor1], [tensor2])
                return
            with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                foreach_op([tensor1], [tensor2])
            if dtype in integral_types_and(torch.bool) and foreach_op == torch._foreach_div:
                with self.assertRaisesRegex(RuntimeError, "result type"):
                    foreach_op_([tensor1], [tensor2])
            else:
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    foreach_op_([tensor1], [tensor2])

    @skipMeta
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
    @ops(foreach_binary_op_db, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_binary_op_list_slow_path(self, device, dtype, op):
        # note(mkozuki): why `n_expected_cudaLaunchKernels=0`?
        # In this test, foreach functions don't go through fast path,
        # but as there is only one tensor in each list of tensors,
        # `cudaLaunchKernel` is 1 so ForeachFuncWrapper internal assert fails.
        foreach_op, native_op, foreach_op_, native_op_ = self._get_funcs(op, n_expected_cudaLaunchKernels=0)
        # 0-strides
        tensor1 = make_tensor((10, 10), dtype=dtype, device=device)
        tensor2 = make_tensor((1,), device=device, dtype=dtype).expand_as(tensor1)
        inputs = ([tensor1], [tensor2])
        self._binary_test(dtype, foreach_op, native_op, inputs, is_fastpath=False, is_inplace=False)
        self._binary_test(dtype, foreach_op_, native_op_, inputs, is_fastpath=False, is_inplace=True)

        # different strides
        tensor1 = torch.zeros(10, 10, device=device, dtype=dtype)
        tensor2 = torch.ones(10, 10, device=device, dtype=dtype)
        inputs = ([tensor1], [tensor2.t()])
        self._binary_test(dtype, foreach_op, native_op, inputs, is_fastpath=False, is_inplace=False)
        self._binary_test(dtype, foreach_op_, native_op_, inputs, is_fastpath=False, is_inplace=True)

        # non contiguous
        tensor1 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype, noncontiguous=True)
        tensor2 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype, noncontiguous=True)
        self.assertFalse(tensor1.is_contiguous())
        self.assertFalse(tensor2.is_contiguous())
        inputs = ([tensor1], [tensor2])
        self._binary_test(dtype, foreach_op, native_op, inputs, is_fastpath=False, is_inplace=False)
        self._binary_test(dtype, foreach_op_, native_op_, inputs, is_fastpath=False, is_inplace=True)

        # sliced tensor
        tensor1 = make_tensor((5, 2, 1, 3), device=device, dtype=dtype)
        tensor2 = make_tensor((5, 2, 1, 3 * 7), device=device, dtype=dtype)[:, :, :, ::7]
        inputs = ([tensor1], [tensor2])
        self._binary_test(dtype, foreach_op, native_op, inputs, is_fastpath=False, is_inplace=False)
        self._binary_test(dtype, foreach_op_, native_op_, inputs, is_fastpath=False, is_inplace=True)

    # note: Below three tests (postfixed with `_tensors_on_different_devices`)
    # checks whether foreach works with lists of tensors on different devices
    # but tensors of the same index are on the same device, e.g., ['cuda', 'cpu].
    @onlyCUDA
    @ops(foreach_unary_op_db)
    def test_unary_op_tensors_on_different_devices(self, device, dtype, op):
        method, ref, inplace_method, ref_inplace = self._get_funcs(op, 1)
        # tensors: ['cuda', 'cpu]
        tensors = op.sample_inputs(device, dtype, 2)
        tensors[1] = tensors[1].to('cpu')
        try:
            actual = method((tensors,), False, False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                ref((tensors,))
        else:
            expected = ref((tensors,))
            self.assertEqual(expected, actual)

        try:
            inplace_method((tensors,), False, False)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), str(e)):
                ref_inplace((tensors,))
        else:
            self.assertEqual(expected, tensors)

    @onlyCUDA
    @ops(foreach_binary_op_db)
    def test_binary_op_tensors_on_different_devices(self, device, dtype, op):
        # `tensors1`: ['cuda', 'cpu']
        # `tensors2`: ['cuda', 'cpu']
        _cuda_tensors = op.sample_inputs(device, dtype, 2, same_size=True)
        _cpu_tensors = op.sample_inputs('cpu', dtype, 2, same_size=True)
        tensors1, tensors2 = list(tensors for tensors in zip(_cuda_tensors, _cpu_tensors))

        foreach_op, foreach_op_ = op.method_variant, op.inplace_variant
        native_op, native_op_ = op.ref, op.ref_inplace
        try:
            actual = foreach_op(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [native_op(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
        else:
            expected = [native_op(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
            self.assertEqual(expected, actual)
        try:
            foreach_op_(tensors1, tensors2)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                [native_op_(t1, t2) for t1, t2 in zip(tensors1, tensors2)]
        else:
            self.assertEqual(actual, tensors1)

    @onlyCUDA
    @ops(foreach_pointwise_op_db, allowed_dtypes=floating_types())
    def test_pointwise_op_tensors_on_different_devices(self, device, dtype, op):
        # tensors1: ['cuda', 'cpu]
        # tensors2: ['cuda', 'cpu]
        # tensors3: ['cuda', 'cpu]
        _cuda_tensors = op.sample_inputs(device, dtype, 3, same_size=True)
        _cpu_tensors = op.sample_inputs('cpu', dtype, 3, same_size=True)
        tensors1, tensors2, tensors3 = list(tensors for tensors in zip(_cuda_tensors, _cpu_tensors))

        foreach_op, foreach_op_, native_op = op.method_variant, op.inplace_variant, op.ref
        actual = foreach_op(tensors1, tensors2, tensors3)
        expected = [native_op(*_cuda_tensors), native_op(*_cpu_tensors)]
        self.assertEqual(expected, actual)

        # note(mkozuki): Limiting dtypes to FP32&FP64, we can safely run inplace ops.
        foreach_op_(tensors1, tensors2, tensors3)
        self.assertEqual(expected, tensors1)

    # note: BFloat16 has the same number of exponent bits as FP32
    # so if squared L2 norm overflows in BF16, then it also overflows in FP32.
    @onlyCUDA
    @ops(foreach_reduce_op_db, allowed_dtypes=(torch.half, torch.bfloat16))
    def test_foreach_l2_large_value_input(self, device, dtype, op):
        ord, N = 2, 10
        max_value = torch.finfo(dtype).max
        scaler = torch.tensor([max_value]).sqrt().to(device=device, dtype=dtype)
        inputs = [t * scaler for t in op.sample_inputs(device, dtype, N, noncontiguous=False, low=1)],
        # make sure that the min. of squared L2 norm value per tensor is greater than the max value of `dtype`.
        self.assertTrue(scaler * scaler * N > max_value)
        fn, ref_fn, *_ = self._get_funcs(op, 3)
        actual = fn(inputs, is_cuda=True, is_fastpath=True, ord=ord)
        expect = ref_fn(inputs, ord=ord)
        if dtype == torch.float16:
            # making sure the reference L2 norm values are in the range of FP16.
            self.assertFalse(any(torch.isinf(e) for e in expect))
        else:
            self.assertTrue(all(torch.isinf(e) for e in expect))
        self.assertEqual(expect, actual, equal_nan=False)


instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
