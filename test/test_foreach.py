import random
import re
import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, TEST_WITH_SLOW
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, skipCUDAIfRocm, skipMeta, ops)
from torch._six import inf, nan
from torch.testing._internal.common_methods_invocations import foreach_unary_op_db, foreach_binary_op_db

# Includes some values such that N * N won't be a multiple of 4,
# which should ensure we test the vectorized and non-vectorized
# kernel code paths.
N_values = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]

_BOOL_SUB_ERR_MSG = "Subtraction, the `-` operator"

class RegularFuncWrapper:

    def __init__(self, func):
        self.func = func

    def __call__(self, inputs, **kwargs):
        return [self.func(*i, **kwargs) for i in zip(*inputs)]


class ForeachFuncWrapper:

    def __init__(self, func, n_expected_cudaLaunchKernels):
        self.func = func
        self.n_expected_cudaLaunchKernels = n_expected_cudaLaunchKernels

    def __call__(self, inputs, is_cuda, is_fastpath, **kwargs):
        if is_cuda and torch.autograd.kineto_available():
            with torch.profiler.profile(activities=(torch.profiler.ProfilerActivity.CPU,)) as p:
                actual = self.func(*inputs, **kwargs)
            for e in p.key_averages():
                if e.key == 'cudaLaunchKernel':
                    if is_fastpath:
                        assert e.count == self.n_expected_cudaLaunchKernels
                    else:
                        assert e.count > self.n_expected_cudaLaunchKernels
            return actual
        else:
            return self.func(*inputs, **kwargs)

class TestForeach(TestCase):

    @property
    def is_cuda(self):
        return self.device_type == 'cuda'

    # note(mkozuki): It might be the case that the expected number of `cudaLaunchKernel`s
    # is greater than 1 once foreach functions internally separate their input `TensorList`s by
    # devices & dtypes into vectors of tensors.
    def _create_funcs(self, op, n_expected_cudaLaunchKernels):
        return (
            ForeachFuncWrapper(op.method_variant, n_expected_cudaLaunchKernels),
            RegularFuncWrapper(op.ref),
            ForeachFuncWrapper(op.inplace_variant, n_expected_cudaLaunchKernels),
            RegularFuncWrapper(op.ref_inplace),
        )

    def _get_funcs(
        self,
        op,
        *,
        n_expected_cudaLaunchKernels=1,
        dtype=None,
        scalars=None,
        n_tensors=None,
        is_fastpath=False,
    ):
        if not is_fastpath:
            return self._create_funcs(op, n_expected_cudaLaunchKernels)
        disable_fast_route = False
        has_integer = dtype in torch.testing.get_all_int_dtypes()
        has_bool = dtype == torch.bool
        has_complex_scalar, has_float_scalar, has_int_scalar, has_bool_scalar = False, False, False, False
        if scalars is not None:
            if not isinstance(scalars, list):
                scalars = [scalars]
            for s in scalars:
                if isinstance(s, complex):
                    has_complex_scalar = True
                if isinstance(s, float):
                    has_float_scalar = True
                if isinstance(s, int):
                    has_int_scalar = True
                if isinstance(s, bool):
                    has_bool_scalar = True
        # mixed scalarlist
        if has_complex_scalar and has_float_scalar and has_int_scalar:
            disable_fast_route = True
            if dtype in torch.testing.get_all_complex_dtypes():
                disable_fast_route = False
        # torch.bool and (bool Scalar or bool ScalarList)
        if has_bool and has_int_scalar and not has_bool_scalar:
            disable_fast_route = True
        # complex Scalar and int/bool/float Tensor
        if has_complex_scalar and dtype not in torch.testing.get_all_complex_dtypes():
            disable_fast_route = True
        # int/bool Tensor and float/complex Scalar
        if (has_integer or has_bool) and (has_complex_scalar or has_float_scalar):
            disable_fast_route = True
        # `_foreach_add.List` with bool tensors & default alpha uses slow path
        if op.ref == torch.add:
            if has_bool and not (has_int_scalar or has_bool_scalar):
                disable_fast_route = True
        # `div` with int/bool Tensor requires type promotion, i.e., slow path
        if op.ref == torch.div:
            if has_bool or has_integer:
                disable_fast_route = True

        if disable_fast_route:
            n_expected_cudaLaunchKernels = n_tensors
        return self._create_funcs(op, n_expected_cudaLaunchKernels)

    # note(mkozuki): Foreach binary ops upcast 16-bits inputs to 32-bits and downcast outputs to original dtype.
    def _requires_cast(self, op, dtype, is_fastpath, inputs, inputs2=None):
        second_arg = inputs[1] if inputs2 is None else inputs2
        return (
            op.func in (torch._foreach_add, torch._foreach_sub, torch._foreach_add_, torch._foreach_sub_) and
            dtype in (torch.bfloat16, torch.float16) and
            self.is_cuda and
            is_fastpath and
            not any(isinstance(a, complex) for a in second_arg)
        )

    # todo(mkozuki): remove this method once `TestForeach` is refactored with `@op` decorator.
    def _get_test_data(self, device, dtype, N):
        if dtype in [torch.bfloat16, torch.bool, torch.float16]:
            tensors = [torch.randn(N, N, device=device).to(dtype) for _ in range(N)]
        elif dtype in torch.testing.get_all_int_dtypes():
            # Constrains the range between 1 and 10 for less stress on int8 tensors.
            tensors = [torch.randint(1, 10, (N, N), device=device, dtype=dtype) for _ in range(N)]
        else:
            tensors = [torch.randn(N, N, device=device, dtype=dtype) for _ in range(N)]

        return tensors

    def _regular_binary_test(self, dtype, op, ref, inputs, is_fastpath, *, inputs2=None, alpha=None):
        requires_cast = self._requires_cast(op, dtype, is_fastpath, inputs, inputs2)
        ref_inputs = [inputs[0], inputs[1] if inputs2 is None else inputs2]
        if (
            op.func == torch._foreach_sub and
            (
                # case 1: test_binary_op_tensorlists_(fast|slow)path with `torch.bool`
                dtype == torch.bool or
                # case 2: test_binary_op_scalar_(fast|slow)path with boolearn scalar
                inputs2 is not None and any(isinstance(a, bool) for a in inputs2) or
                # case 3: test_binary_op_scalarlist_(fast|slow)path with boolean scalar list
                inputs2 is None and any(isinstance(a, bool) for a in inputs[1])
            )
        ):
            with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                op(inputs, self.is_cuda, is_fastpath)
            with self.assertRaisesRegex(RuntimeError, re.escape(_BOOL_SUB_ERR_MSG)):
                ref(ref_inputs)
            return
        if is_fastpath:
            if requires_cast:
                ref_inputs = [[t.to(torch.float32) for t in ref_inputs[0]], ref_inputs[1]]
            expected = ref(ref_inputs)
            if requires_cast:
                expected = [t.to(dtype) for t in expected]
            actual = op(inputs, self.is_cuda, is_fastpath)
            self.assertEqual(expected, actual)
        else:
            try:
                actual = op(inputs, self.is_cuda, is_fastpath)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    ref(ref_inputs)
            else:
                expected = ref(ref_inputs)
                self.assertEqual(actual, expected)
        if alpha is not None:
            # note(mkozuki): Input/Output type casting.
            # For BFloat16/Float16 I/O with add/sub with alpha parameter,
            # foreach implementations cast inputs to float32 and downcast the result
            # to originaly dtype, while regular implementations don't.
            kwargs = {'alpha': alpha}
            requires_cast = dtype in (torch.float16, torch.bfloat16) and self.is_cuda and is_fastpath
            ref_inputs = inputs
            if requires_cast:
                ref_inputs = [[t.to(torch.float32) for t in ref_inputs[0]], ref_inputs[1]]
            try:
                actual = op(inputs, self.is_cuda, is_fastpath, **kwargs)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    ref(ref_inputs, **kwargs)
            else:
                expected = ref(ref_inputs, **kwargs)
                if requires_cast:
                    expected = [t.to(dtype) for t in expected]
                if dtype in (torch.float16, torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(expected, actual, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(expected, actual)

    def _inplace_binary_test(self, dtype, inplace, inplace_ref, inputs, is_fastpath, *, inputs2=None, alpha=None):
        requires_cast = self._requires_cast(inplace, dtype, is_fastpath, inputs, inputs2)
        if inputs2 is not None:
            requires_cast = requires_cast and all(isinstance(a, (int, float, bool)) for a in inputs2)
        ref_inputs = [
            [t.clone().detach() for t in inputs[0]],
            inputs[1] if inputs2 is None else inputs2,
        ]
        if requires_cast:
            ref_inputs = [[t.to(torch.float32) for t in ref_inputs[0]], ref_inputs[1]]
        try:
            inplace(inputs, self.is_cuda, is_fastpath)
        except RuntimeError as e:
            with self.assertRaisesRegex(type(e), re.escape(str(e))):
                inplace_ref(ref_inputs)
        else:
            inplace_ref(ref_inputs)
            expected = ref_inputs[0]
            if requires_cast:
                expected = [t.to(dtype) for t in ref_inputs[0]]
            self.assertEqual(expected, inputs[0])
        if alpha is not None:
            # note(mkozuki): Input/Output type casting.
            # For BFloat16/Float16 I/O with add/sub with alpha parameter,
            # foreach implementations cast inputs to float32 and downcast the result
            # to originaly dtype, while regular implementations don't.
            kwargs = {'alpha': alpha}
            requires_cast = dtype in (torch.float16, torch.bfloat16) and self.is_cuda and is_fastpath
            ref_inputs = inputs
            if requires_cast:
                ref_inputs = [[t.to(torch.float32) for t in tensors] for tensors in inputs]
            try:
                inplace(inputs, self.is_cuda, is_fastpath, **kwargs)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    inplace_ref(ref_inputs, **kwargs)
            else:
                inplace_ref(ref_inputs, **kwargs)
                expected = ref_inputs[0]
                if requires_cast:
                    expected = [t.to(dtype) for t in expected]
                if dtype in (torch.float16, torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(expected, inputs[0], atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(expected, inputs[0])

    def _test_binary_op_tensorlists(self, device, dtype, opinfo, N, is_fastpath):
        op, ref, inplace_op, inplace_ref = self._get_funcs(
            opinfo, dtype=dtype, scalars=None, n_tensors=N, is_fastpath=is_fastpath)
        inputs = [
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
            opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
        ]
        self._regular_binary_test(dtype, op, ref, inputs, is_fastpath)
        self._inplace_binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath)
        if opinfo.supports_alpha_param:
            alpha = 3 if dtype in torch.testing.get_all_int_dtypes() else 3.14
            self._regular_binary_test(dtype, op, ref, inputs, is_fastpath, alpha=alpha)
            self._inplace_binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath, alpha=alpha)

    @ops(foreach_binary_op_db)
    def test_binary_op_tensorlists_fastpath(self, device, dtype, op):
        for N in N_values:
            self._test_binary_op_tensorlists(device, dtype, op, N, True)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_tensorlists_slowpath(self, device, dtype, op):
        for N in N_values:
            self._test_binary_op_tensorlists(device, dtype, op, N, False)

    def _test_binary_op_scalar(self, device, dtype, opinfo, N, scalar, is_fastpath):
        op, ref, inplace_op, inplace_ref = self._get_funcs(
            opinfo, dtype=dtype, scalars=scalar, n_tensors=N, is_fastpath=is_fastpath)
        inputs = [opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath), scalar]
        inputs2 = [scalar for _ in range(N)]
        self._regular_binary_test(dtype, op, ref, inputs, is_fastpath, inputs2=inputs2)
        self._inplace_binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath, inputs2=inputs2)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_int_scalar_fastpath(self, device, dtype, op):
        for N in N_values:
            scalar = random.randint(0, 9) + 1
            self._test_binary_op_scalar(device, dtype, op, N, scalar, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_int_scalar_slowpath(self, device, dtype, op):
        for N in N_values:
            scalar = random.randint(0, 9) + 1
            self._test_binary_op_scalar(device, dtype, op, N, scalar, False)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_float_scalar_fastpath(self, device, dtype, op):
        for N in N_values:
            scalar = 1.0 - random.random()
            self._test_binary_op_scalar(device, dtype, op, N, scalar, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_float_scalar_slowpath(self, device, dtype, op):
        for N in N_values:
            scalar = 1.0 - random.random()
            self._test_binary_op_scalar(device, dtype, op, N, scalar, False)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_complex_scalar_fastpath(self, device, dtype, op):
        for N in N_values:
            scalar = complex(1.0 - random.random(), 1.0 - random.random())
            self._test_binary_op_scalar(device, dtype, op, N, scalar, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_complex_scalar_slowpath(self, device, dtype, op):
        for N in N_values:
            scalar = complex(1.0 - random.random(), 1.0 - random.random())
            self._test_binary_op_scalar(device, dtype, op, N, scalar, False)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_bool_scalar_fastpath(self, device, dtype, op):
        for N in N_values:
            scalar = True
            self._test_binary_op_scalar(device, dtype, op, N, scalar, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_bool_scalar_slowpath(self, device, dtype, op):
        for N in N_values:
            scalar = True
            self._test_binary_op_scalar(device, dtype, op, N, scalar, False)

    def _test_binary_op_scalarlist(self, device, dtype, opinfo, N, scalarlist, is_fastpath):
        op, ref, inplace_op, inplace_ref = self._get_funcs(
            opinfo, dtype=dtype, scalars=scalarlist, n_tensors=N, is_fastpath=is_fastpath)
        inputs = [opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath), scalarlist]
        self._regular_binary_test(dtype, op, ref, inputs, is_fastpath)
        self._inplace_binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_int_scalarlist_fastpath(self, device, dtype, op):
        for N in N_values:
            scalarlist = [random.randint(0, 9) + 1 for _ in range(N)]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_int_scalarlist_slowpath(self, device, dtype, op):
        for N in N_values:
            scalarlist = [random.randint(0, 9) + 1 for _ in range(N)]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, False)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_float_scalarlist_fastpath(self, device, dtype, op):
        for N in N_values:
            scalarlist = [1.0 - random.random() for _ in range(N)]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_float_scalarlist_slowpath(self, device, dtype, op):
        for N in N_values:
            scalarlist = [1.0 - random.random() for _ in range(N)]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, False)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_complex_scalarlist_fastpath(self, device, dtype, op):
        for N in N_values:
            scalarlist = [complex(1.0 - random.random(), 1.0 - random.random()) for _ in range(N)]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_complex_scalarlist_slowpath(self, device, dtype, op):
        for N in N_values:
            scalarlist = [complex(1.0 - random.random(), 1.0 - random.random()) for _ in range(N)]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, False)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_bool_scalarlist_fastpath(self, device, dtype, op):
        for N in N_values:
            scalarlist = [True for _ in range(N)]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_bool_scalarlist_slowpath(self, device, dtype, op):
        for N in N_values:
            scalarlist = [True for _ in range(N)]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, False)

    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_mixed_scalarlist_without_bool_fastpath(self, device, dtype, op):
        _scalarlist = [1, 2.0, 3.0 + 4.5j]
        for N in N_values:
            scalarlist = _scalarlist + [3.0 for _ in range(N - len(_scalarlist))]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_mixed_scalarlist_without_bool_slowpath(self, device, dtype, op):
        _scalarlist = [1, 2.0, 3.0 + 4.5j]
        for N in N_values:
            scalarlist = _scalarlist + [3.0 for _ in range(N - len(_scalarlist))]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, False)

    # note(mkozuki): Why two functions depending on with/without bool?
    # `foreach_sub` & `foreach_sub_` do `sub_check(tensors[i], scalars[i])` from i=1...N.
    # So, if scalarlist has one or more bool values, `foreach_sub` and `foreach_sub_`
    # raise bool subtraction error before doing any math.
    # While regular `sub` and `sub_` do some math until they encounter bool.
    # So, foreach sub's throw bool sub error first. However, regular sub's throw different
    # errors depending on the order of scalarlist. To keep actual unit test impl simple,
    # separating mixed scalarlist tests. By setting the first element of scalarlist to bool,
    # they are expected to throw bool sub error even in inplace test.
    @skipCUDAIfRocm
    @ops(foreach_binary_op_db)
    def test_binary_op_mixed_scalarlist_with_bool_fastpath(self, device, dtype, op):
        _scalarlist = [True, 1, 2.0, 3.0 + 4.5j]
        for N in N_values:
            scalarlist = _scalarlist + [3.0 for _ in range(N - len(_scalarlist))]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, True)

    @skipCUDAIfRocm
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_mixed_scalarlist_with_bool_slowpath(self, device, dtype, op):
        _scalarlist = [True, 1, 2.0, 3.0 + 4.5j]
        for N in N_values:
            scalarlist = _scalarlist + [3.0 for _ in range(N - len(_scalarlist))]
            self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, False)

    def _test_pointwise_op(self, device, dtype, foreach_op, foreach_op_, torch_op):
        for N in N_values:
            # Constrains the range a bit for int8 tensors.
            values = [2 + (i % 5) for i in range(N)]
            for vals in [values[0], values]:
                tensors = self._get_test_data(device, dtype, N)
                tensors1 = self._get_test_data(device, dtype, N)
                tensors2 = self._get_test_data(device, dtype, N)

                # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
                control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                                  (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

                if not isinstance(vals, list):
                    expected = [torch_op(tensors[i].to(dtype=control_dtype),
                                         tensors1[i].to(dtype=control_dtype),
                                         tensors2[i].to(dtype=control_dtype),
                                         value=values[0]).to(dtype=dtype) for i in range(N)]
                else:
                    expected = [torch_op(tensors[i].to(dtype=control_dtype),
                                         tensors1[i].to(dtype=control_dtype),
                                         tensors2[i].to(dtype=control_dtype),
                                         value=values[i]).to(dtype=dtype) for i in range(N)]

                res = foreach_op(tensors, tensors1, tensors2, vals)
                foreach_op_(tensors, tensors1, tensors2, vals)
                self.assertEqual(res, tensors)

                if (dtype is torch.float16 or dtype is torch.bfloat16) and TEST_WITH_ROCM:
                    self.assertEqual(tensors, expected, atol=3.e-3, rtol=self.dtype_precisions[dtype][0])
                else:
                    self.assertEqual(tensors, expected)

                # test error cases
                for op in [torch._foreach_addcmul, torch._foreach_addcmul_, torch._foreach_addcdiv, torch._foreach_addcdiv_]:
                    tensors = self._get_test_data(device, dtype, N)
                    tensors1 = self._get_test_data(device, dtype, N)
                    tensors2 = self._get_test_data(device, dtype, N)

                    with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N + 1)])

                    with self.assertRaisesRegex(RuntimeError, "Tensor list must have same number of elements as scalar list."):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N - 1)])

                    msg = "Tensor lists must have the same number of tensors, got {} and {}".format(N + 1, N)

                    tensors = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, msg):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

                    tensors1 = self._get_test_data(device, dtype, N + 1)
                    with self.assertRaisesRegex(RuntimeError, msg):
                        op(tensors, tensors1, tensors2, [2 for _ in range(N)])

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
        op, ref, inplace_op, inplace_ref = self._get_funcs(opinfo)
        inputs = opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath),
        # note(mkozuki): Complex inputs for `_foreach_abs` go through slowpath.
        if opinfo.name == "_foreach_abs" and dtype in torch.testing.get_all_complex_dtypes():
            is_fastpath = False
        self._regular_unary_test(dtype, op, ref, inputs, is_fastpath)
        self._inplace_unary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath)

    @skipMeta
    @ops(foreach_unary_op_db)
    def test_unary_fastpath(self, device, dtype, op):
        for N in N_values:
            self._test_unary(device, dtype, op, N, is_fastpath=True)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_unary_op_db)
    def test_unary_slowpath(self, device, dtype, op):
        for N in N_values:
            self._test_unary(device, dtype, op, N, is_fastpath=False)

    #
    # Pointwise ops
    #
    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False, include_complex=False))
    def test_addcmul(self, device, dtype):
        if self.device_type == 'cpu':
            if dtype == torch.half:
                with self.assertRaisesRegex(RuntimeError, r"\"addcmul_cpu_out\" not implemented for \'Half\'"):
                    self._test_pointwise_op(device, dtype, torch._foreach_addcmul,
                                            torch._foreach_addcmul_, torch.addcmul)
                return

        self._test_pointwise_op(device, dtype, torch._foreach_addcmul, torch._foreach_addcmul_, torch.addcmul)

    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_bool=False, include_complex=False))
    def test_addcdiv(self, device, dtype):
        if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8]:
            with self.assertRaisesRegex(RuntimeError,
                                        "Integer division with addcdiv is no longer supported, and in a future"):
                self._test_pointwise_op(device, dtype, torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv)
            return

        if self.device_type == 'cpu':
            if dtype == torch.half:
                with self.assertRaisesRegex(RuntimeError, r"\"addcdiv_cpu_out\" not implemented for \'Half\'"):
                    self._test_pointwise_op(device, dtype, torch._foreach_addcdiv,
                                            torch._foreach_addcdiv_, torch.addcdiv)
                return
        self._test_pointwise_op(device, dtype, torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv)

    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=False, include_complex=False))
    def test_min_max(self, device, dtype):
        for N in N_values:
            tensors1 = self._get_test_data(device, dtype, N)
            tensors2 = self._get_test_data(device, dtype, N)

            # Mimics cuda kernel dtype flow.  With fp16/bf16 input, runs in fp32 and casts output back to fp16/bf16.
            control_dtype = torch.float32 if (self.device_type == 'cuda' and
                                              (dtype is torch.float16 or dtype is torch.bfloat16)) else dtype

            expected_max = [torch.max(tensors1[i].to(dtype=control_dtype),
                                      tensors2[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]

            expected_min = [torch.min(tensors1[i].to(dtype=control_dtype),
                                      tensors2[i].to(dtype=control_dtype)).to(dtype=dtype) for i in range(N)]

            res_max = torch._foreach_maximum(tensors1, tensors2)
            self.assertEqual(res_max, expected_max)

            res_min = torch._foreach_minimum(tensors1, tensors2)
            self.assertEqual(res_min, expected_min)

    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_max_min_float_inf_nan(self, device, dtype):
        a = [
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([-float('inf')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype)
        ]

        b = [
            torch.tensor([-float('inf')], device=device, dtype=dtype),
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([float('inf')], device=device, dtype=dtype),
            torch.tensor([float('nan')], device=device, dtype=dtype)
        ]

        expected = [torch.max(a1, b1) for a1, b1 in zip(a, b)]
        res = torch._foreach_maximum(a, b)
        self.assertEqual(expected, res)

        expected = [torch.min(a1, b1) for a1, b1 in zip(a, b)]
        res = torch._foreach_minimum(a, b)
        self.assertEqual(expected, res)

    @dtypes(*(torch.testing.get_all_fp_dtypes(include_half=True, include_bfloat16=False)))
    def test_max_min_inf_nan(self, device, dtype):
        a = [
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([-inf], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype)
        ]

        b = [
            torch.tensor([-inf], device=device, dtype=dtype),
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([inf], device=device, dtype=dtype),
            torch.tensor([nan], device=device, dtype=dtype)
        ]

        expected_max = [torch.max(a1, b1) for a1, b1 in zip(a, b)]
        res_max = torch._foreach_maximum(a, b)
        self.assertEqual(expected_max, res_max)

        expected_min = [torch.min(a1, b1) for a1, b1 in zip(a, b)]
        res_min = torch._foreach_minimum(a, b)
        self.assertEqual(expected_min, res_min)

    @dtypes(*torch.testing.get_all_dtypes())
    def test_add_scalar_with_empty_list_and_empty_tensor(self, device, dtype):
        # TODO: enable empty list case
        for tensors in [[torch.randn([0])]]:
            res = torch._foreach_add(tensors, 1)
            self.assertEqual(res, tensors)

            torch._foreach_add_(tensors, 1)
            self.assertEqual(res, tensors)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
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
    @dtypes(torch.float)
    @ops(foreach_binary_op_db)
    def test_binary_op_scalar_with_different_tensor_dtypes(self, device, _, op):
        foreach_op = op.method_variant
        tensors = [torch.tensor([1.1], dtype=torch.float, device=device),
                   torch.tensor([1], dtype=torch.long, device=device)]
        runtime_error = None
        try:
            foreach_op(tensors, 1)
        except RuntimeError as e:
            runtime_error = e
        self.assertIsNone(runtime_error)

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_list_error_cases(self, device, dtype, op):
        foreach_op, foreach_op_ = op.method_variant, op.inplace_variant
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

        # Corresponding tensors with different sizes
        tensors1 = [torch.zeros(10, 10, device=device, dtype=dtype) for _ in range(10)]
        tensors2 = [torch.ones(11, 11, device=device, dtype=dtype) for _ in range(10)]
        with self.assertRaisesRegex(RuntimeError, "Corresponding tensors in lists must have the same size"):
            foreach_op(tensors1, tensors2)
        with self.assertRaisesRegex(RuntimeError, r", got \[10, 10\] and \[11, 11\]"):
            foreach_op_(tensors1, tensors2)

        # different devices
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
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
            if dtype in torch.testing.get_all_int_dtypes() + [torch.bool] and foreach_op == torch._foreach_div:
                with self.assertRaisesRegex(RuntimeError, "result type"):
                    foreach_op_([tensor1], [tensor2])
            else:
                with self.assertRaisesRegex(RuntimeError, "Expected all tensors to be on the same device"):
                    foreach_op_([tensor1], [tensor2])

    @skipMeta
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not found")
    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_list_slow_path(self, device, dtype, op):
        # note(mkozuki): why `n_expected_cudaLaunchKernels=0`?
        # In this test, foreach functions don't go through fast path,
        # but as there is only one tensor in each list of tensors,
        # `cudaLaunchKernel` is 1 so ForeachFuncWrapper internal assert fails.
        foreach_op, native_op, foreach_op_, native_op_ = self._get_funcs(op, n_expected_cudaLaunchKernels=0)
        # 0-strides
        tensor1 = torch.randn(10, 10, device=device).to(dtype)
        tensor2 = torch.randn(1, device=device).to(dtype).expand_as(tensor1)
        inputs = ([tensor1], [tensor2])
        self._regular_binary_test(dtype, foreach_op, native_op, inputs, False)
        self._inplace_binary_test(dtype, foreach_op_, native_op_, inputs, False)

        # different strides
        tensor1 = torch.zeros(10, 10, device=device, dtype=dtype)
        tensor2 = torch.ones(10, 10, device=device, dtype=dtype)
        inputs = ([tensor1], [tensor2])
        self._regular_binary_test(dtype, foreach_op, native_op, inputs, False)
        self._inplace_binary_test(dtype, foreach_op_, native_op_, inputs, False)

        # non contiguous
        tensor1 = torch.randn(5, 2, 1, 3, device=device).to(dtype)[:, 0]
        tensor2 = torch.randn(5, 2, 1, 3, device=device).to(dtype)[:, 0]
        self.assertFalse(tensor1.is_contiguous())
        self.assertFalse(tensor2.is_contiguous())
        inputs = ([tensor1], [tensor2])
        self._regular_binary_test(dtype, foreach_op, native_op, inputs, False)
        self._inplace_binary_test(dtype, foreach_op_, native_op_, inputs, False)

        # sliced tensor
        tensor1 = torch.randn(5, 2, 1, 3, device=device).to(dtype)
        tensor2 = torch.randn(5, 2, 1, 3 * 7, device=device).to(dtype)[:, :, :, ::7]
        inputs = ([tensor1], [tensor2])
        self._regular_binary_test(dtype, foreach_op, native_op, inputs, False)
        self._inplace_binary_test(dtype, foreach_op_, native_op_, inputs, False)

    # note: Below three tests (postfixed with `_tensors_on_different_devices`)
    # checks whether foreach works with lists of tensors on different devices
    # but tensors of the same index are on the same device, e.g., ['cuda', 'cpu].
    @ops(foreach_unary_op_db)
    def test_unary_op_tensors_on_different_devices(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')
        method, ref, inplace_method, ref_inplace = self._get_funcs(op)
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

    @dtypes(*torch.testing.get_all_dtypes())
    @ops(foreach_binary_op_db)
    def test_binary_op_tensors_on_different_devices(self, device, dtype, op):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')
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

    @dtypes(*torch.testing.get_all_dtypes(include_bfloat16=True))
    def test_pointwise_op_tensors_on_different_devices(self, device, dtype):
        if self.device_type != 'cuda':
            self.skipTest('CUDA is necessary for tests with tensors on different devices')

        pointwise_ops = [
            (torch._foreach_addcmul, torch._foreach_addcmul_, torch.addcmul),
            (torch._foreach_addcdiv, torch._foreach_addcdiv_, torch.addcdiv),
        ]
        for foreach_op, foreach_op_, native_op in pointwise_ops:
            # tensors1: ['cuda', 'cpu]
            # tensors2: ['cuda', 'cpu]
            # tensors3: ['cuda', 'cpu]
            _cuda_tensors = self._get_test_data(device, dtype, 3)
            _cpu_tensors = self._get_test_data('cpu', dtype, 3)
            tensors1, tensors2, tensors3 = list(tensors for tensors in zip(_cuda_tensors, _cpu_tensors))

            try:
                actual = foreach_op(tensors1, tensors2, tensors3)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    expected = [native_op(t1, t2, t3) for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]
            else:
                expected = [native_op(t1, t2, t3) for t1, t2, t3 in zip(tensors1, tensors2, tensors3)]
                self.assertEqual(expected, actual)
            try:
                foreach_op_(tensors1, tensors2, tensors3)
            except RuntimeError as e:
                with self.assertRaisesRegex(type(e), re.escape(str(e))):
                    [getattr(t1, native_op.__name__ + '_')(t2, t3) for t1, t2, t3 in zip(tensors1, tensors3, tensors3)]
            else:
                self.assertEqual(expected, tensors1)


instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
