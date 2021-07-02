import itertools
from numbers import Number
import random
import re
import torch
import unittest
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_ROCM, TEST_WITH_SLOW
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, onlyCUDA, skipCUDAIfRocm, skipMeta, ops)
from torch._six import inf, nan
from torch.testing._internal.common_methods_invocations import foreach_unary_op_db, foreach_binary_op_db, make_tensor

# Includes some values such that N * N won't be a multiple of 4,
# which should ensure we test the vectorized and non-vectorized
# kernel code paths.
N_values = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]

_BOOL_SUB_ERR_MSG = "Subtraction, the `-` operator"

class RegularFuncWrapper:

    def __init__(self, func):
        self.func = func

    def __call__(self, inputs, **kwargs):
        if len(inputs) == 2 and isinstance(inputs[1], Number):
            # binary op with tensorlist and scalar.
            inputs[1] = [inputs[1] for _ in range(len(inputs[0]))]
        return [self.func(*i, **kwargs) for i in zip(*inputs)]


class ForeachFuncWrapper:

    def __init__(self, func, n_expected_cudaLaunchKernels):
        self.func = func
        self.n_expected_cudaLaunchKernels = n_expected_cudaLaunchKernels
        self._is_inplace = func.__name__.endswith('_')

    def __call__(self, inputs, is_cuda, is_fastpath, **kwargs):
        actual = None
        if is_cuda and torch.autograd.kineto_available():
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
    def _get_funcs(self, op, n_expected_cudaLaunchKernels):
        return (
            ForeachFuncWrapper(op.method_variant, n_expected_cudaLaunchKernels),
            RegularFuncWrapper(op.ref),
            ForeachFuncWrapper(op.inplace_variant, n_expected_cudaLaunchKernels),
            RegularFuncWrapper(op.ref_inplace),
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
                    self.assertEqual(expected, actual, atol=1.e-3, rtol=self.dtype_precisions[dtype][0])
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
            if dtype in torch.testing.get_all_int_dtypes():
                alpha = 3
            elif dtype.is_complex:
                dtype = complex(3, 3)
            else:
                alpha = 3.14
            self._binary_test(dtype, op, ref, inputs, is_fastpath, is_inplace=False, alpha=alpha)
            self._binary_test(dtype, inplace_op, inplace_ref, inputs, is_fastpath, is_inplace=True, alpha=alpha)

    # note(mkozuki): Why ROCm?
    # ROCm is supposed to compile slow path as in
    # https://github.com/pytorch/pytorch/blob/7e032f18cf1405804c4f787b05ea2de5e08a091e/aten/src/ATen/native/ForeachUtils.h#L148-L164,  # noqa: E501
    # Therefore `[torch.add(*args, alpha=alpha) for args in zip(tensors1, tensors2)]` and
    # `torch._foreach_add(tensors1, tensors2, alpha=alpha)`
    # are expected to return the same outputs, however, the outputs look unstable for torch.bfloat16 and torch.half.
    # log: https://ci.pytorch.org/jenkins/job/pytorch-builds/job/pytorch-linux-bionic-rocm4.2-py3.6-test1/2741/console
    @skipCUDAIfRocm
    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_op_tensorlists_fastpath(self, device, dtype, op):
        for N in N_values:
            disable_fastpath = op.ref == torch.div and dtype in torch.testing.get_all_int_dtypes() + [torch.bool]
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

    @skipCUDAIfRocm
    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_op_scalar_fastpath(self, device, dtype, op):
        scalars = (
            random.randint(1, 10),
            1.0 - random.random(),
            True,
            complex(1.0 - random.random(), 1.0 - random.random()),
        )
        for N, scalar in itertools.product(N_values, scalars):
            disable_fastpath = op.ref == torch.div and dtype in torch.testing.get_all_int_dtypes() + [torch.bool]
            if isinstance(scalar, int):
                disable_fastpath |= dtype == torch.bool
            if isinstance(scalar, float):
                disable_fastpath |= dtype in torch.testing.get_all_int_dtypes() + [torch.bool]
            if isinstance(scalar, bool):
                disable_fastpath |= dtype == torch.bool
                if op.ref in (torch.add, torch.mul):
                    disable_fastpath = False
            if isinstance(scalar, complex):
                disable_fastpath |= dtype not in torch.testing.get_all_complex_dtypes()
            self._test_binary_op_scalar(device, dtype, op, N, scalar, True, disable_fastpath)

    @ops(foreach_binary_op_db)
    def test_binary_op_scalar_slowpath(self, device, dtype, op):
        scalars = (
            random.randint(1, 10),
            1.0 - random.random(),
            True,
            complex(1.0 - random.random(), 1.0 - random.random()),
        )
        for N, scalar in itertools.product(N_values, scalars):
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
    @skipCUDAIfRocm
    @skipMeta
    @ops(foreach_binary_op_db)
    def test_binary_op_scalarlist_fastpath(self, device, dtype, op):
        for N in N_values:
            for type_str, scalarlist in (
                ("int", [random.randint(0, 9) + 1 for _ in range(N)]),
                ("float", [1.0 - random.random() for _ in range(N)]),
                ("complex", [complex(1.0 - random.random(), 1.0 - random.random()) for _ in range(N)]),
                ("bool", [True for _ in range(N)]),
                ("mixed", [1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(N - 3)]),
                ("mixed", [True, 1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(N - 4)]),
            ):
                bool_int_div = op.ref == torch.div and dtype in torch.testing.get_all_int_dtypes() + [torch.bool]
                disable_fastpath = bool_int_div
                if type_str == "int":
                    disable_fastpath |= dtype == torch.bool
                if type_str == "float":
                    disable_fastpath |= dtype in torch.testing.get_all_int_dtypes() + [torch.bool]
                if type_str == "complex":
                    disable_fastpath |= dtype not in torch.testing.get_all_complex_dtypes()
                if type_str == "mixed":
                    disable_fastpath |= True and dtype not in torch.testing.get_all_complex_dtypes()
                self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, True, disable_fastpath)

    @ops(foreach_binary_op_db)
    def test_binary_op_scalarlist_slowpath(self, device, dtype, op):
        for N in N_values:
            for scalarlist in [
                [random.randint(0, 9) + 1 for _ in range(N)],
                [1.0 - random.random() for _ in range(N)],
                [complex(1.0 - random.random(), 1.0 - random.random()) for _ in range(N)],
                [True for _ in range(N)],
                [1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(N - 3)],
                [True, 1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(N - 4)]
            ]:
                self._test_binary_op_scalarlist(device, dtype, op, N, scalarlist, False, False)

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
        op, ref, inplace_op, inplace_ref = self._get_funcs(opinfo, 1)
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
