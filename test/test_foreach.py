# Owner(s): ["module: mta"]

from numbers import Number
import re
import torch
import unittest

from torch.testing import make_tensor
from torch.testing._comparison import default_tolerances
from torch.testing._internal.common_utils import \
    TestCase, run_tests, TEST_WITH_ROCM, skipIfTorchDynamo, parametrize
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, onlyCUDA, skipMeta, ops, OpDTypes)
from torch.testing._internal.common_methods_invocations import (
    foreach_unary_op_db, foreach_binary_op_db, foreach_pointwise_op_db,
    foreach_reduce_op_db, foreach_lerp_op_db)
from torch.testing._internal.common_dtype import (
    all_types_and_complex_and, integral_types, complex_types,
    floating_types_and, floating_types, integral_types_and,
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

    def __init__(self, func):
        self.func = func
        # Some foreach functions don't have in-place implementations.
        self._is_inplace = False if func is None else func.__name__.endswith('_')

    def __call__(self, inputs, is_cuda, is_fastpath, **kwargs):
        actual = None
        if (
            is_cuda and
            torch.autograd.kineto_available() and
            torch.profiler.ProfilerActivity.CUDA in torch.profiler.supported_activities()
        ):
            with torch.profiler.profile() as p:
                actual = self.func(*inputs, **kwargs)
            keys = tuple([e.key for e in p.key_averages()])
            mta_called = any("multi_tensor_apply_kernel" in k for k in keys)
            assert mta_called == is_fastpath
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
    def _get_funcs(self, op):
        return (
            ForeachFuncWrapper(op.method_variant),
            RegularFuncWrapper(op.ref),
            ForeachFuncWrapper(op.inplace_variant),
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

    @skipMeta
    @ops(foreach_binary_op_db)
    @parametrize("is_fastpath", (True, False))
    def test_binary_op(self, device, dtype, op, is_fastpath):
        for sample in op.sample_inputs(device, dtype, noncontiguous=not is_fastpath):
            rhs_arg, = sample.args
            kwargs = {} or sample.kwargs
            alpha = kwargs.pop("alpha", None)
            disable_fastpath = kwargs.pop("disable_fastpath") if is_fastpath else False

            wrapped_op, ref, inplace_op, inplace_ref = self._get_funcs(op)
            self._binary_test(
                dtype, wrapped_op, ref, [sample.input, rhs_arg], is_fastpath and not disable_fastpath, False, alpha=alpha)
            self._binary_test(
                dtype, inplace_op, inplace_ref, [sample.input, rhs_arg], is_fastpath and not disable_fastpath, True, alpha=alpha)

    @ops(foreach_pointwise_op_db)
    @parametrize("is_fastpath", (True, False))
    def test_pointwise_op(self, device, dtype, op, is_fastpath):
        for sample in op.sample_inputs(device, dtype):
            if not is_fastpath:
                sample = sample.noncontiguous()
            assert isinstance(sample.args, tuple)
            assert len(sample.args) == 2
            inputs = [sample.input, *sample.args]
            kwargs = sample.kwargs
            disable_fastpath = kwargs.pop("disable_fastpath") if is_fastpath else False
            wrapped_op, ref, inplace_op, inplace_ref = self._get_funcs(op)
            values = kwargs.pop("values")
            self._pointwise_test(wrapped_op, ref, inputs, is_fastpath and not disable_fastpath, False, values=values)
            self._pointwise_test(inplace_op, inplace_ref, inputs, is_fastpath and not disable_fastpath, True, values=values)

            if is_fastpath and isinstance(values, list):
                sample = sample.transform(lambda t: t.clone().detach() if torch.is_tensor(t) else t)
                inputs = [sample.input, *sample.args]
                tensor_values = torch.tensor(values)
                # 1D Tensor of scalars
                for is_inplace, op_, ref_ in ((False, wrapped_op, ref), (True, inplace_op, inplace_ref)):
                    self._pointwise_test(op_, ref_, inputs, is_fastpath and not disable_fastpath, is_inplace, values=tensor_values)
                    self._pointwise_test(
                        op_, ref_, inputs, is_fastpath and not disable_fastpath, is_inplace, values=tensor_values[0],
                        custom_values_err="Expected packed scalar Tensor to be of dimension 1. Got 0 instead.")
                    if self.is_cuda:
                        self._pointwise_test(
                            op_, ref_, inputs, is_fastpath and not disable_fastpath, is_inplace, values=tensor_values.cuda(),
                            custom_values_err="Expected scalars to be on CPU, got cuda:0 instead.")
                    self._pointwise_test(
                        op_, ref_, inputs, is_fastpath and not disable_fastpath, is_inplace, values=tensor_values[:2],
                        custom_values_err=f"Expected length of scalars to match input of length {len(values)} but got 2 instead.")
                    self._pointwise_test(
                        op_, ref_, inputs, is_fastpath and not disable_fastpath, is_inplace,
                        values=torch.tensor([[0, 1], [2, 3]])[:, 1],
                        custom_values_err="Expected scalars to be contiguous.")

            # Tests of implicit broadcasting
            N = len(sample.input)
            inputs = [
                [
                    make_tensor((N, N), device=device, dtype=dtype, noncontiguous=not is_fastpath) for _ in range(N)
                ],
                [
                    make_tensor((N - i, 1), device=device, dtype=dtype, noncontiguous=not is_fastpath) for i in range(N)
                ],
                [
                    make_tensor((1, N - i), device=device, dtype=dtype, noncontiguous=not is_fastpath) for i in range(N)
                ],
            ]
            self._pointwise_test(
                wrapped_op, ref, inputs, is_fastpath and disable_fastpath, is_inplace=False, values=values)
            self._pointwise_test(
                inplace_op, inplace_ref, inputs, is_fastpath and disable_fastpath, is_inplace=True, values=values)

    def _pointwise_test(self, op, ref, inputs, is_fastpath, is_inplace, *, values=None, custom_values_err=None):
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

    # note(mkozuki): why `try-except` for both fastpath?
    # - inputs for fastpath can be integer tensors.
    #    - this is because opinfo dtypes are configured for out-place implementation
    # - for integer inputs, trigonometric functions and exponential function returns float outputs,
    #   which causes "result type Float can't be case to the desired type" error.
    # Thus, `try-except` is used even if `is_fastpath` is `True`.
    def _inplace_unary_test(self, inplace, inplace_ref, inputs, is_fastpath):
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

        if opinfo.supports_autograd and dtype in floating_types():
            tensors = opinfo.sample_inputs(device, dtype, N, noncontiguous=not is_fastpath, same_size=True)
            tensors = [t.requires_grad_() for t in tensors]
            ref_tensors = [t.clone().detach().requires_grad_() for t in tensors]

            sum(op.func(tensors)).mean().backward()
            sum([ref.func(t) for t in ref_tensors]).mean().backward()
            self.assertEqual([t.grad for t in tensors], [t.grad for t in ref_tensors])

    @skipMeta
    @ops(foreach_unary_op_db)
    @parametrize("is_fastpath", (True, False))
    def test_unary_op(self, device, dtype, op, is_fastpath):
        wrapped_op, ref, inplace_op, inplace_ref = self._get_funcs(op)
        samples = op.sample_inputs(device, dtype)
        disable_fastpath = op.name == "_foreach_abs" and dtype in complex_types()
        for sample in samples:
            if not is_fastpath:
                sample = sample.noncontiguous()
            inputs = [sample.input]
            disable_fastpath = (
                (op.name == "_foreach_abs" and dtype in complex_types()) or
                sample.kwargs.pop("disable_fastpath")
            )
            self.assertEqual(ref(inputs), wrapped_op(inputs, self.is_cuda, is_fastpath and not disable_fastpath))
            self._inplace_unary_test(inplace_op, inplace_ref, [sample.input], is_fastpath and not disable_fastpath)

    @ops(foreach_reduce_op_db)
    @parametrize("is_fastpath", (True, False))
    def test_reduce_op(self, device, dtype, op, is_fastpath):
        for sample in op.sample_inputs(device, dtype):
            if not is_fastpath:
                sample = sample.noncontiguous()
            ord = sample.kwargs.pop("ord")
            disable_fastpath = sample.kwargs.pop("disable_fastpath", False)

            inputs = (sample.input,)
            wrapped_op, ref, _, _ = self._get_funcs(op)
            self.assertEqual(ref(inputs, ord=ord), wrapped_op(inputs, self.is_cuda, is_fastpath and not disable_fastpath, ord=ord))

    @dtypes(*all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool))
    def test_add_scalar_with_empty_list_and_empty_tensor(self, device, dtype):
        # TODO: enable empty list case
        for tensors in [[torch.randn([0])]]:
            res = torch._foreach_add(tensors, 1)
            self.assertEqual(res, tensors)

            torch._foreach_add_(tensors, 1)
            self.assertEqual(res, tensors)

    @ops(foreach_binary_op_db, dtypes=OpDTypes.supported)
    def test_binary_op_scalar_with_overlapping_tensors(self, device, dtype, op):
        print(op, device, dtype)
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
    @ops(foreach_binary_op_db, dtypes=OpDTypes.supported)
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
    @ops(foreach_binary_op_db, dtypes=OpDTypes.supported)
    def test_binary_op_list_slow_path(self, device, dtype, op):
        foreach_op, native_op, foreach_op_, native_op_ = self._get_funcs(op)
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

    @ops(foreach_binary_op_db, dtypes=floating_types_and(torch.half, torch.bfloat16))
    def test_binary_op_float_inf_nan(self, device, dtype, op):
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
        op, ref, inplace_op, inplace_ref = self._get_funcs(op)
        self._binary_test(dtype, op, ref, inputs, True, False)
        self._binary_test(dtype, inplace_op, inplace_ref, inputs, True, True)

    # note: Below three tests (postfixed with `_tensors_on_different_devices`)
    # checks whether foreach works with lists of tensors on different devices
    # but tensors of the same index are on the same device, e.g., ['cuda', 'cpu].
    @onlyCUDA
    @ops(foreach_unary_op_db)
    def test_unary_op_tensors_on_different_devices(self, device, dtype, op):
        method, ref, inplace_method, ref_inplace = self._get_funcs(op)
        # tensors: ['cuda', 'cpu]
        tensors = list(op.sample_inputs(device, dtype, num_input_tensors=[2]))[0].input
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
        _cuda_tensors = list(op.sample_inputs(device, dtype, num_input_tensors=[2], same_size=True))[0].input
        _cpu_tensors = list(op.sample_inputs("cpu", dtype, num_input_tensors=[2], same_size=True))[0].input
        tensors1, tensors2 = list(zip(_cuda_tensors, _cpu_tensors))

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
        _cuda_tensors = list(op.sample_inputs(device, dtype, num_input_tensors=[3], same_size=True))[0].input
        _cpu_tensors = list(op.sample_inputs("cpu", dtype, num_input_tensors=[3], same_size=True))[0].input
        tensors1, tensors2, tensors3 = list(zip(_cuda_tensors, _cpu_tensors))

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
        inputs = [t * scaler for t in list(op.sample_inputs(device, dtype, [N], low=1))[0].input],
        # make sure that the min. of squared L2 norm value per tensor is greater than the max value of `dtype`.
        self.assertTrue(scaler * scaler * N > max_value)
        fn, ref_fn, *_ = self._get_funcs(op)
        actual = fn(inputs, is_cuda=True, is_fastpath=True, ord=ord)
        expect = ref_fn(inputs, ord=ord)
        if dtype == torch.float16:
            # making sure the reference L2 norm values are in the range of FP16.
            self.assertFalse(any(torch.isinf(e) for e in expect))
        else:
            self.assertTrue(all(torch.isinf(e) for e in expect))
        self.assertEqual(expect, actual, equal_nan=False)

    @parametrize("is_fastpath", (True, False))
    @ops(foreach_lerp_op_db)
    def test_lerp(self, device, dtype, op, is_fastpath):
        for sample in op.sample_inputs(device, dtype, noncontiguous=not is_fastpath):
            wrapped_op, ref, inplace_op, _ = self._get_funcs(op)

            args = [*sample.args]
            inputs = [sample.input, args[0]]

            kwargs, ref_kwargs = {}, {}
            if isinstance(args[1], list):
                inputs.append(args[1])
            else:
                kwargs = ref_kwargs = {"weight": args[1]}

            if (
                dtype in integral_types() or
                dtype == torch.bool or
                (not self.is_cuda and dtype == torch.half)
            ):
                with self.assertRaises(RuntimeError):
                    wrapped_op(inputs, self.is_cuda, is_fastpath, **kwargs)
                return
            actual = wrapped_op(inputs, self.is_cuda, is_fastpath, **kwargs)
            expected = ref(inputs, **ref_kwargs)
            self.assertEqual(actual, expected)

            inplace_inputs = [[t.clone() for t in inputs[0]]] + inputs[1:]
            inplace_actual = inplace_op(inplace_inputs, self.is_cuda, is_fastpath, **kwargs)
            self.assertEqual(inplace_actual, expected)

    @onlyCUDA
    @ops(foreach_reduce_op_db)
    def test_foreach_reduce_large_input(self, device, dtype, op):
        # test inputs larger than kChunkSize = 65536
        ord, N = 2, 65536 * 2
        disable_fastpath = True
        if ord in (1, 2) and dtype in floating_types_and(torch.half, torch.bfloat16):
            disable_fastpath = False
        inputs = [make_tensor((N,), dtype=dtype, device=device, noncontiguous=False)],
        wrapped_op, ref, _, _ = self._get_funcs(op)
        self.assertEqual(ref(inputs, ord=ord), wrapped_op(inputs, self.is_cuda, not disable_fastpath, ord=ord))


instantiate_device_type_tests(TestForeach, globals())

if __name__ == '__main__':
    run_tests()
