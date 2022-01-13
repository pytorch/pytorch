# Owner(s): ["high priority"]

from collections.abc import Sequence
from functools import partial, wraps
import warnings
import unittest
import itertools

import torch

from torch.testing import FileCheck, make_tensor
from torch.testing._internal.common_dtype import floating_and_complex_types_and, get_all_dtypes
from torch.testing._internal.common_utils import \
    (TestCase, is_iterable_of_tensors, run_tests, IS_SANDCASTLE, clone_input_helper,
     gradcheck, gradgradcheck, IS_IN_CI, suppress_warnings, noncontiguous_like,
     TEST_WITH_ASAN, IS_WINDOWS, IS_FBCODE, first_sample)
from torch.testing._internal.common_methods_invocations import \
    (op_db, _NOTHING, UnaryUfuncInfo, ReductionOpInfo, SpectralFuncInfo)
from torch.testing._internal.common_device_type import \
    (deviceCountAtLeast, instantiate_device_type_tests, ops, onlyCPU,
     onlyCUDA, onlyNativeDeviceTypes, skipCUDAIfRocm, OpDTypes, skipMeta)
from torch.testing._internal.common_jit import JitCommonTestCase, check_against_reference
from torch.testing._internal.jit_metaprogramming_utils import create_script_fn, create_traced_fn, \
    check_alias_annotation
from torch.testing._internal.jit_utils import disable_autodiff_subgraph_inlining, is_lambda
import torch.testing._internal.opinfo_helper as opinfo_helper
from torch.testing._internal.composite_compliance import _check_composite_compliance

# TODO: fixme https://github.com/pytorch/pytorch/issues/68972
torch.set_default_dtype(torch.float32)

# variant testing is only done with torch.float and torch.cfloat to avoid
#   excessive test times and maximize signal to noise ratio
_variant_ops = partial(ops, dtypes=OpDTypes.supported,
                       allowed_dtypes=(torch.float, torch.cfloat))

# Get names of all the operators which have ref in their entry in OpInfo (testing infra)
#   except for Unary Ufuncs (separately implemented in test/test_unary_ufuncs.py)
#   and Spectral Functions (separately implemented for only 1D as of now, in test/test_spectral_ops.py)
_ref_test_ops = list(filter(lambda op: not isinstance(op, (UnaryUfuncInfo, ReductionOpInfo,
                     SpectralFuncInfo)) and op.ref is not None and op.ref is not _NOTHING, op_db))


# Tests that apply to all operators and aren't related to any particular
#   system
class TestCommon(TestCase):
    exact_dtype = True

    # Verifies, on teardown, that no OpInfo is still using dynamic dtypes in CI
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()

        if IS_IN_CI:
            err_msg = ("The operator(s) below is(are) using dynamic_dtypes in the OpInfo entries."
                       "This is OK for testing, but be sure to set the dtypes manually before landing your PR!")
            # Assure no opinfo entry has dynamic_dtypes
            filtered_ops = list(filter(opinfo_helper.is_dynamic_dtype_set, op_db))
            for op in filtered_ops:
                fmt_str = opinfo_helper.str_format_dynamic_dtype(op)
                err_msg += "\n" + fmt_str

            assert len(filtered_ops) == 0, err_msg

    # Validates that each OpInfo specifies its forward and backward dtypes
    #   correctly for CPU and CUDA devices
    @skipMeta
    @skipCUDAIfRocm
    @onlyNativeDeviceTypes
    @ops(op_db, dtypes=OpDTypes.none)
    def test_dtypes(self, device, op):
        # dtypes to try to backward in
        allowed_backward_dtypes = floating_and_complex_types_and(torch.bfloat16, torch.float16)

        # lists for (un)supported dtypes
        supported_dtypes = []
        unsupported_dtypes = []
        supported_backward_dtypes = []
        unsupported_backward_dtypes = []

        def unsupported(dtype):
            unsupported_dtypes.append(dtype)
            if dtype in allowed_backward_dtypes:
                unsupported_backward_dtypes.append(dtype)

        for dtype in get_all_dtypes():
            # tries to acquire samples - failure indicates lack of support
            requires_grad = (dtype in allowed_backward_dtypes and op.supports_autograd)
            try:
                samples = list(op.sample_inputs(device, dtype, requires_grad=requires_grad))
            except Exception as e:
                unsupported(dtype)
                continue

            # Counts number of successful backward attempts
            # NOTE: This exists as a kludge because this only understands how to
            #   request a gradient if the output is a tensor or a sequence with
            #   a tensor as its first element.
            num_backward_successes = 0
            for sample in samples:
                # tries to call operator with the sample - failure indicates
                #   lack of support
                try:
                    result = op(sample.input, *sample.args, **sample.kwargs)
                except Exception as e:
                    # NOTE: some ops will fail in forward if their inputs
                    #   require grad but they don't support computing the gradient
                    #   in that type! This is a bug in the op!
                    unsupported(dtype)

                # Short-circuits testing this dtype -- it doesn't work
                if dtype in unsupported_dtypes:
                    break

                # Short-circuits if the dtype isn't a backward dtype or
                #   it's already identified as not supported
                if dtype not in allowed_backward_dtypes or dtype in unsupported_backward_dtypes:
                    continue

                # Checks for backward support in the same dtype
                try:
                    result = sample.output_process_fn_grad(result)
                    if isinstance(result, torch.Tensor):
                        backward_tensor = result
                    elif isinstance(result, Sequence) and isinstance(result[0], torch.Tensor):
                        backward_tensor = result[0]
                    else:
                        continue

                    # Note: this grad may not have the same dtype as dtype
                    # For functions like complex (float -> complex) or abs
                    #   (complex -> float) the grad tensor will have a
                    #   different dtype than the input.
                    #   For simplicity, this is still modeled as these ops
                    #   supporting grad in the input dtype.
                    grad = torch.randn_like(backward_tensor)
                    backward_tensor.backward(grad)
                    num_backward_successes += 1
                except Exception as e:
                    unsupported_backward_dtypes.append(dtype)

            if dtype not in unsupported_dtypes:
                supported_dtypes.append(dtype)
            if num_backward_successes > 0 and dtype not in unsupported_backward_dtypes:
                supported_backward_dtypes.append(dtype)

        # Checks that dtypes are listed correctly and generates an informative
        #   error message
        device_type = torch.device(device).type
        claimed_supported = set(op.supported_dtypes(device_type))
        supported_dtypes = set(supported_dtypes)

        supported_but_unclaimed = supported_dtypes - claimed_supported
        claimed_but_unsupported = claimed_supported - supported_dtypes
        msg = """The supported dtypes for {0} on {1} according to its OpInfo are
        {2}, but the detected supported dtypes are {3}.
        """.format(op.name, device_type, claimed_supported, supported_dtypes)

        if len(supported_but_unclaimed) > 0:
            msg += "The following dtypes should be added to the OpInfo: {0}. ".format(supported_but_unclaimed)
        if len(claimed_but_unsupported) > 0:
            msg += "The following dtypes should be removed from the OpInfo: {0}.".format(claimed_but_unsupported)

        self.assertEqual(supported_dtypes, claimed_supported, msg=msg)

        # Checks that backward dtypes are listed correctly and generates an
        #   informative error message
        # NOTE: this code is nearly identical to the check + msg generation
        claimed_backward_supported = set(op.supported_backward_dtypes(device_type))
        supported_backward_dtypes = set(supported_backward_dtypes)

        supported_but_unclaimed = supported_backward_dtypes - claimed_backward_supported
        claimed_but_unsupported = claimed_backward_supported - supported_backward_dtypes
        msg = """The supported backward dtypes for {0} on {1} according to its OpInfo are
        {2}, but the detected supported backward dtypes are {3}.
        """.format(op.name, device_type, claimed_backward_supported, supported_backward_dtypes)

        if len(supported_but_unclaimed) > 0:
            msg += "The following backward dtypes should be added to the OpInfo: {0}. ".format(supported_but_unclaimed)
        if len(claimed_but_unsupported) > 0:
            msg += "The following backward dtypes should be removed from the OpInfo: {0}.".format(claimed_but_unsupported)

        self.assertEqual(supported_backward_dtypes, claimed_backward_supported, msg=msg)

    # Validates that each OpInfo works correctly on different CUDA devices
    @skipCUDAIfRocm
    @onlyCUDA
    @deviceCountAtLeast(2)
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long))
    def test_multiple_devices(self, devices, dtype, op):
        for cuda_device_str in devices:
            cuda_device = torch.device(cuda_device_str)
            # NOTE: only tests on first sample
            samples = op.sample_inputs(cuda_device, dtype)
            sample = first_sample(self, samples)
            result = op(sample.input, *sample.args, **sample.kwargs)

            if isinstance(result, torch.Tensor):
                self.assertTrue(result.device == cuda_device)
            elif is_iterable_of_tensors(result):
                self.assertTrue(all(map(lambda t: t.device == cuda_device, result)))
            else:
                self.skipTest("Skipped! Only supports single tensor or iterable of tensor outputs.")

    # Tests that the function and its (ndarray-accepting) reference produce the same
    #   values on the tensors from sample_inputs func for the corresponding op.
    # This test runs in double and complex double precision because
    # NumPy does computation internally using double precision for many functions
    # resulting in possible equality check failures.
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(_ref_test_ops, allowed_dtypes=(torch.float64, torch.long, torch.complex128))
    def test_reference_testing(self, device, dtype, op):
        try:
            # Sets the default dtype to NumPy's default dtype of double
            cur_default = torch.get_default_dtype()
            torch.set_default_dtype(torch.double)
            sample_inputs = op.sample_inputs(device, dtype)
            for sample_input in sample_inputs:
                self.compare_with_reference(op, op.ref, sample_input, exact_dtype=(dtype is not torch.long))
        finally:
            torch.set_default_dtype(cur_default)

    @skipMeta
    @onlyNativeDeviceTypes
    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    def test_errors(self, device, op):
        error_inputs = op.error_inputs(device)
        for ei in error_inputs:
            si = ei.sample_input
            with self.assertRaisesRegex(ei.error_type, ei.error_regex):
                op(si.input, *si.args, **si.kwargs)

    # Tests that the function produces the same result when called with
    #   noncontiguous tensors.
    # TODO: get working with Windows by addressing failing operators
    # TODO: get working with ASAN by addressing failing operators
    @unittest.skipIf(IS_WINDOWS, "Skipped under Windows")
    @unittest.skipIf(TEST_WITH_ASAN, "Skipped under ASAN")
    @onlyNativeDeviceTypes
    @suppress_warnings
    @ops(op_db, allowed_dtypes=(torch.float32, torch.long, torch.complex64))
    def test_noncontiguous_samples(self, device, dtype, op):
        test_grad = dtype in op.supported_backward_dtypes(torch.device(device).type)
        sample_inputs = op.sample_inputs(device, dtype, requires_grad=test_grad)
        for sample_input in sample_inputs:
            t_inp, t_args, t_kwargs = sample_input.input, sample_input.args, sample_input.kwargs
            n_inp, n_args, n_kwargs = sample_input.noncontiguous()

            # Verifies sample input tensors should have no grad or history
            sample_tensor = t_inp if isinstance(t_inp, torch.Tensor) else t_inp[0]
            assert sample_tensor.grad is None
            assert sample_tensor.grad_fn is None

            # validates forward
            expected = op(t_inp, *t_args, **t_kwargs)
            actual = op(n_inp, *n_args, **n_kwargs)

            self.assertEqual(actual, expected)

            # validates backward
            # NOTE: only handles single tensor outputs and the first tensor
            #   of ops that output a sequence

            # Short-circuits if the op doesn't support grad in this device x dtype
            if not test_grad:
                continue

            if isinstance(expected, torch.Tensor):
                expected_backward_tensor = expected
                actual_backward_tensor = actual
            elif isinstance(expected, Sequence) and isinstance(expected[0], torch.Tensor):
                expected_backward_tensor = expected[0]
                actual_backward_tensor = actual[0]
            else:
                continue

            grad_for_expected = torch.randn_like(expected_backward_tensor)
            grad_for_actual = noncontiguous_like(grad_for_expected)
            expected_backward_tensor.backward(grad_for_expected)
            actual_backward_tensor.backward(grad_for_actual)

            # Acquires grad (which may be on the first element in a list)
            expected_grad = t_inp.grad if isinstance(t_inp, torch.Tensor) else t_inp[0].grad
            actual_grad = n_inp.grad if isinstance(n_inp, torch.Tensor) else n_inp[0].grad

            # TODO: FIXME: only validates grad on first tensor input
            self.assertEqual(actual_grad, expected_grad)

    # Separates one case from the following test_out because many ops don't properly implement the
    #   incorrectly sized out parameter warning properly yet
    # Cases test here:
    #   - out= with the correct dtype and device, but the wrong shape
    @ops(op_db, dtypes=OpDTypes.none)
    def test_out_warning(self, device, op):
        # TODO: verify the op doesn't support the out= kwarg
        if not op.supports_out:
            self.skipTest("Skipped! Op doesn't support out= kwarg.")

        # Prefers running in float32 but has a fallback for the first listed supported dtype
        supported_dtypes = op.supported_dtypes(self.device_type)
        if len(supported_dtypes) == 0:
            self.skipTest("Skipped! Op has not supported dtypes on this device.")
        dtype = torch.float32 if torch.float32 in supported_dtypes else list(supported_dtypes)[0]

        # NOTE: only tests on first sample
        samples = op.sample_inputs(device, dtype)
        sample = first_sample(self, samples)

        # calls it normally to get the expected result
        expected = op(sample.input, *sample.args, **sample.kwargs)
        op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

        # Short-circuits if output is not a single tensor or an
        #   iterable of tensors

        if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(expected, include_empty=True):
            self.skipTest("Skipped! Only supports single tensor or iterable of tensor outputs.")

        # A wrapper around map that works with single tensors and always
        #   instantiates the map. Used below to apply transforms to
        #   single tensor and iterable tensor outputs.
        def _apply_out_transform(fn, out):
            if isinstance(out, torch.Tensor):
                return fn(out)

            # assumes (see above) that out is an iterable of tensors
            return tuple(map(fn, out))

        # Extracts strides from a tensor or iterable of tensors into a tuple
        def _extract_strides(out):
            if isinstance(out, torch.Tensor):
                return (out.stride(),)

            # assumes (see above) that out is an iterable of tensors
            return tuple(map(lambda t: t.stride(), out))

        # Extracts data pointers from a tensor or iterable of tensors into a tuple
        # NOTE: only extracts on the CPU and CUDA device types since some
        #   device types don't have storage
        def _extract_data_ptrs(out):
            if self.device_type != 'cpu' and self.device_type != 'cuda':
                return ()

            if isinstance(out, torch.Tensor):
                return (out.data_ptr(),)

            # assumes (see above) that out is an iterable of tensors
            return tuple(map(lambda t: t.data_ptr(), out))

        def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
            out = _apply_out_transform(transform, expected)
            original_strides = _extract_strides(out)
            original_ptrs = _extract_data_ptrs(out)

            op_out(out=out)
            final_strides = _extract_strides(out)
            final_ptrs = _extract_data_ptrs(out)

            self.assertEqual(expected, out)

            if compare_strides_and_data_ptrs:
                self.assertEqual(original_strides, final_strides)
                self.assertEqual(original_ptrs, final_ptrs)

        # Case: out= with the correct dtype and device, but the wrong shape
        #   Expected behavior: resize with a warning.
        def _case_two_transform(t):
            wrong_shape = list(t.shape)

            if len(wrong_shape) == 0:
                # Handles scalar tensor case (empty list)
                wrong_shape = [2]
            else:
                wrong_shape[-1] = wrong_shape[-1] + 1
            return make_tensor(wrong_shape, dtype=t.dtype, device=t.device)

        _compare_out(_case_two_transform, compare_strides_and_data_ptrs=False)

        # Additional validates that the appropriate warning is thrown
        out = _apply_out_transform(_case_two_transform, expected)
        msg_fail = "Resized a non-empty tensor but did not warn about it."
        with self.assertWarnsRegex(UserWarning, "An output with one or more elements", msg=msg_fail):
            op_out(out=out)

    # Validates ops implement the correct out= behavior
    # See https://github.com/pytorch/pytorch/wiki/Developer-FAQ#how-does-out-work-in-pytorch
    #   for a description of the correct behavior
    # Validates the following cases:
    #   - Case 0: out has the correct shape, dtype, and device but is full of extremal values
    #   - Case 1: out has the correct shape, dtype, and device but is noncontiguous
    #   - Case 2: out has the correct dtype and device, but is zero elements
    #   - Case 3: out has the correct shape and dtype, but is on a different device type
    #   - Case 4: out has the with correct shape and device, but a dtype that cannot
    #       "safely" cast to
    @ops(op_db, dtypes=OpDTypes.none)
    def test_out(self, device, op):
        # TODO: verify the op doesn't support the out= kwarg
        if not op.supports_out:
            self.skipTest("Skipped! Op doesn't support out= kwarg.")

        # Prefers running in float32 but has a fallback for the first listed supported dtype
        supported_dtypes = op.supported_dtypes(self.device_type)
        if len(supported_dtypes) == 0:
            self.skipTest("Skipped! Op has not supported dtypes on this device.")
        dtype = torch.float32 if torch.float32 in supported_dtypes else list(supported_dtypes)[0]

        # NOTE: only tests on first sample
        samples = op.sample_inputs(device, dtype)
        sample = first_sample(self, samples)

        # calls it normally to get the expected result
        expected = op(sample.input, *sample.args, **sample.kwargs)
        op_out = partial(op, sample.input, *sample.args, **sample.kwargs)

        # Short-circuits if output is not a single tensor or an
        #   iterable of tensors

        if not isinstance(expected, torch.Tensor) and not is_iterable_of_tensors(expected, include_empty=True):
            self.skipTest("Skipped! Only supports single tensor or iterable of tensor outputs.")

        # A wrapper around map that works with single tensors and always
        #   instantiates the map. Used below to apply transforms to
        #   single tensor and iterable tensor outputs.
        def _apply_out_transform(fn, out):
            if isinstance(out, torch.Tensor):
                return fn(out)

            # assumes (see above) that out is an iterable of tensors
            return tuple(map(fn, out))

        # Extracts strides from a tensor or iterable of tensors into a tuple
        def _extract_strides(out):
            if isinstance(out, torch.Tensor):
                return (out.stride(),)

            # assumes (see above) that out is an iterable of tensors
            return tuple(map(lambda t: t.stride(), out))

        # Extracts data pointers from a tensor or iterable of tensors into a tuple
        # NOTE: only extracts on the CPU and CUDA device types since some
        #   device types don't have storage
        def _extract_data_ptrs(out):
            if self.device_type != 'cpu' and self.device_type != 'cuda':
                return ()

            if isinstance(out, torch.Tensor):
                return (out.data_ptr(),)

            # assumes (see above) that out is an iterable of tensors
            return tuple(map(lambda t: t.data_ptr(), out))

        def _compare_out(transform, *, compare_strides_and_data_ptrs=True):
            out = _apply_out_transform(transform, expected)
            original_strides = _extract_strides(out)
            original_ptrs = _extract_data_ptrs(out)

            op_out(out=out)
            final_strides = _extract_strides(out)
            final_ptrs = _extract_data_ptrs(out)

            self.assertEqual(expected, out)

            if compare_strides_and_data_ptrs:
                self.assertEqual(original_strides, final_strides)
                self.assertEqual(original_ptrs, final_ptrs)

        # Case 0: out= with the correct shape, dtype, and device
        #   but NaN values for floating point and complex tensors, and
        #   maximum values for integer tensors.
        #   Expected behavior: out= values have no effect on the computation.
        def _case_zero_transform(t):
            try:
                info = torch.iinfo(t.dtype)
                return torch.full_like(t, info.max)
            except TypeError as te:
                # for non-integer types fills with NaN
                return torch.full_like(t, float('nan'))

        _compare_out(_case_zero_transform)

        # Case 1: out= with the correct shape, dtype, and device,
        #   but noncontiguous.
        #   Expected behavior: strides are respected and `out` storage is not changed.
        def _case_one_transform(t):
            return make_tensor(t.shape,
                               dtype=t.dtype,
                               device=t.device,
                               noncontiguous=True)

        _compare_out(_case_one_transform)

        # Case 2: out= with the correct dtype and device, but has no elements.
        #   Expected behavior: resize without warning.
        def _case_two_transform(t):
            return make_tensor((0,),
                               dtype=t.dtype,
                               device=t.device)

        _compare_out(_case_two_transform, compare_strides_and_data_ptrs=False)

        # Also validates that no warning is thrown when this out is resized
        out = _apply_out_transform(_case_two_transform, expected)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            op_out(out=out)

        # Verifies no warning is a resize warning
        for w in caught:
            if "An output with one or more elements" in str(w.message):
                self.fail("Resizing an out= argument with no elements threw a resize warning!")

        # Case 3: out= with correct shape and dtype, but wrong device.
        wrong_device = None
        if torch.device(device).type != 'cpu':
            wrong_device = 'cpu'
        elif torch.cuda.is_available():
            wrong_device = 'cuda'

        if wrong_device is not None:
            def _case_three_transform(t):
                return make_tensor(t.shape, dtype=t.dtype, device=wrong_device)

            out = _apply_out_transform(_case_three_transform, expected)
            msg_fail = f"Expected RuntimeError when calling with input.device={device} and out.device={wrong_device}"
            with self.assertRaises(RuntimeError, msg=msg_fail):
                op_out(out=out)

        # Case 4: out= with correct shape and device, but a dtype
        #   that output cannot be "safely" cast to (long).
        #   Expected behavior: error.
        # NOTE: this case is filtered by dtype since some ops produce
        #   bool tensors, for example, which can be safely cast to any
        #   dtype. It is applied when single tensors are floating point or complex
        #   dtypes, or if an op returns multiple tensors when at least one such
        #   tensor is a floating point or complex dtype.
        _dtypes = floating_and_complex_types_and(torch.float16, torch.bfloat16)
        if (isinstance(expected, torch.Tensor) and expected.dtype in _dtypes or
                (not isinstance(expected, torch.Tensor) and any(t.dtype in _dtypes for t in expected))):
            def _case_four_transform(t):
                return make_tensor(t.shape, dtype=torch.long, device=t.device)

            out = _apply_out_transform(_case_four_transform, expected)
            msg_fail = "" if not isinstance(expected, torch.Tensor) else \
                       ("Expected RuntimeError when doing an unsafe cast from a result of dtype "
                        f"{expected.dtype} into an out= with dtype torch.long")
            with self.assertRaises(RuntimeError, msg=msg_fail):
                op_out(out=out)

    # Tests that the forward and backward passes of operations produce the
    #   same values for the cross-product of op variants (method, inplace)
    #   against eager's gold standard op function variant
    @_variant_ops(op_db)
    def test_variant_consistency_eager(self, device, dtype, op):
        # Acquires variants (method variant, inplace variant, aliases)

        method = op.method_variant
        inplace = op.inplace_variant

        # list of all inplace ops: inplace variant + alias inplace variants if exist
        inplace_ops = [inplace, ]
        variants = [method, inplace]

        for a_op in op.aliases:
            variants.append(a_op.op)
            variants.append(a_op.method_variant)
            variants.append(a_op.inplace_variant)
            inplace_ops.append(a_op.inplace_variant)

        inplace_variants = tuple(filter(None, inplace_ops))
        variants = tuple(filter(None, variants))

        _requires_grad = (op.supports_autograd and
                          (dtype.is_floating_point or op.supports_complex_autograd(torch.device(device).type)))

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad, include_conjugated_inputs=include_conjugated_inputs)
        samples = list(samples)

        def _test_consistency_helper(samples, variants):
            for sample in samples:
                # TODO: Check grad for all Tensors requiring grad if sample.input is TensorList
                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]

                # Computes function forward and backward values
                tensor.grad = None
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                expected_grad = None

                output_process_fn_grad = sample.output_process_fn_grad if sample.output_process_fn_grad \
                    else lambda x: x

                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                skip_inplace = False
                if (isinstance(expected_forward, torch.Tensor) and
                        expected_forward.dtype is not tensor.dtype):
                    skip_inplace = True

                # TODO: backward consistency only supported for single tensor outputs
                # TODO: backward consistency only checked on sample.input, not all
                #   tensor inputs
                # TODO: update to handle checking grads of all tensor inputs as
                #   derived from each tensor output
                if (op.supports_autograd and isinstance(expected_forward, torch.Tensor)
                        and (dtype.is_floating_point or op.supports_complex_autograd(torch.device(device).type))):
                    output_process_fn_grad(expected_forward).sum().backward()
                    expected_grad = tensor.grad

                # Test eager consistency
                for variant in variants:
                    # Skips inplace ops
                    if variant in inplace_ops and skip_inplace:
                        continue

                    # Compares variant's forward
                    # Note: copies the to-be-modified input when testing the inplace variant
                    tensor.grad = None
                    cloned = clone_input_helper(sample.input) if variant in inplace_ops else sample.input

                    if variant in inplace_ops and sample.broadcasts_input:
                        with self.assertRaises(RuntimeError,
                                               msg=('inplace variant either incorrectly allowed '
                                                    'resizing or you have marked the sample {}'
                                                    ' incorrectly with `broadcasts_self=True'.format(sample.summary()))):
                            variant_forward = variant(cloned,
                                                      *sample.args,
                                                      **sample.kwargs)
                        continue

                    variant_forward = variant(cloned,
                                              *sample.args,
                                              **sample.kwargs)
                    self.assertEqual(expected_forward, variant_forward)

                    # Compares variant's backward
                    if expected_grad is not None and \
                            (variant not in inplace_ops or op.supports_inplace_autograd):
                        output_process_fn_grad(variant_forward).sum().backward()
                        self.assertEqual(expected_grad, tensor.grad)

        _test_consistency_helper(samples, variants)

        def _test_inplace_preserve_storage(samples, variants):
            for sample in samples:
                # Skips inplace variants if the output dtype is not the same as
                #   the input dtype
                expected_forward = op(sample.input, *sample.args, **sample.kwargs)
                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
                skip_inplace = False
                if (isinstance(expected_forward, torch.Tensor) and
                        expected_forward.dtype is not tensor.dtype):
                    skip_inplace = True
                if skip_inplace:
                    return
                for variant in variants:
                    cloned = clone_input_helper(sample.input) if variant in inplace_ops else sample.input
                    inp_tensor = cloned if isinstance(cloned, torch.Tensor) else cloned[0]
                    data_ptr = inp_tensor.data_ptr()
                    variant_forward = variant(cloned,
                                              *sample.args,
                                              **sample.kwargs)
                    # TODO Support non-tensor outputs if they exist for inplace ops
                    if (isinstance(variant_forward, torch.Tensor)):
                        self.assertEqual(data_ptr, variant_forward.data_ptr(), atol=0, rtol=0)
                    else:
                        self.assertTrue(False, "Non-tensor outputs for inplace ops are not supported")

        if len(inplace_ops) > 0:
            inplace_samples = list(filter(lambda sample: not sample.broadcasts_input, samples))
            _test_inplace_preserve_storage(inplace_samples, inplace_variants)

    # Checks if the operator (if it is composite) is written to support most
    # backends and Tensor subclasses. See "CompositeImplicitAutograd Compliance"
    # in aten/src/ATen/native/README.md for more details
    #
    # NB: onlyCPU because CompositeImplicitAutograd ops go through the same
    # codepath on all devices. Ideally we'd use a meta device here but coverage
    # for that is not good yet.
    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, '__torch_dispatch__ does not work in fbcode')
    @onlyCPU
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_composite_compliance(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=False)

        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            _check_composite_compliance(op, args, kwargs)


# gradcheck requires double precision
_gradcheck_ops = partial(ops, dtypes=OpDTypes.supported,
                         allowed_dtypes=[torch.double, torch.cdouble])


class TestGradients(TestCase):
    exact_dtype = True

    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    def _check_helper(self, device, dtype, op, variant, check, *, check_forward_ad=False, check_backward_ad=True,
                      check_batched_grad=None, check_batched_forward_grad=False):
        assert check in ('gradcheck', 'bwgrad_bwgrad', 'fwgrad_bwgrad')
        # NB: check_backward_ad does not affect gradgradcheck (always True)
        if variant is None:
            self.skipTest("Skipped! Variant not implemented.")
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest(f"Skipped! {op.name} does not support dtype {str(dtype)}")

        def is_inplace(variant):
            if hasattr(variant, "__wrapped__"):
                return variant.__wrapped__ is op.get_inplace()
            return variant is op.get_inplace()

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(device, dtype, requires_grad=True, include_conjugated_inputs=include_conjugated_inputs)

        for sample in samples:
            if sample.broadcasts_input and is_inplace(variant):
                continue

            # Note on TensorList inputs
            #
            # gradcheck does not support TensorList inputs so here we pass TensorList
            # inputs of size n as n single Tensor inputs to gradcheck and wrap the op
            # in a function that puts the n Tensor inputs back into a TensorList
            def fn(*inputs):
                # Put tensors back into TensorList since we splat them when passing to gradcheck
                if is_iterable_of_tensors(sample.input):
                    n = len(sample.input)
                    inputs = (inputs[:n], *inputs[n:])
                output = op.gradcheck_wrapper(variant, *inputs, **sample.kwargs)
                if sample.output_process_fn_grad is not None:
                    return sample.output_process_fn_grad(output)
                return output

            # Splat TensorList inputs into single Tensor inputs
            gradcheck_args = (sample.input,) if isinstance(sample.input, torch.Tensor) else tuple(sample.input)
            gradcheck_args += sample.args

            if check == 'gradcheck':
                if check_batched_grad is None:
                    check_batched_grad = op.check_batched_grad
                self.assertTrue(gradcheck(fn, gradcheck_args,
                                          check_batched_grad=check_batched_grad,
                                          check_grad_dtypes=True,
                                          nondet_tol=op.gradcheck_nondet_tol,
                                          fast_mode=op.gradcheck_fast_mode,
                                          check_forward_ad=check_forward_ad,
                                          check_backward_ad=check_backward_ad,
                                          check_undefined_grad=True,
                                          check_batched_forward_grad=check_batched_forward_grad))
            elif check in ('bwgrad_bwgrad', 'fwgrad_bwgrad'):  # gradgrad check
                self.assertFalse(check_forward_ad, msg="Cannot run forward AD check for gradgradcheck")
                for gen_non_contig_grad_outputs in (False, True):
                    kwargs = {
                        "gen_non_contig_grad_outputs": gen_non_contig_grad_outputs,
                        "check_batched_grad": op.check_batched_gradgrad,
                        "check_grad_dtypes": True,
                        "nondet_tol": op.gradcheck_nondet_tol,
                        "fast_mode": op.gradcheck_fast_mode
                    }
                    if check == "fwgrad_bwgrad":
                        kwargs["check_fwd_over_rev"] = True
                        kwargs["check_rev_over_rev"] = False
                        kwargs["check_batched_grad"] = False
                        kwargs["check_undefined_grad"] = False

                    self.assertTrue(gradgradcheck(fn, gradcheck_args, **kwargs))
            else:
                self.assertTrue(False, msg="Unknown check requested!")

    def _grad_test_helper(self, device, dtype, op, variant, *, check_forward_ad=False, check_backward_ad=True,
                          check_batched_grad=None, check_batched_forward_grad=False):
        return self._check_helper(device, dtype, op, variant, 'gradcheck', check_forward_ad=check_forward_ad,
                                  check_backward_ad=check_backward_ad, check_batched_grad=check_batched_grad,
                                  check_batched_forward_grad=check_batched_forward_grad)

    def _skip_helper(self, op, device, dtype):
        if not op.supports_autograd and not op.supports_forward_ad:
            self.skipTest("Skipped! autograd not supported.")
        if not op.supports_complex_autograd(torch.device(device).type) and dtype.is_complex:
            self.skipTest("Skipped! Complex autograd not supported.")

    # Tests that gradients are computed correctly
    @_gradcheck_ops(op_db)
    def test_fn_grad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        self._grad_test_helper(device, dtype, op, op.get_op())

    # Method grad (and gradgrad, see below) tests are disabled since they're
    #   costly and redundant with function grad (and gradgad) tests
    # @_gradcheck_ops(op_db)
    # def test_method_grad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._grad_test_helper(device, dtype, op, op.get_method())

    @_gradcheck_ops(op_db)
    def test_inplace_grad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")
        self._grad_test_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()))

    # Test that gradients of gradients are computed correctly
    @_gradcheck_ops(op_db)
    def test_fn_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.supports_gradgrad:
            self.skipTest("Skipped! Operation does not support gradgrad")
        self._check_helper(device, dtype, op, op.get_op(), 'bwgrad_bwgrad')

    # Test that forward-over-reverse gradgrad is computed correctly
    @_gradcheck_ops(op_db)
    def test_fn_fwgrad_bwgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        if op.supports_fwgrad_bwgrad:
            self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")
        else:
            err_msg = r"Trying to use forward AD with .* that does not support it\."
            hint_msg = ("Running forward-over-backward gradgrad for an OP that has does not support it did not "
                        "raise any error. If your op supports forward AD, you should set supports_fwgrad_bwgrad=True.")
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                self._check_helper(device, dtype, op, op.get_op(), "fwgrad_bwgrad")

    # Test that gradients of gradients are properly raising
    @_gradcheck_ops(op_db)
    def test_fn_fail_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if op.supports_gradgrad:
            self.skipTest("Skipped! Operation does support gradgrad")

        err_msg = r"derivative for .* is not implemented"
        with self.assertRaisesRegex(RuntimeError, err_msg):
            self._check_helper(device, dtype, op, op.get_op(), 'bwgrad_bwgrad')

    # Method gradgrad (and grad, see above) tests are disabled since they're
    #   costly and redundant with function gradgrad (and grad) tests
    # @_gradcheck_ops(op_db)
    # def test_method_gradgrad(self, device, dtype, op):
    #     self._skip_helper(op, device, dtype)
    #     self._gradgrad_test_helper(device, dtype, op, op.get_method())

    @_gradcheck_ops(op_db)
    def test_inplace_gradgrad(self, device, dtype, op):
        self._skip_helper(op, device, dtype)
        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")
        self._check_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()), "bwgrad_bwgrad")

    def _forward_grad_helper(self, device, dtype, op, variant, is_inplace):
        # TODO: clean up how attributes are passed to gradcheck from OpInfos
        def call_grad_test_helper():
            check_batched_forward_grad = ((op.check_batched_forward_grad and not is_inplace) or
                                          (op.check_inplace_batched_forward_grad and is_inplace))
            self._grad_test_helper(device, dtype, op, variant, check_forward_ad=True, check_backward_ad=False,
                                   check_batched_grad=False, check_batched_forward_grad=check_batched_forward_grad)
        if op.supports_forward_ad:
            call_grad_test_helper()
        else:
            err_msg = r"Trying to use forward AD with .* that does not support it\."
            hint_msg = ("Running forward AD for an OP that has does not support it did not "
                        "raise any error. If your op supports forward AD, you should set supports_forward_ad=True")
            with self.assertRaisesRegex(NotImplementedError, err_msg, msg=hint_msg):
                call_grad_test_helper()

    @_gradcheck_ops(op_db)
    def test_forward_mode_AD(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        self._forward_grad_helper(device, dtype, op, op.get_op(), is_inplace=False)

    @_gradcheck_ops(op_db)
    def test_inplace_forward_mode_AD(self, device, dtype, op):
        self._skip_helper(op, device, dtype)

        if not op.inplace_variant or not op.supports_inplace_autograd:
            self.skipTest("Skipped! Operation does not support inplace autograd.")

        self._forward_grad_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()), is_inplace=True)

    # Functions that do not support autograd should not fail in forward mode
    # Inplace functions (such as "resize_") are expected to fail in forward mode and should be skipped
    # Test only when supports_autograd=False and for double dtype
    @ops(filter(lambda op: not op.supports_autograd, op_db), dtypes=OpDTypes.supported, allowed_dtypes=(torch.double,))
    def test_nondifferentiable(self, device, dtype, op):
        # Expecting no errors
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        sample = first_sample(self, samples)
        result = op(sample.input, *sample.args, **sample.kwargs)


# Tests operators for consistency between JIT and eager, also checks
#   correctness of JIT specific alias schemas and intended
#   autodifferentiation behavior.
# Inherits from JitCommonTestCase instead of TestCase directly to share
#   functionality with original test_jit.py method operator tests
class TestJit(JitCommonTestCase):
    exact_dtype = True

    # Tests that the forward and backward passes of operations produce the
    #   same values for the cross-product of op variants (function, method, inplace)
    #   and runtimes (eager, traced, scripted).
    # TODO WARNING: inplace x {traced, scripted} not currently tested
    @_variant_ops(op_db)
    def test_variant_consistency_jit(self, device, dtype, op):
        _requires_grad = op.supports_autograd and (dtype.is_floating_point or
                                                   op.supports_complex_autograd(torch.device(device).type))

        include_conjugated_inputs = op.test_conjugated_samples and dtype.is_complex
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad, include_conjugated_inputs=include_conjugated_inputs)

        # Acquires variants to test
        func = op.get_op()
        method = op.get_method()
        variants = {
            # TODO: inplace tests currently fail, fix and add inplace variant
            'function': func, 'method': method,
        }

        # TODO: find better way to standardize on op registration itself..
        has_fake_function = op.name in ["resize_", 'resize_as_']

        if has_fake_function:
            variants = {'method': getattr(torch.Tensor, op.name)}
            samples = op.sample_inputs(device, dtype, requires_grad=False)

        support_script = op.supports_scripting

        tested = False
        for sample in samples:
            # Test traced and scripted consistency
            for func_type, variant in variants.items():
                if variant is None:
                    continue

                # scripting and check_alias_analysis do not work with lambdas
                # lambdas are typically used as a way to simulate methods without
                # functional variants, so rely on the other variant for testing
                # for now
                if is_lambda(variant):
                    continue

                tested = True

                # Create accessor for script function variant
                name = op.name + '_' if func_type == 'inplace' else op.name

                # run with disable_autodiff_subgraph_inlining(True) to test
                #   autodiff support. Context manager forces the graph to contain
                #   DifferentiableGraph nodes if they are present
                with disable_autodiff_subgraph_inlining():
                    # Check scripted forward, grad, and grad grad
                    if support_script:
                        script_fn = create_script_fn(self, name, func_type)

                    def out_fn(output):
                        # Processes the output for autograd
                        if sample.output_process_fn_grad is not None:
                            return sample.output_process_fn_grad(output)
                        return output

                    def get_sample():
                        return clone_input_helper(sample.input) if op.name[-1] == '_' else sample.input

                    if support_script:
                        check_against_reference(self,
                                                script_fn,
                                                func,
                                                out_fn,
                                                (get_sample(),) + sample.args,
                                                sample.kwargs,
                                                no_grad=not _requires_grad, no_gradgrad=not op.supports_gradgrad)

                    # Check traced forward, grad, and grad grad
                    # TODO: fix tracing here
                    supports_tracing = not has_fake_function
                    if op.assert_jit_shape_analysis:
                        self.assertTrue(supports_tracing)

                    if supports_tracing:
                        traced_fn = create_traced_fn(self, variant)
                        check_against_reference(self,
                                                traced_fn,
                                                func,
                                                out_fn,
                                                (get_sample(),) + sample.args,
                                                sample.kwargs,
                                                no_grad=not _requires_grad, no_gradgrad=not op.supports_gradgrad)

                    # Check alias annotation schema for correctness (make
                    #   sure inputs that aren't supposed to be modified aren't)
                    # Note: only runs in float32 because schema isn't affected by dtype,
                    #   so running it on all dtypes is would be excessive
                    if dtype == torch.float32:
                        # TODO: no reason why we cant run this with tracing graph
                        if support_script and op.name != "rsub":
                            check_alias_annotation(name, (get_sample(),) + sample.args, sample.kwargs,
                                                   func_type=func_type, aten_name=op.aten_name)

                        # TODO: use script graph as well
                        checked_shape_analysis = False
                        if supports_tracing:
                            out = variant(get_sample(), *sample.args, **sample.kwargs)

                            # right now, tuple of outputs and tensor output supported
                            # TODO: list of tensor outputs
                            tuple_of_tensors = isinstance(out, tuple) and all([isinstance(elem, torch.Tensor) for elem in out])

                            if isinstance(out, torch.Tensor) or tuple_of_tensors:
                                if tuple_of_tensors:
                                    sizes = [elem.size() for elem in out]
                                else:
                                    sizes = out.size()
                                self.checkShapeAnalysis(sizes, traced_fn.graph, op.assert_jit_shape_analysis)
                                checked_shape_analysis = True
                        if op.assert_jit_shape_analysis:
                            self.assertTrue(checked_shape_analysis)

                    # Check autodifferentiation of nodes for traced and scripted graphs, only need to check once per sample
                    if dtype is torch.float32:
                        # Sandcastle doesn't fuse nodes
                        if IS_SANDCASTLE:
                            # fusible nodes are expected to be found in FusionGroups in the DifferentiableGraphs
                            nonfusible_nodes = op.autodiff_nonfusible_nodes + op.autodiff_fusible_nodes
                            fusible_nodes = []
                        else:
                            nonfusible_nodes = op.autodiff_nonfusible_nodes
                            fusible_nodes = op.autodiff_fusible_nodes

                        if supports_tracing:
                            self.assertAutodiffNode(traced_fn.last_graph, op.assert_autodiffed, nonfusible_nodes, fusible_nodes)
                        if support_script:
                            self.assertAutodiffNode(script_fn.last_graph, op.assert_autodiffed, nonfusible_nodes, fusible_nodes)
        assert tested, "JIT Test does not execute any logic"

    # alias testing is only done with torch.float for the same reason
    _alias_ops = partial(ops, dtypes=OpDTypes.supported,
                         allowed_dtypes=(torch.float,))

    @_alias_ops((op for op in op_db if op.aliases))
    def test_jit_alias_remapping(self, device, dtype, op):
        # Required to avoid undefined value: tensor error in JIT compilation of the function template
        tensor = torch.tensor

        # NOTE: only tests on first sample
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        sample = first_sample(self, samples)

        # [Scripting Data Preparation]
        # Prepare data for test scripting
        # Below we prepare strings of args/kwargs with and without type annotations.
        # These strings are inserted into function template strings which is then torch scripted.
        # - args string is ["t0"] corresponding to the "input" tensor required by the op
        # - args_kw is the value of args and strings of kwargs used to call the op (without type annotations), for example,
        # ["to", "1.0", "(1,)", "True", "tensor(1.0)"] -> def fn(t0): return variant(t0, 1.0, (1,), True, tensor(1.0))
        args = ["t0"]

        def quote_strs(v):
            if isinstance(v, str):
                return f"'{v}'"

            return str(v)

        args_kw = args + \
            [f"{v}" for v in sample.args] + \
            [f"{k}={quote_strs(v)}" for k, v in sample.kwargs.items()]

        # Prepare data for test tracing
        sample_args_kwargs = ()
        if len(sample.args) > 0:
            sample_args_kwargs += (sample.args, )
        if len(sample.kwargs) > 0:
            sample_args_kwargs += (sample.kwargs, )

        original_name = op.aten_name
        original_name_inplace = original_name + "_"
        expected_dtype = op(sample.input, *sample.args, **sample.kwargs).dtype

        for a_op in op.aliases:
            inplace = a_op.inplace_variant
            method_or_inplace = [a_op.inplace_variant, a_op.method_variant]
            variants = (v for v in (a_op.op, a_op.method_variant, a_op.inplace_variant) if v is not None)

            # Test scripting:
            for variant in variants:
                variant_name = variant.__name__
                op_name = original_name_inplace if variant is inplace else original_name

                if variant in method_or_inplace:
                    fn_template = '''
                        def _fn(t0{c}):
                            return t0.{alias_name}({args_kw})
                    '''
                    # remove the first input tensor
                    script = fn_template.format(
                        c=", " if len(args_kw[1:]) > 1 else "",
                        args_kw=", ".join(args_kw[1:]),
                        alias_name=variant_name,
                    )
                else:
                    fn_template = '''
                        def _fn({args}):
                            return variant({args_kw})
                    '''
                    script = fn_template.format(
                        args=", ".join(args),
                        args_kw=", ".join(args_kw),
                    )
                scripted = torch.jit.CompilationUnit(script)._fn

                if (variant is inplace and not torch.can_cast(expected_dtype, dtype)):
                    try:
                        inp = clone_input_helper(sample.input)
                        scripted(inp)
                    except Exception as e:
                        continue
                    self.fail("Inplace operation on integer tensor that should be promoted to float didn't fail!")

                inp = clone_input_helper(sample.input)
                scripted(inp)
                inp = clone_input_helper(sample.input)
                graph = scripted.graph_for(inp)
                FileCheck().check(op.aten_name).check_not(variant_name).run(graph)

            # Test tracing:
            for variant in variants:
                variant_name = variant.__name__
                op_name = original_name_inplace if variant is inplace else original_name

                def _fn(*sample_args, **sample_kwargs):
                    return variant(*sample_args, **sample_kwargs)

                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                traced = torch.jit.trace(_fn, *inp)
                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                traced(*inp)
                inp = (clone_input_helper(sample.input),) + sample_args_kwargs
                graph = traced.graph_for(*inp)
                FileCheck().check(op_name).check_not(variant_name).run(graph)

class TestMathBits(TestCase):
    # Tests that
    # 1. The operator's output for physically conjugated/negated tensors and conjugate/negative view tensors
    # produces the same value
    # 2. The gradients are same in both cases mentioned in (1)
    # 3. If the operator's inplace variant is supported, tests that the inplace operation
    #    produces the correct value when called on a conjugate/negative view tensor and that the output
    #    has its conj/neg bit set to true
    # This test only runs for C -> R and C -> C functions
    # TODO: add tests for `R->C` functions
    # Note: This test runs for functions that take both tensors and tensorlists as input.
    def _test_math_view(self, device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set, out_type):
        inplace_variant = op.inplace_variant

        # helper function to clone and conjugate/negate the input if its a tensor
        # else clone the sequence and conjugate/negate the first element in the sequence
        # If a requires_grad argument is provided the tensor being conjugated/negated will
        # have its requires_grad set to that value.
        def clone_and_perform_view(input, **kwargs):
            if isinstance(input, torch.Tensor):
                requires_grad = kwargs.get('requires_grad', input.requires_grad)
                with torch.no_grad():
                    # Ensure view represents the original sample input
                    input = math_op_physical(input)
                # Note: .conj() is not called under no_grad mode since it's not allowed to modify a
                # view created in no_grad mode. Here it's ok to do so, so as a workaround we call conj
                # before resetting the requires_grad field for input
                input = math_op_view(input)
                assert input.is_leaf
                return input.requires_grad_(requires_grad)

            if isinstance(input, Sequence):
                out = list(map(clone_input_helper, input))
                out[0] = clone_and_perform_view(out[0])
                return tuple(out)

        for sample in samples:
            tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
            cloned1 = clone_and_perform_view(sample.input)

            # Computes function forward value with a physically conjugated/negated tensor and
            # a conj/neg view tensor and verifies that the output in both case are equal.
            expected_forward = op(sample.input, *sample.args, **sample.kwargs)
            forward_with_mathview = op(cloned1, *sample.args, **sample.kwargs)
            self.assertEqual(expected_forward, forward_with_mathview)

            # If the op has an inplace variant, and the input doesn't require broadcasting
            # and has the same dtype as output, verify that the inplace operation on a conjugated/negated
            # input produces correct output, and the output tensor has the conj/neg bit set to True
            if inplace_variant is not None and not sample.broadcasts_input:
                cloned2 = clone_and_perform_view(tensor, requires_grad=False)
                if (isinstance(expected_forward, torch.Tensor) and
                        expected_forward.dtype is tensor.dtype):
                    inplace_forward = inplace_variant(cloned2, *sample.args, **sample.kwargs)
                    self.assertTrue(is_bit_set(inplace_forward))
                    self.assertEqual(inplace_forward, expected_forward)

            # TODO: backward consistency only supported for single tensor outputs
            # TODO: backward consistency only checked on sample.input, not all
            #   tensor inputs
            # TODO: update to handle checking grads of all tensor inputs as
            #   derived from each tensor output
            if isinstance(expected_forward, torch.Tensor) and expected_forward.requires_grad:
                output_process_fn_grad = sample.output_process_fn_grad or (lambda x: x)
                expected_forward = output_process_fn_grad(expected_forward)
                forward_with_mathview = output_process_fn_grad(forward_with_mathview)

                tensor = sample.input if isinstance(sample.input, torch.Tensor) else sample.input[0]
                expected_forward.sum().backward(retain_graph=True)
                forward_with_mathview.sum().backward(retain_graph=True)
                if tensor.grad is not None:
                    cloned1_tensor = cloned1 if isinstance(cloned1, torch.Tensor) else cloned1[0]
                    self.assertEqual(tensor.grad, cloned1_tensor.grad)

                    tensor.grad, cloned1_tensor.grad = None, None

                    # a repeat of the above test if output is not complex valued
                    if (out_type(expected_forward)):
                        grad = torch.randn_like(expected_forward)
                        expected_forward.backward(grad)
                        forward_with_mathview.backward(math_op_view(math_op_physical(grad)))

                        self.assertEqual(tensor.grad, cloned1_tensor.grad)

    @ops(op_db, allowed_dtypes=(torch.cfloat,))
    def test_conj_view(self, device, dtype, op):
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")
        math_op_physical = torch.conj_physical
        math_op_view = torch.conj
        _requires_grad = (op.supports_autograd and op.supports_complex_autograd(torch.device(device).type))
        is_bit_set = torch.is_conj
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set, torch.is_complex)

    @ops(op_db, allowed_dtypes=(torch.double,))
    def test_neg_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        math_op_physical = torch.neg
        math_op_view = torch._neg_view
        is_bit_set = torch.is_neg
        samples = op.sample_inputs(device, dtype, requires_grad=op.supports_autograd)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set,
                             lambda x: True)

    @ops(op_db, allowed_dtypes=(torch.cdouble,))
    def test_neg_conj_view(self, device, dtype, op):
        if not op.test_neg_view:
            self.skipTest("Operation not tested with tensors with negative bit.")
        if not op.test_conjugated_samples:
            self.skipTest("Operation doesn't support conjugated inputs.")

        def math_op_physical(x):
            return -x.conj_physical()

        def math_op_view(x):
            return torch._neg_view(x).conj()

        def is_bit_set(x):
            return torch.is_neg(x) and torch.is_conj(x)

        _requires_grad = (op.supports_autograd and op.supports_complex_autograd(torch.device(device).type))
        samples = op.sample_inputs(device, dtype, requires_grad=_requires_grad)
        # Only test one sample
        samples = itertools.islice(samples, 1)
        self._test_math_view(device, dtype, op, samples, math_op_physical, math_op_view, is_bit_set,
                             torch.is_complex)


instantiate_device_type_tests(TestCommon, globals())
instantiate_device_type_tests(TestGradients, globals())
instantiate_device_type_tests(TestJit, globals())
instantiate_device_type_tests(TestMathBits, globals())

if __name__ == '__main__':
    run_tests()
