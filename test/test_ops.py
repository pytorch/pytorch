from functools import partial, wraps

import torch

from torch.testing._internal.common_utils import \
    (TestCase, run_tests)
from torch.testing._internal.common_methods_invocations import \
    (op_db)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, ops, dtypes, onlyOnCPUAndCUDA, skipCUDAIfRocm)
from torch.autograd.gradcheck import gradcheck, gradgradcheck


# Tests that apply to all operators

class TestOpInfo(TestCase):
    exact_dtype = True

    # Verifies that ops have their unsupported dtypes
    #   registered correctly by testing that each claimed unsupported dtype
    #   throws a runtime error
    @skipCUDAIfRocm
    @onlyOnCPUAndCUDA
    @ops(op_db, unsupported_dtypes_only=True)
    def test_unsupported_dtypes(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        # NOTE: only tests on first sample
        sample = samples[0]
        with self.assertRaises(RuntimeError):
            op(sample.input, *sample.args, **sample.kwargs)

    # Verifies that ops have their supported dtypes
    #   registered correctly by testing that each claimed supported dtype
    #   does NOT throw a runtime error
    @skipCUDAIfRocm
    @onlyOnCPUAndCUDA
    @ops(op_db)
    def test_supported_dtypes(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype)
        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        # NOTE: only tests on first sample
        sample = samples[0]
        op(sample.input, *sample.args, **sample.kwargs)


class TestGradients(TestCase):
    exact_dtype = True

    # Copies inputs to inplace operations to avoid inplace modifications
    #   to leaves requiring gradient
    def _get_safe_inplace(self, inplace_variant):
        @wraps(inplace_variant)
        def _fn(t, *args, **kwargs):
            return inplace_variant(t.clone(), *args, **kwargs)

        return _fn

    def _check_helper(self, device, dtype, op, variant, check):
        if variant is None:
            self.skipTest("Skipped! Variant not implemented.")
        if not op.supports_dtype(dtype, torch.device(device).type):
            self.skipTest(f"Skipped! {op.name} does not support dtype {str(dtype)}")

        samples = op.sample_inputs(device, dtype, requires_grad=True)
        for sample in samples:
            partial_fn = partial(variant, **sample.kwargs)
            if check == 'gradcheck':
                self.assertTrue(gradcheck(partial_fn, (sample.input,) + sample.args,
                                          check_grad_dtypes=True))
            elif check == 'gradgradcheck':
                self.assertTrue(gradgradcheck(partial_fn, (sample.input,) + sample.args,
                                              gen_non_contig_grad_outputs=False,
                                              check_grad_dtypes=True))
                self.assertTrue(gradgradcheck(partial_fn, (sample.input,) + sample.args,
                                              gen_non_contig_grad_outputs=True,
                                              check_grad_dtypes=True))
            else:
                self.assertTrue(False, msg="Unknown check requested!")

    def _grad_test_helper(self, device, dtype, op, variant):
        return self._check_helper(device, dtype, op, variant, 'gradcheck')

    def _gradgrad_test_helper(self, device, dtype, op, variant):
        return self._check_helper(device, dtype, op, variant, 'gradgradcheck')

    # Tests that gradients are computed correctly
    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_fn_grad(self, device, dtype, op):
        self._grad_test_helper(device, dtype, op, op.get_op())

    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_method_grad(self, device, dtype, op):
        self._grad_test_helper(device, dtype, op, op.get_method())

    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_inplace_grad(self, device, dtype, op):
        if not op.test_inplace_grad:
            self.skipTest("Skipped! Inplace gradcheck marked to skip.")
        self._grad_test_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()))

    # Test that gradients of gradients are computed correctly
    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_fn_gradgrad(self, device, dtype, op):
        self._gradgrad_test_helper(device, dtype, op, op.get_op())

    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_method_gradgrad(self, device, dtype, op):
        self._gradgrad_test_helper(device, dtype, op, op.get_method())

    @dtypes(torch.double, torch.cdouble)
    @ops(op_db)
    def test_inplace_gradgrad(self, device, dtype, op):
        if not op.test_inplace_grad:
            self.skipTest("Skipped! Inplace gradgradcheck marked to skip.")
        self._gradgrad_test_helper(device, dtype, op, self._get_safe_inplace(op.get_inplace()))


class TestCommon(TestCase):
    exact_dtype = True

    def get_call(self, name, func_type, args, kwargs):
        kwargs_str = ', '.join([k + '=' + str(v) for k, v in kwargs.items()])
        self_arg = args[0]

        if (func_type == 'method' or func_type == 'inplace'):
            args = args[1:]

        argument_str = ', '.join(args)
        argument_str += ', ' if len(args) and len(kwargs) else ''
        argument_str += kwargs_str

        if func_type == 'function':
            call = 'torch.{}({})'.format(name, argument_str)
        elif func_type == 'method' or func_type == 'inplace':
            call = '{}.{}({})'.format(self_arg, name, argument_str)
        else:
            raise TypeError('Unsupported function type')

        return call

    def get_script_args(self, args):
        def get_constant(x):
            if x == inf:
                return 'math.inf'
            if x == -inf:
                return '-math.inf'
            return x

        formals: List[str] = []
        tensors: List[torch.Tensor] = []
        actuals: List[str] = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                name = 'i{}'.format(len(formals))
                formals.append(name)
                actuals.append(name)
                tensors.append(arg)
            elif isinstance(arg, str):
                actuals.append("'{}'".format(arg))
            else:
                actuals.append(str(get_constant(arg)))
        return (formals, tensors, actuals)

    # create a script function from (name, func_type, output_process_fn),
    # and returns the compiled function and example inputs
    def gen_script_fn_and_args(self, name, func_type, *args, **kwargs):
        formals, tensors, actuals = self.get_script_args(args)
        call = self.get_call(name, func_type, actuals, kwargs)

        script_template = '''
        def foo({}):
            return {}
        '''

        script = script_template.format(', '.join(formals), call)
        CU = torch.jit.CompilationUnit(script)
        return CU.foo, tensors

    # create a script function from (name, func_type, output_process_fn),
    # returns a function takes in (args, kwargs) and runs the compiled function and
    # then applies the post process fn to the outputs
    def create_script_fn(self, name, func_type):
        def script_fn(*args, **kwargs):
            fn, tensors = self.gen_script_fn_and_args(name, func_type, *args, **kwargs)
            # self.assertExportImport(fn.graph, tensors)
            # output = output_process_fn(fn(*tensors))
            output = fn(*tensors)
            # skip type annotate function attributes for now, see: https://github.com/python/mypy/issues/2087
            script_fn.last_graph = fn.graph_for(*tensors)  # type: ignore[attr-defined]
            return output
        return script_fn

    # make a new function where all non-tensor arguments in 'args' have been partially
    # applied, and all tensor arguments remain.
    # used to trace functions when some arguments are not tensors
    def partial_apply_nontensors(self, fn, args, **kwargs):
        source = ['t' if isinstance(arg, torch.Tensor) else 's' for arg in args]

        def new_fn(*tensors_):
            tensors = iter(tensors_)
            return fn(*(args[i] if s == 's' else next(tensors) for i, s in enumerate(source)), **kwargs)

        return new_fn, [arg for arg in args if isinstance(arg, torch.Tensor)]

    def create_traced_fn(self, fn):
        def traced_fn(*inputs, **kwargs):
            fn_tensors, inputs_tensors = self.partial_apply_nontensors(fn, inputs, **kwargs)
            # `check_trace` is set to False because check_trace is run with @no_grad
            # Also, `check_against_reference` already does all the checks
            # against python function
            traced = torch.jit.trace(fn_tensors, inputs_tensors, check_trace=False)
            # self.assertExportImport(traced.graph, inputs_tensors)
            output = traced(*inputs_tensors)
            # skip type annotate function attributes for now, see: https://github.com/python/mypy/issues/2087
            traced_fn.last_graph = traced.graph_for(*inputs_tensors)  # type: ignore[attr-defined]
            return output
        return traced_fn

    # Compares variant's backward
    # NOTE: verifies it fails when the forward fails
    def check_variant_backward(self, input, forward_result, expected_grad, expected_exception):
        variant_exception_during_backwards = False
        try:
            forward_result.sum().backward()
            variant_grad = input.grad
            input.grad = None
        except Exception as e:
            if not expected_exception:
                self.fail("Unexpected exception during backwards!")
            variant_exception_during_backwards = True

        if expected_exception != variant_exception_during_backwards:
            self.fail("Unexpected success during backwards!")

        if not expected_exception:
            self.assertEqual(variant_grad, expected_grad)

    # Tests that the forward and backward passes of operations produce the
    #   same values for the cross-product of op variants (function, method, inplace)
    #   and runtimes (eager, traced, scripted).
    # WARNING: inplace x {traced, scripted} not currently tested
    @ops(op_db)
    def test_variant_consistency(self, device, dtype, op):
        samples = op.sample_inputs(device, dtype, requires_grad=True)
        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        for sample in samples:
            # Computes expected forward
            expected_forward = op(sample.input, *sample.args, **sample.kwargs)

            # Computes expected backward
            # NOTE: backward may fail for some dtypes
            exception_during_backwards = False
            expected_grad = None
            try:
                expected_forward.sum().backward()
                expected_grad = sample.input.grad
                sample.input.grad = None
            except Exception as e:
                exception_during_backwards = True

            # Acquires variants to test
            method = op.get_method()
            inplace = op.get_inplace()
            # if op.get_inplace() is not None and op.test_inplace_grad:
            #     inplace = self._get_safe_inplace(op.get_inplace())
            variants = (v for v in (method, inplace) if v is not None)

            # Tests eager consistency
            for variant in variants:
                # Verifies that inplace operations that promote int->float fail
                # on tensors with integer dtypes.
                if (variant is inplace and op.promotes_integers_to_float and
                        dtype in (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
                    try:
                        variant_forward = variant(sample.input.clone(), *sample.args, **sample.kwargs)
                    except Exception as e:
                        continue
                    self.fail("Inplace operation on integer tensor that should be promoted to float didn't fail!")

                # Compares variant's forward
                variant_forward = variant(sample.input.clone(), *sample.args, **sample.kwargs)
                self.assertEqual(variant_forward, expected_forward)

                # Compares variant's backward
                if variant is not inplace and op.test_inplace_grad:
                    self.check_variant_backward(sample.input, variant_forward,
                                                expected_grad, exception_during_backwards)

            # Adds function variant to variant list
            # TODO: investigate why inplace test fails
            # variants = (v for v in (op, method, inplace) if v is not None)
            variants = (v for v in (op, method) if v is not None)

            # Tests traced and scripted consistency
            for variant in variants:
                # Creates the script function
                if variant is op:
                    name = op.name
                    func_type = 'function'
                elif variant is method:
                    name = op.name
                    func_type = 'method'
                else:  # variant is inplace
                    assert variant is inplace
                    name = op.name + "_"
                    func_type = 'inplace'

                # Checks scripted forward
                script_fn = self.create_script_fn(name, func_type)
                scripted_forward = script_fn(sample.input.clone(), *sample.args, **sample.kwargs)
                self.assertEqual(scripted_forward, expected_forward)

                # Checks scripted backwards
                self.check_variant_backward(sample.input, scripted_forward,
                                            expected_grad, exception_during_backwards)

                # Checks traced
                traced_fn = self.create_traced_fn(variant)
                traced_forward = traced_fn(sample.input.clone(), *sample.args, **sample.kwargs)
                self.assertEqual(traced_forward, expected_forward)

                # Checks traced backwards
                self.check_variant_backward(sample.input, traced_forward,
                                            expected_grad, exception_during_backwards)

    @ops(op_db)
    def test_out(self, device, dtype, op):
        if not op.supports_tensor_out:
            self.skipTest("Skipped! Operator %s does not support out=..." % op.name)

        samples = op.sample_inputs(device, dtype)
        if len(samples) == 0:
            self.skipTest("Skipped! No sample inputs!")

        # NOTE: only tests on first sample
        sample = samples[0]
        # call it normally to get the expected result
        expected = op(sample.input, *sample.args, **sample.kwargs)
        # call it with out=... and check we get the expected result
        out_kwargs = sample.kwargs.copy()
        out_kwargs['out'] = out = torch.empty_like(expected)
        op(sample.input, *sample.args, **out_kwargs)
        self.assertEqual(expected, out)


instantiate_device_type_tests(TestOpInfo, globals())
instantiate_device_type_tests(TestGradients, globals())
instantiate_device_type_tests(TestCommon, globals())

if __name__ == '__main__':
    run_tests()
