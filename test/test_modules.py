from itertools import product
from inspect import signature, isgenerator
from copy import deepcopy
import tempfile

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import (
    TestCase, run_tests, freeze_rng_state, mock_wrapper, get_tensors_from, gradcheck, gradgradcheck)
from unittest.mock import patch


class TestModule(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    precision = 1e-5
    rel_tol = 1e-5

    @modules(module_db)
    def test_forward(self, device, dtype, module_info):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)

                # === Do forward pass. ===
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                outputs = m(*args, **kwargs)

                # === Compare outputs to a reference if one is specified. ===
                # TODO: Handle precision
                reference_fn = module_input.reference_fn
                if reference_fn is not None:
                    ref_outputs = reference_fn(m, *args, **kwargs)
                    self.assertEqual(outputs, ref_outputs)

    # Tests passing factory kwargs (e.g. device / dtype) during module instantiation.
    # They should be applied to any created parameters and buffers.
    @modules(module_db)
    def test_factory_kwargs(self, device, dtype, module_info):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False)
        for module_input in module_inputs:
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            # Check if this module creates parameters or registers buffers.
            # The mock magic here passes through to the real Parameter / register_buffer
            # logic and is only used to check call inputs.
            module_creates_params_or_buffers = False
            parameter_new = mock_wrapper(torch.nn.Parameter.__new__)
            with patch.object(torch.nn.Parameter, '__new__', parameter_new):
                register_buffer = mock_wrapper(torch.nn.Module.register_buffer)
                with patch.object(torch.nn.Module, 'register_buffer', register_buffer):
                    m = module_cls(*args, **kwargs)

                    # Check if a parameter or buffer was created with a tensor not passed to the constructor.
                    constructor_tensors = get_tensors_from(args, kwargs)
                    for mock in [parameter_new.mock, register_buffer.mock]:
                        for call_args, call_kwargs in mock.call_args_list:
                            call_tensors = get_tensors_from(call_args, call_kwargs)
                            if len(call_tensors) > 0 and not constructor_tensors.intersection(call_tensors):
                                module_creates_params_or_buffers = True
                                break

            if not module_creates_params_or_buffers:
                continue

            # Instantiate module with the factory kwargs.
            kwargs.update({
                'device': device,
                'dtype': dtype,
            })

            if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
                # Ensure device and dtype are passed to all UninitializedParameters and UninitializedBuffers.
                uninit_param_new = mock_wrapper(torch.nn.UninitializedParameter.__new__)
                with patch.object(torch.nn.UninitializedParameter, '__new__', uninit_param_new):
                    uninit_buffer_new = mock_wrapper(torch.nn.UninitializedBuffer.__new__)
                    with patch.object(torch.nn.UninitializedBuffer, '__new__', uninit_buffer_new):
                        m = module_cls(*args, **kwargs)
                        uninit_param_new.mock.assert_has_calls(
                            [mock.call(device=device, dtype=dtype) for _ in uninit_param_new.mock.mock_calls])
                        uninit_buffer_new.mock.assert_has_calls(
                            [mock.call(device=device, dtype=dtype) for _ in uninit_buffer_new.mock.mock_calls])
            else:
                # Check device placement and dtype for created parameters and buffers.
                # Only verify floating point dtypes since that's what the kwarg applies to.
                m = module_cls(*args, **kwargs)
                for name, param in m.named_parameters():
                    self.assertEqual(
                        str(param.device), device,
                        f'Parameter {name} is on {param.device.type} instead of the expected device {device}')
                    if param.dtype.is_floating_point:
                        self.assertEqual(
                            param.dtype, dtype,
                            f'Parameter {name} is of dtype {param.dtype} instead of the expected dtype {dtype}')
                for name, buffer in m.named_buffers():
                    self.assertEqual(
                        str(buffer.device), device,
                        f'Buffer {name} is on {buffer.device.type} instead of the expected device {device}')
                    if buffer.dtype.is_floating_point:
                        self.assertEqual(
                            buffer.dtype, dtype,
                            f'Buffer {name} is of dtype {buffer.dtype} instead of the expected dtype {dtype}')

    @modules(module_db)
    def test_repr(self, device, dtype, module_info):
        # Test module can be represented with repr and str without errors.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False)
        for module_input in module_inputs:
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)

            # Check that these methods do not raise errors
            m.__repr__()
            str(m)

    @modules(module_db)
    def test_pickle(self, device, dtype, module_info):
        # Test that module can be pickled and unpickled.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)

                # === Do forward pass. ===
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                output = m(*args, **kwargs)

                # === Check unpickled module gives the same output. ===
                with tempfile.TemporaryFile() as f:
                    torch.save(m, f)
                    f.seek(0)
                    m_copy = torch.load(f)
                    output_from_copy = m_copy(*args, **kwargs)
                    self.assertEqual(output, output_from_copy)

    @modules([module_info for module_info in module_db
              if 'inplace' in signature(module_info.module_cls).parameters])
    def test_check_inplace(self, device, dtype, module_info):
        # Check if the inplace variant of the module gives the same result as the out of place
        # variant.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m_op = module_cls(*args, **kwargs, inplace=False)
            m_op.to(device).to(dtype)
            m_inplace = module_cls(*args, **kwargs, inplace=True)
            m_inplace.to(device).to(dtype)

            # === Inplace modules only supports inplace operations on the first argument ===
            input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs

            # ===  Do not allow the first input to be in input_kwargs ===
            forward_sig = signature(m_op).parameters
            self.assertGreaterEqual(len(forward_sig), 1)
            first_param_name = next(iter(forward_sig.items()))
            self.assertNotIn(first_param_name, input_kwargs)

            # === Out of place operation does not write to original tensor ===
            self.assertGreaterEqual(len(input_args), 1)
            input_version = input_args[0]._version
            with freeze_rng_state():
                output_op = m_op(*input_args, **input_kwargs)
            self.assertEqual(input_args[0]._version, input_version)

            # === Check that the inplace operation gives the same result ===
            input_arg_copy = deepcopy(input_args)
            input_arg_clone = tuple(i.clone() for i in input_arg_copy)
            with freeze_rng_state():
                output_ip = m_inplace(*input_arg_clone, **input_kwargs)
            self.assertNotEqual(input_arg_clone[0]._version, input_version)
            self.assertEqual(output_op, output_ip)

            # === Check that the gradients are the same ===
            grad = output_op.data.clone().normal_()
            output_op.backward(grad)
            output_ip.backward(grad)
            self.assertEqual(input_args[0].grad, input_arg_copy[0].grad)

    def _traverse_obj(self, obj, func):
        if isinstance(obj, (tuple, list)):
            return type(obj)(self._traverse_obj(o, func) for o in obj)
        elif isgenerator(obj):
            return tuple(self._traverse_obj(o, func) for o in obj)
        elif isinstance(obj, dict):
            return {name: self._traverse_obj(o, func) for name, o in obj.items()}
        elif isinstance(obj, (torch.Tensor, torch.nn.Parameter)):
            return func(obj)

    def _retain_grad(self, obj):
        # gradients needs to be retained to check for grad. This is useful when
        # non-leafs are present in the graph.
        def inner_retain_grad(obj):
            if obj.requires_grad:
                obj.retain_grad()
        self._traverse_obj(obj, inner_retain_grad)

    def _get_grads(self, obj):
        def inner_get_grad(obj):
            if obj.requires_grad:
                return obj.grad
        return self._traverse_obj(obj, inner_get_grad)

    def _zero_grad(self, obj):
        def inner_zero_grad(obj):
            if obj.grad is not None:
                obj.grad = None
        self._traverse_obj(obj, inner_zero_grad)

    @modules(module_db)
    def test_non_contiguous_tensors(self, device, dtype, module_info):
        # Check modules work with non-contiguous tensors

        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True)

        def _make_non_contiguous(obj):
            def inner_make_non_contiguous(obj):
                # Scalar tensors can not be made non-contiguous
                if not isinstance(obj, torch.Tensor) or obj.dim() == 0:
                    return obj

                out = torch.repeat_interleave(obj, 2, dim=-1)
                out = out[..., ::2].detach()
                out.requires_grad = obj.requires_grad
                return out
            return self._traverse_obj(obj, inner_make_non_contiguous)

        def _can_be_noncontiguous(obj):
            if isinstance(obj, (tuple, list)):
                return any(_can_be_noncontiguous(o) for o in obj)
            elif isinstance(obj, dict):
                return any(_can_be_noncontiguous(o) for o in obj.values())
            # scalar tensors can not be non-contiguous
            if not isinstance(obj, torch.Tensor) or obj.dim() == 0:
                return False
            return True


        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
            if not (_can_be_noncontiguous(input_args) or _can_be_noncontiguous(input_kwargs)):
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)

            self._retain_grad((input_args, input_kwargs))

            # === Forward with default input
            with freeze_rng_state():
                default_output = m(*input_args, **input_kwargs)
                grad_output = default_output.clone().detach_().normal_()
                default_output.backward(grad_output, retain_graph=True)

            default_input_args_grad, default_input_kwargs_grad = deepcopy(self._get_grads((input_args, input_kwargs)))
            default_param_grad = deepcopy([p.grad for p in m.parameters()])

            # === Construct non-contiguous tensors ===
            nc_input_args, nc_input_kwargs = _make_non_contiguous((input_args, input_kwargs))
            nc_grad_output = _make_non_contiguous(grad_output)

            # === Compare results with non-contiguous and contiguous tensors ===
            inputs = [(input_args, input_kwargs), (nc_input_args, nc_input_kwargs)]
            grads = [grad_output, nc_grad_output]

            for (in_args, in_kwargs), g_out in product(inputs, grads):
                g_out_copy = deepcopy(g_out)
                self._zero_grad((in_args, in_kwargs))
                self._zero_grad(m.parameters())

                with freeze_rng_state():
                    out = m(*in_args, **in_kwargs)
                    out.backward(g_out_copy, retain_graph=True)

                input_args_grad, input_kwargs_grad = self._get_grads((in_args, in_kwargs))
                self.assertEqual(out, default_output)
                self.assertEqual(input_args_grad, default_input_args_grad, atol=1e-4, rtol=0)
                self.assertEqual(input_kwargs_grad, default_input_kwargs_grad, atol=1e-4, rtol=0)

                param_grad = [p.grad for p in m.parameters()]
                self.assertEqual(param_grad, default_param_grad)


    def _test_gradients_helper(self, device, dtype, module_info, check):
        # Check gradients
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True)

        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)

            params = tuple(m.parameters())

            # === Perform gradient check on the input_args ===
            input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs

            other_kwargs = {}
            kwarg_tensors = []
            for name, obj in input_kwargs.items():
                if isinstance(obj, torch.Tensor):
                    kwarg_tensors.append((name, obj))
                else:
                    other_kwargs[name] = obj

            grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)

            def fn_to_gradcheck(*input_and_params):
                new_input_args = input_and_params[:len(input_args)]
                kwarg_args = input_and_params[-len(kwarg_tensors):]
                new_kwargs = {name: obj for (name, _), obj in zip(kwarg_tensors, kwarg_args)}

                with freeze_rng_state():
                    return m(*new_input_args, **new_kwargs, **other_kwargs)

            self.assertTrue(check(fn_to_gradcheck, grad_input))


    @modules(module_db, allowed_dtypes=[torch.double])
    def test_grad(self, device, dtype, module_info):
        self._test_gradients_helper(device, dtype, module_info, gradcheck)

    @modules([m for m in module_db if m.supports_gradgrad],
             allowed_dtypes=[torch.double])
    def test_gradgrad(self, device, dtype, module_info):
        self._test_gradients_helper(device, dtype, module_info, gradgradcheck)


instantiate_device_type_tests(TestModule, globals())

if __name__ == '__main__':
    run_tests()
