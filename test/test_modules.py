from copy import deepcopy
import tempfile

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCUDA
from torch.testing._internal.common_modules import module_db, modules
from torch.testing._internal.common_utils import (
    TestCase, run_tests, freeze_rng_state, mock_wrapper, get_tensors_from)
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

    @onlyCUDA
    @modules(module_db)
    def test_cpu_cuda_equal(self, device, dtype, module_info):
        # Test cpu and cuda results are the same
        module_cls = module_info.module_cls
        module_inputs_cpu = module_info.module_inputs_func(module_info, device="cpu", dtype=dtype,
                                                           requires_grad=True)

        def set_zero_grad(items):
            if isinstance(items, dict):
                items = items.values()

            for item in items:
                if not isinstance(item, torch.Tensor):
                    continue
                requires_grad = item.requires_grad
                if not item.is_leaf:
                    item.detach_()
                    item.requires_grad = requires_grad
                if item.requires_grad and item.grad is not None:
                    item.grad.detach_()
                    item.grad.zero_()

        def set_zero_grad_parameters(module):
            for p in module.parameters():
                if p.grad is None:
                    continue
                with torch.no_grad():
                    p.grad.zero_()
                p.grad.detach_()

        def to_device(obj):
            if isinstance(obj, torch.Tensor):
                with torch.no_grad():
                    res = obj.detach().to(device=device)
                    res.requires_grad = obj.requires_grad
                return res
            elif isinstance(obj, tuple):
                return tuple(to_device(o) for o in obj)
            elif isinstance(obj, dict):
                return {key: to_device(o) for key, o in obj.items()}
            else:
                return deepcopy(obj)

        if dtype == torch.float32:
            atol = 5e-2
        else:
            atol = 4e-4

        for module_input in module_inputs_cpu:

            # === Move input from cpu to device ===
            cpu_forward_args = module_input.forward_input.args
            cpu_forward_kwargs = module_input.forward_input.kwargs

            gpu_forward_args = to_device(cpu_forward_args)
            gpu_forward_kwargs = to_device(cpu_forward_kwargs)

            set_zero_grad(cpu_forward_args)
            set_zero_grad(cpu_forward_kwargs)
            set_zero_grad(gpu_forward_args)
            set_zero_grad(gpu_forward_kwargs)

            # === Construct module on cpu and gpu ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            cpu_module = module_cls(*args, **kwargs).to(dtype).to("cpu")
            gpu_module = module_cls(*args, **kwargs).to(dtype).to(device)

            set_zero_grad_parameters(cpu_module)
            set_zero_grad_parameters(gpu_module)

            for cpu_p, gpu_p in zip(cpu_module.parameters(), gpu_module.parameters()):
                gpu_p.data.copy_(cpu_p)

            # === Compare forward resputs for cpu and gpu ===
            with freeze_rng_state():
                cpu_output = cpu_module(*cpu_forward_args, **cpu_forward_kwargs)
            with freeze_rng_state():
                gpu_output = gpu_module(*gpu_forward_args, **gpu_forward_kwargs)

            self.assertEqual(cpu_output, gpu_output, atol=atol, rtol=0)

            # === Run backwards on CPU and GPU and compare results ===
            for _ in range(5):
                cpu_grad_output = cpu_output.clone().normal_()
                gpu_grad_output = cpu_grad_output.type_as(gpu_output)

                cpu_output.backward(cpu_grad_output, retain_graph=True)
                gpu_output.backward(gpu_grad_output, retain_graph=True)

                cpu_grad_input = tuple(i.grad.data if i.grad is not None else None for i in cpu_forward_args)
                gpu_grad_input = tuple(i.grad.data if i.grad is not None else None for i in gpu_forward_args)
                self.assertEqual(cpu_grad_input, gpu_grad_input, atol=atol, rtol=0)

                for cpu_p, gpu_p in zip(cpu_module.parameters(), gpu_module.parameters()):
                    self.assertEqual(cpu_p.grad, gpu_p.grad, atol=atol, rtol=0)

                cpu_grad_kwarg_input = {name: i.grad.data if i.grad is not None else None
                                        for name, i in cpu_forward_kwargs.items()
                                        if isinstance(i, torch.Tensor)}
                gpu_grad_kwarg_input = {name: i.grad.data if i.grad is not None else None
                                        for name, i in gpu_forward_kwargs.items()
                                        if isinstance(i, torch.Tensor)}
                self.assertEqual(cpu_grad_kwarg_input, gpu_grad_kwarg_input, atol=atol, rtol=0)


instantiate_device_type_tests(TestModule, globals())

if __name__ == '__main__':
    run_tests()
