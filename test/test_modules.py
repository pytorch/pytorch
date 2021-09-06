import tempfile

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
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


instantiate_device_type_tests(TestModule, globals())

if __name__ == '__main__':
    run_tests()
