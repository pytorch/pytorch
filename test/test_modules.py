import torch
from copy import deepcopy
from itertools import chain
from tempfile import TemporaryFile
from torch.testing import floating_types
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, dtypes, _TestParametrizer, onlyCUDA)
from torch.testing._internal.common_modules import (
    module_db, MODULE_CLASSES, MODULE_CLASS_NAMES, formatted_module_name, mock_wrapper, modules)
from torch.testing._internal.common_utils import TestCase, run_tests, freeze_rng_state
from unittest import mock
from unittest.mock import patch


class TestModule(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    @modules(module_db)
    def test_basics(self, device, dtype, module_info):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                      requires_grad=False)
        for module_input in module_inputs:
            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)

            # TODO: Check that the module is on the correct device and dtype.

            # === Check that the module can be printed. ===
            m.__repr__()
            str(m)

            if module_input.forward_input is None:
                continue

            # === Run forward and backward. ===
            with freeze_rng_state():
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                outputs = m(*args, **kwargs)

                # Compare outputs to a reference if one is specified.
                # TODO: Handle precision
                reference_fn = module_input.reference_fn
                if reference_fn is not None:
                    ref_args, ref_kwargs = deepcopy(args), deepcopy(kwargs)
                    ref_outputs = reference_fn(*ref_args, **ref_kwargs)
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        assert isinstance(ref_outputs, tuple) or isinstance(ref_outputs, list), \
                            "Expected reference_fn to return multiple outputs, but got: {}".format(ref_outputs)
                        for output, ref_output in zip(outputs, ref_outputs):
                            self.assertTrue(torch.allclose(output, ref_output))
                    else:
                        self.assertTrue(torch.allclose(outputs, ref_outputs))

                # Check gradients.
                if module_info.has_sparse_gradients:
                    # TODO: Handle this!
                    # gradcheck doesn't support operators that take in dense inputs but
                    # return sparse parameters. This only happens in the case of nn.Embedding
                    # and nn.EmbeddingBag. Instead, we call `self.check_jacobian`, which
                    # is a slightly different version of gradcheck that can handle this.
                    #assert len(args) == 1 and len(kwargs) == 0
                    #test_input_jacobian = torch.is_floating_point(args[0])
                    #self.check_jacobian(module, args[0], test_input_jacobian)
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                    # TODO: Handle this!
                    # params = tuple(x for x in module.parameters())
                    # num_inputs = len(input_tuple)

                    # def fn_to_gradcheck(*inputs_and_params, **kwargs):
                    #     assert not kwargs
                    #     return test_case._forward(module, inputs_and_params[:num_inputs])

                    # self.assertTrue(gradcheck(fn_to_gradcheck, input_tuple + params,
                    #                           check_batched_grad=self.check_batched_grad))

                    # if self.check_gradgrad:
                    #     test_case.assertTrue(gradgradcheck(fn_to_gradcheck, input_tuple + params,
                    #                                        check_batched_grad=self.check_batched_grad))

    @modules([m for m in module_db if m.has_inplace_variant])
    def test_inplace_variant(self, device, dtype, module_info):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                      requires_grad=False)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                return

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)

                # === Instantiate an in-place version of the module. ===
                ip_args, ip_kwargs = deepcopy(args), deepcopy(kwargs)
                m_inplace = module_cls(*ip_args, **ip_kwargs, inplace=True)

                # === Forward pass. ===
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                # TODO: explain why this assert is here
                assert len(args) == 1 and len(kwargs) == 0

                input = args[0]
                before_version = input._version
                outputs = m(*args, **kwargs)
                self.assertEqual(input._version, before_version)

                ip_args, ip_kwargs = deepcopy(args), deepcopy(kwargs)
                input = ip_args[0]
                before_version = input._version
                outputs_inplace = m_inplace(*ip_args, **ip_kwargs)

                self.assertNotEqual(input._version, before_version)
                self.assertEqual(outputs, outputs_inplace)

                # TODO: Check backward pass

    @modules([m for m in module_db if m.is_pickleable])
    def test_pickle_unpickle(self, device, dtype, module_info):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                      requires_grad=False)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                return

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)

            with TemporaryFile() as f:
                # Compare module outputs to outputs of a saved and reloaded form of the module.
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                m(*args, **kwargs)
                torch.save(m, f)
                f.seek(0)
                m_copy = torch.load(f)
                self.assertEqual(m(*args, **kwargs), m_copy(*args, **kwargs))

    @modules(module_db)
    def test_factory_kwargs(self, device, dtype, module_info):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False)
        for module_input in module_inputs:
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            # Some modules need to explicitly pass factory_kwargs to avoid conflict with
            # existing args such as dtype.
            if module_info.needs_factory_kwargs:
                extra_kwargs = {
                    'factory_kwargs': {
                        'device': device,
                        'dtype': dtype,
                    }
                }
            else:
                extra_kwargs = {
                    'device': device,
                    'dtype': dtype,
                }

            # Check if this module creates parameters or registers buffers.
            # The mock magic here passes through to the real Parameter / register_buffer
            # logic and is only used to check for calls.
            parameter_new = mock_wrapper(torch.nn.Parameter.__new__)
            with patch.object(torch.nn.Parameter, '__new__', parameter_new):
                register_buffer = mock_wrapper(torch.nn.Module.register_buffer)
                with patch.object(torch.nn.Module, 'register_buffer', register_buffer):
                    m = module_cls(*args, **kwargs)
                    module_creates_params_or_buffers = parameter_new.mock.called or register_buffer.mock.called

            if module_creates_params_or_buffers:
                kwargs.update(extra_kwargs)

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
                    # Check device placement and dtype for parameters and buffers.
                    # Only verify floating point dtypes since that's what the kwarg applies to.
                    # Note that dtype verification is also skipped if the module requires factory_kwargs.
                    m = module_cls(*args, **kwargs)
                    for name, param in m.named_parameters():
                        self.assertEqual(
                            str(param.device), device,
                            f'Parameter {name} is on {param.device.type} instead of the expected device {device}')
                        if param.dtype.is_floating_point and not module_info.needs_factory_kwargs:
                            self.assertEqual(
                                param.dtype, dtype,
                                f'Parameter {name} is of dtype {param.dtype} instead of the expected dtype {dtype}')
                    for name, buffer in m.named_buffers():
                        self.assertEqual(
                            str(buffer.device), device,
                            f'Buffer {name} is on {buffer.device.type} instead of the expected device {device}')
                        if buffer.dtype.is_floating_point and not module_info.needs_factory_kwargs:
                            self.assertEqual(
                                buffer.dtype, dtype,
                                f'Buffer {name} is of dtype {buffer.dtype} instead of the expected dtype {dtype}')

    @onlyCUDA
    @modules(module_db)
    def test_cpu_gpu_parity(self, device, dtype, module_info):
        module_cls = module_info.module_cls
        cpu_inputs = module_info.module_inputs_func(module_info, device='cpu', dtype=dtype, requires_grad=True)
        gpu_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype, requires_grad=True)
        for cpu_input, gpu_input in zip(cpu_inputs, gpu_inputs):
            if cpu_input.forward_input is None or gpu_input.forward_input is None:
                continue

            with freeze_rng_state():
                # === Instantiate the module on CPU and GPU. ===
                cpu_args, cpu_kwargs = cpu_input.constructor_input.args, cpu_input.constructor_input.kwargs
                m_cpu = module_cls(*cpu_args, **cpu_kwargs)
                gpu_args, gpu_kwargs = gpu_input.constructor_input.args, gpu_input.constructor_input.kwargs
                m_gpu = module_cls(*gpu_args, **gpu_kwargs)

                # === Run forward and backward. ===
                cpu_args, cpu_kwargs = cpu_input.forward_input.args, gpu_input.forward_input.kwargs
                cpu_outputs = m_cpu(*cpu_args, **cpu_kwargs)
                gpu_args, gpu_kwargs = gpu_input.forward_input.args, gpu_input.forward_input.kwargs
                gpu_outputs = m_gpu(*gpu_args, **gpu_kwargs)
                self.assertEqual(cpu_outputs, gpu_outputs)

                # TODO: Check backwards as well.


instantiate_device_type_tests(TestModule, globals())

if __name__ == '__main__':
    run_tests()
