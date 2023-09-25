# Owner(s): ["module: nn"]

from itertools import chain, product
from inspect import signature, isgenerator
from copy import deepcopy
import tempfile
from operator import methodcaller

import torch

from torch._subclasses.meta_utils import assert_metadata_eq
from torch.testing._internal.common_cuda import with_tf32_off
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, onlyCPU, onlyCUDA, toleranceOverride, tol, skipMeta)
from torch.testing._internal.common_modules import module_db, modules, ModuleErrorEnum, TrainEvalMode
from torch.testing._internal.common_utils import (
    TestCase, run_tests, freeze_rng_state, mock_wrapper, get_tensors_from, gradcheck,
    gradgradcheck)
from unittest.mock import patch, call


class TestModule(TestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    precision = 1e-5
    rel_tol = 1e-5

    def _assert_module_parameters_and_buffer_are(self, module, device, dtype):
        # Check device placement and dtype for created parameters and buffers.
        # Only verify floating point dtypes since that's what the kwarg or methods
        # such as `float()` applies to.
        if not isinstance(device, torch.device):
            device = torch.device(device)

        def _check_module(items, name, device=device, dtype=dtype):
            for item_name, item in items:
                self.assertEqual(
                    item.device, device,
                    f'{name} {item_name} is on device {item.device} instead of the expected device {device}')
                if item.dtype.is_floating_point:
                    self.assertEqual(
                        item.dtype, dtype,
                        f'{name} {item_name} is of dtype {item.dtype} instead of the expected dtype {dtype}')
        _check_module(module.named_parameters(), "Parameter")
        _check_module(module.named_buffers(), "Buffer")

    @modules(module_db)
    def test_forward(self, device, dtype, module_info, training):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        dtype_to_method_caller = {
            torch.float32: methodcaller("float"),
            torch.float64: methodcaller("double"),
        }
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)
                m.train(training)

                # === Do forward pass. ===
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                outputs = m(*args, **kwargs)

                # === Compare outputs to a reference if one is specified. ===
                # TODO: Handle precision
                reference_fn = module_input.reference_fn
                if reference_fn is not None:
                    ref_outputs = reference_fn(m, *args, **kwargs)
                    self.assertEqual(outputs, ref_outputs)

                # === Use the method call and verify the parameters and buffers ===
                if dtype in dtype_to_method_caller:
                    dtype_to_method_caller[dtype](m)
                    m(*args, **kwargs)
                    self._assert_module_parameters_and_buffer_are(m, device, dtype)

    # Tests passing factory kwargs (e.g. device / dtype) during module instantiation.
    # They should be applied to any created parameters and buffers.
    @modules(module_db)
    def test_factory_kwargs(self, device, dtype, module_info, training):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
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
                    m.train(training)

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
                        m.train(training)
                        uninit_param_new.mock.assert_has_calls(
                            [call(device=device, dtype=dtype) for _ in uninit_param_new.mock.mock_calls])
                        uninit_buffer_new.mock.assert_has_calls(
                            [call(device=device, dtype=dtype) for _ in uninit_buffer_new.mock.mock_calls])
            else:
                # Check device placement and dtype for created parameters and buffers.
                # Only verify floating point dtypes since that's what the kwarg applies to.
                m = module_cls(*args, **kwargs)
                m.train(training)
                self._assert_module_parameters_and_buffer_are(m, device, dtype)

    @onlyCUDA
    @modules(module_db)
    def test_multiple_device_transfer(self, device, dtype, module_info, training):
        module_cls = module_info.module_cls
        module_inputs_device = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                              requires_grad=False, training=training)
        module_inputs_cpu = module_info.module_inputs_func(module_info, device="cpu", dtype=dtype,
                                                           requires_grad=False, training=training)
        for module_input_device, module_input_cpu in zip(module_inputs_device, module_inputs_cpu):
            if module_input_device.forward_input is None:
                continue

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input_device.constructor_input.args, module_input_device.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)
                m.train(training)

                # === Do forward pass on GPU ===
                input_device_args = module_input_device.forward_input.args
                input_device_kwargs = module_input_device.forward_input.kwargs
                m(*input_device_args, **input_device_kwargs)
                self._assert_module_parameters_and_buffer_are(m, device, dtype)

                # === Move to CPU ===
                input_cpu_args = module_input_cpu.forward_input.args
                input_cpu_kwargs = module_input_cpu.forward_input.kwargs
                m.cpu()
                m(*input_cpu_args, **input_cpu_kwargs)
                self._assert_module_parameters_and_buffer_are(m, "cpu", dtype)

                # === Move back to GPU and forward pass ===
                m.cuda()
                m(*input_device_args, **input_device_kwargs)
                self._assert_module_parameters_and_buffer_are(m, device, dtype)

                if torch.cuda.device_count() >= 2:
                    # === test cross-GPU transfer works
                    def _to_device1(objs):
                        if isinstance(objs, (tuple, list)):
                            return type(objs)(_to_device1(item) for item in objs)
                        elif isinstance(objs, dict):
                            return {name: _to_device1(item) for name, item in objs.items()}
                        elif isinstance(objs, torch.Tensor):
                            return objs.cuda(1)
                        else:
                            return objs
                    input_device_1_args = _to_device1(input_device_args)
                    input_device_1_kwargs = _to_device1(input_device_kwargs)

                    m.cuda(1)
                    with torch.cuda.device(1):
                        m(*input_device_1_args, **input_device_1_kwargs)
                    self._assert_module_parameters_and_buffer_are(m, torch.device("cuda:1"), dtype)

    @modules(module_db)
    def test_repr(self, device, dtype, module_info, training):
        # Test module can be represented with repr and str without errors.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        for module_input in module_inputs:
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            m.train(training)

            # Check that these methods do not raise errors
            m.__repr__()
            str(m)

    @modules(module_db)
    def test_pickle(self, device, dtype, module_info, training):
        # Test that module can be pickled and unpickled.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)
                m.train(training)

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

    @skipMeta
    @modules([module_info for module_info in module_db
              if 'inplace' in signature(module_info.module_cls).parameters])
    def test_check_inplace(self, device, dtype, module_info, training):
        # Check if the inplace variant of the module gives the same result as the out of place
        # variant.
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m_op = module_cls(*args, **kwargs, inplace=False)
            m_op.to(device).to(dtype)
            m_op.train(training)
            m_inplace = module_cls(*args, **kwargs, inplace=True)
            m_inplace.to(device).to(dtype)
            m_inplace.train(training)

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
            input_clone_version = input_arg_clone[0]._version
            with freeze_rng_state():
                output_ip = m_inplace(*input_arg_clone, **input_kwargs)
            self.assertGreater(input_arg_clone[0]._version, input_clone_version)
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
    def test_non_contiguous_tensors(self, device, dtype, module_info, training):
        # Check modules work with non-contiguous tensors

        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)

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
            m.train(training)

            self._retain_grad((input_args, input_kwargs))

            # === Forward with default input
            with freeze_rng_state():
                default_output = m(*input_args, **input_kwargs)
                if isinstance(default_output, torch.Tensor):
                    grad_output = default_output.clone().detach_().normal_()
                    default_output.backward(grad_output, retain_graph=True)
                else:
                    grad_output = tuple(self._traverse_obj(o, lambda o: o.clone().detach_().normal_() if o.requires_grad else None)
                                        for o in default_output)
                    flattened_default_output, _ = torch.utils._pytree.tree_flatten(default_output)
                    flattened_grad_output, _ = torch.utils._pytree.tree_flatten(grad_output)
                    for o, g_o in zip(flattened_default_output, flattened_grad_output):
                        if (o.requires_grad):
                            o.backward(g_o, retain_graph=True)

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
                    if isinstance(out, torch.Tensor):
                        out.backward(g_out_copy, retain_graph=True)
                    else:
                        flattened_out, _ = torch.utils._pytree.tree_flatten(out)
                        flattened_g_out_copy, _ = torch.utils._pytree.tree_flatten(g_out_copy)
                        for o, g_o in zip(flattened_out, flattened_g_out_copy):
                            if o.requires_grad:
                                o.backward(g_o, retain_graph=True)

                input_args_grad, input_kwargs_grad = self._get_grads((in_args, in_kwargs))
                self.assertEqual(out, default_output)
                self.assertEqual(input_args_grad, default_input_args_grad, atol=1e-4, rtol=0)
                self.assertEqual(input_kwargs_grad, default_input_kwargs_grad, atol=1e-4, rtol=0)

                param_grad = [p.grad for p in m.parameters()]
                self.assertEqual(param_grad, default_param_grad)

    def _test_gradients_helper(self, device, dtype, module_info, training, check):
        # Check gradients
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)
        # === Set nondet tol for gradcheck to user-defined value if on CUDA and cudNN is enabled
        gradcheck_nondet_tol = 0.0
        if (torch.device(device).type == 'cuda' and torch.backends.cudnn.enabled):
            gradcheck_nondet_tol = module_info.gradcheck_nondet_tol

        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            m.train(training)

            params = tuple(m.parameters())

            # === Lazy modules need to see an input to initialize params before gradcheck is run. ===
            input_args, input_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
            if issubclass(module_info.module_cls, torch.nn.modules.lazy.LazyModuleMixin):
                with torch.no_grad():
                    m(*input_args, **input_kwargs)

            # === Perform gradient check on the input_args ===
            other_kwargs = {}
            kwarg_tensors = []
            for name, obj in input_kwargs.items():
                if isinstance(obj, torch.Tensor):
                    kwarg_tensors.append((name, obj))
                else:
                    other_kwargs[name] = obj

            def fn_to_gradcheck(*flat_input_and_params):
                input_and_params = torch.utils._pytree.tree_unflatten(flat_input_and_params, flat_spec)
                new_input_args = input_and_params[:len(input_args)]
                kwarg_args = input_and_params[-len(kwarg_tensors):]
                new_kwargs = {name: obj for (name, _), obj in zip(kwarg_tensors, kwarg_args)}

                with freeze_rng_state():
                    output = m(*new_input_args, **new_kwargs, **other_kwargs)
                    output_flattened, _ = torch.utils._pytree.tree_flatten(output)
                    return output_flattened

            # check total derivative
            grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
            flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)

            self.assertTrue(check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol))

            # check partial derivatives
            old_params_requires_grad = [p.requires_grad for p in params]
            for p in params:
                p.requires_grad = False

            old_kwargs_requires_grad = [obj.requires_grad for (_, obj) in kwarg_tensors]
            for (_, obj) in kwarg_tensors:
                obj.requires_grad = False

            for p, old in zip(params, old_params_requires_grad):
                p.requires_grad = old
                grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
                flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
                self.assertTrue(check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol))
                p.requires_grad = False

            for (_, obj), old in zip(kwarg_tensors, old_kwargs_requires_grad):
                obj.requires_grad = old
                grad_input = input_args + params + tuple(obj for (_, obj) in kwarg_tensors)
                flat_input, flat_spec = torch.utils._pytree.tree_flatten(grad_input)
                self.assertTrue(check(fn_to_gradcheck, flat_input, nondet_tol=gradcheck_nondet_tol))
                obj.requires_grad = False

    @modules(module_db, allowed_dtypes=[torch.double])
    def test_grad(self, device, dtype, module_info, training):
        self._test_gradients_helper(device, dtype, module_info, training, gradcheck)

    @modules([m for m in module_db if m.supports_gradgrad],
             allowed_dtypes=[torch.double])
    def test_gradgrad(self, device, dtype, module_info, training):
        self._test_gradients_helper(device, dtype, module_info, training, gradgradcheck)

    @onlyCUDA
    @with_tf32_off  # Turn off TF32 to compute at full precision https://github.com/pytorch/pytorch/issues/86798
    @toleranceOverride({torch.float32: tol(5e-2, 0),
                        torch.float64: tol(4e-4, 0)})
    @modules(module_db)
    def test_cpu_gpu_parity(self, device, dtype, module_info, training):
        # TODO: RNN / GRU / LSTM don't support backwards on eval mode for cuDNN; skip this in a
        # nicer way for eval mode only.
        # See https://github.com/pytorch/pytorch/issues/79161
        rnn_modules = {torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM}
        if (module_info.module_cls in rnn_modules
                and not training
                and 'cuda' in device
                and torch.backends.cudnn.enabled):
            return

        # Test cpu and gpu results are the same
        module_cls = module_info.module_cls
        module_inputs_cpu = module_info.module_inputs_func(module_info, device="cpu", dtype=dtype,
                                                           requires_grad=True, training=training)

        def _to_device(obj):
            if isinstance(obj, torch.Tensor):
                res = obj.detach().to(device=device)
                res.requires_grad = obj.requires_grad
                return res
            elif isinstance(obj, tuple):
                return tuple(_to_device(o) for o in obj)
            elif isinstance(obj, dict):
                return {key: _to_device(o) for key, o in obj.items()}
            else:
                return deepcopy(obj)

        for module_input in module_inputs_cpu:
            # === Move input from cpu to device ===
            cpu_forward_args = module_input.forward_input.args
            cpu_forward_kwargs = module_input.forward_input.kwargs

            gpu_forward_args, gpu_forward_kwargs = _to_device((cpu_forward_args, cpu_forward_kwargs))

            self._retain_grad((cpu_forward_args, cpu_forward_kwargs, gpu_forward_args, gpu_forward_kwargs))

            # === Construct module on cpu and gpu ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

            cpu_module = module_cls(*args, **kwargs).to(dtype).to("cpu")
            cpu_module.train(training)
            gpu_module = module_cls(*args, **kwargs).to(dtype).to(device)
            gpu_module.train(training)

            # === Lazy modules need to see an input to initialize params ===
            if issubclass(module_cls, torch.nn.modules.lazy.LazyModuleMixin):
                with torch.no_grad():
                    cpu_module(*cpu_forward_args, **cpu_forward_kwargs)
                    gpu_module(*gpu_forward_args, **gpu_forward_kwargs)

            for cpu_p, gpu_p in zip(cpu_module.parameters(), gpu_module.parameters()):
                gpu_p.data.copy_(cpu_p)

            # === Compare forward output between cpu and gpu ===
            cpu_outputs = cpu_module(*cpu_forward_args, **cpu_forward_kwargs)
            gpu_outputs = gpu_module(*gpu_forward_args, **gpu_forward_kwargs)

            self.assertEqual(cpu_outputs, gpu_outputs)

            # === Run backwards on CPU and GPU and compare results ===
            def check_backward(cpu_output, gpu_output):
                cpu_grad_output = cpu_output.clone().normal_()
                gpu_grad_output = cpu_grad_output.type_as(gpu_output)

                cpu_output.backward(cpu_grad_output, retain_graph=True)
                gpu_output.backward(gpu_grad_output, retain_graph=True)

                cpu_grad_input = self._get_grads(cpu_forward_args)
                gpu_grad_input = self._get_grads(gpu_forward_args)
                self.assertEqual(cpu_grad_input, gpu_grad_input)

                for cpu_p, gpu_p in zip(cpu_module.parameters(), gpu_module.parameters()):
                    self.assertEqual(cpu_p.grad, gpu_p.grad)

                cpu_grad_kwarg_input = self._get_grads(cpu_forward_kwargs)
                gpu_grad_kwarg_input = self._get_grads(gpu_forward_kwargs)
                self.assertEqual(cpu_grad_kwarg_input, gpu_grad_kwarg_input)

            for _ in range(5):
                if isinstance(cpu_outputs, torch.Tensor):
                    check_backward(cpu_outputs, gpu_outputs)
                else:
                    flatten_cpu_outputs, _ = torch.utils._pytree.tree_flatten(cpu_outputs)
                    flatten_gpu_outputs, _ = torch.utils._pytree.tree_flatten(gpu_outputs)
                    for cpu_output, gpu_output in zip(flatten_cpu_outputs, flatten_gpu_outputs):
                        if cpu_output.requires_grad:
                            check_backward(cpu_output, gpu_output)

    @with_tf32_off
    @modules(module_db)
    def test_memory_format(self, device, dtype, module_info, training):
        is_sm86or80 = device.startswith("cuda") and (torch.cuda.get_device_capability(0) == (8, 6)
                                                     or torch.cuda.get_device_capability(0) == (8, 0))
        # TODO tighten it to a specific module
        atol, rtol = (3e-3, 7e-3) if is_sm86or80 else (None, None)
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=True, training=training)
        module_memformat_affects_out = module_info.module_memformat_affects_out

        def _get_mem_formats(channels_last=False, channels_last_3d=False):
            if channels_last:
                return ([torch.contiguous_format, torch.channels_last],
                        [torch.preserve_format, torch.contiguous_format, torch.channels_last])
            elif channels_last_3d:
                return ([torch.contiguous_format, torch.channels_last_3d],
                        [torch.preserve_format, torch.contiguous_format, torch.channels_last_3d])
            else:
                return ([torch.contiguous_format],
                        [torch.preserve_format, torch.contiguous_format])

        # Check that at least one Tensor input has dim == n
        def _check_dims(obj, n):
            if isinstance(obj, torch.Tensor):
                return obj.dim() == n
            elif isinstance(obj, (tuple, list)):
                return any(_check_dims(o, n) for o in obj)
            else:
                return False

        # Called after _check_dims, when we know that >= 1 tensor can be converted to mem_format
        def _to_mem_format(mem_format, obj):
            def inner_to_mem_format(obj):
                d = obj.dim()
                if ((mem_format == torch.channels_last and d != 4)
                   or (mem_format == torch.channels_last_3d and d != 5)):
                    return obj.clone().detach().requires_grad_(obj.requires_grad)
                return obj.clone().to(memory_format=mem_format).detach().requires_grad_(obj.requires_grad)

            return self._traverse_obj(obj, inner_to_mem_format)

        def _check_out_mem_format(output, input_mem_format, module_mem_format):
            def inner_check_out_mem_format(output):
                d = output.dim()
                if (d == 4 and ((input_mem_format == torch.channels_last)
                                or (module_mem_format == torch.channels_last and module_memformat_affects_out))):
                    self.assertTrue(output.is_contiguous(memory_format=torch.channels_last))
                elif (d == 5 and ((input_mem_format == torch.channels_last_3d)
                                  or (module_mem_format == torch.channels_last_3d and module_memformat_affects_out))):
                    self.assertTrue(output.is_contiguous(memory_format=torch.channels_last_3d))
                else:
                    self.assertTrue(output.is_contiguous())
            return self._traverse_obj(output, inner_check_out_mem_format)

        def _req_grad(t):
            return isinstance(t, torch.Tensor) and t.requires_grad

        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            supports_channels_last = _check_dims(module_input.forward_input.args, 4)
            supports_channels_last_3d = _check_dims(module_input.forward_input.args, 5)
            input_mem_formats, module_mem_formats = _get_mem_formats(supports_channels_last, supports_channels_last_3d)

            with freeze_rng_state():
                # === Instantiate the module. ===
                args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs

                m = module_cls(*args, **kwargs)
                m.to(device).to(dtype)
                m.train(training)

                # === Get output in (contiguous, contiguous) configuration. ===
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                desired_outputs = m(*args, **kwargs)
                # === Do backward pass. ===
                ref_diff_outputs = tuple(t for t in torch.utils._pytree.tree_flatten(desired_outputs)[0] if _req_grad(t))
                if training and len(ref_diff_outputs) > 0:
                    params = tuple(p for p in m.parameters())
                    ref_diff_inputs = tuple(
                        t
                        for t in torch.utils._pytree.tree_flatten((args, kwargs, params))[0]
                        if _req_grad(t)
                    )
                    ref_grad_outputs = tuple(
                        torch.rand_like(t)
                        for t in ref_diff_outputs
                    )
                    ref_grad_inputs = torch.autograd.grad(
                        ref_diff_outputs,
                        ref_diff_inputs,
                        grad_outputs=ref_grad_outputs,
                    )

                for input_mem_format in input_mem_formats:
                    # === Change memformat of input. ===
                    d_args = _to_mem_format(input_mem_format, module_input.forward_input.args)
                    d_kwargs = _to_mem_format(input_mem_format, module_input.forward_input.kwargs)

                    # See https://github.com/pytorch/pytorch/issues/107861
                    # When inductor tests are turned on, the setting of requires_grad will be lost
                    for t1, t2 in zip(
                        torch.utils._pytree.tree_flatten(d_args)[0],
                        torch.utils._pytree.tree_flatten(module_input.forward_input.args)[0],
                    ):
                        t1.requires_grad_(t2.requires_grad)
                    for t1, t2 in zip(
                        torch.utils._pytree.tree_flatten(d_kwargs)[0],
                        torch.utils._pytree.tree_flatten(module_input.forward_input.kwargs)[0],
                    ):
                        t1.requires_grad_(t2.requires_grad)

                    module_input.forward_input.args = d_args
                    module_input.forward_input.kwargs = d_kwargs

                    for module_mem_format in module_mem_formats:
                        # === Change memformat of module ===
                        m.to(memory_format=module_mem_format)

                        # === Do forward pass. ===
                        args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                        outputs = m(*args, **kwargs)

                        # === Compare outputs to (contiguous, contiguous) output. ===
                        if input_mem_format != torch.contiguous_format or module_mem_format != torch.contiguous_format:
                            self.assertEqual(outputs, desired_outputs, rtol=rtol, atol=atol)

                        # === Check mem format of output. ===
                        _check_out_mem_format(outputs, input_mem_format, module_mem_format)

                        # === Do backward pass. ===
                        diff_outputs = tuple(t for t in torch.utils._pytree.tree_flatten(outputs)[0] if _req_grad(t))
                        if training and len(diff_outputs) > 0:
                            params = tuple(p for p in m.parameters())
                            diff_inputs = tuple(
                                t
                                for t in torch.utils._pytree.tree_flatten((args, kwargs, params))[0]
                                if _req_grad(t)
                            )
                            grad_outputs = tuple(
                                torch.empty_like(t1).copy_(t2)
                                for (t1, t2) in zip(diff_outputs, ref_grad_outputs)
                            )

                            grad_inputs = torch.autograd.grad(
                                diff_outputs,
                                diff_inputs,
                                grad_outputs=grad_outputs,
                            )

                            if (
                                input_mem_format != torch.contiguous_format
                                or module_mem_format != torch.contiguous_format
                            ):
                                self.assertEqual(
                                    grad_inputs, ref_grad_inputs, rtol=rtol, atol=atol
                                )

                            # === Check mem format of grad_inputs. ===
                            _check_out_mem_format(grad_inputs, input_mem_format, module_mem_format)

    # Test whether train and eval modes differ for each module. Use to verify
    # that the ModuleInfo entry flag is correct.
    @modules(module_db, train_eval_mode=TrainEvalMode.train_only)
    def test_if_train_and_eval_modes_differ(self, device, dtype, module_info, training):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)

        # Run forward inputs through to see if the training flag is accessed during forward.
        for module_input in module_inputs:
            if module_input.forward_input is None:
                continue

            # === Instantiate the module. ===
            args, kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            m = module_cls(*args, **kwargs)
            m.to(device).to(dtype)
            m.train(training)

            # Remove training attribute and see if forward still works.
            delattr(m, 'training')

            # === Do forward pass. ===
            try:
                args, kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                m(*args, **kwargs)
            except AttributeError as e:
                if "'training'" in str(e):
                    self.assertTrue(module_info.train_and_eval_differ,
                                    f"The ModuleInfo entry for {module_info.name} has "
                                    "train_and_eval_differ=False, but the training mode was found to "
                                    "affect the forward pass. Consider setting train_and_eval_differ=True "
                                    "for this ModuleInfo entry.")
                else:
                    raise e


    @onlyCPU
    @modules(module_db)
    def test_device_ctx_init(self, device, dtype, module_info, training):
        module_cls = module_info.module_cls
        module_inputs = module_info.module_inputs_func(module_info, device=device, dtype=dtype,
                                                       requires_grad=False, training=training)
        with torch.device('meta'):
            module_inputs_meta = module_info.module_inputs_func(module_info, device=None, dtype=dtype,
                                                                requires_grad=False, training=training)

        for module_input, module_input_meta in zip(module_inputs, module_inputs_meta):
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            fw_args, fw_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs

            c_args_meta, c_kwargs_meta = module_input_meta.constructor_input.args, module_input_meta.constructor_input.kwargs
            fw_args_meta, fw_kwargs_meta = module_input_meta.forward_input.args, module_input_meta.forward_input.kwargs

            m_cpu = module_cls(*c_args, **c_kwargs)

            with torch.device('meta'):
                m = module_cls(*c_args_meta, **c_kwargs_meta)

            for (p_meta, p_cpu) in chain(zip(m.parameters(), m_cpu.parameters()),
                                         zip(m.buffers(), m_cpu.buffers())):
                if torch.nn.parameter.is_lazy(p_meta):
                    continue
                self.assertTrue(p_meta.is_meta)
                assert_metadata_eq(self.assertEqual, p_meta, p_cpu)


    @modules([module for module in module_db if module.module_error_inputs_func is not None])
    def test_errors(self, device, dtype, module_info, training):
        module_cls = module_info.module_cls
        error_inputs = module_info.module_error_inputs_func(module_info, device=device, dtype=dtype,
                                                            requires_grad=False, training=training)
        for error_input in error_inputs:
            module_input = error_input.module_error_input
            c_args, c_kwargs = module_input.constructor_input.args, module_input.constructor_input.kwargs
            if error_input.error_on == ModuleErrorEnum.CONSTRUCTION_ERROR:
                with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                    m = module_cls(*c_args, **c_kwargs)
            elif error_input.error_on == ModuleErrorEnum.FORWARD_ERROR:
                m = module_cls(*c_args, **c_kwargs)
                fw_args, fw_kwargs = module_input.forward_input.args, module_input.forward_input.kwargs
                with self.assertRaisesRegex(error_input.error_type, error_input.error_regex):
                    m(*fw_args, **fw_kwargs)
            else:
                raise NotImplementedError(f"Unknown error type {error_input.error_on}")


instantiate_device_type_tests(TestModule, globals(), allow_mps=True)

if __name__ == '__main__':
    run_tests()
