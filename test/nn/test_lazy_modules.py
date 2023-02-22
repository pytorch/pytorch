# Owner(s): ["module: nn"]
import unittest
import pickle

import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter, UninitializedBuffer
from torch.nn import Parameter
from torch.testing._internal.common_utils import TestCase, run_tests, suppress_warnings
from torch.testing._internal.common_cuda import TEST_CUDA

class LazyModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
    pass


class TestLazyModules(TestCase):

    @suppress_warnings
    def test_lazy_module_parameter(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        state_dict = module.state_dict()
        self.assertIsInstance(state_dict['test_param'], UninitializedParameter)
        new_module = LazyModule()
        # An error is raised when there is an attempt to replace an existing parameter
        # with an uninitialized one
        new_module.register_parameter('test_param', nn.Parameter(torch.ones(5, 5)))
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            new_module.load_state_dict(state_dict)
        # Uninitialized parameters are overriden when the state dict to be loaded contains a valid one
        new_module = LazyModule()
        new_module.register_parameter('test_param', nn.Parameter(torch.ones(5, 5)))
        module.load_state_dict(new_module.state_dict())
        self.assertEqual(module.test_param, torch.ones((5, 5)))

        # Uninitialized parameters are left unchanged
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())

        new_module = LazyModule()
        new_module.register_parameter('test_param', UninitializedParameter())
        module.load_state_dict(new_module.state_dict())
        self.assertTrue(module.has_uninitialized_params())

    @suppress_warnings
    def test_lazy_module_buffer(self):
        module = LazyModule()
        module.register_buffer('test_buffer', UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())
        state_dict = module.state_dict()
        self.assertIsInstance(state_dict['test_buffer'], UninitializedBuffer)
        new_module = LazyModule()
        # An error is raised when there is an attempt to replace an existing parameter
        # with an uninitialized one
        new_module.register_buffer('test_buffer', torch.ones(5, 5))
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            new_module.load_state_dict(state_dict)
        # Uninitialized parameters are overriden when the state dict to be loaded contains a valid one
        new_module = LazyModule()
        new_module.register_buffer('test_buffer', torch.ones(5, 5))
        module.load_state_dict(new_module.state_dict())
        self.assertEqual(module.test_buffer, torch.ones((5, 5)))

        # Uninitialized parameters are left unchanged
        module = LazyModule()
        module.register_buffer('test_buffer', UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())

        new_module = LazyModule()
        new_module.register_buffer('test_buffer', UninitializedBuffer())
        module.load_state_dict(new_module.state_dict())
        module.load_state_dict(new_module.state_dict())
        self.assertTrue(module.has_uninitialized_params())

    @suppress_warnings
    def test_lazy_module_jit_param(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, 'run a forward pass'):
            torch.jit.script(module)

    @suppress_warnings
    def test_lazy_module_jit_buffer(self):
        module = LazyModule()
        module.register_buffer('test_buffer', UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, 'run a forward pass'):
            torch.jit.script(module)

    @suppress_warnings
    def test_lazy_share_memory_param(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, 'share memory on an uninitialized'):
            module.share_memory()

    @suppress_warnings
    def test_lazy_share_memory_buffer(self):
        module = LazyModule()
        module.register_buffer('test_buffer', UninitializedBuffer())
        self.assertTrue(module.has_uninitialized_params())
        with self.assertRaisesRegex(RuntimeError, 'share memory on an uninitialized'):
            module.share_memory()

    @suppress_warnings
    def test_linear(self):
        module = nn.LazyLinear(10)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(5, 5)
        module(input)
        self.assertIsInstance(module, nn.Linear)
        self.assertNotIsInstance(module, nn.LazyLinear)
        self.assertTrue(module.weight.shape == (10, 5))
        self.assertTrue(module.bias.shape == (10,))
        y = module(input)
        self.assertTrue(torch.equal(torch.nn.functional.linear(input, module.weight, module.bias), y))

    @suppress_warnings
    def test_lazy_linear_pickle(self):
        module = nn.LazyLinear(10)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(module, nn.LazyLinear)
        self.assertIsInstance(module.weight, UninitializedParameter)
        self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(5, 5)
        module(input)  # fully materialized
        new_module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(new_module, nn.Linear)
        self.assertNotIsInstance(new_module, nn.LazyLinear)
        self.assertTrue(new_module.weight.shape == (10, 5))
        self.assertNotIsInstance(new_module.weight, UninitializedParameter)
        self.assertTrue(new_module.bias.shape == (10,))
        self.assertNotIsInstance(new_module.bias, UninitializedParameter)

    @suppress_warnings
    def test_linear_state(self):
        module = nn.Linear(5, 10)
        lazy_module = nn.LazyLinear(10)
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Linear one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertTrue(lazy_module.weight.shape == (10, 5))
        self.assertTrue(lazy_module.bias.shape == (10,))

        module = nn.Linear(5, 10)
        lazy_module = nn.LazyLinear(10)
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            module.load_state_dict(lazy_module.state_dict())

    def _check_lazy_conv(self, cls, lazy_cls, func, init_args, input_shape,
                         expected_weight_shape, expected_bias_shape):
        module = lazy_cls(*init_args)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(*input_shape)
        module(input)
        self.assertIsInstance(module, cls)
        self.assertNotIsInstance(module, lazy_cls)
        self.assertEqual(module.weight.shape, expected_weight_shape)
        if module.bias is not None:
            self.assertEqual(module.bias.shape, expected_bias_shape)
        y = module(input)
        self.assertTrue(torch.equal(func(input, module.weight, module.bias), y))

    def _check_lazy_conv_pickle(self, cls, lazy_cls, init_args, input_shape,
                                expected_weight_shape, expected_bias_shape):
        module = lazy_cls(*init_args)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(module, lazy_cls)
        self.assertIsInstance(module.weight, UninitializedParameter)
        if module.bias is not None:
            self.assertIsInstance(module.bias, UninitializedParameter)
        input = torch.ones(*input_shape)
        module(input)  # fully materialized
        new_module = pickle.loads(pickle.dumps(module))
        self.assertIsInstance(new_module, cls)
        self.assertNotIsInstance(new_module, lazy_cls)
        self.assertEqual(new_module.weight.shape, expected_weight_shape)
        self.assertNotIsInstance(new_module.weight, UninitializedParameter)
        if new_module.bias is not None:
            self.assertEqual(new_module.bias.shape, expected_bias_shape)
            self.assertNotIsInstance(new_module.bias, UninitializedParameter)

    def _check_lazy_conv_state(self, gen_module, gen_lazy_module,
                               expected_weight_shape, expected_bias_shape):
        module = gen_module()
        lazy_module = gen_lazy_module()
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Conv one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertEqual(lazy_module.weight.shape, expected_weight_shape)
        if lazy_module.bias is not None:
            self.assertEqual(lazy_module.bias.shape, expected_bias_shape)

        module = gen_module()
        lazy_module = gen_lazy_module()
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            module.load_state_dict(lazy_module.state_dict())


    def test_lazy_pre_forward_hook(self):
        """
        This test is to test whether lazymodule can register other pre-forward hook
        functions successfully.
        """
        class TestModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def initialize_parameters(self, input):
                return None

            def forward(self, input):
                return input

        def hook_function(module, input):
            return input[0] + 1

        module = TestModule()
        module.register_forward_pre_hook(hook_function)
        output = module(torch.zeros(2, 2))
        self.assertEqual(output, torch.ones(2, 2))

    def test_lazy_forward_hook(self):
        """
        This test is to test whether lazymodule can register other forward hook
        functions successfully.
        """
        class TestModule(torch.nn.modules.lazy.LazyModuleMixin, torch.nn.Module):
            def initialize_parameters(self, input):
                return None

            def forward(self, input):
                return input

        def hook_function(module, input, output):
            return input[0] + 1

        module = TestModule()
        module.register_forward_hook(hook_function)
        output = module(torch.zeros(2, 2))
        self.assertEqual(output, torch.ones(2, 2))

    @suppress_warnings
    def test_lazy_conv1d(self):
        self._check_lazy_conv(nn.Conv1d, nn.LazyConv1d, torch.nn.functional.conv1d,
                              (32, 2), (192, 16, 50), (32, 16, 2), (32,))

    @suppress_warnings
    def test_lazy_conv1d_pickle(self):
        self._check_lazy_conv_pickle(nn.Conv1d, nn.LazyConv1d, (32, 2), (192, 16, 50),
                                     (32, 16, 2), (32,))

    @suppress_warnings
    def test_lazy_conv1d_state(self):
        self._check_lazy_conv_state(lambda: nn.Conv1d(16, 32, 2),
                                    lambda: nn.LazyConv1d(32, 2),
                                    (32, 16, 2), (32,))

    @suppress_warnings
    def test_lazy_conv2d(self):
        self._check_lazy_conv(nn.Conv2d, nn.LazyConv2d, torch.nn.functional.conv2d,
                              (32, 2), (192, 16, 8, 6), (32, 16, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv2d_pickle(self):
        self._check_lazy_conv_pickle(nn.Conv2d, nn.LazyConv2d, (32, 2), (192, 16, 8, 6),
                                     (32, 16, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv2d_state(self):
        self._check_lazy_conv_state(lambda: nn.Conv2d(16, 32, 2),
                                    lambda: nn.LazyConv2d(32, 2),
                                    (32, 16, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv3d(self):
        self._check_lazy_conv(nn.Conv3d, nn.LazyConv3d, torch.nn.functional.conv3d,
                              (32, 2), (192, 16, 8, 7, 6), (32, 16, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv3d_pickle(self):
        self._check_lazy_conv_pickle(nn.Conv3d, nn.LazyConv3d, (32, 2), (192, 16, 8, 7, 6),
                                     (32, 16, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv3d_state(self):
        self._check_lazy_conv_state(lambda: nn.Conv3d(16, 32, 2),
                                    lambda: nn.LazyConv3d(32, 2),
                                    (32, 16, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transposed1d(self):
        self._check_lazy_conv(nn.ConvTranspose1d, nn.LazyConvTranspose1d, torch.nn.functional.conv_transpose1d,
                              (32, 2), (192, 16, 50), (16, 32, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose1d_pickle(self):
        self._check_lazy_conv_pickle(nn.ConvTranspose1d, nn.LazyConvTranspose1d, (32, 2),
                                     (192, 16, 50), (16, 32, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose1d_state(self):
        self._check_lazy_conv_state(lambda: nn.ConvTranspose1d(16, 32, 2),
                                    lambda: nn.LazyConvTranspose1d(32, 2),
                                    (16, 32, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose2d(self):
        self._check_lazy_conv(nn.ConvTranspose2d, nn.LazyConvTranspose2d, torch.nn.functional.conv_transpose2d,
                              (32, 2), (192, 16, 8, 6), (16, 32, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose2d_pickle(self):
        self._check_lazy_conv_pickle(nn.ConvTranspose2d, nn.LazyConvTranspose2d, (32, 2),
                                     (192, 16, 8, 6), (16, 32, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose2d_state(self):
        self._check_lazy_conv_state(lambda: nn.ConvTranspose2d(16, 32, 2),
                                    lambda: nn.LazyConvTranspose2d(32, 2),
                                    (16, 32, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose3d(self):
        self._check_lazy_conv(nn.ConvTranspose3d, nn.LazyConvTranspose3d, torch.nn.functional.conv_transpose3d,
                              (32, 2), (192, 16, 8, 7, 6), (16, 32, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose3d_pickle(self):
        self._check_lazy_conv_pickle(nn.ConvTranspose3d, nn.LazyConvTranspose3d, (32, 2),
                                     (192, 16, 8, 7, 6), (16, 32, 2, 2, 2), (32,))

    @suppress_warnings
    def test_lazy_conv_transpose3d_state(self):
        self._check_lazy_conv_state(lambda: nn.ConvTranspose3d(16, 32, 2),
                                    lambda: nn.LazyConvTranspose3d(32, 2),
                                    (16, 32, 2, 2, 2), (32,))

    def _check_lazy_norm(self, cls, lazy_cls, input_shape):
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                lazy_module = lazy_cls(affine=affine, track_running_stats=track_running_stats)

                if affine:
                    self.assertIsInstance(lazy_module.weight, UninitializedParameter)
                    self.assertIsInstance(lazy_module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertIsInstance(lazy_module.running_mean, UninitializedBuffer)
                    self.assertIsInstance(lazy_module.running_var, UninitializedBuffer)

                input = torch.ones(*input_shape)
                lazy_output = lazy_module(input)
                self.assertIsInstance(lazy_module, cls)
                self.assertNotIsInstance(lazy_module, lazy_cls)

                num_features = input_shape[1]
                module = cls(num_features, affine=affine, track_running_stats=track_running_stats)
                expected_output = module(input)

                self.assertEqual(lazy_output, expected_output)
                if module.weight is not None:
                    self.assertEqual(lazy_module.weight.shape, module.weight.shape)
                    self.assertEqual(lazy_module.weight, module.weight)
                if module.bias is not None:
                    self.assertEqual(lazy_module.bias.shape, module.bias.shape)
                    self.assertEqual(lazy_module.bias, module.bias)
                if module.running_mean is not None:
                    self.assertEqual(lazy_module.running_mean.shape, module.running_mean.shape)
                    self.assertEqual(lazy_module.running_mean, module.running_mean)
                if module.running_var is not None:
                    self.assertEqual(lazy_module.running_var.shape, module.running_var.shape)
                    self.assertEqual(lazy_module.running_var, module.running_var)
                if module.num_batches_tracked is not None:
                    self.assertEqual(lazy_module.num_batches_tracked.shape, module.num_batches_tracked.shape)
                    self.assertEqual(lazy_module.num_batches_tracked, module.num_batches_tracked)

    def _check_lazy_norm_pickle(self, cls, lazy_cls, input_shape):
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                module = lazy_cls(affine=affine, track_running_stats=track_running_stats)
                module = pickle.loads(pickle.dumps(module))

                self.assertIsInstance(module, lazy_cls)
                if affine:
                    self.assertIsInstance(module.weight, UninitializedParameter)
                    self.assertIsInstance(module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertIsInstance(module.running_mean, UninitializedBuffer)
                    self.assertIsInstance(module.running_var, UninitializedBuffer)

                input = torch.ones(*input_shape)
                module(input)  # fully materialized
                module = pickle.loads(pickle.dumps(module))

                self.assertNotIsInstance(module, lazy_cls)
                self.assertIsInstance(module, cls)
                if affine:
                    self.assertNotIsInstance(module.weight, UninitializedParameter)
                    self.assertNotIsInstance(module.bias, UninitializedParameter)
                if track_running_stats:
                    self.assertNotIsInstance(module.running_mean, UninitializedBuffer)
                    self.assertNotIsInstance(module.running_var, UninitializedBuffer)

    def _check_lazy_batchnorm_state(self, cls, lazy_cls):
        module = cls(10)
        lazy_module = lazy_cls(affine=True, track_running_stats=True)
        lazy_module.load_state_dict(module.state_dict())
        # Parameters have been initialized but the module won't become a full
        # Conv one until the first iteration. This is due to
        # limitations on the state_dict loading logic
        self.assertFalse(lazy_module.has_uninitialized_params())
        self.assertEqual(lazy_module.weight.shape, (10,))
        self.assertEqual(lazy_module.bias.shape, (10,))
        self.assertEqual(lazy_module.running_mean.shape, (10,))
        self.assertEqual(lazy_module.running_var.shape, (10,))

        module = cls(10)
        lazy_module = lazy_cls()
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            module.load_state_dict(lazy_module.state_dict())

    def _check_lazy_instancenorm_state(self, cls, lazy_cls):
        for affine in [False, True]:
            for track_running_stats in [False, True]:
                module = cls(10, affine=affine, track_running_stats=track_running_stats)
                lazy_module = lazy_cls(affine=affine, track_running_stats=track_running_stats)
                lazy_module.load_state_dict(module.state_dict())
                # Parameters have been initialized but the module won't become a full
                # InstanceNorm one until the first iteration. This is due to
                # limitations on the state_dict loading logic
                self.assertFalse(lazy_module.has_uninitialized_params())
                if affine:
                    self.assertEqual(lazy_module.weight.shape, (10,))
                    self.assertEqual(lazy_module.bias.shape, (10,))
                if track_running_stats:
                    self.assertEqual(lazy_module.running_mean.shape, (10,))
                    self.assertEqual(lazy_module.running_var.shape, (10,))

        module = cls(10, affine=True, track_running_stats=True)
        lazy_module = lazy_cls(affine=True, track_running_stats=True)
        with self.assertRaisesRegex(RuntimeError, 'shape of an uninitialized'):
            module.load_state_dict(lazy_module.state_dict())

    def test_lazy_batchnorm1d(self):
        self._check_lazy_norm(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6))
        self._check_lazy_norm(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 6))

    def test_lazy_batchnorm1d_pickle(self):
        self._check_lazy_norm_pickle(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 3, 6))
        self._check_lazy_norm_pickle(nn.BatchNorm1d, nn.LazyBatchNorm1d, (16, 6))

    def test_lazy_batchnorm1d_state(self):
        self._check_lazy_batchnorm_state(nn.BatchNorm1d, nn.LazyBatchNorm1d)
        self._check_lazy_batchnorm_state(nn.BatchNorm1d, nn.LazyBatchNorm1d)

    def test_lazy_batchnorm2d(self):
        self._check_lazy_norm(nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7))

    def test_lazy_batchnorm2d_pickle(self):
        self._check_lazy_norm_pickle(nn.BatchNorm2d, nn.LazyBatchNorm2d, (16, 3, 6, 7))

    def test_lazy_batchnorm2d_state(self):
        self._check_lazy_batchnorm_state(nn.BatchNorm2d, nn.LazyBatchNorm2d)
        self._check_lazy_batchnorm_state(nn.BatchNorm2d, nn.LazyBatchNorm2d)

    def test_lazy_batchnorm3d(self):
        self._check_lazy_norm(nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8))

    def test_lazy_batchnorm3d_pickle(self):
        self._check_lazy_norm_pickle(nn.BatchNorm3d, nn.LazyBatchNorm3d, (16, 3, 6, 7, 8))

    def test_lazy_batchnorm3d_state(self):
        self._check_lazy_batchnorm_state(nn.BatchNorm3d, nn.LazyBatchNorm3d)
        self._check_lazy_batchnorm_state(nn.BatchNorm3d, nn.LazyBatchNorm3d)

    def test_lazy_instancenorm1d(self):
        self._check_lazy_norm(nn.InstanceNorm1d, nn.LazyInstanceNorm1d, (16, 3, 6))

    def test_lazy_instancenorm1d_pickle(self):
        self._check_lazy_norm_pickle(nn.InstanceNorm1d, nn.LazyInstanceNorm1d, (16, 3, 6))

    def test_lazy_instancenorm1d_state(self):
        self._check_lazy_instancenorm_state(nn.InstanceNorm1d, nn.LazyInstanceNorm1d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm1d, nn.LazyInstanceNorm1d)

    def test_lazy_instancenorm2d(self):
        self._check_lazy_norm(nn.InstanceNorm2d, nn.LazyInstanceNorm2d, (16, 3, 6, 7))

    def test_lazy_instancenorm2d_pickle(self):
        self._check_lazy_norm_pickle(nn.InstanceNorm2d, nn.LazyInstanceNorm2d, (16, 3, 6, 7))

    def test_lazy_instancenorm2d_state(self):
        self._check_lazy_instancenorm_state(nn.InstanceNorm2d, nn.LazyInstanceNorm2d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm2d, nn.LazyInstanceNorm2d)

    def test_lazy_instancenorm3d(self):
        self._check_lazy_norm(nn.InstanceNorm3d, nn.LazyInstanceNorm3d, (16, 3, 6, 7, 8))

    def test_lazy_instancenorm3d_pickle(self):
        self._check_lazy_norm_pickle(nn.InstanceNorm3d, nn.LazyInstanceNorm3d, (16, 3, 6, 7, 8))

    def test_lazy_instancenorm3d_state(self):
        self._check_lazy_instancenorm_state(nn.InstanceNorm3d, nn.LazyInstanceNorm3d)
        self._check_lazy_instancenorm_state(nn.InstanceNorm3d, nn.LazyInstanceNorm3d)

    @suppress_warnings
    def test_materialize_dtype(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.dtype == torch.get_default_dtype())
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        module.half()
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.dtype == torch.float16)

    @unittest.skipIf(not TEST_CUDA, 'CUDA not available')
    @suppress_warnings
    def test_materialize_device(self):
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.device.type == 'cpu')
        module = LazyModule()
        module.register_parameter('test_param', UninitializedParameter())
        module.cuda()
        module.test_param.materialize(10)
        self.assertTrue(module.test_param.device.type == 'cuda')

    @suppress_warnings
    def test_chained_initialization(self):
        class MyNetwork(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_1 = torch.nn.LazyLinear(15)
                self.linear_2 = torch.nn.LazyLinear(10)

            def forward(self, x):
                y = self.linear_1(x)
                return self.linear_2(y)

        net = MyNetwork()
        net(torch.ones(5, 10))
        self.assertTrue(net.linear_1.weight.shape == (15, 10))
        self.assertTrue(net.linear_1.bias.shape == (15,))
        self.assertTrue(net.linear_2.weight.shape == (10, 15))
        self.assertTrue(net.linear_2.bias.shape == (10,))

    @suppress_warnings
    def test_optimizer_pass(self):
        optimizers = [torch.optim.Adadelta, torch.optim.Adagrad, torch.optim.Adam,
                      torch.optim.AdamW, torch.optim.Adamax,
                      torch.optim.ASGD, torch.optim.SGD, torch.optim.Rprop,
                      torch.optim.RMSprop, torch.optim.LBFGS]

        def run_step(module, optim):
            self.assertIsInstance(optim.param_groups[0]['params'][0], UninitializedParameter)
            module.test_param.materialize(10)
            self.assertIsInstance(optim.param_groups[0]['params'][0], Parameter)
            self.assertNotIsInstance(optim.param_groups[0]['params'][0], UninitializedParameter)
            for p in module.parameters():
                p.grad = torch.rand_like(p)
            if isinstance(optim, torch.optim.LBFGS):
                optim.step(lambda: 1.0)
            else:
                optim.step()

        for optim_cls in optimizers:
            module = LazyModule()
            module.register_parameter('test_param', UninitializedParameter())
            if optim_cls is torch.optim.SGD:
                optim = optim_cls(module.parameters(), lr=0.0)
            elif optim_cls is torch.optim.Adagrad:
                with self.assertRaisesRegex(ValueError, 'uninitialized parameter'):
                    optim = optim_cls(module.parameters())
                continue
            else:
                optim = optim_cls(module.parameters())
            run_step(module, optim)

    @suppress_warnings
    def test_weight_norm(self):
        m = nn.LazyLinear(7)
        with self.assertRaisesRegex(ValueError, 'have uninitialized parameters.'):
            m = torch.nn.utils.weight_norm(m)

    @suppress_warnings
    def test_spectral_norm(self):
        m = nn.LazyLinear(7)
        with self.assertRaisesRegex(ValueError, 'have uninitialized parameters.'):
            m = torch.nn.utils.spectral_norm(m)

    @suppress_warnings
    def test_invalid_functions(self):
        param = torch.nn.parameter.UninitializedParameter()
        with self.assertRaisesRegex(ValueError, 'uninitialized parameter'):
            torch.empty_like(param)

        with self.assertRaisesRegex(ValueError, 'uninitialized parameter'):
            torch.add(param, param)

        with self.assertRaisesRegex(ValueError, 'uninitialized parameter'):
            param + param

if __name__ == '__main__':
    run_tests()
