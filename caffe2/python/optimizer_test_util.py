## @package optimizer_test_util
# Module caffe2.python.optimizer_test_util
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np
from caffe2.python import brew, core, workspace, cnn, optimizer
from caffe2.proto import caffe2_pb2
from caffe2.python.modeling.initializers import (
    Initializer, PseudoFP16Initializer)

from caffe2.python.model_helper import ModelHelper


class OptimizerTestBase(object):
    """
    This is an abstract base class.
    Don't inherit from unittest.TestCase, and don't name it 'Test*'.
    Do, however, do these things in classes which inherit from this.
    """

    def _createDense(self, dtype=core.DataType.FLOAT):
        perfect_model = np.array([2, 6, 5, 0, 1]).astype(np.float32)
        np.random.seed(123)  # make test deterministic
        numpy_dtype = np.float32 if dtype == core.DataType.FLOAT else np.float16
        initializer = Initializer if dtype == core.DataType.FLOAT else \
            PseudoFP16Initializer
        data = np.random.randint(
            2,
            size=(20, perfect_model.size)).astype(numpy_dtype)
        label = np.dot(data, perfect_model)[:, np.newaxis]

        model = ModelHelper(name="test", arg_scope={'order': 'NCHW'})
        out = brew.fc(
            model,
            'data', 'fc', perfect_model.size, 1, ('ConstantFill', {}),
            ('ConstantFill', {}), axis=0,
            WeightInitializer=initializer, BiasInitializer=initializer
        )
        if dtype == core.DataType.FLOAT16:
            out = model.HalfToFloat(out, out + "_fp32")
        sq = model.SquaredL2Distance([out, 'label'])
        loss = model.AveragedLoss(sq, "avg_loss")
        grad_map = model.AddGradientOperators([loss])
        self.assertIsInstance(grad_map['fc_w'], core.BlobReference)
        return (model, perfect_model, data, label)

    def testDense(self):
        model, perfect_model, data, label = self._createDense()
        optimizer = self.build_optimizer(model)
        workspace.FeedBlob('data', data[0])
        workspace.FeedBlob('label', label[0])
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net, True)
        for _ in range(2000):
            idx = np.random.randint(data.shape[0])
            workspace.FeedBlob('data', data[idx])
            workspace.FeedBlob('label', label[idx])
            workspace.RunNet(model.net.Proto().name)

        np.testing.assert_allclose(
            perfect_model[np.newaxis, :],
            workspace.FetchBlob('fc_w'),
            atol=1e-2
        )
        self.check_optimizer(optimizer)

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    def testGPUDense(self, dtype=core.DataType.FLOAT):
        device_opt = core.DeviceOption(workspace.GpuDeviceType, 0)
        with core.DeviceScope(device_opt):
            model, _perfect_model, data, label = self._createDense(dtype)
            if dtype == core.DataType.FLOAT16:
                fc_fp32_for_host = model.HalfToFloat('fc', 'fc_fp32_for_host')
                model.CopyGPUToCPU(fc_fp32_for_host, 'fc_cpu')
            else:
                model.CopyGPUToCPU('fc', 'fc_cpu')
            workspace.FeedBlob('data', data[0])
            workspace.FeedBlob('label', label[0])

        # Add some CPU ops
        brew.fc(model, 'fc_cpu', 'fc2', dim_in=1, dim_out=10, axis=0)

        # Create optimizer in default device scope
        self.build_optimizer(model)

        if self._skip_gpu:
            return

        # Run net to see it does not crash
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net, True)
        workspace.RunNet(model.net.Proto().name)

    def testSparse(self):
        # to test duplicated indices we assign two indices to each weight and
        # thus each weight might count once or twice
        DUPLICATION = 2
        perfect_model = np.array([2, 6, 5, 0, 1]).astype(np.float32)
        np.random.seed(123)  # make test deterministic
        data = np.random.randint(
            2,
            size=(20, perfect_model.size * DUPLICATION)).astype(np.float32)
        label = np.dot(data, np.repeat(perfect_model, DUPLICATION))

        model = cnn.CNNModelHelper("NCHW", name="test")
        # imitate what model wrapper does
        w = model.param_init_net.ConstantFill(
            [], 'w', shape=[perfect_model.size], value=0.0)
        model.params.append(w)
        picked = model.net.Gather([w, 'indices'], 'gather')
        out = model.ReduceFrontSum(picked, 'sum')

        sq = model.SquaredL2Distance([out, 'label'])
        loss = model.AveragedLoss(sq, "avg_loss")
        grad_map = model.AddGradientOperators([loss])
        self.assertIsInstance(grad_map['w'], core.GradientSlice)
        optimizer = self.build_optimizer(model)

        workspace.CreateBlob('indices')
        workspace.CreateBlob('label')

        for indices_type in [np.int32, np.int64]:
            workspace.RunNetOnce(model.param_init_net)
            workspace.CreateNet(model.net, True)
            for _ in range(2000):
                idx = np.random.randint(data.shape[0])
                # transform into indices of binary features
                indices = np.repeat(np.arange(perfect_model.size),
                                    DUPLICATION)[data[idx] == 1]
                if indices.size == 0:
                    continue
                workspace.FeedBlob(
                    'indices',
                    indices.reshape((indices.size,)).astype(indices_type)
                )
                workspace.FeedBlob('label',
                                   np.array(label[idx]).astype(np.float32))
                workspace.RunNet(model.net.Proto().name)

            np.testing.assert_allclose(
                perfect_model,
                workspace.FetchBlob('w'),
                atol=1e-2
            )
        self.check_optimizer(optimizer)


class LRModificationTestBase(object):
    """
    This is an abstract base class.
    Don't inherit from unittest.TestCase, and don't name it 'Test*'.
    Do, however, do these things in classes which inherit from this.
    """

    def _gradient_ratio_reference(self, model, params, max_gradient_norm):
        from caffe2.python import core
        sum_squared_norms = 0.0
        for param in params:
            grad = (
                model.param_to_grad[param]
                if not isinstance(
                    model.param_to_grad[param],
                    core.GradientSlice,
                ) else model.param_to_grad[param].values
            )
            val = workspace.FetchBlob(grad)
            sum_squared_norms += np.power(np.linalg.norm(val), 2.0)
        global_norm = np.sqrt(sum_squared_norms)
        clip_norm = max_gradient_norm
        norm_ratio = clip_norm / np.maximum(clip_norm, global_norm)
        return norm_ratio

    def test_global_norm_based_gradient_clipping(self):
        max_gradient_norm = 1.0
        model, perfect_model, data, label = self._createDense()
        opt = self.build_optimizer(model, max_gradient_norm=max_gradient_norm)

        params = []
        for param in model.GetParams(top_scope=True):
            if param in model.param_to_grad:
                if not isinstance(
                    model.param_to_grad[param],
                    core.GradientSlice,
                ):
                    params.append(param)

        workspace.FeedBlob('data', data[0])
        workspace.FeedBlob('label', label[0])
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net, True)
        self.assertIsNotNone(opt._lr_multiplier)

        # Run net once
        idx = np.random.randint(data.shape[0])
        workspace.FeedBlob('data', data[idx])
        workspace.FeedBlob('label', label[idx])
        workspace.RunNet(model.net.Proto().name)

        reference = self._gradient_ratio_reference(
            model,
            params,
            max_gradient_norm,
        )
        norm_ratio = workspace.FetchBlob(
            'norm_clipped_grad_update/norm_ratio')
        np.testing.assert_almost_equal(norm_ratio, reference)
        self.assertTrue(
            reference < 1.0, "Bad test, gradient not being scaled."
        )

    def test_lr_injection(self):
        model, perfect_model, data, label = self._createDense()
        opt = self.build_optimizer(
            model, max_gradient_norm=1, allow_lr_injection=True
        )

        workspace.FeedBlob('data', data[0])
        workspace.FeedBlob('label', label[0])
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net, True)

        # Test LR injection initialized properly
        self.assertIsNotNone(opt._lr_multiplier)
        self.assertEqual(optimizer.get_lr_injection(), 1)

        # Test that we're able to modify the value of the lr_injection
        optimizer.set_lr_injection(0)
        self.assertEqual(optimizer.get_lr_injection(), 0)

        # Test that setting the lr_injector properly propagates to the
        # lr_multiplier. Here, we have both lr_injector and norm_ratio that
        # affect the lr_multiplier
        workspace.RunNet(model.net.Proto().name)
        self.assertEqual(workspace.FetchBlob('lr_multiplier'), 0)
