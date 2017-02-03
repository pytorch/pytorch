from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, data_parallel_model, cnn
from caffe2.python.test_util import TestCase


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
@unittest.skipIf(workspace.NumCudaDevices() < 2, "Need at least 2 GPUs.")
class GPUDataParallelModelTest(TestCase):

    def run_model(self, gpu_devices):
        '''
        Helper function for test_equiv
        '''
        def input_builder_fun(model):
            return None

        def model_build_fun(model, loss_scale):
            fc = model.FC("data", "fc", 16, 1,
                          ("ConstantFill", {}), ("ConstantFill", {}))
            fc_fl = model.FlattenToVec(fc, "fc_fl")
            sigm = model.Sigmoid(fc_fl, "sigm")
            sq = model.SquaredL2Distance([sigm, "label"], "sq")
            loss = model.AveragedLoss(sq, "loss")
            loss = model.Scale(loss, scale=loss_scale)
            return [loss]

        def param_update_fun(model):
            ITER = model.Iter("ITER")
            LR = model.net.LearningRate(
                [ITER],
                "LR",
                base_lr=(-0.1),
                policy="fixed",
            )
            ONE = model.param_init_net.ConstantFill(
                [], "ONE", shape=[1], value=1.0,
            )
            for param in model.GetParams():
                grad = model.param_to_grad[param]
                model.WeightedSum([param, ONE, grad, LR], param)

        workspace.ResetWorkspace()
        model = cnn.CNNModelHelper(
            order="NHWC",
            name="test{}".format(gpu_devices),
        )
        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=input_builder_fun,
            forward_pass_builder_fun=model_build_fun,
            param_update_builder_fun=param_update_fun,
            devices=gpu_devices,
        )

        np.random.seed(2603)

        # Each run has same input, independent of number of gpus
        batch_size = 64
        for i in range(0, 10):
            full_data = np.random.rand(batch_size, 16)
            full_labels = np.round(full_data[:, 0])
            batch_per_device = batch_size // len(gpu_devices)

            for (j, g) in enumerate(gpu_devices):
                st = j * batch_per_device
                en = st + batch_per_device
                data = full_data[st:en, :].astype(np.float32)
                labels = full_labels[st:en].astype(np.float32)
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, g)):
                    workspace.FeedBlob("gpu_{}/data".format(g), data)
                    workspace.FeedBlob("gpu_{}/label".format(g), labels)

            if i == 0:
                workspace.RunNetOnce(model.param_init_net)
                workspace.CreateNet(model.net)

            workspace.RunNet(model.net.Proto().name)

        return workspace.FetchBlob("gpu_0/fc_w")

    def test_equiv(self):
        '''
        Test that the model produces exactly same results given
        total batchsize, independent of number of GPUs.
        '''
        result_2gpus = self.run_model([0, 1])
        result_1gpus = self.run_model([0])

        self.assertTrue(np.allclose(result_1gpus, result_2gpus))

        if workspace.NumCudaDevices() >= 4:
            result_4gpus = self.run_model(range(4))
            self.assertTrue(np.allclose(result_1gpus, result_4gpus))

        if workspace.NumCudaDevices() >= 8:
            result_8gpus = self.run_model(range(8))
            self.assertTrue(np.allclose(result_1gpus, result_8gpus))
