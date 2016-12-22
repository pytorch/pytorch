from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import unittest
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, cnn


class CopyOpsTest(unittest.TestCase):

    @unittest.skipIf(workspace.NumCudaDevices() < 1, "Need at least 1 GPU.")
    def test_cpu2gpu_gpu2cpu_gradients(self):
        model = cnn.CNNModelHelper(name="copy_test")

        batch = 32
        cpu_opt = core.DeviceOption(caffe2_pb2.CPU, 0)
        gpu_opt = core.DeviceOption(caffe2_pb2.CUDA, 0)

        with core.NameScope("cpu"):
            with core.DeviceScope(cpu_opt):
                x_cpu = model.FC('data', 'x_cpu', 16, 8)

        with core.NameScope("gpu_0"):
            with core.DeviceScope(gpu_opt):
                x_gpu = model.CopyCPUToGPU(x_cpu, "x_gpu")
                pred_gpu = model.FC(x_gpu, "pred_gpu", 8, 4)
                pred_cpu = model.CopyGPUToCPU(pred_gpu, "pred_cpu")

        with core.DeviceScope(cpu_opt):
            with core.NameScope("cpu"):
                (softmax, loss) = model.SoftmaxWithLoss(
                    [pred_cpu, "label"],
                    ["softmax", "loss"],
                )

        gradient_map = model.AddGradientOperators([loss])

        # Add param updates (for cpu and gpu)
        init_net = model.param_init_net
        with core.DeviceScope(cpu_opt):
            with core.NameScope("cpu"):
                ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
                LR = init_net.ConstantFill([], "LR", shape=[1], value=-2.0)
                for param in model.GetParams():
                    model.WeightedSum(
                        [param, ONE, gradient_map[param], LR],
                        param,
                    )

        with core.NameScope("gpu_0"):
            with core.DeviceScope(gpu_opt):
                ONE = init_net.ConstantFill([], "ONE", shape=[1], value=1.)
                LR = init_net.ConstantFill([], "LR", shape=[1], value=-2.0)
                for param in model.GetParams():
                    model.WeightedSum(
                        [param, ONE, gradient_map[param], LR],
                        param,
                    )

        with core.DeviceScope(cpu_opt):
            workspace.FeedBlob(
                'cpu/data',
                np.random.rand(batch, 16).astype(np.float32),
            )
            workspace.FeedBlob(
                'cpu/label',
                np.random.randint(4, size=batch).astype(np.int32),
            )

        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net)

        initial_params = {p: workspace.FetchBlob(p) for p in model.GetParams()}
        workspace.RunNet(model.net.Proto().name)
        updated_params = {p: workspace.FetchBlob(p) for p in model.GetParams()}

        for p in model.GetParams():
            g = gradient_map[p]
            expected = initial_params[p] - 2.0 * workspace.FetchBlob(g)
            actual = updated_params[p]
            self.assertTrue(
                np.array_equal(expected, updated_params[p]),
                "Mismatch: {}: {}, {}".format(p, expected, actual),
            )
