from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import unittest
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, model_helper, brew, test_util


class CopyOpsTest(test_util.TestCase):

    def tearDown(self):
        # Reset workspace after each test
        # Otherwise, the multi-GPU test will use previously created tensors,
        #   which may have been placed on the wrong device
        workspace.ResetWorkspace()

    def run_test_copy_gradient(self, device_opt):
        model = model_helper.ModelHelper(name="copy_test")
        with core.DeviceScope(device_opt):
            x = model.net.AddExternalInputs("x")
            y = model.Copy(x, "y")
            loss = model.AveragedLoss(y, "loss")
            gradient_map = model.AddGradientOperators([loss])
            workspace.FeedBlob(x, np.random.rand(32).astype(np.float32))
            workspace.RunNetOnce(model.param_init_net)
            workspace.RunNetOnce(model.net)
            self.assertTrue(np.array_equal(
                workspace.FetchBlob(x),
                workspace.FetchBlob(y),
            ))
            self.assertTrue(np.array_equal(
                workspace.FetchBlob(gradient_map[x]),
                workspace.FetchBlob(gradient_map[y]),
            ))

    def test_copy_gradient_cpu(self):
        self.run_test_copy_gradient(core.DeviceOption(caffe2_pb2.CPU, 0))

    @unittest.skipIf(workspace.NumGpuDevices() < 1, "Need at least 1 GPU.")
    def test_copy_gradient_gpu(self):
        self.run_test_copy_gradient(core.DeviceOption(workspace.GpuDeviceType, 0))

    @unittest.skipIf(workspace.NumGpuDevices() < 2, "Need at least 2 GPU.")
    def test_copy_gradient_multiple_gpus(self):
        model = model_helper.ModelHelper(name="copy_test")

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
            x_cpu = model.net.AddExternalInputs("x_cpu")

        with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, 0)):
            x_gpu_1 = model.CopyCPUToGPU(x_cpu, "x_gpu_1")

        with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, 1)):
            x_gpu_2 = model.Copy(x_gpu_1, "x_gpu_2")
            loss = model.AveragedLoss(x_gpu_2, "loss")
            gradient_map = model.AddGradientOperators([loss])

        workspace.FeedBlob("x_cpu", np.random.rand(32).astype(np.float32))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        self.assertTrue(np.array_equal(
            workspace.FetchBlob("x_gpu_1"),
            workspace.FetchBlob("x_gpu_2"),
        ))
        self.assertTrue(np.array_equal(
            workspace.FetchBlob(gradient_map["x_gpu_1"]),
            workspace.FetchBlob(gradient_map["x_gpu_2"]),
        ))

        def get_op_with_output(model, output_blob_name):
            for op in model.net.Proto().op:
                if len(op.output) == 1 and op.output[0] == output_blob_name:
                    return op
            return None

        self.assertEqual(
            get_op_with_output(model, "x_gpu_2_grad").device_option,
            core.DeviceOption(workspace.GpuDeviceType, 1),
        )
        self.assertEqual(
            get_op_with_output(model, "x_cpu_grad").device_option,
            core.DeviceOption(workspace.GpuDeviceType, 0),
        )

    @unittest.skipIf(workspace.NumGpuDevices() < 1, "Need at least 1 GPU.")
    def test_cpu2gpu_gpu2cpu_sparse_gradients(self):
        model = model_helper.ModelHelper(name="copy_test")
        v = model.param_init_net.UniformFill([], ["v"], shape=[16, 4])
        indices = model.param_init_net.UniformFill([], ["v"], shape=[16, 4])
        cpu_opt = core.DeviceOption(caffe2_pb2.CPU, 0)
        gpu_opt = core.DeviceOption(workspace.GpuDeviceType, 0)

        with core.DeviceScope(gpu_opt):
            vcpu = model.CopyGPUToCPU(v, "vcpu")

        with core.DeviceScope(cpu_opt):
            g = model.Gather([vcpu, indices], "g")

        with core.DeviceScope(gpu_opt):
            ggpu = model.CopyCPUToGPU(g, "ggpu")
            f = brew.fc(model, ggpu, "out", dim_in=4, dim_out=6)
            (softmax, loss) = model.SoftmaxWithLoss(
                [f, "label"],
                ["softmax", "loss"],
            )
        gradient_map = model.AddGradientOperators([loss])
        self.assertTrue("v" in gradient_map)
        self.assertTrue(isinstance(gradient_map['v'], core.GradientSlice))

    @unittest.skipIf(workspace.NumGpuDevices() < 1, "Need at least 1 GPU.")
    def test_cpu2gpu_gpu2cpu_gradients(self):
        model = model_helper.ModelHelper(name="copy_test")

        batch = 32
        cpu_opt = core.DeviceOption(caffe2_pb2.CPU, 0)
        gpu_opt = core.DeviceOption(workspace.GpuDeviceType, 0)

        with core.NameScope("cpu"):
            with core.DeviceScope(cpu_opt):
                x_cpu = brew.fc(model, 'data', 'x_cpu', 16, 8)

        with core.NameScope("gpu_0"):
            with core.DeviceScope(gpu_opt):
                x_gpu = model.CopyCPUToGPU(x_cpu, "x_gpu")
                pred_gpu = brew.fc(model, x_gpu, "pred_gpu", 8, 4)
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
