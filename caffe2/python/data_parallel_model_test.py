from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, data_parallel_model, cnn, recurrent
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

            print(i, workspace.FetchBlob("gpu_0/fc_w").flatten()[:5])
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


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
@unittest.skipIf(workspace.NumCudaDevices() < 2, "Need at least 2 GPUs.")
class RecurrentNetworkParallelTest(TestCase):

    def run_model(self, gpu_devices):

        '''
        Helper function for test_equiv
        '''
        def input_builder_fun(model):
            return None

        def model_build_fun(model, loss_scale):
            workspace.FeedBlob(
                core.ScopedBlobReference("seq_lengths"),
                np.array([self.T] * self.batch_per_device, dtype=np.int32)
            )
            model.param_init_net.ConstantFill(
                [],
                "hidden_init",
                value=0.0,
                shape=[1, self.batch_per_device, self.hidden_dim]
            )
            model.param_init_net.ConstantFill(
                [],
                "cell_init",
                value=0.0,
                shape=[1, self.batch_per_device, self.hidden_dim]
            )

            output, _last_hidden, _, _last_state, = recurrent.LSTM(
                model=model,
                input_blob="data",
                seq_lengths="seq_lengths",
                initial_states=("hidden_init", "cell_init"),
                dim_in=self.input_dim,
                dim_out=self.hidden_dim,
                scope="partest",
            )

            # A silly loss function
            loss = model.AveragedLoss(
                model.Sub([output, "target"], "dist"),
                "loss",
            )
            loss = model.Scale(loss, "loss_scaled", scale=loss_scale)
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
                param_grad = model.param_to_grad[param]
                model.WeightedSum([param, ONE, param_grad, LR], param)

            assert len(model.GetParams()) == len(model.params) // len(model._devices)

        workspace.ResetWorkspace()
        model = cnn.CNNModelHelper(
            name="recurrent_test{}".format(gpu_devices),
        )

        self.T = 8
        self.batch_size = 64
        self.input_dim = 8
        self.hidden_dim = 31
        self.batch_per_device = self.batch_size // len(gpu_devices)

        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=input_builder_fun,
            forward_pass_builder_fun=model_build_fun,
            param_update_builder_fun=param_update_fun,
            devices=gpu_devices,
            optimize_gradient_memory=True,
        )

        # Change all initialization to be ConstantFills so that
        # the everything is deterministic
        for op in model.param_init_net.Proto().op:
            if op.type.endswith('Fill'):
                op.type = 'ConstantFill'

        # Each run has same input, independent of number of gpus
        np.random.seed(20150210)
        for i in range(0, 10):
            full_data = np.random.rand(self.T, self.batch_size, self.input_dim)
            full_target = np.random.rand(
                self.T, self.batch_size, self.hidden_dim
            )

            for (j, g) in enumerate(gpu_devices):
                st = j * self.batch_per_device
                en = st + self.batch_per_device
                data = full_data[:, st:en, :].astype(np.float32)
                targets = full_target[:, st:en, :].astype(np.float32)
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, g)):
                    workspace.FeedBlob("gpu_{}/data".format(g), data)
                    workspace.FeedBlob("gpu_{}/target".format(g), targets)

            if i == 0:
                workspace.RunNetOnce(model.param_init_net)
                workspace.CreateNet(model.net)

            workspace.RunNet(model.net.Proto().name)

        return workspace.FetchBlob("gpu_0/partest/i2h_w")

    def test_equiv_recurrent(self):
        '''
        Test that the model produces exactly same results given
        total batchsize, independent of number of GPUs.
        '''
        result_2gpus = self.run_model([0, 1])
        result_1gpus = self.run_model([0])

        print("result 1", result_1gpus.flatten()[:5])
        print("result 2", result_2gpus.flatten()[:5])

        self.assertTrue(np.allclose(result_1gpus, result_2gpus))

        if workspace.NumCudaDevices() >= 4:
            result_4gpus = self.run_model(range(4))
            self.assertTrue(np.allclose(result_1gpus, result_4gpus))

        if workspace.NumCudaDevices() >= 8:
            result_8gpus = self.run_model(range(8))
            self.assertTrue(np.allclose(result_1gpus, result_8gpus))


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
@unittest.skipIf(workspace.NumCudaDevices() < 2, "Need at least 2 GPUs.")
class SparseDataParallelModelTest(TestCase):

    '''
    Create and run the model. We try with both storing indices for gather
    on CPU and on GPU
    '''
    def run_model(self, V, gpu_devices, cpu_indices):

        def input_builder_fun(model):
            return None

        def model_build_fun(model, loss_scale):
            if cpu_indices:
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                    gathered_cpu = model.net.Gather(
                        [self.vecs, 'indices'], 'gathered_cpu')

                gathered = model.CopyCPUToGPU(gathered_cpu, "gathered")
            else:
                gpu_vecs = model.param_init_net.CopyCPUToGPU(
                    self.vecs, "gpuvecs",
                )
                model.params.append(gpu_vecs)
                gathered = model.net.Gather([gpu_vecs, 'indices'], 'gathered')
            flattened = model.Flatten(gathered, "flattened")
            fc = model.FC(flattened, "fc", 16 * 16, 1,
                          ("ConstantFill", {}), ("ConstantFill", {}))
            fc_fl = model.FlattenToVec(fc, "fc_fl")
            sigm = model.Sigmoid(fc_fl, "sigm")
            sq = model.SquaredL2Distance([sigm, "label"], "sq")
            loss = model.AveragedLoss(sq, "loss")
            loss = model.Scale(loss, scale=loss_scale)
            return [loss]

        def param_update_fun(model):
            ONE = model.param_init_net.ConstantFill(
                [], "ONE", shape=[1], value=1.0,
            )
            LR = model.CopyCPUToGPU(self.LR, "LR")
            for param in model.GetParams():
                param_grad = model.param_to_grad[param]
                if not isinstance(param_grad, core.GradientSlice):
                    model.WeightedSum([param, ONE, param_grad, LR], param)
                else:
                    param_momentum = model.param_init_net.ConstantFill(
                        [param],
                        '{}_momentum'.format(param),
                        value=0.0,
                    )
                    model.net.SparseMomentumSGDUpdate(
                        [
                            param_grad.values,
                            param_momentum,
                            LR,
                            param,
                            param_grad.indices,
                        ],
                        [
                            param_grad.values, param_momentum, param
                        ],
                        momentum=0.1,
                        nesterov=0,
                    )

        workspace.ResetWorkspace()
        model = cnn.CNNModelHelper(
            order="NHWC",
            name="sparse_test{}".format(gpu_devices),
        )

        with core.NameScope("cpu"):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                self.ITER = model.Iter("ITER")
                self.LR = model.net.LearningRate(
                    [self.ITER],
                    "LR",
                    base_lr=(-0.1),
                    policy="fixed",
                )
                self.vecs = model.param_init_net.UniformFill(
                    [], "vecs", shape=[V, 16])
                if cpu_indices:
                    model.params.append(self.vecs)
                self.ONE_CPU = model.param_init_net.ConstantFill(
                    [], "ONE_CPU", shape=[1], value=1.0,
                )

        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=input_builder_fun,
            forward_pass_builder_fun=model_build_fun,
            param_update_builder_fun=param_update_fun,
            devices=gpu_devices,
        )

        # Update the vecs
        if cpu_indices:
            with core.NameScope("cpu"):
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                    for param in model.GetParams():
                        param_grad = model.param_to_grad[param]
                        model.ScatterWeightedSum([param, self.ONE_CPU,
                                                  param_grad.indices,
                                                  param_grad.values,
                                                  self.LR],
                                                  self.vecs)
        else:
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
                model.CopyGPUToCPU("gpu_0/gpuvecs", self.vecs)

        np.random.seed(2603)

        # Each run has same input, independent of number of gpus
        batch_size = 64
        for i in range(0, 10):
            full_indices = np.random.permutation(V)[:batch_size * 16].reshape(
                batch_size, 16
            )
            full_labels = full_indices[:, 0] % 2
            batch_per_device = batch_size // len(gpu_devices)

            for (j, g) in enumerate(gpu_devices):
                st = j * batch_per_device
                en = st + batch_per_device
                indices = full_indices[st:en, :].astype(np.int32)
                labels = full_labels[st:en].astype(np.float32)

                device_for_indices = core.DeviceOption(caffe2_pb2.CPU)
                if not cpu_indices:
                    device_for_indices = core.DeviceOption(caffe2_pb2.CUDA, g)

                with core.DeviceScope(device_for_indices):
                    workspace.FeedBlob("gpu_{}/indices".format(g), indices)

                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, g)):
                    workspace.FeedBlob("gpu_{}/label".format(g), labels)

            if i == 0:
                workspace.RunNetOnce(model.param_init_net)
                # Force vecs to be same on all runs
                orig_vecs = np.random.rand(V, 16).astype(np.float32)
                workspace.FeedBlob(
                    self.vecs,
                    orig_vecs
                )
                if not cpu_indices:
                    for g in gpu_devices:
                        workspace.FeedBlob(
                            "gpu_{}/gpuvecs".format(g),
                            orig_vecs,
                            device_option=core.DeviceOption(caffe2_pb2.CUDA, g),
                        )
                workspace.CreateNet(model.net)

            workspace.RunNet(model.net.Proto().name)
            if len(gpu_devices) == 2:
                open("dump.txt", "w").write(str(model.net.Proto()))
                if not cpu_indices:
                    idx = workspace.FetchBlob("gpu_0/indices")
                    idx = list(idx.flatten())
                    n = len(idx)
                    nu = len(set(idx))
                    assert n == nu, "We cannot have duplicate indices"

        # Sanity check to see the vecs were updated
        self.assertFalse(
            np.allclose(workspace.FetchBlob(self.vecs), orig_vecs))
        return [workspace.FetchBlob(self.vecs if cpu_indices else "gpu_0/gpuvecs"),
                workspace.FetchBlob("gpu_0/fc_w")]

    def _test_equiv_sparse(self, cpu_indices):
        '''
            Test that the model produces exactly same results given
            total batchsize, independent of number of GPUs.
        '''
        V = 10000
        result_2gpus = self.run_model(V, [0, 1], cpu_indices)
        result_1gpus = self.run_model(V, [0], cpu_indices)

        self.assertTrue(np.allclose(result_1gpus[0], result_2gpus[0]))
        self.assertTrue(np.allclose(result_1gpus[1], result_2gpus[1]))

        if workspace.NumCudaDevices() >= 4:
            result_4gpus = self.run_model(V, range(4), cpu_indices)
            self.assertTrue(np.allclose(result_1gpus[0], result_4gpus[0]))
            self.assertTrue(np.allclose(result_1gpus[1], result_4gpus[1]))

        if workspace.NumCudaDevices() >= 8:
            result_8gpus = self.run_model(V, range(8), cpu_indices)
            self.assertTrue(np.allclose(result_1gpus[0], result_8gpus[0]))
            self.assertTrue(np.allclose(result_1gpus[1], result_8gpus[1]))

    def test_equiv_sparse(self):
        self._test_equiv_sparse(True)
        self._test_equiv_sparse(False)
