from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, data_parallel_model, cnn, rnn_cell
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

    def test_checkpoint_params(self):
        def add_input_ops(model):
            pass

        def add_model_ops(model):
            model = cnn.CNNModelHelper(name="convtest", order="NCHW")
            model.NHWC2NCHW("data", "data_nchw")
            model.Conv("data_nchw", 'conv1', 3, 64,
                       weight_init=("MSRAFill", {}), kernel=7,
                       stride=2, pad=3, no_bias=0)
            model.SpatialBN('conv1', 'conv1_spatbn_relu', 64, epsilon=1e-3)
            model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')
            model.MaxPool('conv1_spatbn_relu', 'pool1', kernel=3, stride=2)
            model.FC('pool1', 'fc', dim_in=(64 * 56 * 56), dim_out=100)
            model.Sigmoid('fc', 'fc_sigm')
            model.Softmax('fc_sigm', 'softmax')
            model.LabelCrossEntropy(['softmax', 'label'], 'xent')
            loss = model.AveragedLoss('xent', 'loss')

            model.AddGradientOperators([loss])

        def add_parameter_update_ops(model):
            model.Iter("ITER")
            LR = model.param_init_net.ConstantFill(
                [], 'LR', shape=[1], value=0.1
            )
            for param in model.GetParams():
                param_grad = model.param_to_grad[param]
                param_momentum = model.param_init_net.ConstantFill(
                    [param], param + '_momentum', value=0.0
                )
                model.net.MomentumSGDUpdate(
                    [param_grad, param_momentum, LR, param],
                    [param_grad, param_momentum, param],
                )
            model = cnn.CNNModelHelper(
                order="NHWC",
                name="test",
            )
            data_parallel_model.Parallelize_GPU(
                model,
                input_builder_fun=add_input_ops,
                forward_pass_builder_fun=add_model_ops,
                param_update_builder_fun=add_parameter_update_ops,
                devices=[1, 2, 3],
            )

            # Only gpu_1 params should be returned (gpu_1 is the first gpu)
            checkpoint_params = data_parallel_model.GetCheckpointParams(model)
            for p in model.GetParams("gpu_1/"):
                self.assertTrue(p in checkpoint_params)
                self.assertTrue(p + "_momentum" in checkpoint_params)
            for p in model.GetParams("gpu_2/"):
                self.assertTrue(p in checkpoint_params)
            for c in model.GetComputedParams("gpu_1/"):
                self.assertFalse(c in checkpoint_params)
            for c in model.GetComputedParams("gpu_2/"):
                self.assertFalse(c in checkpoint_params)
            self.assertFalse(core.BlobReference("gpu_1/data") in checkpoint_params)
            self.assertTrue(core.BlobReference("gpu_1/ITER") in checkpoint_params)



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

            output, _last_hidden, _, _last_state, = rnn_cell.LSTM(
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
                        param + '_momentum',
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


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
@unittest.skipIf(workspace.NumCudaDevices() < 2, "Need at least 2 GPUs.")
class ParallelizeGPUBMUFTest(TestCase):

    def _run_model(self, gpu_devices):
        '''
        Helper function for test_equiv
        '''
        def input_builder_fun(model):
            return None

    def _model_build_fun(self, model, loss_scale):
        fc = model.FC(
            "data", "fc", 16, 1, ("ConstantFill", {}), ("ConstantFill", {})
        )
        fc_fl = model.FlattenToVec(fc, "fc_fl")
        sigm = model.Sigmoid(fc_fl, "sigm")
        sq = model.SquaredL2Distance([sigm, "label"], "sq")
        loss = model.AveragedLoss(sq, "loss")
        loss = model.Scale(loss, scale=loss_scale)
        return [loss]

    def _param_update_fun(self, model):
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

    def _generate_data(self, gpu_devices):
        np.random.seed(26)
        # Each run has same input, independent of number of gpus
        batch_size = 64
        for _ in range(0, 10):
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

    def test_parallelize_gpu_bmuf(self):
        model = cnn.CNNModelHelper(
            order="NHWC",
            name="test"
        )
        gpu_ids = [0, 1]

        def input_builder_fun(model):
            return None

        self._generate_data(gpu_ids)

        data_parallel_model.Parallelize_GPU_BMUF(
            model,
            input_builder_fun,
            self._model_build_fun,
            self._param_update_fun,
            devices=gpu_ids,
        )

        data_parallel_model.RunInitNet(model)

        # Check initial momentum params are zeros
        self.assertEqual(model._device_grouped_blobs.keys(), ['fc_w', 'fc_b'])
        self.assertEqual(workspace.FetchBlob('gpu_0/fc_b_v'), 0)
        np.testing.assert_equal(
            workspace.FetchBlob('gpu_0/fc_w_v'),
            np.zeros(16).astype(np.float32).reshape(1, 16)
        )

        # Run the algorithm for one iteration to have non-zero params.
        data_parallel_model.RunNet(model, 1)

        # Save iteration momentum and post local update params
        v_b_ = workspace.FetchBlob('gpu_0/fc_b_v')
        v_w_ = workspace.FetchBlob('gpu_0/fc_w_v')

        workspace.RunNetOnce(model.net)

        b_0_ = workspace.FetchBlob('gpu_0/fc_b')
        w_0_ = workspace.FetchBlob('gpu_0/fc_w')
        b_1_ = workspace.FetchBlob('gpu_1/fc_b')
        w_1_ = workspace.FetchBlob('gpu_1/fc_w')

        def getBlockAvg(param_name):
            param_0 = workspace.FetchBlob("gpu_0/{}".format(param_name))
            param_1 = workspace.FetchBlob("gpu_1/{}".format(param_name))
            return (param_0 + param_1) / 2

        # Compute block gradients.
        b_g_ = workspace.FetchBlob('gpu_0/fc_b_g')
        w_g_ = workspace.FetchBlob('gpu_0/fc_w_g')
        workspace.RunNetOnce(model._global_model_param_updates_net)

        g_b = (b_0_ + b_1_) / 2 - b_g_
        g_w = (w_0_ + w_1_) / 2 - w_g_
        v_b = workspace.FetchBlob('gpu_0/fc_b_v')
        v_w = workspace.FetchBlob('gpu_0/fc_w_v')

        w_g = workspace.FetchBlob('gpu_0/fc_w_g')
        b_g = workspace.FetchBlob('gpu_0/fc_b_g')
        w_0 = workspace.FetchBlob('gpu_0/fc_w')
        b_0 = workspace.FetchBlob('gpu_0/fc_b')
        w_1 = workspace.FetchBlob('gpu_1/fc_w')
        b_1 = workspace.FetchBlob('gpu_1/fc_b')

        # Check momentum update step
        np.testing.assert_equal(v_b, 0.5 * v_b_ + g_b)
        np.testing.assert_equal(v_w, 0.5 * v_w_ + g_w)

        np.testing.assert_equal(w_g, w_0)
        np.testing.assert_equal(w_g, w_1)
        np.testing.assert_equal(b_g, b_0)
        np.testing.assert_equal(b_g, b_1)

        # Check params update step
        np.testing.assert_equal(w_0, w_g_ + v_w)
        np.testing.assert_equal(b_0, b_g_ + v_b)
