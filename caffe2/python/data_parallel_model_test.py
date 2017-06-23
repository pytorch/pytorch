from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import unittest
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, data_parallel_model, cnn, rnn_cell
from caffe2.python import optimizer
from caffe2.python.test_util import TestCase


class DataParallelModelTest(TestCase):

    def run_model(self, devices, gpu):
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

        def add_optimizer(model):
            optimizer.build_sgd(model, 0.1, policy="fixed")

        workspace.ResetWorkspace()
        model = cnn.CNNModelHelper(
            order="NHWC",
            name="test{}".format(devices),
        )
        data_parallel_model.Parallelize(
            model,
            input_builder_fun=input_builder_fun,
            forward_pass_builder_fun=model_build_fun,
            optimizer_builder_fun=add_optimizer,
            devices=devices,
            cpu_device=not gpu,
        )

        np.random.seed(2603)

        # Each run has same input, independent of number of gpus
        batch_size = 64
        for i in range(0, 10):
            full_data = np.random.rand(batch_size, 16)
            full_labels = np.round(full_data[:, 0])
            batch_per_device = batch_size // len(devices)

            for (j, g) in enumerate(devices):
                st = j * batch_per_device
                en = st + batch_per_device
                data = full_data[st:en, :].astype(np.float32)
                labels = full_labels[st:en].astype(np.float32)
                with core.DeviceScope(core.DeviceOption(model._device_type, g)):
                    workspace.FeedBlob(
                        "{}_{}/data".format(model._device_prefix, g), data
                    )
                    workspace.FeedBlob(
                        "{}_{}/label".format(model._device_prefix, g), labels
                    )

            if i == 0:
                workspace.RunNetOnce(model.param_init_net)
                workspace.CreateNet(model.net)

            workspace.RunNet(model.net.Proto().name)
        return workspace.FetchBlob("{}_0/fc_w".format(model._device_prefix))

    def test_equiv(self):
        '''
        Test that the model produces exactly same results given
        total batchsize, independent of number of GPUs.
        '''
        for gpu in [True, False]:
            if gpu and not workspace.has_gpu_support:
                continue
            result_2gpus = self.run_model([0, 1], gpu=gpu)
            result_1gpus = self.run_model([0], gpu=gpu)

            self.assertTrue(np.allclose(result_1gpus, result_2gpus))

            if not gpu or workspace.NumCudaDevices() >= 4:
                result_4gpus = self.run_model(list(range(4)), gpu=gpu)
                self.assertTrue(np.allclose(result_1gpus, result_4gpus))

            if not gpu or workspace.NumCudaDevices() >= 8:
                result_8gpus = self.run_model(list(range(8)), gpu=gpu)
                self.assertTrue(np.allclose(result_1gpus, result_8gpus))

    def test_checkpoint_params(self):
        def add_input_ops(model):
            pass

        def add_model_ops(model, loss_scale):
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

            # Add a duplicate param init to ensure it does not cause issues
            model.param_init_net.ConstantFill(
                [], ["fc_w"], shape=((64 * 56 * 56), 1000)
            )
            return [loss]

        def add_optimizer(model):
            optimizer.build_sgd(model, 0.1, policy="fixed", momentum=0.9)

        model = cnn.CNNModelHelper(
            order="NHWC",
            name="test",
        )
        data_parallel_model.Parallelize_CPU(
            model,
            input_builder_fun=add_input_ops,
            forward_pass_builder_fun=add_model_ops,
            optimizer_builder_fun=add_optimizer,
            devices=[1, 2, 3],
        )

        # Only gpu_1 params should be returned (gpu_1 is the first gpu)
        checkpoint_params = data_parallel_model.GetCheckpointParams(model)
        for p in model.GetParams("cpu_1/"):
            self.assertTrue(p in checkpoint_params)
            self.assertTrue(p + "_momentum" in checkpoint_params)
        for p in model.GetParams("cpu_2/"):
            self.assertFalse(p in checkpoint_params)
        self.assertTrue(
            core.BlobReference("cpu_1/fc_w_momentum") in checkpoint_params)
        for c in model.GetComputedParams("cpu_1/"):
            self.assertTrue(c in checkpoint_params)
        for c in model.GetComputedParams("cpu_2/"):
            self.assertFalse(c in checkpoint_params)
        self.assertFalse(core.BlobReference("cpu_1/data") in checkpoint_params)
        self.assertTrue(core.BlobReference("optimizer_iteration") in checkpoint_params)


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
@unittest.skipIf(workspace.NumCudaDevices() < 2, "Need at least 2 GPUs.")
class RecurrentNetworkParallelTest(TestCase):

    def run_model(self, devices, gpu):

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
            name="recurrent_test{}".format(devices),
        )

        self.T = 8
        self.batch_size = 64
        self.input_dim = 8
        self.hidden_dim = 31
        self.batch_per_device = self.batch_size // len(devices)

        data_parallel_model.Parallelize(
            model,
            input_builder_fun=input_builder_fun,
            forward_pass_builder_fun=model_build_fun,
            param_update_builder_fun=param_update_fun,
            devices=devices,
            optimize_gradient_memory=True,
            cpu_device=not gpu,
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

            for (j, g) in enumerate(devices):
                st = j * self.batch_per_device
                en = st + self.batch_per_device
                data = full_data[:, st:en, :].astype(np.float32)
                targets = full_target[:, st:en, :].astype(np.float32)
                with core.DeviceScope(core.DeviceOption(model._device_type, g)):
                    workspace.FeedBlob(
                        "{}_{}/data".format(model._device_prefix, g), data
                    )
                    workspace.FeedBlob(
                        "{}_{}/target".format(model._device_prefix, g), targets
                    )

            if i == 0:
                workspace.RunNetOnce(model.param_init_net)
                workspace.CreateNet(model.net)

            workspace.RunNet(model.net.Proto().name)

        return workspace.FetchBlob("{}_0/partest/i2h_w".format(model._device_prefix))

    def test_equiv_recurrent(self):
        '''
        Test that the model produces exactly same results given
        total batchsize, independent of number of GPUs/CPUs.
        '''
        for gpu in [True, False]:
            if gpu and not workspace.has_gpu_support:
                continue
            result_2gpus = self.run_model([0, 1], gpu)
            result_1gpus = self.run_model([0], gpu)

            self.assertTrue(np.allclose(result_1gpus, result_2gpus))

            if not gpu or workspace.NumCudaDevices() >= 4:
                result_4gpus = self.run_model(list(range(4)), gpu)
                self.assertTrue(np.allclose(result_1gpus, result_4gpus))

            if not gpu or workspace.NumCudaDevices() >= 8:
                result_8gpus = self.run_model(list(range(8)), gpu)
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
                with open("/tmp/dump.txt", "w") as f:
                    f.write(str(model.net.Proto()))
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
            result_4gpus = self.run_model(V, list(range(4)), cpu_indices)
            self.assertTrue(np.allclose(result_1gpus[0], result_4gpus[0]))
            self.assertTrue(np.allclose(result_1gpus[1], result_4gpus[1]))

        if workspace.NumCudaDevices() >= 8:
            result_8gpus = self.run_model(V, list(range(8)), cpu_indices)
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


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
@unittest.skipIf(workspace.NumCudaDevices() < 2, "Need at least 2 GPUs.")
class SparseDataParallelModelTestWithSharedIndices(TestCase):

    '''
    Create and run the model. We try with both storing indices for gather
    on CPU and on GPU
    '''
    def run_model(self, V, gpu_devices):

        def input_builder_fun(model):
            return None

        def model_build_fun(model, loss_scale):
            gpu_vecs_gathered = []
            gpu_vecs = []
            for num, vec in enumerate(self.vecs):
                gpu_vec = model.param_init_net.CopyCPUToGPU(
                    vec, 'gpuvec_{}'.format(num),
                )
                if num != 2:
                    model.params.append(gpu_vec)
                gpu_vecs.append(gpu_vec)
            for num, gpu_vec in enumerate(gpu_vecs):
                gpu_vec_gathered = model.net.Gather(
                    [gpu_vec, 'indices'],
                    ['gpu_vec_gathered_{}'.format(num)]
                )
                gpu_vecs_gathered.append(gpu_vec_gathered)

            assert len(gpu_vecs_gathered) == 3

            fc = model.net.FC(
                [
                    gpu_vecs_gathered[2],
                    gpu_vecs_gathered[0],
                    gpu_vecs_gathered[1],
                ],
                ['fc'],
            )
            _, loss = model.net.SoftmaxWithLoss(
                [fc, 'label'],
                ['ce_loss', 'avg_loss'],
                only_loss=True,
            )
            loss = model.Scale(loss, scale=loss_scale)
            model.net.Print(loss, [], limit=10)
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
                    model.net.ScatterWeightedSum(
                        [
                            param,
                            ONE,
                            param_grad.indices,
                            param_grad.values,
                            ONE,
                        ],
                        param,
                    )

        workspace.ResetWorkspace()
        model = cnn.CNNModelHelper(
            order="NHWC",
            name="sparse_test{}".format(gpu_devices),
        )
        batch_size = 32
        batch_per_device = batch_size // len(gpu_devices)

        with core.NameScope("cpu"):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                self.ITER = model.Iter("ITER")
                self.LR = model.net.LearningRate(
                    [self.ITER],
                    "LR",
                    base_lr=(-0.1),
                    policy="fixed",
                )
                '''
                self.vecs consists of 3 big blobs on which we call Gather:
                1) FC weights, shape=(V, 16)
                2) FC bias, shape=(V)
                3) FC input, shape=(batch_per_device, 16)
                '''
                self.vecs = [
                    model.param_init_net.UniformFill(
                        [], "vec_{}".format(num), shape=[V, 16])
                    for num in range(2)
                ]
                self.vecs.append(
                    model.param_init_net.UniformFill(
                        [],
                        "vec_2", shape=[batch_per_device, 16]
                    )
                )
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
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            for num, vec in enumerate(self.vecs[:-1]):
                model.CopyGPUToCPU("gpu_0/gpuvec_{}".format(num), vec)

        # Each run has same input, independent of number of gpus
        for i in range(0, 10):
            np.random.seed(2603)
            full_indices = np.random.permutation(V)[:batch_size].reshape(
                batch_size
            )
            full_labels = full_indices[:] % batch_per_device

            for (j, g) in enumerate(gpu_devices):
                st = j * batch_per_device
                en = st + batch_per_device
                indices = full_indices[st:en].astype(np.int32)
                labels = full_labels[st:en].astype(np.int32)

                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, g)):
                    workspace.FeedBlob("gpu_{}/indices".format(g), indices)
                    workspace.FeedBlob("gpu_{}/label".format(g), labels)

            if i == 0:
                workspace.RunNetOnce(model.param_init_net)
                # Force vecs to be same on all runs
                orig_vecs = [
                    np.random.rand(V, 16).astype(np.float32),
                    np.random.rand(V).astype(np.float32),
                    np.random.rand(V, 16).astype(np.float32),
                ]
                for vec, orig_vec in zip(self.vecs, orig_vecs):
                    workspace.FeedBlob(
                        vec,
                        orig_vec
                    )
                for g in gpu_devices:
                    for num, orig_vec in enumerate(orig_vecs):
                        workspace.FeedBlob(
                            "gpu_{}/gpuvec_{}".format(g, num),
                            orig_vec,
                            device_option=core.DeviceOption(
                                caffe2_pb2.CUDA, g),
                        )
                workspace.CreateNet(model.net)

            workspace.RunNet(model.net.Proto().name)

            idx = workspace.FetchBlob('gpu_0/indices')
            grad_slices = [
                workspace.FetchBlob(
                    'gpu_{}/gpu_vec_gathered_{}_grad'.format(g, num))
                for g in gpu_devices for num in range(2)
            ]
            for grad_slice in grad_slices:
                # print (len(idx), len(grad_slice))
                assert len(idx) == len(grad_slice), (
                    'Number of indices {} is not same as number of gradient '
                    'slices {}. This might lead to illegal memory access'.format(
                        len(idx), len(grad_slice)
                    )
                )

    def test_sparse_shared_indices_gpu(self):
        '''
            Test that the model has same number of indices and gradient rows
            given total batchsize, independent of number of GPUs.
        '''
        V = 10000
        self.run_model(V, [0, 1])
        self.run_model(V, [0])

        if workspace.NumCudaDevices() >= 4:
            self.run_model(V, list(range(4)))

        if workspace.NumCudaDevices() >= 8:
            self.run_model(V, list(range(8)))
