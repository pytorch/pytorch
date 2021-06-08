



from future.utils import viewkeys
from multiprocessing import Process, Queue
import numpy as np
import os
import shutil
import tempfile
import unittest
import time
from mock import Mock
from hypothesis import assume, given, settings
import hypothesis.strategies as st

from caffe2.proto import caffe2_pb2
from caffe2.python import brew, core, cnn, data_parallel_model, dyndep, \
    model_helper, optimizer, rnn_cell, workspace
from caffe2.python.test_util import TestCase


dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:file_store_handler_ops")


class TemporaryDirectory:
    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp()
        return self.tmpdir

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.tmpdir)

# Note(jiayq): we are yet to find out why Travis gives out an error in gloo
# like:
# RuntimeError: [enforce fail at /home/travis/build/caffe2/caffe2/third_party/gloo/gloo/transport/tcp/device.cc:113] ifa != nullptr. Unable to find interface for: [127.0.1.1]
# See for example https://travis-ci.org/caffe2/caffe2/jobs/262433866
# As a result, we will check if this is travis, and if yes, disable it.
@unittest.skipIf(os.environ.get("TRAVIS"), "DPMTest has a known issue with Travis.")
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

            # For testing explicit sync
            model.param_init_net.UniformFill([], ["sync_num"], shape=[1])
            return [loss]

        def add_optimizer(model):
            return optimizer.build_sgd(
                model,
                0.1,
                policy="fixed",
                max_gradient_norm=5.0,
                allow_lr_injection=True,
            )

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
            shared_model=not gpu,
            combine_spatial_bn=not gpu,
        )
        data_parallel_model.AddBlobSync(model, ["sync_num"])

        # Light test for LR names
        lr_names = data_parallel_model.GetLearningRateBlobNames(model)
        self.assertGreater(len(lr_names), 0)

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

            workspace.FeedBlob(
                model._device_prefix + "_0/sync_num",
                np.array([i * 2]).astype(np.float32),
                device_option=core.DeviceOption(model._device_type, 0))
            workspace.RunNet(model.net.Proto().name)

            # Test AddBlobSync
            for j in model._devices:
                sync = workspace.FetchBlob(
                    model._device_prefix + "_{}/sync_num".format(j))[0]
                self.assertTrue(abs(sync - i * 2) < 0.01)

        return workspace.FetchBlob("{}_0/fc_w".format(model._device_prefix))

    def run_test_locally(self, fn, device_option=None, **kwargs):
        # Queue for assertion errors on subprocesses
        queue = Queue()

        # Capture any exception thrown by the subprocess
        def run_fn(*args, **kwargs):
            try:
                if device_option is None:
                    fn(*args, **kwargs)
                    workspace.ResetWorkspace()
                else:
                    with core.DeviceScope(device_option):
                        fn(*args, **kwargs)
                        workspace.ResetWorkspace()
            except Exception as ex:
                queue.put(ex)

        # Start N processes in the background
        procs = []
        for i in range(kwargs['comm_size']):
            kwargs['comm_rank'] = i
            proc = Process(
                target=run_fn,
                kwargs=kwargs)
            proc.start()
            procs.append(proc)

        # Test complete, join background processes
        while len(procs) > 0:
            proc = procs.pop(0)
            while proc.is_alive():
                proc.join(1)

                # Raise exception if we find any.
                # Note that the following is executed ALSO after
                # the last process was joined, so if ANY exception
                # was raised, it will be re-raised here.
                if not queue.empty():
                    raise queue.get()

    def test_equiv(self):
        '''
        Test that the model produces exactly same results given
        total batchsize, independent of number of GPUs.
        '''
        for gpu in [True, False]:
            if gpu and (not workspace.has_gpu_support or
                        workspace.NumCudaDevices() < 2):
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

            if not gpu or workspace.NumCudaDevices() >= 16:
                result_16gpus = self.run_model(list(range(16)), gpu=gpu)
                self.assertTrue(np.allclose(result_1gpus, result_16gpus))

    def test_checkpoint_params(self):
        def add_input_ops(model):
            pass

        def add_model_ops(model, loss_scale):
            model.NHWC2NCHW("data", "data_nchw")
            model.Conv("data_nchw", 'conv1', 3, 64,
                       weight_init=("MSRAFill", {}), kernel=7,
                       stride=2, pad=3, no_bias=0)
            model.SpatialBN('conv1', 'conv1_spatbn_relu', 64, epsilon=1e-3, is_test=False)
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

    def test_net_conversion_and_append_net(self):
        other = model_helper.ModelHelper()
        fc1 = brew.fc(other, "data", "other_fc1", dim_in=3*227*227, dim_out=10)
        fc2 = brew.fc(other, fc1, "other_fc2", dim_in=10, dim_out=10)
        brew.fc(other, fc2, "other_fc3", dim_in=10, dim_out=10)

        def add_input_ops(model):
            model.net.UniformFill([], ["data"], shape=[4, 227, 227, 3])
            model.net.UniformFill([], ["label"], shape=[4])

        def add_model_ops(model, loss_scale):
            model.NHWC2NCHW("data", "data_nchw")
            model.Conv("data_nchw", 'conv1', 3, 64,
                       weight_init=("MSRAFill", {}), kernel=7,
                       stride=2, pad=3, no_bias=0)
            model.SpatialBN('conv1', 'conv1_spatbn_relu', 64, epsilon=1e-3, is_test=False)
            model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')
            model.MaxPool('conv1_spatbn_relu', 'pool1', kernel=3, stride=2)
            model.FC('pool1', 'fc', dim_in=(64 * 56 * 56), dim_out=10)

            # Append the net and param_init_net of the other model
            appendnet = data_parallel_model.ConvertNetForDevice(other.net)
            model.net.AppendNet(appendnet)

            model.param_init_net.AppendNet(
                data_parallel_model.ConvertNetForDevice(other.param_init_net))

            model.Sigmoid('fc', 'fc_sigm')
            model.Softmax('fc_sigm', 'softmax')
            loss = model.AveragedLoss('softmax', 'loss')
            return [loss]

        def add_optimizer(model):
            optimizer.build_sgd(model, 0.1, policy="fixed", momentum=0.9)

        model = cnn.CNNModelHelper(
            order="NCHW",
            name="test",
        )
        data_parallel_model.Parallelize_CPU(
            model,
            input_builder_fun=add_input_ops,
            forward_pass_builder_fun=add_model_ops,
            optimizer_builder_fun=add_optimizer,
            devices=range(4)
        )

        # Just create and run net and confirm no exception is thrown
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net)
        workspace.RunNet(model.net)

    @unittest.skip("Test fails on GPU/RE")
    def test_synchronization_barrier(self):
        def run(comm_rank, comm_size, tmpdir):
            def add_input_ops(model):
                pass

            def add_model_ops(model, loss_scale):
                return []

            def add_optimizer(model):
                pass

            store_handler = "store_handler"
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "FileStoreHandlerCreate",
                    [],
                    [store_handler],
                    path=tmpdir))
            rendezvous = dict(
                kv_handler=store_handler,
                shard_id=comm_rank,
                num_shards=comm_size,
                engine='GLOO',
            )

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
                rendezvous=rendezvous
            )
            data_parallel_model.RunInitNet(model)

            for _ in range(2):
                data_parallel_model.Synchronize(model)

        with TemporaryDirectory() as tmpdir:
            self.run_test_locally(
                run,
                comm_size=2,
                device_option=None,
                tmpdir=tmpdir)

    @unittest.skip("Test fails on GPU/RE")
    def test_pre_train_synchronization_barrier(self):
        def run(comm_rank, comm_size, tmpdir):
            def add_input_ops(model):
                pass

            def add_model_ops(model, loss_scale):
                return []

            def add_optimizer(model):
                pass

            workspace.ResetWorkspace()
            store_handler = "store_handler"
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    "FileStoreHandlerCreate",
                    [],
                    [store_handler],
                    path=tmpdir))
            rendezvous = dict(
                kv_handler=store_handler,
                shard_id=comm_rank,
                num_shards=comm_size,
                engine='GLOO',
            )

            model = cnn.CNNModelHelper(
                order="NHWC",
                name="test",
            )
            # Set network timeout to 2 seconds, and add a 3 seconds
            # sleep for 1 host.  Make sure there is no timeout on the
            # second RunNet.
            data_parallel_model._DEFAULT_TIMEOUT_SEC = 2
            data_parallel_model.Parallelize_CPU(
                model,
                input_builder_fun=add_input_ops,
                forward_pass_builder_fun=add_model_ops,
                optimizer_builder_fun=add_optimizer,
                devices=[1, 2, 3],
                rendezvous=rendezvous,
                barrier_net_timeout_sec=5
            )
            data_parallel_model.RunInitNet(model)
            data_parallel_model.RunNet(model, 2)
            if comm_rank == 0:
                time.sleep(data_parallel_model._DEFAULT_TIMEOUT_SEC)
            data_parallel_model.RunNet(model, 2)

        with TemporaryDirectory() as tmpdir:
            self.run_test_locally(
                run,
                comm_size=2,
                device_option=None,
                tmpdir=tmpdir)

    def test_device_scope_check(self):
        with self.assertRaises(AssertionError):
            with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, 0)):
                data_parallel_model.Parallelize_GPU(None, None, None)

    def test_net_transformer_function(self):
        devices = [1, 2, 3]

        def add_input_ops(model):
            model.param_init_net.UniformFill([], ["data"], shape=[32, 8])

        def add_optimizer(model):
            optimizer.build_sgd(model, 0.1)

        def add_model_ops(model, loss_scale):
            fc1 = brew.fc(model, "data", "fc1", dim_in=8, dim_out=8)
            return [fc1]

        kwargs = {
            'input_builder_fun': add_input_ops,
            'forward_pass_builder_fun': add_model_ops,
            'devices': devices,
        }

        # assert that the transformer is called for both train and test cases
        transform = Mock()
        kwargs['net_transformer_fun'] = transform
        model = model_helper.ModelHelper(name="r", init_params=False)
        data_parallel_model.Parallelize_CPU(model, **kwargs)
        self.assertTrue(transform.called)
        self.assertEqual(transform.call_count, 1)

        transform = Mock()
        kwargs['net_transformer_fun'] = transform
        kwargs['optimizer_builder_fun'] = add_optimizer
        model = model_helper.ModelHelper(name="r", init_params=True)
        data_parallel_model.Parallelize_CPU(model, **kwargs)
        self.assertTrue(transform.called)
        self.assertEqual(transform.call_count, 1)

    @given(seed=st.integers(0, 65535), batch_size=st.integers(1, 20))
    @settings(deadline=2000)
    def test_multi_device_bn_op_level_cpu(self, seed, batch_size):
        self._bn_check_op_level("cpu", seed, batch_size)

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
    @unittest.skipIf(workspace.NumCudaDevices() < 2, "Need at least 2 GPUs.")
    @given(seed=st.integers(0, 65535), batch_size=st.integers(1, 20))
    @settings(deadline=2000)
    def test_multi_device_bn_op_level_gpu(self, seed, batch_size):
        self._bn_check_op_level("gpu", seed, batch_size)

    def _bn_check_op_level(self, device_type, seed, batch_size):
        '''
        Test multi device batch normalization at the operation level. This is
        done by checking the outputs of batch normalization and its gradient
        operator. We compare values produced with our manually calculated
        batch normalization values and gradients.
        '''
        devices = [0, 1]
        epsilon = 1e-3
        tolerance = 1e-3

        def _test_forward_pass(x, devices, device_type, scale, bias, epsilon):
            x_concat = np.concatenate(x)
            mean = np.mean(x_concat, axis=0)
            var = np.var(x_concat, axis=0)
            for device in devices:
                x_i = x[device]
                x_hat = (x_i - mean) / (np.sqrt(var + epsilon))
                expected_out = scale * x_hat + bias
                spatial_out = workspace.FetchBlob(
                    "{}_{}/bn_out".format(device_type, device))
                rel_error = np.linalg.norm(spatial_out - expected_out) \
                            / np.linalg.norm(expected_out)
                self.assertTrue(rel_error < 0.005)

        def _test_backward_pass(x, devices, device_type, scale, tolerance):
            dBias_arr = []
            dY_arr = []
            dGamma_arr = []
            num_devices = len(devices)
            mean = np.array(workspace.FetchBlob(
                "{}_0/bn_out_sm".format(device_type)), dtype=np.float32)
            inv_var = np.array(workspace.FetchBlob(
                "{}_0/bn_out_siv".format(device_type)), dtype=np.float32)

            # dBias
            # Sum dBias values over all devices to find the average gradient
            for device in devices:
                dY_blob = workspace.FetchBlob(
                    "{}_{}/bn_out_grad".format(device_type, device))
                dY = np.array(dY_blob, dtype=np.float32)
                dY_arr.append(dY)
                dBias_arr.append(np.array(np.sum(dY, axis=0), dtype=np.float32))
            dBias = np.sum(dBias_arr, dtype=np.float32)
            dBias_avg = dBias / num_devices
            for device in devices:
                dBiasActual = np.sum(workspace.FetchBlob("{}_{}/bn_out_b_grad"
                    .format(device_type, device)), dtype=np.float32)
                self.assertTrue(np.isclose([dBiasActual], [dBias], atol=tolerance))

            # dGamma
            # Sum dGamma values over all devices to find the average gradient
            for device in devices:
                dGamma = np.sum((x[device] - mean) * inv_var * dY_arr[device],
                    axis=0, dtype=np.float32)
                dGamma_arr.append(dGamma)
            dGamma = np.sum(dGamma_arr, axis=0, dtype=np.float32)
            dGamma_avg = dGamma / num_devices
            for device in devices:
                dGammaActual = workspace.FetchBlob(
                    "{}_{}/bn_out_s_grad".format(device_type, device))
                self.assertTrue(np.isclose([dGamma], [dGammaActual], atol=tolerance))

            # dX
            scale_inv_var = scale * inv_var / batch_size
            for device in devices:
                dX = scale_inv_var * (dY_arr[device] * batch_size - dBias_avg
                    - (x[device] - mean) * dGamma_avg * inv_var)
                dX_actual = workspace.FetchBlob(
                    "{}_{}/tanh_grad".format(device_type, device))
                self.assertTrue(np.isclose([dX], [dX_actual], atol=tolerance).all())

        def add_input_ops(model):
            for device in devices:
                data = np.random.rand(batch_size, 1, 1, 1).astype(np.float32)
                workspace.FeedBlob("{}_{}/data".format(device_type, device), data)

        def add_model_ops(model, loss_scale):
            if device_type == "gpu":
                model.CopyCPUToGPU("data", "device_data")
                model.Tanh("device_data", "tanh")
            else:
                model.Tanh("data", "tanh")
            model.SpatialBN("tanh", "bn_out", 1, epsilon=epsilon, is_test=False)
            model.Sqr("bn_out", "sqr")
            loss = model.SumElements("sqr", "loss")
            return [loss]

        def add_optimizer(model):
            return optimizer.build_sgd(model, 0.1)

        np.random.seed(seed)
        workspace.ResetWorkspace()
        model = cnn.CNNModelHelper(
            order="NCHW",
            name="test"
        )
        data_parallel_model.Parallelize(
            model,
            input_builder_fun=add_input_ops,
            forward_pass_builder_fun=add_model_ops,
            optimizer_builder_fun=add_optimizer,
            devices=devices,
            cpu_device=device_type == "cpu",
            shared_model=False,
            combine_spatial_bn=True,
        )

        workspace.RunNetOnce(model.param_init_net)
        scale = workspace.FetchBlob("{}_0/bn_out_s".format(device_type))
        bias = workspace.FetchBlob("{}_0/bn_out_b".format(device_type))
        workspace.RunNetOnce(model.net)

        x = []
        for device in devices:
            x_blob = workspace.FetchBlob("{}_{}/tanh".format(device_type, device))
            x_i = np.array(x_blob, dtype=np.float32)
            x.append(x_i)

        _test_forward_pass(x, devices, device_type, scale, bias, epsilon)
        _test_backward_pass(x, devices, device_type, scale, tolerance)

    @given(seed=st.integers(0, 65535), batch_size=st.integers(1, 20))
    @settings(deadline=2000)
    def test_multi_device_bn_net_lvl_cpu(self, seed, batch_size):
        if batch_size % 2 == 1:
            batch_size += 1
        self._test_multi_device_bn_net_lvl("cpu", seed, batch_size)

    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
    @unittest.skipIf(workspace.NumCudaDevices() < 2, "Need at least 2 GPUs.")
    @given(seed=st.integers(0, 65535), batch_size=st.integers(1, 20))
    @settings(deadline=2000)
    def test_multi_device_bn_net_lvl_gpu(self, seed, batch_size):
        if batch_size % 2 == 1:
            batch_size += 1
        self._test_multi_device_bn_net_lvl("gpu", seed, batch_size)

    def _test_multi_device_bn_net_lvl(self, device_type, seed, batch_size):
        '''
        Test multi device batch normalization at the net level. This is done
        by verifying that the final batch normalization outputs and the
        gradient outputs from multiple devices are the same as those produced
        from a single device
        '''

        # Verify that the gradients calculated over multiple devices are the
        # same as the gradients calculated over one device. These values should
        # be equivalent because combine_spatial_bn sums values over all devices
        def _verify_bn_outputs(
            devices,
            device_type,
            tolerance,
            single_device_bn_out,
            two_device_bn_out_vals,
            single_device_grads,
            two_device_grads,
        ):
            two_device_bn_out = np.concatenate(two_device_bn_out_vals)
            self.assertTrue(np.isclose(
                [single_device_bn_out], [two_device_bn_out], atol=tolerance).all())

            # Scalar and Bias gradients should be the same across devices
            gradient_names = ["bn_out_s_grad", "bn_out_b_grad"]
            for name in gradient_names:
                expected_grad = single_device_grads[name]
                for device in devices:
                    actual_grad = two_device_grads[device][name]
                    self.assertTrue(
                        np.isclose([actual_grad], [expected_grad], atol=tolerance))

            # Expected tanh_grad should be the combined tanh_grad vectors
            # across the devices
            first_grad = two_device_grads[0]["tanh_grad"]
            second_grad = two_device_grads[1]["tanh_grad"]
            actual_grad = np.concatenate([first_grad, second_grad])
            expected_grad = single_device_grads["tanh_grad"]
            rel_error = np.linalg.norm(actual_grad - expected_grad) \
                / np.linalg.norm(expected_grad)
            self.assertTrue(rel_error < 1e-3)

        def _create_model(multiple_devices):
            def add_input_ops_no_combine(model):
                workspace.FeedBlob("{}_0/data".format(device_type), data)

            def add_input_ops_combine(model):
                half = int(batch_size / 2)
                workspace.FeedBlob("{}_0/data".format(device_type), data[:half])
                workspace.FeedBlob("{}_1/data".format(device_type), data[half:])

            def add_model_ops(model, loss_scale):
                if device_type == "gpu":
                    model.CopyCPUToGPU("data", "device_data")
                    model.Tanh("device_data", "tanh")
                else:
                    model.Tanh("data", "tanh")
                model.SpatialBN("tanh", "bn_out", 1, epsilon=epsilon, is_test=False)
                model.Sqr("bn_out", "sqr")
                loss = model.SumElements("sqr", "loss")
                return [loss]

            def add_optimizer(model):
                return optimizer.build_sgd(model, 0.1)

            if multiple_devices:
                input_fun = add_input_ops_combine
                devices = [0, 1]
                combine_spatial_bn = True
            else:
                input_fun = add_input_ops_no_combine
                devices = [0]
                combine_spatial_bn = False
            model = cnn.CNNModelHelper(
                order="NCHW",
                name="test"
            )
            data_parallel_model.Parallelize(
                model,
                input_builder_fun=input_fun,
                forward_pass_builder_fun=add_model_ops,
                optimizer_builder_fun=add_optimizer,
                devices=devices,
                cpu_device=device_type == "cpu",
                shared_model=False,
                combine_spatial_bn=combine_spatial_bn,
            )
            return model

        devices = [0, 1]
        epsilon = 1e-3
        tolerance = 1e-3
        # We are generating random data
        np.random.seed(seed)
        data = np.random.rand(batch_size, 1, 1, 1).astype(np.float32)
        data = np.reshape(data, (batch_size, 1, 1, 1))

        # Get values calculated without combine_spatial_bn
        workspace.ResetWorkspace()
        model_no_combine = _create_model(multiple_devices=False)
        workspace.RunNetOnce(model_no_combine.param_init_net)
        workspace.RunNetOnce(model_no_combine.net)
        single_device_bn_out = workspace.FetchBlob("{}_0/bn_out".format(device_type))
        single_device_grads = {}
        single_device_grads["bn_out_s_grad"] = workspace.FetchBlob(
            "{}_0/bn_out_s_grad".format(device_type))
        single_device_grads["bn_out_b_grad"] = workspace.FetchBlob(
            "{}_0/bn_out_b_grad".format(device_type))
        single_device_grads["tanh_grad"] = workspace.FetchBlob(
            "{}_0/tanh_grad".format(device_type))

        # Get values calculated over multiple devices with combine_spatial_bn true
        workspace.ResetWorkspace()
        model_combine = _create_model(multiple_devices=True)
        workspace.RunNetOnce(model_combine.param_init_net)
        workspace.RunNetOnce(model_combine.net)
        two_device_bn_out_vals = []
        two_device_grads = {}
        for device in devices:
            bn_out_blob = "{}_{}/bn_out".format(device_type, device)
            two_device_bn_out_vals.append(workspace.FetchBlob(bn_out_blob))
            two_device_grads[device] = {}
            two_device_grads[device]["bn_out_s_grad"] = workspace.FetchBlob(
                "{}_{}/bn_out_s_grad".format(device_type, device))
            two_device_grads[device]["bn_out_b_grad"] = workspace.FetchBlob(
                "{}_{}/bn_out_b_grad".format(device_type, device))
            two_device_grads[device]["tanh_grad"] = workspace.FetchBlob(
                "{}_{}/tanh_grad".format(device_type, device))

        # Check to see if the combined values are equivalent
        _verify_bn_outputs(
            devices,
            device_type,
            tolerance,
            single_device_bn_out,
            two_device_bn_out_vals,
            single_device_grads,
            two_device_grads
        )

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

    @unittest.skip("Test is flaky: https://github.com/pytorch/pytorch/issues/10322")
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
            with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, 0)):
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
                    device_for_indices = core.DeviceOption(workspace.GpuDeviceType, g)

                with core.DeviceScope(device_for_indices):
                    workspace.FeedBlob("gpu_{}/indices".format(g), indices)

                with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, g)):
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
                            device_option=core.DeviceOption(workspace.GpuDeviceType, g),
                        )
                workspace.CreateNet(model.net)

            workspace.RunNet(model.net.Proto().name)
            if len(gpu_devices) == 2:
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
@unittest.skipIf(workspace.NumGpuDevices() < 2, "Need at least 2 GPUs.")
class ParallelizeBMUFTest(TestCase):

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

    def _generate_data(self, devices, device_type, device_prefix):
        np.random.seed(26)
        # Each run has same input, independent of number of gpus
        batch_size = 64
        for _ in range(0, 10):
            full_data = np.random.rand(batch_size, 16)
            full_labels = np.round(full_data[:, 0])
            batch_per_device = batch_size // len(devices)

            for (j, g) in enumerate(devices):
                st = j * batch_per_device
                en = st + batch_per_device
                data = full_data[st:en, :].astype(np.float32)
                labels = full_labels[st:en].astype(np.float32)
                with core.DeviceScope(core.DeviceOption(device_type, g)):
                    workspace.FeedBlob("{}_{}/data".format(device_prefix, g), data)
                    workspace.FeedBlob("{}_{}/label".format(device_prefix, g), labels)

    @given(
        cpu_device=st.booleans()
    )
    @settings(deadline=2000)
    def test_parallelize_bmuf(self, cpu_device):
        assume(cpu_device or workspace.has_gpu_support or workspace.has_hip_support)

        workspace.ResetWorkspace()

        model = cnn.CNNModelHelper(
            order="NHWC",
            name="test"
        )
        devices = [0, 1]

        def input_builder_fun(model):
            return None

        if not cpu_device:
            device_type = workspace.GpuDeviceType
            device_prefix = "gpu"
        else:
            device_type = caffe2_pb2.CPU
            device_prefix = "cpu"
        self._generate_data(devices, device_type, device_prefix)

        data_parallel_model.Parallelize_BMUF(
            model,
            input_builder_fun,
            self._model_build_fun,
            self._param_update_fun,
            devices=devices,
            cpu_device=cpu_device
        )

        data_parallel_model.RunInitNet(model)

        # Check initial momentum params are zeros
        self.assertEqual(
            list(viewkeys(model._device_grouped_blobs)), ['fc_w', 'fc_b']
        )
        self.assertEqual(workspace.FetchBlob('{}_0/fc_b_v'.format(device_prefix)), 0)
        np.testing.assert_equal(
            workspace.FetchBlob('{}_0/fc_w_v'.format(device_prefix)),
            np.zeros(16).astype(np.float32).reshape(1, 16)
        )

        # Run the algorithm for one iteration to have non-zero params.
        data_parallel_model.RunNet(model, 1)

        # Save iteration momentum and post local update params
        v_b_ = workspace.FetchBlob('{}_0/fc_b_v'.format(device_prefix))
        v_w_ = workspace.FetchBlob('{}_0/fc_w_v'.format(device_prefix))

        workspace.RunNetOnce(model.net)

        b_0_ = workspace.FetchBlob('{}_0/fc_b'.format(device_prefix))
        w_0_ = workspace.FetchBlob('{}_0/fc_w'.format(device_prefix))
        b_1_ = workspace.FetchBlob('{}_1/fc_b'.format(device_prefix))
        w_1_ = workspace.FetchBlob('{}_1/fc_w'.format(device_prefix))

        # Compute block gradients.
        b_g_ = workspace.FetchBlob('{}_0/fc_b_g'.format(device_prefix))
        w_g_ = workspace.FetchBlob('{}_0/fc_w_g'.format(device_prefix))
        workspace.RunNetOnce(model._global_model_param_updates_net)

        g_b = (b_0_ + b_1_) / 2 - b_g_
        g_w = (w_0_ + w_1_) / 2 - w_g_
        v_b = workspace.FetchBlob('{}_0/fc_b_v'.format(device_prefix))
        v_w = workspace.FetchBlob('{}_0/fc_w_v'.format(device_prefix))

        w_g = workspace.FetchBlob('{}_0/fc_w_g'.format(device_prefix))
        b_g = workspace.FetchBlob('{}_0/fc_b_g'.format(device_prefix))
        w_0 = workspace.FetchBlob('{}_0/fc_w'.format(device_prefix))
        b_0 = workspace.FetchBlob('{}_0/fc_b'.format(device_prefix))
        w_1 = workspace.FetchBlob('{}_1/fc_w'.format(device_prefix))
        b_1 = workspace.FetchBlob('{}_1/fc_b'.format(device_prefix))

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
@unittest.skipIf(workspace.NumGpuDevices() < 2, "Need at least 2 GPUs.")
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
        with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, 0)):
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

                with core.DeviceScope(core.DeviceOption(workspace.GpuDeviceType, g)):
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
                                workspace.GpuDeviceType, g),
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

        if workspace.NumGpuDevices() >= 4:
            self.run_model(V, list(range(4)))

        if workspace.NumGpuDevices() >= 8:
            self.run_model(V, list(range(8)))


if __name__ == "__main__":
    import unittest
    unittest.main()
