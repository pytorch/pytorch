from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Manager

import numpy as np
import unittest
import tempfile
import shutil
import logging

log = logging.getLogger("parallelize_gpu_bmuf_distributed_test")
log.setLevel(logging.INFO)


def bmuf_process(filestore_dir, process_id, shared_results):
    # We need to import caffe2 in every process to initialize CUDA independently.
    from caffe2.python import core, cnn, data_parallel_model, workspace, dyndep
    from caffe2.proto import caffe2_pb2
    dyndep.InitOpsLibrary("@/caffe2/caffe2/distributed:file_store_handler_ops")

    if not workspace.has_gpu_support or workspace.NumCudaDevices() < 2:
        log.info('No GPU support test is Ignored.')
        return

    model = cnn.CNNModelHelper(
        order="NHWC",
        name="test"
    )

    gpu_ids = [0, 1] if process_id == 0 else [2, 3]

    def _model_build_fun(model, loss_scale):
        fc = model.FC(
            "data", "fc", 16, 1, ("ConstantFill", {}), ("ConstantFill", {})
        )
        fc_fl = model.FlattenToVec(fc, "fc_fl")
        sigm = model.Sigmoid(fc_fl, "sigm")
        sq = model.SquaredL2Distance([sigm, "label"], "sq")
        loss = model.AveragedLoss(sq, "loss")
        loss = model.Scale(loss, scale=loss_scale)
        return [loss]

    def _input_builder_fun(model):
        return None

    def _param_update_fun(model):
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

    def _generate_data(gpu_devices, process_id):
        np.random.seed(26 + process_id * 10)
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

    _generate_data(gpu_ids, process_id)

    workspace.RunOperatorOnce(
        core.CreateOperator(
            "FileStoreHandlerCreate", [], ["store_handler"],
            path=filestore_dir
        )
    )
    rendezvous = dict(
        kv_handler="store_handler",
        shard_id=process_id,
        num_shards=2,
        engine="GLOO",
        exit_nets=None
    )

    data_parallel_model.Parallelize_GPU_BMUF(
        model,
        _input_builder_fun,
        _model_build_fun,
        _param_update_fun,
        devices=gpu_ids,
        rendezvous=rendezvous
    )

    data_parallel_model.RunInitNet(model)

    def _gpu_pid(gpu_id, pid):
        if pid == 1:
            return gpu_id + 2
        return gpu_id

    np.testing.assert_equal(
        workspace.FetchBlob("gpu_{}/fc_w_v".format(_gpu_pid(0, process_id))),
        np.zeros(16).astype(np.float32).reshape(1, 16)
    )

    # Run the algorithm for one iteration to have non-zero params.
    data_parallel_model.RunNet(model, 1)

    # Save iteration momentum and post local update params
    results = {}
    v_b_ = workspace.FetchBlob("gpu_{}/fc_b_v".format(_gpu_pid(0, process_id)))
    v_w_ = workspace.FetchBlob("gpu_{}/fc_w_v".format(_gpu_pid(0, process_id)))

    results['v_b_'] = v_b_
    results['v_w_'] = v_w_

    workspace.RunNetOnce(model.net)

    b_0_ = workspace.FetchBlob("gpu_{}/fc_b".format(_gpu_pid(0, process_id)))
    w_0_ = workspace.FetchBlob("gpu_{}/fc_w".format(_gpu_pid(0, process_id)))
    b_1_ = workspace.FetchBlob("gpu_{}/fc_b".format(_gpu_pid(1, process_id)))
    w_1_ = workspace.FetchBlob("gpu_{}/fc_w".format(_gpu_pid(1, process_id)))

    results['b_0_'] = b_0_
    results['w_0_'] = w_0_
    results['b_1_'] = b_1_
    results['w_1_'] = w_1_

    # Compute block gradients.
    b_g_ = workspace.FetchBlob("gpu_{}/fc_b_g".format(_gpu_pid(0, process_id)))
    w_g_ = workspace.FetchBlob("gpu_{}/fc_w_g".format(_gpu_pid(0, process_id)))
    results['b_g_'] = b_g_
    results['w_g_'] = w_g_
    workspace.RunNetOnce(model._global_model_param_updates_net)

    #  g_b = (b_0_ + b_1_) / 2 - b_g_
    #  g_w = (w_0_ + w_1_) / 2 - w_g_
    v_b = workspace.FetchBlob("gpu_{}/fc_b_v".format(_gpu_pid(0, process_id)))
    v_w = workspace.FetchBlob("gpu_{}/fc_w_v".format(_gpu_pid(0, process_id)))
    w_g = workspace.FetchBlob("gpu_{}/fc_w_g".format(_gpu_pid(0, process_id)))
    b_g = workspace.FetchBlob("gpu_{}/fc_b_g".format(_gpu_pid(0, process_id)))
    w_0 = workspace.FetchBlob("gpu_{}/fc_w".format(_gpu_pid(0, process_id)))
    b_0 = workspace.FetchBlob("gpu_{}/fc_b".format(_gpu_pid(0, process_id)))
    w_1 = workspace.FetchBlob("gpu_{}/fc_w".format(_gpu_pid(1, process_id)))
    b_1 = workspace.FetchBlob("gpu_{}/fc_b".format(_gpu_pid(1, process_id)))
    results['v_b'] = v_b
    results['v_w'] = v_w
    results['w_g'] = w_g
    results['b_g'] = b_g
    results['w_0'] = w_0
    results['b_0'] = b_0
    results['w_1'] = w_1
    results['b_1'] = b_1

    shared_results[process_id] = results


class DistrubitedTest(unittest.TestCase):

    def test_bmuf_distributed(self):
        processes = []
        filestore_dir = tempfile.mkdtemp()
        results = Manager().dict()
        for idx in range(0, 2):
            process = Process(
                target=bmuf_process,
                args=(filestore_dir, idx, results)
            )
            processes.append(process)
            process.start()

        while len(processes) > 0:
            process = processes.pop()
            process.join()
        shutil.rmtree(filestore_dir)

        if len(results) == 0:
            return

        w_0 = results[0]['w_0']
        w_1 = results[0]['w_1']
        b_0 = results[0]['b_0']
        b_1 = results[0]['b_1']
        # Check parameters are in sync.
        np.testing.assert_equal(w_0, w_1)
        np.testing.assert_equal(w_0, results[1]['w_0'])
        np.testing.assert_equal(w_0, results[1]['w_1'])
        np.testing.assert_equal(b_0, b_1)
        np.testing.assert_equal(b_0, results[1]['b_0'])
        np.testing.assert_equal(b_0, results[1]['b_1'])

        w_g_ = results[0]['w_g_']
        b_g_ = results[0]['b_g_']

        g_b = (results[0]['b_0_'] + results[1]['b_0_'] + results[0]['b_1_'] +
               results[1]['b_1_']) / 4 - b_g_
        g_w = (results[0]['w_0_'] + results[1]['w_0_'] + results[0]['w_1_'] +
               results[1]['w_1_']) / 4 - w_g_
        v_b_ = results[0]['v_b_']
        v_b = results[0]['v_b']
        v_w_ = results[0]['v_w_']
        v_w = results[0]['v_w']

        # Check block gradients are correct.
        np.testing.assert_almost_equal(v_b, 0.75 * v_b_ + g_b)
        np.testing.assert_almost_equal(v_w, 0.75 * v_w_ + g_w)

        # Check params update step
        np.testing.assert_equal(w_0, w_g_ + v_w)
        np.testing.assert_equal(b_0, b_g_ + v_b)
