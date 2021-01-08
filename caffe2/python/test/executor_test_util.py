




from caffe2.python import (
    brew, cnn, core, workspace, data_parallel_model,
    timeout_guard, model_helper, optimizer)
from caffe2.python.test_util import TestCase
import caffe2.python.models.resnet as resnet
from caffe2.python.modeling.initializers import Initializer
from caffe2.python import convnet_benchmarks as cb
from caffe2.python import hypothesis_test_util as hu

import time
import numpy as np


CI_MAX_EXAMPLES = 2
CI_TIMEOUT = 600


def executor_test_settings(func):
    if hu.is_sandcastle() or hu.is_travis():
        return hu.settings(
            max_examples=CI_MAX_EXAMPLES,
            deadline=CI_TIMEOUT * 1000  # deadline is in ms
        )(func)
    else:
        return func


def gen_test_resnet50(_order, _cudnn_ws):
    model = cnn.CNNModelHelper(
        order="NCHW",
        name="resnet_50_test",
        cudnn_exhaustive_search=True,
    )
    data = model.net.AddExternalInput("data")
    label = model.net.AddExternalInput("label")
    (_softmax, loss) = resnet.create_resnet50(
        model,
        data,
        num_input_channels=3,
        num_labels=1000,
        label=label,
        is_test=False,
    )
    return model, 227


def conv_model_generators():
    return {
        'AlexNet': cb.AlexNet,
        'OverFeat': cb.OverFeat,
        'VGGA': cb.VGGA,
        'Inception': cb.Inception,
        'MLP': cb.MLP,
        'Resnet50': gen_test_resnet50,
    }


def executor_test_model_names():
    if hu.is_sandcastle() or hu.is_travis():
        return ["MLP"]
    else:
        return sorted(conv_model_generators().keys())


def build_conv_model(model_name, batch_size):
    model_gen_map = conv_model_generators()
    assert model_name in model_gen_map, "Model " + model_name + " not found"
    model, input_size = model_gen_map[model_name]("NCHW", None)

    input_shape = [batch_size, 3, input_size, input_size]
    if model_name == "MLP":
        input_shape = [batch_size, input_size]

    model.param_init_net.GaussianFill(
        [],
        "data",
        shape=input_shape,
        mean=0.0,
        std=1.0
    )
    model.param_init_net.UniformIntFill(
        [],
        "label",
        shape=[batch_size, ],
        min=0,
        max=999
    )

    model.AddGradientOperators(["loss"])

    ITER = brew.iter(model, "iter")
    LR = model.net.LearningRate(
        ITER, "LR", base_lr=-1e-8, policy="step", stepsize=10000, gamma=0.999)
    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    for param in model.params:
        param_grad = model.param_to_grad[param]
        model.net.WeightedSum([param, ONE, param_grad, LR], param)

    return model


def build_resnet50_dataparallel_model(
        num_gpus,
        batch_size,
        epoch_size,
        cudnn_workspace_limit_mb=64,
        num_channels=3,
        num_labels=1000,
        weight_decay=1e-4,
        base_learning_rate=0.1,
        image_size=227,
        use_cpu=False):

    batch_per_device = batch_size // num_gpus

    train_arg_scope = {
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustive_search': False,
        'ws_nbytes_limit': (cudnn_workspace_limit_mb * 1024 * 1024),
        'deterministic': True,
    }
    train_model = model_helper.ModelHelper(
        name="test_resnet50", arg_scope=train_arg_scope
    )

    def create_resnet50_model_ops(model, loss_scale):
        with brew.arg_scope([brew.conv, brew.fc],
                            WeightInitializer=Initializer,
                            BiasInitializer=Initializer,
                            enable_tensor_core=0):
            pred = resnet.create_resnet50(
                model,
                "data",
                num_input_channels=num_channels,
                num_labels=num_labels,
                no_bias=True,
                no_loss=True,
            )

        softmax, loss = model.SoftmaxWithLoss([pred, 'label'],
                                              ['softmax', 'loss'])
        loss = model.Scale(loss, scale=loss_scale)
        brew.accuracy(model, [softmax, "label"], "accuracy")
        return [loss]

    def add_optimizer(model):
        stepsz = int(30 * epoch_size / batch_size)
        optimizer.add_weight_decay(model, weight_decay)
        opt = optimizer.build_multi_precision_sgd(
            model,
            base_learning_rate,
            momentum=0.9,
            nesterov=1,
            policy="step",
            stepsize=stepsz,
            gamma=0.1
        )
        return opt

    def add_image_input(model):
        model.param_init_net.GaussianFill(
            [],
            ["data"],
            shape=[batch_per_device, 3, image_size, image_size],
            dtype='float',
        )
        model.param_init_net.ConstantFill(
            [],
            ["label"],
            shape=[batch_per_device],
            value=1,
            dtype=core.DataType.INT32,
        )

    def add_post_sync_ops(model):
        for param_info in model.GetOptimizationParamInfo(model.GetParams()):
            if param_info.blob_copy is not None:
                model.param_init_net.HalfToFloat(
                    param_info.blob,
                    param_info.blob_copy[core.DataType.FLOAT])

    # Create parallelized model
    data_parallel_model.Parallelize(
        train_model,
        input_builder_fun=add_image_input,
        forward_pass_builder_fun=create_resnet50_model_ops,
        optimizer_builder_fun=add_optimizer,
        post_sync_builder_fun=add_post_sync_ops,
        devices=list(range(num_gpus)),
        rendezvous=None,
        optimize_gradient_memory=True,
        cpu_device=use_cpu,
        shared_model=use_cpu,
    )

    return train_model


def run_resnet50_epoch(train_model, batch_size, epoch_size, skip_first_n_iter=0):
    epoch_iters = int(epoch_size / batch_size)
    prefix = "{}_{}".format(
        train_model._device_prefix,
        train_model._devices[0])
    train_time = 0.0
    train_examples = 0
    for i in range(epoch_iters):
        timeout = 600.0 if i == 0 else 60.0
        with timeout_guard.CompleteInTimeOrDie(timeout):
            t1 = time.time()
            workspace.RunNet(train_model.net.Proto().name)
            t2 = time.time()
            dt = t2 - t1
            if i >= skip_first_n_iter:
                train_time += dt
                train_examples += batch_size

        fmt = "Finished iteration {}/{} ({:.2f} images/sec)"
        print(fmt.format(i + 1, epoch_iters, batch_size / dt))

    accuracy = workspace.FetchBlob(prefix + '/accuracy')
    loss = workspace.FetchBlob(prefix + '/loss')

    assert loss < 40, "Exploded gradients"

    return (
        train_examples,
        train_time,
        accuracy, loss)


class ExecutorTestBase(TestCase):
    def compare_executors(self, model, ref_executor, test_executor, model_run_func):
        model.Proto().type = ref_executor
        model.param_init_net.set_rand_seed(seed=0xCAFFE2)
        model.net.set_rand_seed(seed=0xCAFFE2)

        workspace.ResetWorkspace()
        workspace.RunNetOnce(model.param_init_net)

        workspace.CreateNet(model.net)
        model_run_func()
        ref_ws = {str(k): workspace.FetchBlob(k) for k in workspace.Blobs()}
        ref_ws = {k: v for k, v in ref_ws.items() if type(v) is np.ndarray}

        workspace.ResetWorkspace()
        workspace.RunNetOnce(model.param_init_net)

        model.Proto().type = test_executor
        workspace.CreateNet(model.net, overwrite=True)
        model_run_func()
        test_ws = {str(k): workspace.FetchBlob(k) for k in workspace.Blobs()}
        test_ws = {k: v for k, v in test_ws.items() if type(v) is np.ndarray}

        for blob_name, ref_val in ref_ws.items():
            self.assertTrue(
                blob_name in test_ws,
                "Blob {} not found in {} run".format(blob_name, test_executor))
            val = test_ws[blob_name]
            np.testing.assert_array_equal(
                val, ref_val,
                "Blob {} differs in {} run".format(blob_name, test_executor))
