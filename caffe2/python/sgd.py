from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from caffe2.python import core
from caffe2.proto import caffe2_pb2


def _build_lr(model, base_learning_rate, policy="fixed", iter_val=0,
              **other_lr_params):

    # Add training operators.
    with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
        ITER = model.param_init_net.ConstantFill([], "ITER", shape=[1],
                                                 value=iter_val,
                                                 dtype=core.DataType.INT32)

    model.net.Iter(ITER, ITER)

    # There is one interesting thing here: since we are minimizing, we are
    # doing "descent" so the learning rate is set to be negative.
    LR = model.net.LearningRate(
        [ITER],
        "LR",
        base_lr=-base_learning_rate,
        policy=policy,
        **other_lr_params
    )
    return LR, ITER


def build_sgd(model, base_learning_rate, policy="fixed", **other_lr_params):
    LR, _ = _build_lr(model, base_learning_rate, policy, **other_lr_params)

    ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)
    for param, grad in model.GetOptimizationPairs().items():
        if isinstance(grad, core.GradientSlice):
            model.ScatterWeightedSum(
                [param, ONE, grad.indices, grad.values, LR], param
            )
        else:
            model.WeightedSum([param, ONE, grad, LR], param)
