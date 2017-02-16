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


def _dedup(model, dedup_indices, grad):
    assert (isinstance(grad, core.GradientSlice))
    # TODO(dzhulgakov): find a better place to do deduplication
    if dedup_indices:
        return model.net.DeduplicateGradientSlices(grad)
    else:
        return grad


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


def build_adagrad(model, base_learning_rate, dedup_indices=False,
                  parameters=None, **params):
    LR, _ = _build_lr(model, base_learning_rate, policy="fixed")
    param_to_grad = model.GetOptimizationPairs(parameters)

    for param, grad in param_to_grad.items():
        # allocate additional args of the same shape as main weights
        moment = model.param_init_net.ConstantFill(
            [param],
            param + "_square_sum",
            value=0.0
        )
        if isinstance(grad, core.GradientSlice):
            g = _dedup(model, dedup_indices, grad)
            model.SparseAdagrad(
                [param, moment, g.indices, g.values, LR], [param, moment],
                **params
            )

        else:
            model.Adagrad([param, moment, grad, LR], [param, moment], **params)


def build_adam(model, base_learning_rate, dedup_indices=False, iter_val=0,
               **params):
    LR, ITER = _build_lr(model, base_learning_rate, policy="fixed",
                         iter_val=iter_val)
    for param, grad in model.GetOptimizationPairs().items():
        # allocate additional args of the same shape as main weights
        # TODO(nvivek): Fuse input moments if perf critical.
        # Currently keeping it separate to keep the math cleaner
        m1 = model.param_init_net.ConstantFill(
            [param],
            param + "_first_moment",
            value=0.0
        )
        m2 = model.param_init_net.ConstantFill(
            [param],
            param + "_second_moment",
            value=0.0
        )
        if isinstance(grad, core.GradientSlice):
            g = _dedup(model, dedup_indices, grad)
            model.SparseAdam(
                [param, m1, m2, g.indices, g.values, LR, ITER], [param, m1, m2],
                **params
            )

        else:
            model.Adam([param, m1, m2, grad, LR, ITER], [param, m1, m2],
                        **params)
