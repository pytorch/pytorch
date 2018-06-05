# @package optimizer
# Module caffe2.python.optimizer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from caffe2.python import core, utils


class RegularizationBy(object):
    AFTER_OPTIMIZER = "after_optimizer"
    ON_LOSS = "on_loss"


class Regularizer(object):
    def __init__(self):
        self.kEpsilon = 1e-9

    '''
    Adds regularization to train_net for given parameter. Its factor ahead of
    regularization is given when initialization.
    The param should be a BlobReference.
    '''

    def __call__(self, net, param_init_net, param, grad=None, by=None):
        assert isinstance(param, core.BlobReference)
        return {
            RegularizationBy.AFTER_OPTIMIZER: self._run_after_optimizer,
            RegularizationBy.ON_LOSS: self._run_on_loss,
        }[by](net, param_init_net, param, grad)

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        return None

    def _run_after_optimizer(self, net, param_init_net, param, grad):
        return None

    def _ensure_clipped(
        self,
        net,
        param,
        grad=None,
        min=None,
        max=None,
        open_range=False,
        left_open=False,
        right_open=False,
    ):
        min = (
            min + self.kEpsilon
            if min is not None and (open_range or left_open)
            else min
        )
        max = (
            max - self.kEpsilon
            if max is not None and (open_range or right_open)
            else max
        )
        input_blobs = (
            [param, grad.indices, grad.values]
            if isinstance(grad, core.GradientSlice)
            else [param]
        )
        net.EnsureClipped(input_blobs, [param], min=min, max=max)


class L1Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L1Norm, self).__init__()
        assert reg_lambda >= 0,\
            'factor ahead of regularization should be 0 or positive'

        self.reg_lambda = reg_lambda

<<<<<<< HEAD
    def _run(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + '_l1_regularization')
=======
    def _run_on_loss(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + "_l1_regularization")
>>>>>>> 2a1b245... [C2/D2][1/n]: Nonnegative-Constrained Optimization -- log barrier
        net.LpNorm([param], [output_blob], p=1)
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob


class L2Norm(Regularizer):
    def __init__(self, reg_lambda):
        super(L2Norm, self).__init__()
        assert reg_lambda >= 0,\
            'factor ahead of regularization should be 0 or positive'

        self.reg_lambda = reg_lambda

<<<<<<< HEAD
    def _run(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + '_l2_regularization')
=======
    def _run_on_loss(self, net, param_init_net, param, grad=None):
        output_blob = net.NextScopedBlob(param + "_l2_regularization")
>>>>>>> 2a1b245... [C2/D2][1/n]: Nonnegative-Constrained Optimization -- log barrier
        net.LpNorm([param], [output_blob], p=2)
        net.Scale([output_blob], [output_blob], scale=self.reg_lambda)
        return output_blob


class MaxNorm(Regularizer):
    def __init__(self, norm=1.0):
        super(MaxNorm, self).__init__()
        self.norm = norm

<<<<<<< HEAD
    def _run(self, net, param_init_net, param, grad):
        assert self.norm > 0, 'norm should be bigger than 0.'
=======
    def _run_after_optimizer(self, net, param_init_net, param, grad):
        assert self.norm > 0, "norm should be bigger than 0."
>>>>>>> 2a1b245... [C2/D2][1/n]: Nonnegative-Constrained Optimization -- log barrier
        if isinstance(grad, core.GradientSlice):
            net.SparseNormalize(
                [param, grad.indices, grad.values],
                [param],
                use_max_norm=True,
                norm=self.norm,
            )
        else:
            raise NotImplementedError(
                "MaxNorm is not supported for dense parameters"
            )


class ConstantNorm(Regularizer):
    def __init__(self, norm=1.0):
        super(ConstantNorm, self).__init__()
        self.norm = norm

<<<<<<< HEAD
    def _run(self, net, param_init_net, param, grad):
        assert self.norm > 0, 'norm should be bigger than 0.'
=======
    def _run_after_optimizer(self, net, param_init_net, param, grad):
        assert self.norm > 0, "norm should be bigger than 0."
>>>>>>> 2a1b245... [C2/D2][1/n]: Nonnegative-Constrained Optimization -- log barrier
        if isinstance(grad, core.GradientSlice):
            net.SparseNormalize(
                [param, grad.indices, grad.values],
                [param],
                use_max_norm=False,
                norm=self.norm,
            )
        else:
            raise NotImplementedError(
                "ConstantNorm is not supported for dense parameters"
            )


class LogBarrier(Regularizer):
    def __init__(self, reg_lambda, discount_policy="inv", discount_options=None):
        super(LogBarrier, self).__init__()
        assert reg_lambda > 0, "factor ahead of regularization should be 0 or positive"
        self.reg_lambda = reg_lambda
        self.discount_policy = discount_policy
        self.discount_options = discount_options or {"gamma": 1.0, "power": 1.0}

    def _run_on_loss(self, net, param_init_net, param, grad=None):
        iteration = utils.BuildUniqueMutexIter(param_init_net, net)
        # Since we are most likely to do a minimization
        discount = net.NextScopedBlob(param + "_log_barrier_discount")
        net.LearningRate(
            [iteration],
            [discount],
            base_lr=-self.reg_lambda,
            policy=self.discount_policy,
            **self.discount_options
        )
        # TODO(xlwang): param might still be negative at the initialization time or
        # slighly negative due to the distributed training. Enforce it's non-negativity
        # for now (at least above machine epsilon)
        param_non_neg = net.NextScopedBlob(param + "_non_neg")
        net.Clip([param], [param_non_neg], min=self.kEpsilon)
        param_log = net.NextScopedBlob(param + "_log")
        net.Log([param_non_neg], [param_log])
        param_log_sum = net.NextScopedBlob(param + "_log_sum")
        net.SumElements([param_log], [param_log_sum])
        output_blob = net.NextScopedBlob(param + "_log_barrier")
        net.Mul([param_log_sum, discount], [output_blob], broadcast=1)
        return output_blob

    def _run_after_optimizer(self, net, param_init_net, param, grad):
        self._ensure_clipped(net, param, grad, min=0, open_range=True)
