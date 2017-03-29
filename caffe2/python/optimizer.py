## @package optimizer
# Module caffe2.python.optimizer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.proto import caffe2_pb2


class Optimizer(object):
    def __init__(self):
        pass

    def __call__(self, net, param_init_net, param, grad):
        raise NotImplementedError()

    @staticmethod
    def build_lr(net, param_init_net, base_learning_rate,
                 learning_rate_blob="lr", policy="fixed",
                 iter_val=0, **kwargs):
        # Add training operators.
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            iterator = param_init_net.ConstantFill(
                [], "iterator", shape=[1],
                value=iter_val,
                dtype=core.DataType.INT32)

        net.Iter(iterator, iterator)

        # There is one interesting thing here: since we are minimizing, we are
        # doing "descent" so the learning rate is set to be negative.
        lr = net.LearningRate(
            [iterator],
            learning_rate_blob,
            base_lr=-base_learning_rate,
            policy=policy,
            **kwargs
        )
        return lr, iterator

    @staticmethod
    def dedup(net, sparse_dedup_aggregator, grad):
        assert (isinstance(grad, core.GradientSlice))
        if sparse_dedup_aggregator:
            return net.DeduplicateGradientSlices(
                grad, aggregator=sparse_dedup_aggregator)
        else:
            return grad


class SgdOptimizer(Optimizer):
    def __init__(self, base_learning_rate=0.01, policy='fixed',
                 momentum=0.0, **kwargs):
        self.base_learning_rate = base_learning_rate
        self.policy = policy
        self.momentum = momentum
        self.init_kwargs = kwargs

    def __call__(self, net, param_init_net, param, grad):
        if self.base_learning_rate <= 0:
            return

        lr, _ = self.build_lr(
            net, param_init_net,
            base_learning_rate=self.base_learning_rate,
            learning_rate_blob=str(param) + "_lr",
            policy=self.policy,
            **(self.init_kwargs)
        )

        ONE = param_init_net.ConstantFill([], "ONE", shape=[1], value=1.0)

        if self.momentum > 0:
            momentum_data = param_init_net.ConstantFill(
                param, str(param) + "_momentum", value=0.)

        if isinstance(grad, core.GradientSlice):
            assert self.momentum == 0., "Doesn't support momentum for sparse"
            net.ScatterWeightedSum(
                [param, ONE, grad.indices, grad.values, lr],
                param
            )
        else:
            if self.momentum > 0.:
                net.MomentumSGD(
                    [grad, momentum_data, lr], [grad, momentum_data],
                    momentum=self.momentum,
                    nesterov=1)
                coeff = ONE
            else:
                coeff = lr

            net.WeightedSum(
                [param, ONE, grad, coeff],
                param
            )


class AdagradOptimizer(Optimizer):
    def __init__(self, alpha=0.01, epsilon=1e-4, policy="fixed",
                 sparse_dedup_aggregator=None, engine='', **kwargs):
        self.alpha = alpha
        self.epsilon = epsilon
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
        self.init_kwargs = kwargs

    def __call__(self, net, param_init_net, param, grad):
        if self.alpha <= 0:
            return

        lr, _ = self.build_lr(
            net, param_init_net,
            base_learning_rate=self.alpha,
            learning_rate_blob=str(param) + "_lr",
            policy=self.policy,
            **(self.init_kwargs)
        )

        param_square_sum = param_init_net.ConstantFill(
            [param],
            str(param) + "_square_sum",
            value=0.0
        )

        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseAdagrad(
                [param, param_square_sum, grad.indices, grad.values, lr],
                [param, param_square_sum],
                epsilon=self.epsilon,
                engine=self.engine
            )
        else:
            net.Adagrad(
                [param, param_square_sum, grad, lr],
                [param, param_square_sum],
                epsilon=self.epsilon,
                engine=self.engine
            )


class FtrlOptimizer(Optimizer):
    def __init__(self, alpha=0.01, beta=1e-4, lambda1=0, lambda2=0,
                 sparse_dedup_aggregator=None, engine=''):
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine

    def __call__(self, net, param_init_net, param, grad):
        if self.alpha <= 0:
            return

        nz = param_init_net.ConstantFill(
            [param],
            str(param) + "_ftrl_nz",
            extra_shape=[2],
            value=0.0
        )
        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseFtrl(
                [param, nz, grad.indices, grad.values],
                [param, nz],
                engine=self.engine,
                alpha=self.alpha,
                beta=self.beta,
                lambda1=self.lambda1,
                lambda2=self.lambda2
            )
        else:
            net.Ftrl(
                [param, nz, grad],
                [param, nz],
                engine=self.engine,
                alpha=self.alpha,
                beta=self.beta,
                lambda1=self.lambda1,
                lambda2=self.lambda2
            )


class AdamOptimizer(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 policy='fixed', sparse_dedup_aggregator=None,
                 engine='', **kwargs):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
        self.init_kwargs = kwargs

    def __call__(self, net, param_init_net, param, grad):
        if self.alpha <= 0:
            return

        lr, iterator = self.build_lr(
            net, param_init_net,
            base_learning_rate=self.alpha,
            learning_rate_blob=str(param) + "_lr",
            policy=self.policy,
            **(self.init_kwargs)
        )

        m1 = param_init_net.ConstantFill(
            [param],
            param + "_first_moment",
            value=0.0
        )
        m2 = param_init_net.ConstantFill(
            [param],
            param + "_second_moment",
            value=0.0
        )
        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseAdam(
                [param, m1, m2, grad.indices, grad.values, lr, iterator],
                [param, m1, m2],
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon
            )

        else:
            net.Adam(
                [param, m1, m2, grad, lr, iterator],
                [param, m1, m2],
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon)


def build_sgd(model, base_learning_rate, **kwargs):
    sgd_optimizer = SgdOptimizer(base_learning_rate, **kwargs)
    for param, grad in model.GetOptimizationPairs().items():
        sgd_optimizer(model.net, model.param_init_net, param, grad)


def build_ftrl(model, engine="SIMD", **kwargs):
    if engine == "SIMD":
        assert core.IsOperator('Ftrl_ENGINE_SIMD')
        assert core.IsOperator('SparseFtrl_ENGINE_SIMD')
    ftrl_optimizer = FtrlOptimizer(engine=engine, **kwargs)
    for param, grad in model.GetOptimizationPairs().items():
        ftrl_optimizer(model.net, model.param_init_net, param, grad)


def build_adagrad(model, base_learning_rate, parameters=None, **kwargs):
    adagrad_optimizer = AdagradOptimizer(alpha=base_learning_rate, **kwargs)
    param_to_grad = model.GetOptimizationPairs(parameters)

    for param, grad in param_to_grad.items():
        adagrad_optimizer(model.net, model.param_init_net, param, grad)


def build_adam(model, base_learning_rate, **kwargs):
    adam_optimizer = AdamOptimizer(alpha=base_learning_rate, **kwargs)
    for param, grad in model.GetOptimizationPairs().items():
        adam_optimizer(model.net, model.param_init_net, param, grad)
