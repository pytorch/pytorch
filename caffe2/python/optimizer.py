## @package optimizer
# Module caffe2.python.optimizer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import namedtuple
from past.builtins import basestring

from caffe2.python import core, scope
from caffe2.python.modeling import parameter_info
from caffe2.proto import caffe2_pb2


_OPTIMIZER_ITERATION_NAME = "optimizer_iteration"

AuxOptimizerParams = namedtuple("AuxOptimizerParams", ["local", "shared"])

class Optimizer(object):
    def __init__(self):
        self._aux_params = AuxOptimizerParams(local=[], shared=[])

    '''
    Adds optimization operators to the net for given parameter and its gradient
    Parameter is specified by either 'param' being a ParameterInfo object.
    In this case  param.grad has to be set

    Or by 'param' being a BlobReference and 'grad' being a BlobReference for its
    gradient.
    '''
    def __call__(self, net, param_init_net, param, grad=None):
        if grad is None:
            assert isinstance(param, parameter_info.ParameterInfo)
            assert param.grad is not None
        else:
            if isinstance(param, basestring):
                param = core.BlobReference(param)
            param = parameter_info.ParameterInfo(
                param_id=None, param=param, grad=grad)

        self._run(net, param_init_net, param)

    def _run(self, net, param_init_net, param_info):
        raise Exception("Not Impelemented")

    @staticmethod
    def build_lr(net, param_init_net, base_learning_rate,
                 learning_rate_blob="lr", policy="fixed",
                 iter_val=0, **kwargs):
        if not param_init_net.BlobIsDefined(_OPTIMIZER_ITERATION_NAME):
            # Add training operators.
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                iteration = param_init_net.ConstantFill(
                    [], _OPTIMIZER_ITERATION_NAME, shape=[1],
                    value=iter_val,
                    dtype=core.DataType.INT64)
                iter_mutex = param_init_net.CreateMutex([], ["iteration_mutex"])
                net.AtomicIter([iter_mutex, iteration], [iteration])
        else:
            iteration = param_init_net.GetBlobRef(_OPTIMIZER_ITERATION_NAME)

        # There is one interesting thing here: since we are minimizing, we are
        # doing "descent" so the learning rate is set to be negative.
        lr = net.LearningRate(
            [iteration],
            learning_rate_blob,
            base_lr=-base_learning_rate,
            policy=policy,
            **kwargs
        )
        return lr, iteration

    @staticmethod
    def dedup(net, sparse_dedup_aggregator, grad):
        assert (isinstance(grad, core.GradientSlice))
        if sparse_dedup_aggregator:
            return net.DeduplicateGradientSlices(
                grad, aggregator=sparse_dedup_aggregator)
        else:
            return grad

    def get_auxiliary_parameters(self):
        """Returns a list of auxiliary parameters.

        Returns:
            aux_params: A namedtuple, AuxParams.

            aux_params.local stores a list of blobs. Each blob is a local
            auxiliary parameter. A local auxiliary parameter is a parameter in
            parallel to a learning rate parameter. Take adagrad as an example,
            the local auxiliary parameter is the squared sum parameter, because
            every learning rate has a squared sum associated with it.

            aux_params.shared also stores a list of blobs. Each blob is a shared
            auxiliary parameter. A shared auxiliary parameter is a parameter
            that is shared across all the learning rate parameters. Take adam as
            an example, the iteration parameter is a shared parameter, because
            all the learning rates share the same iteration parameter.
        """
        return self._aux_params

    # TODO(xlwang): In transfer learning, parameter initialized from pretrained
    # model might require a different learning rate than otherwise initialized.
    # To this end, here we implement a python solution where
    # `base_learning_rate` is scaled by `scale`, by calling
    # `scale_learning_rate`; Alternatively, we can achieve same effect by
    # rewriting the LearningRate operator in C++
    # Note that it is the responsibility of specific optimizer to decide what
    # logic should be used for `scale_learning_rate`
    def scale_learning_rate(self, *args, **kwargs):
        raise NotImplementedError(
            "Optimizer Need to Implement `scale_learning_rate` method.")


class SgdOptimizer(Optimizer):
    def __init__(self, base_learning_rate=0.01, policy='fixed',
                 momentum=0.0, nesterov=1, **kwargs):
        super(SgdOptimizer, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.policy = policy
        self.momentum = momentum
        self.nesterov = nesterov
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.base_learning_rate == 0:
            return
        assert self.base_learning_rate > 0

        # We need negative sign for LR when used directly with WeightedSum
        # below.
        lr_sign = -1 if self.momentum else 1
        lr, _ = self.build_lr(
            net, param_init_net,
            base_learning_rate=self.base_learning_rate * lr_sign,
            learning_rate_blob=str(param) + "_lr",
            policy=self.policy,
            **(self.init_kwargs)
        )

        dev = scope.CurrentDeviceScope()
        if dev is None:
            dev = core.DeviceOption(caffe2_pb2.CPU)

        # Each GPU/CPU must have its own ONE blob, thus modify the name
        # to include device information.
        ONE = param_init_net.ConstantFill(
            [],
            "ONE_{}_{}".format(dev.device_type, dev.cuda_gpu_id),
            shape=[1],
            value=1.0
        )

        self._aux_params.shared.append(ONE)

        if self.momentum > 0:
            momentum_data = param_init_net.ConstantFill(
                param, str(param) + "_momentum", value=0.)
            self._aux_params.local.append(momentum_data)

        if isinstance(grad, core.GradientSlice):
            assert self.momentum == 0., "Doesn't support momentum for sparse"
            net.ScatterWeightedSum(
                [param, ONE, grad.indices, grad.values, lr],
                param
            )
        else:
            if self.momentum > 0.:
                net.MomentumSGDUpdate(
                    [grad, momentum_data, lr, param],
                    [grad, momentum_data, param],
                    momentum=self.momentum,
                    nesterov=self.nesterov)
            else:
                coeff = lr

                net.WeightedSum(
                    [param, ONE, grad, coeff],
                    param
                )

    def scale_learning_rate(self, scale):
        self.base_learning_rate *= scale
        return


class MultiPrecisionSgdOptimizer(SgdOptimizer):
    def __init__(self, base_learning_rate=0.1, momentum=0.0,
                 policy="fixed", nesterov=1, **kwargs):
        super(SgdOptimizer, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.momentum = momentum
        self.policy = policy
        self.nesterov = nesterov
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        param_fp32 = param_info.blob_copy[core.DataType.FLOAT] \
                if param_info.blob_copy is not None else None

        # If we have a straight fp32 parameter, run the base class
        if param_fp32 == None:
            return SgdOptimizer._run(self, net, param_init_net, param_info)

        grad = param_info.grad
        if self.base_learning_rate == 0:
            return
        assert self.base_learning_rate > 0

        lr, _ = self.build_lr(
            net, param_init_net,
            base_learning_rate=-self.base_learning_rate,
            learning_rate_blob=param + "_lr",
            policy=self.policy,
            **(self.init_kwargs)
        )

        momentum_data = param_init_net.ConstantFill(
            param_fp32, str(param) + "_momentum", value=0.)
        self._aux_params.local.append(momentum_data)

        assert not isinstance(grad, core.GradientSlice), \
                "Doesn't support sparse gradients"

        # Copy gradient to fp32
        grad_fp32 = net.HalfToFloat(grad, grad + "_fp32")

        # update (fused) in fp32
        net.MomentumSGDUpdate(
            [grad_fp32, momentum_data, lr, param_fp32],
            [grad_fp32, momentum_data, param_fp32],
            momentum=self.momentum,
            nesterov=self.nesterov)

        # Copy updated param back to fp16
        net.FloatToHalf(param_fp32, param)


class WeightDecayBuilder(Optimizer):
    def __init__(self, weight_decay):
        self.weight_decay = weight_decay

    def _run(self, net, param_init_net, param_info):
        dev = scope.CurrentDeviceScope()
        if dev is None:
            dev = core.DeviceOption(caffe2_pb2.CPU)

        ONE = param_init_net.ConstantFill(
            [],
            "ONE_{}_{}".format(dev.device_type, dev.cuda_gpu_id),
            shape=[1],
            value=1.0
        )
        WD = param_init_net.ConstantFill(
            [], "wd_{}_{}".format(dev.device_type, dev.cuda_gpu_id),
            shape=[1], value=self.weight_decay
        )

        if isinstance(param_info.grad, core.GradientSlice):
            assert "Weight decay does not yet support sparse gradients"
        else:
            net.WeightedSum(
                [param_info.grad, ONE, param_info.blob, WD],
                param_info.grad,
            )


class AdagradOptimizer(Optimizer):
    def __init__(self, alpha=0.01, epsilon=1e-4, policy="fixed",
                 sparse_dedup_aggregator=None, engine='', **kwargs):
        super(AdagradOptimizer, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        if self.alpha <= 0:
            return

        lr, _ = self.build_lr(
            net, param_init_net,
            base_learning_rate=self.alpha,
            learning_rate_blob=str(param) + "_lr",
            policy=self.policy,
            **(self.init_kwargs)
        )

        param_squared_sum = param_init_net.ConstantFill(
            [param],
            str(param) + "_squared_sum",
            value=0.0
        )
        self._aux_params.local.append(param_squared_sum)

        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseAdagrad(
                [param, param_squared_sum, grad.indices, grad.values, lr],
                [param, param_squared_sum],
                epsilon=self.epsilon,
                engine=self.engine
            )
        else:
            net.Adagrad(
                [param, param_squared_sum, grad, lr],
                [param, param_squared_sum],
                epsilon=self.epsilon,
                engine=self.engine
            )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


class FtrlOptimizer(Optimizer):
    def __init__(self, alpha=0.01, beta=1e-4, lambda1=0, lambda2=0,
                 sparse_dedup_aggregator=None, engine=''):
        super(FtrlOptimizer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        if self.alpha <= 0:
            return

        nz = param_init_net.ConstantFill(
            [param],
            str(param) + "_ftrl_nz",
            extra_shape=[2],
            value=0.0
        )
        self._aux_params.local.append(nz)
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

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


class AdamOptimizer(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 policy='fixed', sparse_dedup_aggregator=None,
                 engine='', **kwargs):
        super(AdamOptimizer, self).__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        if self.alpha <= 0:
            return

        lr, iteration = self.build_lr(
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
        self._aux_params.shared.append(iteration)
        self._aux_params.local.append(m1)
        self._aux_params.local.append(m2)
        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseAdam(
                [param, m1, m2, grad.indices, grad.values, lr, iteration],
                [param, m1, m2],
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon
            )

        else:
            net.Adam(
                [param, m1, m2, grad, lr, iteration],
                [param, m1, m2],
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon)

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


def _get_param_to_device(model):
    # Infer blob devices by going through the net and param_init_net
    # ops and observing the device used to create or use the blob.
    param_to_device = core.InferBlobDevices(model.net)
    param_to_device.update(core.InferBlobDevices(model.param_init_net))
    return param_to_device


def _build(model, optimizer, weights_only=False):
    param_to_device = _get_param_to_device(model)

    # Validate there are no duplicate params
    model.Validate()

    # Call optimizer for each param
    for param_info in model.GetOptimizationParamInfo():
        if weights_only:
            if param_info.blob not in model.weights:
                continue
        param_name = str(param_info.blob)

        # We first check if parameter's device has been inferred. If not,
        # we check the gradient. This can happen if parameter is not output
        # by any blob but created by a FetchBlob.
        device = None
        if param_name in param_to_device:
            device = param_to_device[param_name]
        else:
            if isinstance(param_info.grad, core.GradientSlice):
                grad = param_info.grad
                if str(grad.values) in param_to_device:
                    device = param_to_device[str(grad.values)]
                elif str(grad.indices) in param_to_device:
                    device = param_to_device[str(grad.indices)]
            else:
                grad_name = str(param_info.grad)
                if grad_name in param_to_device:
                    device = param_to_device[grad_name]

        assert device is not None,\
            "Cannot infer device for {}: no op creates it".format(param_name)

        with core.DeviceScope(device):
            optimizer(model.net, model.param_init_net, param_info)
    return optimizer


def add_weight_decay(model, weight_decay):
    """Adds a decay to weights in the model.

    This is a form of L2 regularization.

    Args:
        weight_decay: strength of the regularization
    """
    _build(
        model,
        WeightDecayBuilder(weight_decay=weight_decay),
        weights_only=True,
    )


def build_sgd(model, base_learning_rate, **kwargs):
    sgd_optimizer = SgdOptimizer(base_learning_rate, **kwargs)
    return _build(model, sgd_optimizer)


def build_multi_precision_sgd(model, base_learning_rate, **kwargs):
    multi_prec_sgd_optimizer = MultiPrecisionSgdOptimizer(
        base_learning_rate, **kwargs
    )
    return _build(model, multi_prec_sgd_optimizer)


def build_ftrl(model, engine="SIMD", **kwargs):
    if engine == "SIMD":
        assert core.IsOperator('Ftrl_ENGINE_SIMD')
        assert core.IsOperator('SparseFtrl_ENGINE_SIMD')
    ftrl_optimizer = FtrlOptimizer(engine=engine, **kwargs)
    return _build(model, ftrl_optimizer)


def build_adagrad(model, base_learning_rate, parameters=None, **kwargs):
    adagrad_optimizer = AdagradOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, adagrad_optimizer)


def build_adam(model, base_learning_rate, **kwargs):
    adam_optimizer = AdamOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, adam_optimizer)
