# @package optimizer
# Module caffe2.python.optimizer


import copy
import logging
from collections import defaultdict, namedtuple
from typing import Any, Dict

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, scope, utils, workspace
from caffe2.python.modeling import parameter_info
from past.builtins import basestring


_LEARNING_RATE_INJECTION = "lr_injection"

AuxOptimizerParams = namedtuple("AuxOptimizerParams", ["local", "shared"])
_optimizer_instance_count = defaultdict(int)

FP16_ENGINES = ["SIMD_Q_FP16", "SIMD_Q_STOC_FP16", "SIMD_Q_STOC_MKL_FP16"]

logger = logging.getLogger(__name__)

def reset_optimizer_instance_count():
    """
    This function clears the _optimizer_instance_count. And keeps it
    empty. This functionality is needed in some situations where
    optimizer instance count might not reset even though the workplace is reset.
    """
    _optimizer_instance_count.clear()


class Optimizer:
    def __init__(self):
        self._aux_params = AuxOptimizerParams(local=[], shared=[])
        self._instance_num = _optimizer_instance_count[self.__class__.__name__]
        _optimizer_instance_count[self.__class__.__name__] += 1
        self._lr_multiplier = None
        self._local_lr_multiplier = None
        self._local_lr_multiplier_on_gpu = False
        self._use_dedicated_lr_iteration_counter = False

    """
    Adds optimization operators to the net for given parameter and its gradient
    Parameter is specified by either 'param' being a ParameterInfo object.
    In this case  param.grad has to be set

    Or by 'param' being a BlobReference and 'grad' being a BlobReference for its
    gradient.
    """

    def __call__(self, net, param_init_net, param, grad=None):
        if grad is None:
            assert isinstance(
                param, parameter_info.ParameterInfo
            ), "Expected parameter to be of type ParameterInfo, got {}".format(param)
            assert param.grad is not None
        else:
            if isinstance(param, basestring):
                param = core.BlobReference(param)
            param = parameter_info.ParameterInfo(param_id=None, param=param, grad=grad)

        self._run(net, param_init_net, param)

    def _run(self, net, param_init_net, param_info):
        raise Exception("Not Implemented")

    def get_cpu_blob_name(self, base_str, node_name=""):
        classname = self.__class__.__name__
        return "%s_%d_%s%s_cpu" % (classname, self._instance_num, base_str, node_name)

    def get_gpu_blob_name(self, base_str, gpu_id, node_name):
        classname = self.__class__.__name__
        return "%s_%d_%s%s_gpu%d" % (
            classname,
            self._instance_num,
            base_str,
            node_name,
            gpu_id,
        )

    @property
    def attributes(self):
        # return a dict that contains attributes related to init args only
        attr = copy.deepcopy(self.__dict__)
        del attr["_instance_num"]
        return attr

    @property
    def use_dedicated_lr_iteration_counter(self):
        return self._use_dedicated_lr_iteration_counter

    @use_dedicated_lr_iteration_counter.setter
    def use_dedicated_lr_iteration_counter(self, val):
        self._use_dedicated_lr_iteration_counter = val

    def make_unique_blob_name(self, base_str):
        """
        Returns a blob name that will be unique to the current device
        and optimizer instance.
        """
        current_scope = scope.CurrentDeviceScope()
        if current_scope is None:
            return self.get_cpu_blob_name(base_str)

        if core.IsGPUDeviceType(current_scope.device_type):
            return self.get_gpu_blob_name(
                base_str, current_scope.device_id, current_scope.node_name
            )
        else:
            return self.get_cpu_blob_name(base_str, current_scope.node_name)

    def build_lr(
        self,
        net,
        param_init_net,
        base_learning_rate,
        learning_rate_blob=None,
        policy="fixed",
        iter_val=0,
        **kwargs
    ):
        if learning_rate_blob is None:
            learning_rate_blob = self.make_unique_blob_name("lr")

        if self._use_dedicated_lr_iteration_counter:
            iteration = utils.BuildUniqueMutexIter(
                param_init_net,
                net,
                iter=utils.OPTIMIZER_ITERATION_LR_NAME,
                iter_mutex=utils.ITERATION_MUTEX_LR_NAME,
                iter_val=iter_val,
            )
            logger.info(f"Created dedicated learning rate iteration counter: {iteration}")
        else:
            iteration = utils.BuildUniqueMutexIter(param_init_net, net, iter_val=iter_val)

        if not net.BlobIsDefined(learning_rate_blob):
            # There is one interesting thing here: since we are minimizing, we are
            # doing "descent" so the learning rate is set to be negative.
            lr = net.LearningRate(
                [iteration],
                learning_rate_blob,
                base_lr=-base_learning_rate,
                policy=policy,
                **kwargs
            )
        else:
            lr = net.GetBlobRef(learning_rate_blob)

        if self._lr_multiplier is not None:
            lr_multiplier = net.CopyFromCPUInput(
                self._lr_multiplier, self.make_unique_blob_name("lr_multiplier")
            )

            lr = net.Mul(
                [lr, lr_multiplier],
                self.make_unique_blob_name("scaled_lr"),
                broadcast=1,
            )

        if self._local_lr_multiplier is not None:
            current_scope = scope.CurrentDeviceScope()
            if (
                current_scope is not None
                and core.IsGPUDeviceType(current_scope.device_type)
                and not self._local_lr_multiplier_on_gpu
            ):
                local_lr_multiplier = net.CopyFromCPUInput(
                    self._local_lr_multiplier,
                    self.make_unique_blob_name("local_lr_multiplier"),
                )
            else:
                local_lr_multiplier = self._local_lr_multiplier

            lr = net.Mul(
                [lr, local_lr_multiplier],
                self.make_unique_blob_name("local_scaled_lr"),
                broadcast=1,
            )

        return lr, iteration

    def build_non_lr_iter(
        self,
        net,
        param_init_net,
        iter_val=0,
    ):
        assert (
            self._use_dedicated_lr_iteration_counter
        ), "This method should be only called when dedicated learning rate iteration counter is used."

        iteration = utils.BuildUniqueMutexIter(param_init_net, net, iter_val=iter_val)
        logger.info(f"Created iteration counter for non learning rate purposes: {iteration}")

        # We need to create a dummy learning rate operator to enforce that
        # iteration counter blob being placed in the trainer nodes. Otherwise,
        # the Automatic Device Placement (ADP) algorithm for Hierachical
        # Training (HT) will encounter issues to distribute blobs across group
        # parameter servers. Note that this learning rate operator will not be
        # used for any other purpose.
        learning_rate_blob = self.make_unique_blob_name("iter_placement_hint")
        if not net.BlobIsDefined(learning_rate_blob):
            net.LearningRate(
                [iteration],
                learning_rate_blob,
                base_lr=1.0,
                policy="fixed",
            )

        return iteration

    def add_lr_multiplier(self, lr_multiplier):
        """
        Set the global learning rate multiplier. If a multiplier already
        existed, this will overwrite the existing multiplier. The multiplier is
        used for all future calls to _run(), unless it is overwritten.
        """
        self._lr_multiplier = lr_multiplier

    def _add_local_lr_multiplier(self, local_lr_multiplier, is_gpu_blob=False):
        """
        Set the local learning rate multiplier. This local multiplier is
        multiplied with the global learning rate multiplier if it exists. As
        with the global learning rate multiplier, this multiplier will be
        used for all future calls to _run(), so please call
        _clear_local_lr_multiplier() at the beginning of the optimizer's _run()
        before optionally calling this function.
        """
        self._local_lr_multiplier = local_lr_multiplier
        self._local_lr_multiplier_on_gpu = is_gpu_blob

    def _clear_local_lr_multiplier(self):
        self._local_lr_multiplier = None
        self._local_lr_multiplier_on_gpu = False

    @staticmethod
    def dedup(net, sparse_dedup_aggregator, grad):
        assert isinstance(
            grad, core.GradientSlice
        ), "Dedup only works for sparse gradient, got {}".format(grad)
        if sparse_dedup_aggregator:
            return net.DeduplicateGradientSlices(
                grad, aggregator=sparse_dedup_aggregator
            )
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
            "Optimizer Need to Implement `scale_learning_rate` method."
        )

    def create_lars_inputs(self, param_init_net, weight_decay, trust, lr_max):
        wd = param_init_net.ConstantFill(
            [], "weight_decay", shape=[1], value=weight_decay
        )
        trust = param_init_net.ConstantFill([], "trust", shape=[1], value=trust)
        lr_max = param_init_net.ConstantFill([], "lr_max", shape=[1], value=lr_max)
        return wd, trust, lr_max


class SgdOptimizer(Optimizer):
    def __init__(
        self,
        base_learning_rate=0.01,
        policy="fixed",
        momentum=0.0,
        nesterov=True,
        sparse_dedup_aggregator=None,
        lars=None,
        **kwargs
    ):
        super().__init__()
        self.base_learning_rate = base_learning_rate
        self.policy = policy
        self.momentum = momentum
        self.nesterov = nesterov
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.lars = lars
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.base_learning_rate == 0:
            return
        assert (
            self.base_learning_rate > 0
        ), "Expect positive base learning rate, got {}".format(self.base_learning_rate)

        self._clear_local_lr_multiplier()

        # TODO(zqq): support LARS for sparse parameters
        if self.lars is not None and not isinstance(grad, core.GradientSlice):
            assert self.lars >= 0, "Lars offset must be nonnegative, got {}".format(
                self.lars
            )
            wd, trust, lr_max = self.create_lars_inputs(
                param_init_net, 0.0, 1.0, np.finfo(np.float32).max
            )
            lr_lars_multiplier = net.Lars(
                [param, grad, wd, trust, lr_max],
                self.make_unique_blob_name(str(param) + "_lars"),
                offset=self.lars,
                lr_min=0.0,
            )
            current_scope = scope.CurrentDeviceScope()
            self._add_local_lr_multiplier(
                lr_lars_multiplier,
                is_gpu_blob=(
                    current_scope is not None
                    and core.IsGPUDeviceType(current_scope.device_type)
                ),
            )

        # We need negative sign for LR when used directly with WeightedSum
        # below.
        lr_sign = -1 if self.momentum else 1
        lr, _ = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=self.base_learning_rate * lr_sign,
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
            "ONE_{}_{}{}".format(dev.device_type, dev.device_id, dev.node_name),
            shape=[1],
            value=1.0,
        )

        self._aux_params.shared.append(ONE)

        if self.momentum > 0:
            momentum_data = param_init_net.ConstantFill(
                param, str(param) + "_momentum", value=0.0
            )
            self._aux_params.local.append(momentum_data)

        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            if self.momentum > 0.0:
                net.SparseMomentumSGDUpdate(
                    [grad.values, momentum_data, lr, param, grad.indices],
                    [grad.values, momentum_data, param],
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                )
            else:
                net.ScatterWeightedSum(
                    [param, ONE, grad.indices, grad.values, lr], param
                )
        else:
            if self.momentum > 0.0:
                net.MomentumSGDUpdate(
                    [grad, momentum_data, lr, param],
                    [grad, momentum_data, param],
                    momentum=self.momentum,
                    nesterov=self.nesterov,
                )
            else:
                coeff = lr

                net.WeightedSum([param, ONE, grad, coeff], param)

    def scale_learning_rate(self, scale):
        self.base_learning_rate *= scale
        return


class MultiPrecisionSgdOptimizer(SgdOptimizer):
    def __init__(
        self,
        base_learning_rate=0.1,
        momentum=0.0,
        policy="fixed",
        nesterov=True,
        sparse_dedup_aggregator=None,
        **kwargs
    ):
        super().__init__(
            base_learning_rate=base_learning_rate,
            policy=policy,
            momentum=momentum,
            nesterov=nesterov,
            sparse_dedup_aggregator=sparse_dedup_aggregator,
            **kwargs
        )

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        param_fp32 = (
            param_info.blob_copy[core.DataType.FLOAT]
            if param_info.blob_copy is not None
            else None
        )

        # If we have a straight fp32 parameter, run the base class
        if param_fp32 is None:
            return SgdOptimizer._run(self, net, param_init_net, param_info)

        grad = param_info.grad
        if self.base_learning_rate == 0:
            return
        assert (
            self.base_learning_rate > 0
        ), "Expect positive base learning rate, got {}".format(self.base_learning_rate)

        lr, _ = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=-self.base_learning_rate,
            policy=self.policy,
            **(self.init_kwargs)
        )

        momentum_data = param_init_net.ConstantFill(
            param_fp32, str(param) + "_momentum", value=0.0
        )
        self._aux_params.local.append(momentum_data)

        assert not isinstance(
            grad, core.GradientSlice
        ), "MultiPrecisionSgd does not support sparse gradients"

        # Copy gradient to fp32
        grad_fp32 = net.HalfToFloat(grad, grad + "_fp32")

        # update (fused) in fp32
        net.MomentumSGDUpdate(
            [grad_fp32, momentum_data, lr, param_fp32],
            [grad_fp32, momentum_data, param_fp32],
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

        # Copy updated param back to fp16
        net.FloatToHalf(param_fp32, param)


class FP16SgdOptimizer(SgdOptimizer):
    def __init__(
        self,
        base_learning_rate=0.1,
        momentum=0.0,
        policy="fixed",
        nesterov=True,
        weight_decay=0.0001,
        sparse_dedup_aggregator=None,
        **kwargs
    ):
        super().__init__(
            base_learning_rate=base_learning_rate,
            policy=policy,
            momentum=momentum,
            nesterov=nesterov,
            sparse_dedup_aggregator=sparse_dedup_aggregator,
            **kwargs
        )
        self.weight_decay = weight_decay

    def _run(self, net, param_init_net, param_info, fp32_update=False):

        fp32_update_flag = 0
        param_name = str(param_info.blob)

        # should only be triggered in FP16 training by SpatialBN, which
        # requires FP32 params in CuDNN.
        if param_name.find("spatbn") != -1:
            fp32_update = True

        if fp32_update:
            # doing a 32bit update
            # Have to assume param_info.blob is FP32 as there is no way
            # (that i currently know of) to query a blob's type in python
            fp32_update_flag = 1
            param = param_info.blob
            param_fp32 = param_info.blob
        else:
            if param_info.blob_copy is None:
                # doing a 32bit update
                # Have to assume param_info.blob is FP32 as there is no way
                # (that i currently know of) to query a blob's type in python
                fp32_update_flag = 1
                param = param_info.blob
                param_fp32 = param_info.blob
            else:
                if core.DataType.FLOAT in param_info.blob_copy:
                    param = param_info.blob
                    param_fp32 = param_info.blob_copy[core.DataType.FLOAT]
                elif core.DataType.FLOAT16 in param_info.blob_copy:
                    param = param_info.blob_copy[core.DataType.FLOAT16]
                    param_fp32 = param_info.blob
                else:
                    AssertionError(
                        "Unrecognized parameter format to be updated "
                        "by FP16 Optimizer. Parameter: {}".format(param_info.name)
                    )

        grad = param_info.grad

        if self.base_learning_rate == 0:
            return
        assert (
            self.base_learning_rate > 0
        ), "Expect positive base learning rate, got {}".format(self.base_learning_rate)

        lr, _ = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=-self.base_learning_rate,
            policy=self.policy,
            **(self.init_kwargs)
        )

        momentum_data_fp32 = param_init_net.ConstantFill(
            param_fp32, str(param) + "_momentum_fp32", value=0.0
        )

        momentum_data = param_init_net.FloatToHalf(
            momentum_data_fp32, str(param) + "_momentum"
        )

        self._aux_params.local.append(momentum_data)

        assert not isinstance(
            grad, core.GradientSlice
        ), "FP16Sgd does not support sparse gradients"

        if fp32_update_flag == 0:
            net.FP16MomentumSGDUpdate(
                [grad, momentum_data, lr, param],
                [grad, momentum_data, param],
                momentum=self.momentum,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
            )
        else:
            # flag set to 1, therefore doing FP32 update
            net.FP32MomentumSGDUpdate(
                [grad, momentum_data_fp32, lr, param],
                [grad, momentum_data_fp32, param],
                momentum=self.momentum,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
            )


class WeightDecayBuilder(Optimizer):
    def __init__(self, weight_decay):
        self.weight_decay = weight_decay

    def _run(self, net, param_init_net, param_info):
        dev = scope.CurrentDeviceScope()
        if dev is None:
            dev = core.DeviceOption(caffe2_pb2.CPU)

        ONE = param_init_net.ConstantFill(
            [], "ONE_{}_{}".format(dev.device_type, dev.device_id), shape=[1], value=1.0
        )
        WD = param_init_net.ConstantFill(
            [],
            "wd_{}_{}".format(dev.device_type, dev.device_id),
            shape=[1],
            value=self.weight_decay,
        )

        if isinstance(param_info.grad, core.GradientSlice):
            raise ValueError("Weight decay does not yet support sparse gradients")
        else:
            net.WeightedSum(
                [param_info.grad, ONE, param_info.blob, WD], param_info.grad
            )


class AdagradOptimizer(Optimizer):
    def __init__(
        self,
        alpha=0.01,
        epsilon=1e-4,
        decay=1,
        weight_decay=0.0,
        policy="fixed",
        sparse_dedup_aggregator=None,
        rowWise=False,
        engine="",
        lars=None,
        output_effective_lr=False,
        output_effective_lr_and_update=False,
        pruning_options=None,
        swa_options=None,
        ema_options=None,
        weight_scale=None,
        counter_halflife=-1,
        use_dedicated_lr_iteration_counter=False,
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay = decay
        self.weight_decay = float(weight_decay)
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.rowWise = rowWise
        self.engine = engine
        self.lars = lars
        self.output_effective_lr = output_effective_lr
        self.output_effective_lr_and_update = output_effective_lr_and_update
        self.counter_halflife = counter_halflife
        self.init_kwargs = kwargs
        self.weight_scale = weight_scale
        self.use_dedicated_lr_iteration_counter = use_dedicated_lr_iteration_counter

        self._process_pruning_options(pruning_options)
        self._process_swa_options(swa_options)
        self._process_ema_options(ema_options)

    def set_mapping_for_param2ema_teacher_param(self, param_mapping: Dict[str, Any]) -> None:
        self.param2ema_teacher_param = param_mapping

    def _process_swa_options(self, swa_options):
        self.swa_enabled = True if swa_options else False
        if self.swa_enabled:
            self.swa_avg_start_it = swa_options.get("swa_avg_start_it", None)
            self.swa_avg_end_it = swa_options.get("swa_avg_end_it", None)
            self.swa_feedback_start_it = swa_options.get("swa_feedback_start_it", None)
            self.swa_feedback_step = swa_options.get("swa_feedback_step", None)
            self.swa_feedback_end_it = swa_options.get("swa_feedback_end_it", None)

    def _process_ema_options(self, ema_options):
        logger.info(f"ema_options: {str(ema_options)}")
        self.ema_enabled = ema_options and ema_options.get("ema_alpha", None) is not None
        self.ema_teacher_enabled = ema_options and ema_options.get("ema_teacher_alpha", None) is not None
        self.param2ema_teacher_param = {}
        if self.ema_enabled or self.ema_teacher_enabled:
            self.ema_start = ema_options.get("ema_start", None)
            self.ema_end = ema_options.get("ema_end", None)
            self.ema_step = ema_options.get("ema_step", None)
            self.ema_alpha = ema_options.get("ema_alpha", None)
            self.ema_teacher_alpha = ema_options.get("ema_teacher_alpha", None)
            self.ema_teacher_module_name = ema_options.get(
                "ema_teacher_module_name", "ema_teacher_arch"
            )

    def _process_pruning_options(self, pruning_options):
        self.use_mask = False

        if pruning_options is None:
            pruning_options = {}
        else:
            assert isinstance(pruning_options, dict), (
                "pruning_options can only "
                "be provided as a dictionary, currently: {}".format(pruning_options)
            )

        self.mask_tensor = pruning_options.get("mask_tensor", None)
        self.mask_db_path = pruning_options.get("mask_db_path", None)
        self.mask_db_type = pruning_options.get("mask_db_type", None)
        self.mask_blob_name = pruning_options.get("mask_blob_name", None)
        self.prune_delays = pruning_options.get("prune_delays", [])
        self.prune_ratios = pruning_options.get("prune_ratios", [])
        self.prune_block_size = pruning_options.get("prune_block_size", 1)

        if self.mask_tensor is not None:
            assert (
                type(self.mask_tensor) is np.ndarray
            ), "mask_tensor must be a numpy array!"
            assert self.mask_db_path is None, (
                "mask can be provided through either a numpy array "
                "or a db path, not both"
            )
            assert self.mask_db_type is None, (
                "mask can be provided through either a numpy array "
                "or a db path, not both"
            )
            assert self.mask_blob_name is None, (
                "mask can be provided through either a numpy array "
                "or a db path, not both"
            )
            self.use_mask = True

        if self.mask_db_path is not None or self.mask_db_type is not None:
            assert self.mask_db_path is not None, (
                "when mask is provided through db, "
                "db path, db type, and blob name are all needed"
            )
            assert self.mask_db_type is not None, (
                "when mask is provided through db, "
                "db path, db type, and blob name are all needed"
            )
            assert self.mask_tensor is None, (
                "mask can be provided through either a numpy array "
                "or a db path, not both"
            )
            self.use_mask = True

        if self.prune_delays:
            assert self.prune_ratios is not None and len(self.prune_delays) == len(
                self.prune_ratios
            ), "Prune Delays and prune ratios should be of the same length"
            assert (
                self.mask_tensor is None
            ), "Mask Tensor should be None with prune ratios"
            assert (
                self.mask_db_path is None
            ), "Mask DB Path should be None with prune ratios"
            self.use_mask = True

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        if self.alpha <= 0:
            return

        self._clear_local_lr_multiplier()

        if self.lars is not None and not isinstance(grad, core.GradientSlice):
            assert (
                self.weight_decay == 0
            ), "weight decay is not implemented for LARS yet"
            assert self.lars >= 0, "Lars offset must be nonnegative, got {}".format(
                self.lars
            )
            wd, trust, lr_max = self.create_lars_inputs(
                param_init_net, 0.0, 1.0, np.finfo(np.float32).max
            )
            lr_lars_multiplier = net.Lars(
                [param, grad, wd, trust, lr_max],
                self.make_unique_blob_name(str(param) + "_lars"),
                offset=self.lars,
                lr_min=0.0,
            )

            current_scope = scope.CurrentDeviceScope()
            self._add_local_lr_multiplier(
                lr_lars_multiplier,
                is_gpu_blob=(
                    current_scope is not None
                    and core.IsGPUDeviceType(current_scope.device_type)
                ),
            )

        lr, lr_iteration = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=self.alpha,
            policy=self.policy,
            **(self.init_kwargs)
        )
        iteration = (
            self.build_non_lr_iter(net, param_init_net, iter_val=0)
            if self._use_dedicated_lr_iteration_counter
            else lr_iteration
        )

        if self.counter_halflife > 0:
            self._aux_params.shared.append(iteration)

        if self.rowWise:
            logger.debug(
                "Using engine {} for rowWise Adagrad to train param {}".format(
                    self.engine, param
                )
            )

            shapes, types = workspace.InferShapesAndTypes([param_init_net])
            if str(param) not in shapes:
                # Type/shape inference is not available for this param, fallback
                # on Shape/Slice logic
                shape = param_init_net.Shape(param, str(param) + "_shape")
                num_rows = param_init_net.Slice(
                    [shape], str(shape) + "_numrows", starts=[0], ends=[1]
                )
                param_squared_sum = param_init_net.ConstantFill(
                    num_rows,
                    str(param) + "_avg_squared_sum",
                    input_as_shape=1,
                    value=0.0,
                )
            else:
                param_squared_sum = param_init_net.ConstantFill(
                    [],
                    str(param) + "_avg_squared_sum",
                    shape=[shapes[str(param)][0]],
                    value=0.0,
                )
        else:
            logger.debug(
                "Using engine {} for regular Adagrad to train param {}".format(
                    self.engine, param
                )
            )

            if self.engine in FP16_ENGINES:
                assert (
                    self.weight_decay == 0
                ), "weight decay is not tested for engine: {}".format(self.engine)

                shapes, types = workspace.InferShapesAndTypes([param_init_net])
                assert str(param) in shapes, shapes
                shape = shapes[str(param)]

                param_squared_sum = param_init_net.Float16ConstantFill(
                    [], str(param) + "_squared_sum", value=0.0, shape=shape
                )
            else:
                param_squared_sum = param_init_net.ConstantFill(
                    [param], str(param) + "_squared_sum", value=0.0
                )

        if self.use_mask is True:
            assert (
                self.weight_decay == 0
            ), "weight decay is not implemented for use_mask yet"

            if self.mask_tensor is not None:
                if not isinstance(grad, core.GradientSlice):
                    mask_blob = param_init_net.GivenTensorFill(
                        [],
                        [str(param) + "_mask"],
                        values=self.mask_tensor,
                        shape=self.mask_tensor.shape,
                    )
                else:
                    self.mask_tensor = self.mask_tensor.astype(np.uint8)
                    mask_blob = param_init_net.GivenTensorBoolFill(
                        [],
                        [str(param) + "_mask"],
                        values=self.mask_tensor,
                        shape=self.mask_tensor.shape,
                    )
                    mask_blob = param_init_net.Cast(mask_blob, to=core.DataType.UINT8)
                    mask_changed_blob = param_init_net.ConstantFill(
                        [],
                        [str(param) + "_mask_changed_blob"],
                        value=False,
                        dtype=core.DataType.BOOL,
                        shape=[1],
                    )
            elif (
                self.mask_db_path is not None or self.mask_db_type is not None
            ):  # mask is provided through a db file
                # if mask_blob_name is not given use the param name to derive mask name
                self.mask_blob_name = self.mask_blob_name or str(param) + "_mask"

                mask_blob = param_init_net.Load(
                    [],
                    self.mask_blob_name,
                    db=self.mask_db_path,
                    db_type=self.mask_db_type,
                    absolute_path=True,
                )

                if isinstance(grad, core.GradientSlice):
                    mask_changed_blob = param_init_net.ConstantFill(
                        [],
                        [str(param) + "_mask_changed_blob"],
                        value=False,
                        dtype=core.DataType.BOOL,
                        shape=[1],
                    )
            elif self.prune_delays:
                last_mask_updated_iter = param_init_net.ConstantFill(
                    [],
                    [str(param) + "_last_mask_updated_iter"],
                    value=-1,
                    dtype=core.DataType.INT64,
                    shape=[1],
                )

                if isinstance(grad, core.GradientSlice):
                    AssertionError(
                        "Prune Delays and Prune Ratios are currently not supported"
                        "for sparse operators"
                    )
                else:
                    mask_blob = param_init_net.GivenTensorFill(
                        [],
                        [str(param) + "_empty_mask"],
                        values=[],
                        dtype=core.DataType.FLOAT,
                        shape=[0],
                    )
            else:
                raise NotImplementedError(
                    "If mask is used, it needs a numpy array or a db file or"
                    "a delay iter needs to be provided"
                )

        self._aux_params.local.append(param_squared_sum)
        if self.counter_halflife > 0:
            shapes, types = workspace.InferShapesAndTypes([param_init_net])
            if str(param) not in shapes:
                shape = param_init_net.Shape(param, str(param) + "_shape")
                num_rows = param_init_net.Slice(
                    [shape], str(shape) + "_numrows", starts=[0], ends=[1]
                )
                update_counter = param_init_net.ConstantFill(
                    num_rows,
                    str(param) + "_update_counter",
                    input_as_shape=1,
                    value=0.0,
                    dtype=core.DataType.DOUBLE,
                )
                prev_update_iter = param_init_net.ConstantFill(
                    num_rows,
                    str(param) + "_prev_update_iter",
                    input_as_shape=1,
                    value=0,
                    dtype=core.DataType.INT64,
                )
            else:
                update_counter = param_init_net.ConstantFill(
                    [],
                    str(param) + "_update_counter",
                    shape=[shapes[str(param)][0]],
                    value=0.0,
                    dtype=core.DataType.DOUBLE,
                )
                prev_update_iter = param_init_net.ConstantFill(
                    [],
                    str(param) + "_prev_update_iter",
                    shape=[shapes[str(param)][0]],
                    value=0,
                    dtype=core.DataType.INT64,
                )
            self._aux_params.local.append(update_counter)
            self._aux_params.local.append(prev_update_iter)

        if self.rowWise:
            assert isinstance(grad, core.GradientSlice), (
                "If SparseAdagrad with rowWise=True, gradient must be "
                "a gradientslice. PLease ensure that rowWise is not enabled "
                "for the dense Adagrad optimizer, as it is not supported."
            )

        shapes, _ = workspace.InferShapesAndTypes([param_init_net])
        param_shape = shapes[str(param)]
        weight_decay = 0.0
        if isinstance(grad, core.GradientSlice):
            if len(param_shape) == 1:
                weight_decay = 0.0
                logger.warn(
                    "SKIPPING weight decay on 1d sparse param: {}.shape is {}".format(
                        str(param), param_shape
                    )
                )
            else:
                weight_decay = self.weight_decay
        else:
            # Skip weight decay for 1d parameters
            if len(param_shape) == 1:
                weight_decay = 0.0
                logger.warning(
                    "SKIPPING weight decay on 1d dense param: {}.shape is {}".format(
                        str(param), param_shape
                    )
                )
            else:
                weight_decay = self.weight_decay
        logger.debug(
            "weight_decay for {} (shape:{}): {}".format(
                str(param), param_shape, weight_decay
            )
        )

        if isinstance(grad, core.GradientSlice):
            assert (
                self.decay == 1.0
            ), "Decay is not implemented for SparseAdagrad and must be set to 1"
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)

            input_args = [param, param_squared_sum, grad.indices, grad.values, lr]
            output_args = [param, param_squared_sum]
            if self.rowWise:
                if self.use_mask is True:
                    op = "MaskedRowWiseSparseAdagrad"
                    assert (
                        weight_decay == 0
                    ), "weight decay is not implemented for {} yet".format(op)
                    input_args += [mask_blob, mask_changed_blob]
                else:
                    if self.counter_halflife > 0:
                        input_args += [update_counter]
                    op = "RowWiseSparseAdagrad"
            else:
                if self.use_mask is True:
                    op = "MaskedSparseAdagrad"
                    assert (
                        weight_decay == 0
                    ), "weight decay is not implemented for {} yet".format(op)
                    input_args += [mask_blob, mask_changed_blob]
                else:
                    op = "SparseAdagrad"
            logger.debug("using {} for {}".format(op, str(param)))

            if self.prune_delays:
                input_args += [iteration, last_mask_updated_iter]
                output_args += [mask_blob, last_mask_updated_iter]

            if weight_decay > 0 and self.counter_halflife == -1:
                net.__getattr__(op)(
                    input_args,
                    output_args,
                    epsilon=self.epsilon,
                    weight_decay=weight_decay,
                    engine=self.engine,
                )
            elif weight_decay > 0 and self.counter_halflife != -1:
                net.__getattr__(op)(
                    input_args,
                    output_args,
                    epsilon=self.epsilon,
                    weight_decay=weight_decay,
                    engine=self.engine,
                    counter_halflife=self.counter_halflife,
                )
            else:
                net.__getattr__(op)(
                    input_args, output_args, epsilon=self.epsilon, engine=self.engine
                )
            if self.counter_halflife > 0:
                net.RowWiseCounter(
                    [prev_update_iter, update_counter, grad.indices, iteration],
                    [prev_update_iter, update_counter],
                    counter_halflife=self.counter_halflife,
                )
        else:
            input_args = [param, param_squared_sum, grad, lr]
            output_args = [param, param_squared_sum]

            if self.output_effective_lr_and_update:
                assert (
                    self.use_mask is False
                ), "MaskedAdagrad doesn't support outputting effective_lr_and_update"
                output_args.append(str(param) + "_effective_lr")
                output_args.append(str(param) + "_update")
            elif self.output_effective_lr:
                assert (
                    self.use_mask is False
                ), "MaskedAdagrad doesn't support outputting effective_lr"
                output_args.append(str(param) + "_effective_lr")

            if self.use_mask is True:
                input_args += [mask_blob]

            if self.prune_delays:
                input_args += [iteration, last_mask_updated_iter]
                output_args += [mask_blob, last_mask_updated_iter]

            if self.use_mask:
                assert (
                    weight_decay == 0
                ), "weight decay is not implemented for use_mask yet"
                net.MaskedAdagrad(
                    input_args,
                    output_args,
                    epsilon=self.epsilon,
                    decay=float(self.decay),
                    block_size=self.prune_block_size,
                    delays=self.prune_delays,
                    prune_ratios=self.prune_ratios,
                    engine=self.engine,
                )
            else:
                if weight_decay > 0:
                    net.Adagrad(
                        input_args,
                        output_args,
                        epsilon=self.epsilon,
                        decay=float(self.decay),
                        weight_decay=weight_decay,
                        engine=self.engine,
                    )
                else:
                    net.Adagrad(
                        input_args,
                        output_args,
                        epsilon=self.epsilon,
                        decay=float(self.decay),
                        engine=self.engine,
                    )

                if self.swa_enabled:
                    param_swa = str(param) + "_swa"
                    if not param_init_net.BlobIsDefined(param_swa):
                        param_init_net.ConstantFill([param], param_swa, value=0.0)
                        self._aux_params.local.append(param_swa)

                    net.SWA(
                        [param, param_swa, iteration],
                        [param, param_swa],
                        avg_start=self.swa_avg_start_it,
                        avg_end=self.swa_avg_end_it,
                        feedback_start=self.swa_feedback_start_it,
                        feedback_step=self.swa_feedback_step,
                        feedback_end=self.swa_feedback_end_it,
                    )

        if self.ema_enabled:
            param_ema = str(param) + "_ema"
            if not param_init_net.BlobIsDefined(param_ema):
                param_init_net.ConstantFill([param], param_ema, value=0.0)
                self._aux_params.local.append(param_ema)

            net.EMA(
                [param, param_ema, iteration],
                [param, param_ema],
                ema_start=self.ema_start,
                ema_end=self.ema_end,
                ema_step=self.ema_step,
                ema_alpha=self.ema_alpha,
            )


        if self.ema_teacher_enabled:
            if param in self.param2ema_teacher_param:
                param_ema_teacher = self.param2ema_teacher_param[param]
                if not param_init_net.BlobIsDefined(param_ema_teacher):
                    param_init_net.ConstantFill([param], param_ema_teacher, value=0.0)
                    self._aux_params.local.append(param_ema_teacher)

                net.EMA(
                    [param, param_ema_teacher, iteration],
                    [param, param_ema_teacher],
                    ema_start=self.ema_start,
                    ema_end=self.ema_end,
                    ema_step=self.ema_step,
                    ema_alpha=self.ema_teacher_alpha,
                )

        if self.weight_scale:
            net.WeightScale(
                [param, iteration],
                [param],
                stepsize=self.weight_scale.stepsize,
                upper_bound_iter=self.weight_scale.upper_bound_iter,
                scale=float(self.weight_scale.scale),
            )
            if self.weight_scale.to_aux:
                net.WeightScale(
                    [param_squared_sum, iteration],
                    [param_squared_sum],
                    stepsize=self.weight_scale.stepsize,
                    upper_bound_iter=self.weight_scale.upper_bound_iter,
                    scale=float(self.weight_scale.scale),
                )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


class WngradOptimizer(Optimizer):
    def __init__(
        self,
        alpha=1.0,
        epsilon=1e-9,
        policy="fixed",
        sparse_dedup_aggregator=None,
        engine="",
        moment_init=100.0,
        lars=None,
        output_effective_lr=False,
        output_effective_lr_and_update=False,
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
        self.moment_init = moment_init
        self.lars = lars
        self.output_effective_lr = output_effective_lr
        self.output_effective_lr_and_update = output_effective_lr_and_update
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        if self.alpha <= 0:
            return

        self._clear_local_lr_multiplier()

        if self.lars is not None and not isinstance(grad, core.GradientSlice):
            assert self.lars >= 0, "Lars offset must be nonnegative, got {}".format(
                self.lars
            )
            wd, trust, lr_max = self.create_lars_inputs(
                param_init_net, 0.0, 1.0, np.finfo(np.float32).max
            )
            lr_lars_multiplier = net.Lars(
                [param, grad, wd, trust, lr_max],
                self.make_unique_blob_name(str(param) + "_lars"),
                offset=self.lars,
                lr_min=0.0,
            )
            current_scope = scope.CurrentDeviceScope()
            self._add_local_lr_multiplier(
                lr_lars_multiplier,
                is_gpu_blob=(
                    current_scope is not None
                    and core.IsGPUDeviceType(current_scope.device_type)
                ),
            )

        lr, _ = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=self.alpha,
            policy=self.policy,
            **(self.init_kwargs)
        )

        moment = param_init_net.ConstantFill(
            [], str(param) + "_moment", shape=[1], value=self.moment_init
        )

        self._aux_params.local.append(moment)

        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseWngrad(
                [param, moment, grad.indices, grad.values, lr],
                [param, moment],
                epsilon=self.epsilon,
                engine=self.engine,
            )
        else:
            output_args = [param, moment]
            if self.output_effective_lr_and_update:
                output_args.append(str(param) + "_effective_lr")
                output_args.append(str(param) + "_update")
            elif self.output_effective_lr:
                output_args.append(str(param) + "_effective_lr")

            net.Wngrad(
                [param, moment, grad, lr],
                output_args,
                epsilon=self.epsilon,
                engine=self.engine,
            )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


class StormOptimizer(Optimizer):
    def __init__(
        self,
        lr=0.1,
        momentum=10.0,
        beta=0.1,
        grad_sq_init=0.01,
        policy="fixed",
        sparse_dedup_aggregator=None,
        lars=None,
        **kwargs
    ):
        """Constructor function to add STORM Optimizer

        Args:
            lr: learning rate scaling (called k in the original paper)
            momentum: momentum scaling (called c in the original paper)
            beta: initial value of denominator in adaptive learning rate (
              called c in the original paper)
            grad_sq_init: initial value of gradient squared accumulator.
            policy: specifies how learning rate should be applied, options are
              'fixed', 'step', 'exp', etc.
            sparse_dedup_aggregator: specifies deduplication strategy for
              gradient slices. Works while using sparse gradients. Options
              include 'mean' and 'sum'.
            lars: lars offset.
        """
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.grad_sq_init = grad_sq_init
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.lars = lars
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        if self.lr <= 0:
            return

        self._clear_local_lr_multiplier()

        if self.lars is not None and not isinstance(grad, core.GradientSlice):
            assert self.lars >= 0, "Lars offset must be nonnegative, got {}".format(
                self.lars
            )
            wd, trust, lr_max = self.create_lars_inputs(
                param_init_net, 0.0, 1.0, np.finfo(np.float32).max
            )
            lr_lars_multiplier = net.Lars(
                [param, grad, wd, trust, lr_max],
                self.make_unique_blob_name(str(param) + "_lars"),
                offset=self.lars,
                lr_min=0.0,
            )
            current_scope = scope.CurrentDeviceScope()
            self._add_local_lr_multiplier(
                lr_lars_multiplier,
                is_gpu_blob=(
                    current_scope is not None
                    and core.IsGPUDeviceType(current_scope.device_type)
                ),
            )

        lr, _ = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=self.lr,
            policy=self.policy,
            **(self.init_kwargs)
        )

        moment = param_init_net.ConstantFill(param, str(param) + "_moment", value=0.0)
        self._aux_params.local.append(moment)

        grad_sq_sum = param_init_net.ConstantFill(
            [], str(param) + "_grad_sq_sum", shape=[1], value=self.grad_sq_init
        )
        self._aux_params.local.append(grad_sq_sum)

        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseStorm(
                [param, moment, grad_sq_sum, grad.values, grad.indices, lr],
                [param, moment, grad_sq_sum],
                momentum=self.momentum,
                beta=self.beta,
            )
        else:
            net.Storm(
                [param, moment, grad_sq_sum, grad, lr],
                [param, moment, grad_sq_sum],
                momentum=self.momentum,
                beta=self.beta,
            )

    def scale_learning_rate(self, scale):
        self.lr *= scale


class AdadeltaOptimizer(Optimizer):
    def __init__(
        self,
        alpha=0.01,
        epsilon=1e-4,
        decay=0.95,
        policy="fixed",
        sparse_dedup_aggregator=None,
        engine="",
        **kwargs
    ):
        """Constructor function to add Adadelta Optimizer

        Args:
            alpha: learning rate
            epsilon: attribute of Adadelta to avoid numerical issues
            decay: attribute of Adadelta to decay the squared gradient sum
            policy: specifies how learning rate should be applied, options are
              "fixed", "step", "exp", etc.
            sparse_dedup_aggregator: specifies deduplication strategy for
              gradient slices. Works while using sparse gradients. Options
              include "mean" and "sum".
            engine: the engine used, options include "", "CUDNN", etc.
        """
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay = decay
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
            net,
            param_init_net,
            base_learning_rate=self.alpha,
            policy=self.policy,
            **(self.init_kwargs)
        )

        moment = param_init_net.ConstantFill(
            [param], str(param) + "_squared_moment", value=0.0
        )

        moment_update = param_init_net.ConstantFill(
            [param], str(param) + "_squared_moment_update", value=0.0
        )

        self._aux_params.local.append(moment)
        self._aux_params.local.append(moment_update)

        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseAdadelta(
                [param, moment, moment_update, grad.indices, grad.values, lr],
                [param, moment, moment_update],
                epsilon=self.epsilon,
                decay=self.decay,
                engine=self.engine,
            )
        else:
            net.Adadelta(
                [param, moment, moment_update, grad, lr],
                [param, moment, moment_update],
                epsilon=self.epsilon,
                decay=self.decay,
                engine=self.engine,
            )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


class FtrlOptimizer(Optimizer):
    def __init__(
        self,
        alpha=0.01,
        beta=1e-4,
        lambda1=0,
        lambda2=0,
        sparse_dedup_aggregator=None,
        engine="",
    ):
        super().__init__()
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
            [param], str(param) + "_ftrl_nz", extra_shape=[2], value=0.0
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
                lambda2=self.lambda2,
            )
        else:
            net.Ftrl(
                [param, nz, grad],
                [param, nz],
                engine=self.engine,
                alpha=self.alpha,
                beta=self.beta,
                lambda1=self.lambda1,
                lambda2=self.lambda2,
            )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


class GFtrlOptimizer(Optimizer):
    """Group Lasso FTRL Optimizer."""

    def __init__(
        self,
        alpha=0.01,
        beta=1e-4,
        lambda1=0,
        lambda2=0,
        sparse_dedup_aggregator=None,
        engine="",
    ):
        super().__init__()
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
            [param], str(param) + "_gftrl_nz", extra_shape=[2], value=0.0
        )
        self._aux_params.local.append(nz)
        net.GFtrl(
            [param, nz, grad],
            [param, nz],
            engine=self.engine,
            alpha=self.alpha,
            beta=self.beta,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
        )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        policy="fixed",
        use_lr_adaption=False,
        lr_alpha=0.01,
        normalized_lr_adaption=True,
        sparse_dedup_aggregator=None,
        rowWise=False,
        engine="",
        enableRAdam=False,
        use_smart_decay=False,  # See https://fburl.com/2jdiwrhy for context.
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.policy = policy
        self.use_lr_adaption = use_lr_adaption
        self.lr_alpha = lr_alpha
        self.normalized_lr_adaption = normalized_lr_adaption
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.rowWise = rowWise
        self.engine = engine
        self.enableRAdam = enableRAdam
        if use_smart_decay:
            if rowWise:
                raise NotImplementedError(('Smart decay is not implemented for rowWise Adam.  '
                                           'Set rowWise or use_smart_decay to False.'))
            if enableRAdam:
                raise NotImplementedError(('Smart decay is not implemented for RAdam.  '
                                           'Set enableRAdam or use_smart_decay to False.'))
            if use_lr_adaption:
                raise NotImplementedError(('Smart decay is not implemented with lr_adaption.  '
                                           'Set use_lr_adaption or use_smart_decay to False.'))

        self.use_smart_decay = use_smart_decay
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        if self.alpha <= 0:
            return

        lr, iteration = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=self.alpha,
            policy=self.policy,
            **(self.init_kwargs)
        )

        m1 = param_init_net.ConstantFill([param], param + "_first_moment", value=0.0)

        if self.rowWise:
            shapes, types = workspace.InferShapesAndTypes([param_init_net])
            m2 = param_init_net.ConstantFill(
                [], param + "_avg_second_moment", shape=[shapes[param][0]], value=0.0
            )
        else:
            m2 = param_init_net.ConstantFill(
                [param], param + "_second_moment", value=0.0
            )

        # Initialize "minibatch in which this parameter was last seen" for smart decay.
        if self.use_smart_decay:
            shapes, _ = workspace.InferShapesAndTypes([param_init_net])
            last_seen = param_init_net.ConstantFill(
                [], param + "_last_seen", shape=[shapes[param][0]], value=0, dtype=core.DataType.INT64
            )
            self._aux_params.local.append(last_seen)

        self._aux_params.shared.append(iteration)
        self._aux_params.local.append(m1)
        self._aux_params.local.append(m2)

        if self.rowWise:
            assert isinstance(grad, core.GradientSlice), (
                "If SparseAdam with rowWise=True, gradient must be "
                "a gradientslice. PLease ensure that rowWise is not enabled "
                "for the dense Adam optimizer, as it is not supported."
            )

        output_blobs = [param, m1, m2]

        if self.use_smart_decay:
            output_blobs.append(last_seen)

        if self.use_lr_adaption:
            effective_grad = str(param) + "_effective_grad"
            output_blobs.append(effective_grad)

        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            if self.rowWise:
                op = "RowWiseSparseAdam"
            elif self.use_smart_decay:
                op = "SmartDecaySparseAdam"
            else:
                op = "SparseAdam"

            # Currently, only SparseAdam support RAdam, other Adam Ops will support later
            if op == "SparseAdam":
                net.__getattr__(op)(
                    [param, m1, m2, grad.indices, grad.values, lr, iteration],
                    output_blobs,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                    enableRAdam=self.enableRAdam,
                )
            elif op == "SmartDecaySparseAdam":
                net.__getattr__(op)(
                    [param, m1, m2, last_seen, grad.indices, grad.values, lr, iteration],
                    output_blobs,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                )
            else:
                assert (
                    not self.enableRAdam
                ), "Currently, RowWiseSparseAdam is not supported by RAdam!"
                net.__getattr__(op)(
                    [param, m1, m2, grad.indices, grad.values, lr, iteration],
                    output_blobs,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon,
                )

            if self.use_lr_adaption:
                net.LearningRateAdaption(
                    [lr, grad.values, effective_grad],
                    [lr],
                    lr_alpha=self.lr_alpha,
                    normalized_lr_adaption=self.normalized_lr_adaption,
                )

        else:
            net.Adam(
                [param, m1, m2, grad, lr, iteration],
                output_blobs,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon,
            )
            if self.use_lr_adaption:
                net.LearningRateAdaption(
                    [lr, grad, effective_grad],
                    [lr],
                    lr_alpha=self.lr_alpha,
                    normalized_lr_adaption=self.normalized_lr_adaption,
                )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return

class DecayAdagradOptimizer(Optimizer):
    def __init__(
        self,
        alpha=0.01,
        beta1=0.0,
        beta2=0.999,
        epsilon=0.1,
        weight_decay=0.0,
        ema_options=None,
        bias_correction_first=True,
        policy="fixed",
        engine="",
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.bias_correction_first = bias_correction_first
        self.policy = policy
        self.engine = engine
        self.init_kwargs = kwargs
        self._process_ema_options(ema_options)

    def set_mapping_for_param2ema_teacher_param(self, param_mapping: Dict[str, Any]) -> None:
        self.param2ema_teacher_param = param_mapping

    def _process_ema_options(self, ema_options):
        self.ema_enabled = True if ema_options and "ema_alpha" in ema_options else False
        self.ema_teacher_enabled = True if ema_options and "ema_teacher_alpha" in ema_options else False
        self.param2ema_teacher_param = {}
        if self.ema_enabled or self.ema_teacher_enabled:
            self.ema_start = ema_options.get("ema_start", None)
            self.ema_end = ema_options.get("ema_end", None)
            self.ema_step = ema_options.get("ema_step", None)
            self.ema_alpha = ema_options.get("ema_alpha", None)
            self.ema_teacher_alpha = ema_options.get("ema_alpha", None)
            self.ema_teacher_module_name = ema_options.get(
                "ema_teacher_module_name", "ema_teacher_arch"
            )

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        if self.alpha <= 0:
            return

        lr, iteration = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=self.alpha,
            policy=self.policy,
            **(self.init_kwargs)
        )

        if isinstance(grad, core.GradientSlice):
            # hack for position weighted.
            param_squared_sum = param_init_net.ConstantFill([param], param + "_squared_sum", value=0.0)
            self._aux_params.local.append(param_squared_sum)
            output_blobs = [param, param_squared_sum]
            net.SparseAdagrad(
                [param, param_squared_sum, grad.indices, grad.values, lr],
                output_blobs,
                epsilon=self.epsilon,
            )
        else:
            m1 = param_init_net.ConstantFill([param], param + "_first_mo1ment", value=0.0)
            m2 = param_init_net.ConstantFill([param], param + "_second_moment", value=0.0)
            self._aux_params.shared.append(iteration)
            self._aux_params.local.append(m1)
            self._aux_params.local.append(m2)
            output_blobs = [param, m1, m2]
            net.DecayAdagrad(
                [param, m1, m2, grad, lr, iteration],
                output_blobs,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon,
                weight_decay=self.weight_decay,
                bias_correction_first=self.bias_correction_first,
            )

            if self.ema_enabled:
                param_ema = str(param) + "_ema"
                if not param_init_net.BlobIsDefined(param_ema):
                    param_init_net.ConstantFill([param], param_ema, value=0.0)
                    self._aux_params.local.append(param_ema)

                net.EMA(
                    [param, param_ema, iteration],
                    [param, param_ema],
                    ema_start=self.ema_start,
                    ema_end=self.ema_end,
                    ema_step=self.ema_step,
                    ema_alpha=self.ema_alpha,
                )

            if self.ema_teacher_enabled:
                if param in self.param2ema_teacher_param:
                    param_ema_teacher = self.param2ema_teacher_param[param]
                    if not param_init_net.BlobIsDefined(param_ema_teacher):
                        param_init_net.ConstantFill([param], param_ema_teacher, value=0.0)
                        self._aux_params.local.append(param_ema_teacher)

                    net.EMA(
                        [param, param_ema_teacher, iteration],
                        [param, param_ema_teacher],
                        ema_start=self.ema_start,
                        ema_end=self.ema_end,
                        ema_step=self.ema_step,
                        ema_alpha=self.ema_teacher_alpha,
                    )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return

class YellowFinOptimizer(Optimizer):
    """YellowFin: An automatic tuner for momentum SGD

    See https://arxiv.org/abs/1706.03471 for more details. This implementation
    has separate learning rate and momentum per each parameter."""

    def __init__(
        self,
        alpha=0.1,
        mu=0.0,
        beta=0.999,
        curv_win_width=20,
        zero_debias=True,
        epsilon=0.1 ** 6,
        policy="fixed",
        sparse_dedup_aggregator=None,
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.mu = mu
        self.beta = beta
        self.curv_win_width = curv_win_width
        self.zero_debias = zero_debias
        self.epsilon = epsilon
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):

        # Note: This is number of persistent scalars in YellowFin optimizer.
        #       It should always be the number of scalars being used. The same
        #       number should be used in class for the operation.
        SCALARS_MEMORY_SIZE = 5

        param = param_info.blob
        grad = param_info.grad
        moment = param_init_net.ConstantFill([param], param + "_moment", value=0.0)
        curv_win = param_init_net.ConstantFill(
            [], param + "_curv_win", shape=[self.curv_win_width], value=0.0
        )
        g_avg = param_init_net.ConstantFill([param], param + "_g_avg", value=0.0)
        g2_avg = param_init_net.ConstantFill([param], param + "_g2_avg", value=0.0)
        lr_avg = param_init_net.ConstantFill(
            [], param + "_lr_avg", shape=[1], value=self.alpha
        )
        mu_avg = param_init_net.ConstantFill(
            [], param + "_mu_avg", shape=[1], value=self.mu
        )
        scalars_memory = param_init_net.ConstantFill(
            [], param + "_scalars_memory", shape=[SCALARS_MEMORY_SIZE], value=0.0
        )

        assert self.alpha > 0
        assert not isinstance(
            grad, core.GradientSlice
        ), "YellowFin does not support sparse gradients"

        iteration = utils.BuildUniqueMutexIter(param_init_net, net, iter_val=0)

        self._aux_params.shared.append(iteration)
        self._aux_params.local.append(moment)
        self._aux_params.local.append(lr_avg)
        self._aux_params.local.append(mu_avg)
        self._aux_params.local.append(curv_win)
        self._aux_params.local.append(g_avg)
        self._aux_params.local.append(g2_avg)
        self._aux_params.local.append(scalars_memory)

        yf_in_out_args = [
            param,
            moment,
            lr_avg,
            mu_avg,
            curv_win,
            g_avg,
            g2_avg,
            scalars_memory,
        ]

        net.YellowFin(
            yf_in_out_args + [grad, iteration],
            yf_in_out_args,
            beta=self.beta,
            epsilon=self.epsilon,
            curv_win_width=self.curv_win_width,
            zero_debias=self.zero_debias,
        )

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


class RmsPropOptimizer(Optimizer):
    def __init__(
        self,
        alpha=0.01,
        decay=0.9,
        momentum=0.0,
        epsilon=1e-5,
        policy="fixed",
        engine="",
        **kwargs
    ):
        super().__init__()
        self.alpha = alpha
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.policy = policy
        self.engine = engine
        self.init_kwargs = kwargs

    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad

        assert self.alpha > 0
        assert not isinstance(
            grad, core.GradientSlice
        ), "RmsPropOptimizer doesn't support sparse gradients"

        dev = scope.CurrentDeviceScope()
        if dev is None:
            dev = core.DeviceOption(caffe2_pb2.CPU)

        ONE = param_init_net.ConstantFill(
            [], "ONE_{}_{}".format(dev.device_type, dev.device_id), shape=[1], value=1.0
        )

        lr, _ = self.build_lr(
            net,
            param_init_net,
            base_learning_rate=-self.alpha,
            policy=self.policy,
            **(self.init_kwargs)
        )

        grad_o = param_init_net.ConstantFill(
            [param], str(param) + "_grad_o", values=0.0
        )

        ms = param_init_net.ConstantFill(
            [param], str(param) + "_mean_squares", values=0.0
        )

        mom = param_init_net.ConstantFill([param], str(param) + "_momentum", values=0.0)

        self._aux_params.local.append(ms)
        self._aux_params.local.append(mom)

        net.RmsProp(
            [grad, ms, mom, ONE],
            [grad_o, ms, mom],
            decay=self.decay,
            momentum=self.momentum,
            epsilon=self.epsilon,
            engine=self.engine,
        )

        net.MomentumSGDUpdate([grad_o, mom, lr, param], [grad_o, mom, param])

    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


def _get_param_to_device(model):
    # Infer blob devices by going through the net and param_init_net
    # ops and observing the device used to create or use the blob.
    param_to_device = core.InferBlobDevices(model.net)
    param_to_device.update(core.InferBlobDevices(model.param_init_net))
    return param_to_device


def get_param_device(param_name, grad, param_to_device=None, default_device=None):
    device = default_device
    param_to_device = param_to_device or {}
    # We first check if parameter's device has been inferred. If not,
    # we check the gradient. This can happen if parameter is not output
    # by any blob but created by a FetchBlob.
    if param_name in param_to_device:
        device = param_to_device[param_name]
    else:
        if isinstance(grad, core.GradientSlice):
            grad = grad
            if str(grad.values) in param_to_device:
                device = param_to_device[str(grad.values)]
            elif str(grad.indices) in param_to_device:
                device = param_to_device[str(grad.indices)]
        else:
            grad_name = str(grad)
            if grad_name in param_to_device:
                device = param_to_device[grad_name]

    assert device is not None, "Cannot infer device for {}: no op creates it".format(
        param_name
    )
    return device


def get_lr_injection():
    """
    Gets current value for lr_injection, a multiplier for all base
    learning rates.
    Must set allow_lr_injection=True when building optimizer, as it
    relies on synchronization over CPU.
    """
    return workspace.FetchBlob(_LEARNING_RATE_INJECTION)


def set_lr_injection(lr_injection_value):
    """
    Sets lr_injection, a multiplier for all base learning rates.
    Must set allow_lr_injection=True when building optimizer, as it
    relies on synchronization over CPU.
    """
    workspace.FeedBlob(
        _LEARNING_RATE_INJECTION,
        np.array([float(lr_injection_value)], dtype=np.float32),
    )


def _calc_norm_ratio(model, params, name_scope, param_to_device, max_gradient_norm):
    with core.NameScope(name_scope):
        grad_squared_sums = []
        for i, param in enumerate(params):
            device = get_param_device(str(param.blob), param.grad, param_to_device)

            with core.DeviceScope(device):
                grad = (
                    param.grad
                    if not isinstance(param.grad, core.GradientSlice)
                    else param.grad.values
                )

                grad_squared_sum_name = "grad_{}_squared_sum".format(i)
                grad_squared_sum = model.net.SumSqrElements(grad, grad_squared_sum_name)
                grad_squared_sum_cpu = model.net.EnsureCPUOutput(grad_squared_sum)
                grad_squared_sums.append(grad_squared_sum_cpu)

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            grad_squared_full_sum = model.net.Sum(
                grad_squared_sums, "grad_squared_full_sum"
            )
            global_norm = model.net.Pow(
                grad_squared_full_sum, "global_norm", exponent=0.5
            )
            clip_norm = model.param_init_net.ConstantFill(
                [], "clip_norm", shape=[], value=float(max_gradient_norm)
            )
            max_norm = model.net.Max([global_norm, clip_norm], "max_norm")
            norm_ratio = model.net.Div([clip_norm, max_norm], "norm_ratio")
            return norm_ratio


def _build(
    model,
    optimizer,
    weights_only=False,
    use_param_info_optim=True,
    max_gradient_norm=None,
    allow_lr_injection=False,
):
    param_to_device = _get_param_to_device(model)

    # Validate there are no duplicate params
    model.Validate()

    params = []
    for param_info in model.GetOptimizationParamInfo():
        if weights_only and param_info.blob not in model.weights:
            continue
        params.append(param_info)

    lr_multiplier = None
    if max_gradient_norm is not None:
        lr_multiplier = _calc_norm_ratio(
            model,
            params,
            "norm_clipped_grad_update",
            param_to_device,
            max_gradient_norm,
        )

    if allow_lr_injection:
        if not model.net.BlobIsDefined(_LEARNING_RATE_INJECTION):
            lr_injection = model.param_init_net.ConstantFill(
                [], _LEARNING_RATE_INJECTION, shape=[1], value=1.0
            )
        else:
            lr_injection = _LEARNING_RATE_INJECTION

        if lr_multiplier is None:
            lr_multiplier = lr_injection
        else:
            lr_multiplier = model.net.Mul(
                [lr_multiplier, lr_injection], "lr_multiplier", broadcast=1
            )
    optimizer.add_lr_multiplier(lr_multiplier)

    for param_info in params:
        param_name = str(param_info.blob)
        device = get_param_device(param_name, param_info.grad, param_to_device)
        with core.DeviceScope(device):
            if param_info.optimizer and use_param_info_optim:
                param_info.optimizer(model.net, model.param_init_net, param_info)
            else:
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
        use_param_info_optim=False,
    )


def build_sgd(
    model,
    base_learning_rate,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    sgd_optimizer = SgdOptimizer(base_learning_rate, **kwargs)
    return _build(
        model,
        sgd_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )


def build_multi_precision_sgd(
    model,
    base_learning_rate,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    multi_prec_sgd_optimizer = MultiPrecisionSgdOptimizer(base_learning_rate, **kwargs)
    return _build(
        model,
        multi_prec_sgd_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )


def build_fp16_sgd(model, base_learning_rate, **kwargs):
    fp16_sgd_optimizer = FP16SgdOptimizer(base_learning_rate, **kwargs)
    return _build(model, fp16_sgd_optimizer)


def build_ftrl(model, engine="SIMD", **kwargs):
    if engine == "SIMD":
        assert core.IsOperator("Ftrl_ENGINE_SIMD")
        assert core.IsOperator("SparseFtrl_ENGINE_SIMD")
    ftrl_optimizer = FtrlOptimizer(engine=engine, **kwargs)
    return _build(model, ftrl_optimizer)


def build_gftrl(model, engine="", **kwargs):
    if engine == "SIMD":
        assert core.IsOperator("GFtrl_ENGINE_SIMD")
    gftrl_optimizer = GFtrlOptimizer(engine=engine, **kwargs)
    return _build(model, gftrl_optimizer)


def build_adagrad(
    model,
    base_learning_rate,
    parameters=None,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    adagrad_optimizer = AdagradOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(
        model,
        adagrad_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )


def build_wngrad(
    model,
    base_learning_rate,
    parameters=None,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    wngrad_optimizer = WngradOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(
        model,
        wngrad_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )


def build_storm(
    model,
    base_learning_rate,
    parameters=None,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    storm_optimizer = StormOptimizer(lr=base_learning_rate, **kwargs)
    return _build(
        model,
        storm_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )


def build_adadelta(
    model,
    base_learning_rate,
    parameters=None,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    adadelta_optimizer = AdadeltaOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(
        model,
        adadelta_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )


def build_adam(
    model,
    base_learning_rate,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    adam_optimizer = AdamOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(
        model,
        adam_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )

def build_decay_adagrad(
    model,
    base_learning_rate,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    decay_adagrad_optimizer = DecayAdagradOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(
        model,
        decay_adagrad_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )

def build_yellowfin(model, base_learning_rate=0.1, **kwargs):
    yellowfin_optimizer = YellowFinOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, yellowfin_optimizer)


def build_rms_prop(
    model,
    base_learning_rate,
    max_gradient_norm=None,
    allow_lr_injection=False,
    **kwargs
):
    rms_prop_optimizer = RmsPropOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(
        model,
        rms_prop_optimizer,
        max_gradient_norm=max_gradient_norm,
        allow_lr_injection=allow_lr_injection,
    )
