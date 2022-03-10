from collections import namedtuple
from typing import Optional, Any

import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FakeQuantizeBase,
    default_fake_quant,
    default_dynamic_fake_quant,
    default_per_channel_weight_fake_quant,
    default_weight_fake_quant,
    default_fused_act_fake_quant,
    default_fused_wt_fake_quant,
    FusedMovingAvgObsFakeQuantize,
    default_fused_per_channel_wt_fake_quant,
    default_embedding_fake_quant,
    default_embedding_fake_quant_4bit,
)

from .observer import (
    HistogramObserver,
    MovingAverageMinMaxObserver,
    NoopObserver,
    PlaceholderObserver,
    ReuseInputObserver,
    default_debug_observer,
    default_dynamic_quant_observer,
    default_float_qparams_observer,
    default_float_qparams_observer_4bit,
    default_observer,
    default_per_channel_weight_observer,
    default_placeholder_observer,
    default_weight_observer,
    default_reuse_input_observer,
)
import warnings


class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfig(
          activation=MinMaxObserver.with_args(dtype=torch.qint8),
          weight=default_observer.with_args(dtype=torch.qint8))

    """
    def __new__(cls, activation, weight):
        # catch common mistakes
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("QConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        return super(QConfig, cls).__new__(cls, activation, weight)


class QConfigDynamic(namedtuple('QConfigDynamic', ['activation', 'weight'])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial)::

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """
    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity):
        # catch common mistakes
        if isinstance(weight, nn.Module):
            raise ValueError("QConfigDynamic received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        warnings.warn("QConfigDynamic is going to be deprecated in PyTorch 1.12, please use QConfig instead")
        return super(QConfigDynamic, cls).__new__(cls, activation, weight)


default_qconfig = QConfig(activation=default_observer,
                          weight=default_weight_observer)
"""
Default qconfig configuration.
"""

default_debug_qconfig = QConfig(weight=default_weight_observer,
                                activation=default_debug_observer)
"""
Default qconfig configuration for debugging.
"""

default_per_channel_qconfig = QConfig(activation=default_observer,
                                      weight=default_per_channel_weight_observer)
"""
Default qconfig configuration for per channel weight quantization.
"""

default_dynamic_qconfig = QConfig(activation=default_dynamic_quant_observer,
                                  weight=default_weight_observer)
"""
Default dynamic qconfig.
"""

float16_dynamic_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float32, compute_dtype=torch.float16),
                                  weight=PlaceholderObserver.with_args(dtype=torch.float16))
"""
Dynamic qconfig with weights quantized to `torch.float16`.
"""

float16_static_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float16),
                                 weight=PlaceholderObserver.with_args(dtype=torch.float16))
"""
Dynamic qconfig with both activations and weights quantized to `torch.float16`.
"""

per_channel_dynamic_qconfig = QConfig(activation=default_dynamic_quant_observer,
                                      weight=default_per_channel_weight_observer)
"""
Dynamic qconfig with weights quantized per channel.
"""

float_qparams_weight_only_qconfig = QConfig(
    activation=default_placeholder_observer,
    weight=default_float_qparams_observer)
"""
Dynamic qconfig with weights quantized with a floating point zero_point.
"""

float_qparams_weight_only_qconfig_4bit = QConfig(
    activation=default_placeholder_observer,
    weight=default_float_qparams_observer_4bit)

default_qat_qconfig = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant)
"""
Default qconfig for QAT.
"""

default_dynamic_qat_qconfig = QConfig(activation=default_dynamic_fake_quant,
                                      weight=default_weight_fake_quant)
"""
Default qconfig for dynamic QAT.
"""

default_weight_only_qconfig = QConfig(activation=torch.nn.Identity,
                                      weight=default_weight_fake_quant)
"""
Default qconfig for quantizing weights only.
"""

default_activation_only_qconfig = QConfig(activation=default_fake_quant,
                                          weight=torch.nn.Identity)
"""
Default qconfig for quantizing activations only.
"""

# QAT config that uses a fused observer + fake quant modules for optimized training performance.
# to modify the activation/weight observers, the default entries in fake_quantize.py can be modified.
default_qat_qconfig_v2 = QConfig(activation=default_fused_act_fake_quant, weight=default_fused_wt_fake_quant)
"""
Fused version of `default_qat_config`, has performance benefits.
"""

default_reuse_input_qconfig = QConfig(activation=default_reuse_input_observer,
                                      weight=NoopObserver)
"""
Default qconfig for operators that reuse the observers from input Tensor, e.g. reshape
"""

def get_default_qconfig(backend='fbgemm', version=0):
    """
    Returns the default PTQ qconfig for the specified backend.

    Args:
      * `backend`: a string representing the target backend. Currently supports `fbgemm`
        and `qnnpack`.

    Return:
        qconfig
    """
    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                              weight=default_per_channel_weight_observer)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                              weight=default_weight_observer)
        else:
            qconfig = default_qconfig
    else:
        raise AssertionError("Version number: " + str(version) +
                             " in get_default_qconfig is not supported. Version number must be 0")

    return qconfig

default_embedding_qat_qconfig = QConfig(activation=NoopObserver.with_args(dtype=torch.float32),
                                        weight=default_embedding_fake_quant)

default_embedding_qat_qconfig_4bit = QConfig(activation=NoopObserver.with_args(dtype=torch.float32),
                                             weight=default_embedding_fake_quant_4bit)

def get_default_qat_qconfig(backend='fbgemm', version=1):
    """
    Returns the default QAT qconfig for the specified backend.

    Args:
      * `backend`: a string representing the target backend. Currently supports `fbgemm`
        and `qnnpack`.
      * `version`: version, for backwards compatibility. Can be `None` or `1`.

    Return:
        qconfig
    """
    # Histogram observer is too slow for quantization aware training
    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255,
                                                                reduce_range=True),
                              weight=default_per_channel_weight_fake_quant)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255,
                                                                reduce_range=False),
                              weight=default_weight_fake_quant)
        else:
            qconfig = default_qat_qconfig
    # Use the fused observe + fake_quant modules for doing QAT.
    elif version == 1:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=True),
                              weight=default_fused_per_channel_wt_fake_quant)
        elif backend == 'qnnpack':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=False),
                              weight=default_fused_wt_fake_quant)
        else:
            qconfig = default_qat_qconfig_v2
    else:
        raise AssertionError("Version number: " + str(version) +
                             "in get_default_qat_qconfig is not supported. Version number must be 0 or 1")

    return qconfig

def _get_default_qconfig_dict_helper(qconfig, qconfig_transpose):
    return {
        "": qconfig,
        "object_type": [("reshape", default_reuse_input_qconfig),
                        (torch.nn.Conv1d, qconfig),
                        (torch.nn.Conv2d, qconfig),
                        (torch.nn.Conv3d, qconfig),
                        (torch.nn.ConvTranspose1d, qconfig_transpose),
                        (torch.nn.ConvTranspose2d, qconfig_transpose),
                        (torch.nn.ConvTranspose3d, qconfig_transpose),
                        (torch.nn.Linear, qconfig),
                        (torch.nn.functional.conv1d, qconfig),
                        (torch.nn.functional.conv2d, qconfig),
                        (torch.nn.functional.conv3d, qconfig),
                        (torch.nn.functional.conv_transpose1d, qconfig_transpose),
                        (torch.nn.functional.conv_transpose2d, qconfig_transpose),
                        (torch.nn.functional.conv_transpose3d, qconfig_transpose),
                        (torch.nn.functional.linear, qconfig)]}

def get_default_qconfig_dict(backend='fbgemm', version=0):
    qconfig = get_default_qconfig(backend, version)
    qconfig_transpose = qconfig
    # default_per_channel_weight_observer is not currently compatible with fbgemm backend
    # so we have to modify the weight observer to default_weight_observer or another
    # per tensor supported observer.
    # see https://github.com/pytorch/pytorch/issues/47535
    if backend == "fbgemm":
        qconfig_transpose = QConfig(activation=qconfig.activation, weight=default_weight_observer)
    return _get_default_qconfig_dict_helper(qconfig, qconfig_transpose)

def get_default_qat_qconfig_dict(backend='fbgemm', version=1):
    qconfig = get_default_qat_qconfig(backend, version)
    qconfig_transpose = qconfig
    # default_per_channel_weight_observer is not currently compatible with fbgemm backend
    # so we have to modify the weight observer to default_weight_observer or another
    # per tensor supported observer
    # see https://github.com/pytorch/pytorch/issues/47535
    if backend == "fbgemm":
        qconfig_transpose = QConfig(activation=qconfig.activation, weight=default_weight_fake_quant)
    return _get_default_qconfig_dict_helper(qconfig, qconfig_transpose)

def assert_valid_qconfig(qconfig: Optional[QConfig],
                         mod: torch.nn.Module) -> None:
    """
    Verifies that this `qconfig` is valid.
    """
    if qconfig is None:
        return
    is_conv_transpose_mod = (
        isinstance(mod, torch.nn.ConvTranspose1d) or
        isinstance(mod, torch.nn.ConvTranspose2d) or
        isinstance(mod, torch.nn.ConvTranspose3d))
    if is_conv_transpose_mod:
        if qconfig.weight is None:
            # for now, we assume that any qconfig for ConvTranspose without a weight is valid
            return
        example_observer = qconfig.weight()
        is_per_channel = (
            isinstance(example_observer, torch.ao.quantization.PerChannelMinMaxObserver) or
            isinstance(example_observer, torch.ao.quantization.MovingAveragePerChannelMinMaxObserver)
        )
        assert not is_per_channel, \
            'Per channel weight observer is not supported yet for ConvTranspose{n}d.'

# TODO: remove QConfigAny and replace it with Optional[QConfig]
QConfigAny = Optional[QConfig]

def add_module_to_qconfig_obs_ctr(
        qconfig: QConfigAny,
        module: Optional[nn.Module]) -> Any:
    r"""This is a helper function for use in quantization prepare that updates a qconfig so that
    the constructors stored in the qconfig will create observers on the same device that
    'module' is on. This is intended to be used when the qconfigs are propagated to each
    module in order to avoid potential device alignment issues.

    Args:
        qconfig: QConfig with obs constructors stored in activation and weight
        module: module which the qconfig is related to

    Return:
        qconfig: configured so that obs constructors set to construct on the same device as module
    """

    if module is None or qconfig is None or qconfig._fields != ('activation', 'weight'):
        return qconfig

    def get_factory_kwargs_based_on_module_device():
        assert isinstance(module, torch.nn.Module)
        devices = {p.device for p in module.parameters()} | \
            {p.device for p in module.buffers()}
        device = next(iter(devices)) if len(devices) > 0 else None
        return None if device is None else {'device': device}

    def configure_constructor_to_put_obs_on_module_device(original_constructor):
        try:
            # check if constructor can accept factory_kwargs
            check = original_constructor.with_args(factory_kwargs=None)
            check()
            return original_constructor.with_callable_args(factory_kwargs=get_factory_kwargs_based_on_module_device)
        except AttributeError:  # qconfig doesn't have activation or weight
            return original_constructor
        except TypeError:  # the class doesn't accept factory_kwargs argument
            return original_constructor

    activation = configure_constructor_to_put_obs_on_module_device(qconfig.activation)
    weight = configure_constructor_to_put_obs_on_module_device(qconfig.weight)

    return QConfig(activation, weight)

def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
    """
    Returns `True` if `q1` equals `q2`, and `False` otherwise.
    """
    # functools.partial has no __eq__ operator defined so '==' defaults to 'is'
    def partial_equals(p1, p2):
        same = p1.func == p2.func
        same = same and p1.args == p2.args
        return same and p1.keywords == p2.keywords

    if q1 is None or q2 is None:
        return q1 == q2
    else:
        assert q1 is not None and q2 is not None
        try:
            # Qconfig weight and activation can be either a partial wrapper,
            # or an observer class. Special handling is required (above) for
            # comparing partial wrappers.
            if(isinstance(q1.activation, torch.ao.quantization.observer._PartialWrapper)):
                activation_same = partial_equals(q1.activation.p, q2.activation.p)
            else:
                activation_same = q1.activation == q2.activation
            if(isinstance(q1.weight, torch.ao.quantization.observer._PartialWrapper)):
                weight_same = partial_equals(q1.weight.p, q2.weight.p)
            else:
                weight_same = q1.weight == q2.weight

            return activation_same and weight_same
        except AttributeError:
            return q1 == q2

def activation_is_memoryless(qconfig: QConfig):
    """
    Return whether the observer for activations defined in the given QConfig is memoryless.
    """
    def _is_memoryless(observer):
        return hasattr(observer, "memoryless") and observer.memoryless
    act = qconfig.activation()
    if isinstance(act, FakeQuantizeBase) and hasattr(act, "activation_post_process"):
        return _is_memoryless(act.activation_post_process)
    else:
        return _is_memoryless(act)

def is_reuse_input_qconfig(qconfig: Optional[QConfig]):
    return qconfig is not None and \
        isinstance(qconfig.activation(), ReuseInputObserver) and \
        isinstance(qconfig.weight(), NoopObserver)
