from collections import namedtuple
from typing import Optional, Any, Union

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
    fused_wt_fake_quant_range_neg_127_to_127,
    fused_per_channel_wt_fake_quant_range_neg_127_to_127,
)

from .observer import (
    _PartialWrapper,
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
    weight_observer_range_neg_127_to_127,
    per_channel_weight_observer_range_neg_127_to_127,
    default_reuse_input_observer,
    ObserverBase,
)
import warnings
import copy

__all__ = [
    "QConfig",
    # TODO: deprecated, remove
    "QConfigDynamic",
    "default_qconfig",
    "default_debug_qconfig",
    "default_per_channel_qconfig",
    "default_dynamic_qconfig",
    "float16_dynamic_qconfig",
    "float16_static_qconfig",
    "per_channel_dynamic_qconfig",
    "float_qparams_weight_only_qconfig",
    "float_qparams_weight_only_qconfig_4bit",
    "default_qat_qconfig",
    "default_dynamic_qat_qconfig",
    "default_weight_only_qconfig",
    "default_activation_only_qconfig",
    "default_qat_qconfig_v2",
    "default_reuse_input_qconfig",
    "default_symmetric_qnnpack_qconfig",
    "default_per_channel_symmetric_qnnpack_qconfig",
    "default_symmetric_qnnpack_qat_qconfig",
    "default_per_channel_symmetric_qnnpack_qat_qconfig",
    "default_embedding_qat_qconfig",
    "default_embedding_qat_qconfig_4bit",
    "get_default_qconfig",
    "get_default_qat_qconfig",
    "get_default_qconfig_dict",
    "get_default_qat_qconfig_dict",
    "QConfigAny",
    "qconfig_equals",
]

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

float16_dynamic_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float16, compute_dtype=torch.float16),
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

def get_default_qconfig(backend='x86', version=0):
    """
    Returns the default PTQ qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    if backend not in supported_backends:
        raise AssertionError(
            "backend: " + str(backend) +
            " not supported. backend must be one of {}".format(supported_backends)
        )

    if version == 0:
        if backend == 'fbgemm':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                              weight=default_per_channel_weight_observer)
        elif backend == 'qnnpack':
            # TODO: make this compatible with xnnpack constraints
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                              weight=default_weight_observer)
        elif backend == 'onednn':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                              weight=default_per_channel_weight_observer)
        elif backend == 'x86':
            qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                              weight=default_per_channel_weight_observer)
        else:
            # won't reach
            qconfig = default_qconfig
    else:
        raise AssertionError("Version number: " + str(version) +
                             " in get_default_qconfig is not supported. Version number must be 0")

    return qconfig

"""
Default, symmetric PTQ qconfig for the specified backend. And a per_channel
variant of the same.

Symmetric here applies to signed weights with zero point = 0, and additional
value restrictions. The activations are also signed 8-bit integers with this
qconfig.

    * Once this change is merged [as of 3/17/22], with backend or qengine =
    'qnnpack', some quantized operators with this symmetric qconfig may use
    operators from xnnpack library.

        ** Support to use xnnpack ops with `qnnpack` backed for asymmetric
        qconfig (returned by get_default_qconfig()) is not available yet.

    * This qconfig uses signed activations and weights. Weights have added
    restrictions such as zero point is forced to be 0, making the weights
    symmetric, hence the name. And the 8-bit quantized values are
    restricting to to [-127, +127], excluding -128.

    * xnnpack has a requantization scale value restriction, 0x1p-32 <=
    requantization_scale < 256.0 where, `requantization_scale = (input_scale
    * kernel_scale) / (output_scale)`. Using this eps (w/ assumed max value
    of 256) is to prevent requantization_scale to go below xnnpack lower
    threshold.
"""
default_symmetric_qnnpack_qconfig = QConfig(activation=HistogramObserver.with_args(dtype=torch.qint8,
                                                                                   reduce_range=False,
                                                                                   eps=2 ** -12),
                                            weight=weight_observer_range_neg_127_to_127)

default_per_channel_symmetric_qnnpack_qconfig = QConfig(activation=HistogramObserver.with_args(dtype=torch.qint8,
                                                                                               reduce_range=False,
                                                                                               eps=2 ** -12),
                                                        weight=per_channel_weight_observer_range_neg_127_to_127)

default_embedding_qat_qconfig = QConfig(activation=NoopObserver.with_args(dtype=torch.float32),
                                        weight=default_embedding_fake_quant)

default_embedding_qat_qconfig_4bit = QConfig(activation=NoopObserver.with_args(dtype=torch.float32),
                                             weight=default_embedding_fake_quant_4bit)

def get_default_qat_qconfig(backend='x86', version=1):
    """
    Returns the default QAT qconfig for the specified backend.

    Args:
      * `backend` (str): a string representing the target backend. Currently supports
        `x86` (default), `fbgemm`, `qnnpack` and `onednn`.
      * `version`: version, for backwards compatibility. Can be `None` or `1`.

    Return:
        qconfig
    """
    supported_backends = ["fbgemm", "x86", "qnnpack", "onednn"]
    if backend not in supported_backends:
        raise AssertionError(
            "backend: " + str(backend) +
            " not supported. backend must be one of {}".format(supported_backends)
        )

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
        elif backend == 'onednn':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255),
                              weight=default_per_channel_weight_fake_quant)
        if backend == 'x86':
            qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                quant_min=0,
                                                                quant_max=255,
                                                                reduce_range=True),
                              weight=default_per_channel_weight_fake_quant)
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
            # TODO: make this compatible with xnnpack constraints
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=False),
                              weight=default_fused_wt_fake_quant)
        elif backend == 'onednn':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255),
                              weight=default_fused_per_channel_wt_fake_quant)
        elif backend == 'x86':
            qconfig = QConfig(activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                                                 quant_min=0,
                                                                                 quant_max=255,
                                                                                 reduce_range=True),
                              weight=default_fused_per_channel_wt_fake_quant)
        else:
            qconfig = default_qat_qconfig_v2
    else:
        raise AssertionError("Version number: " + str(version) +
                             "in get_default_qat_qconfig is not supported. Version number must be 0 or 1")

    return qconfig

"""
Default symmetric QAT qconfig for qnnpack. And its per channel weight variant.
"""
default_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                       quant_min=-128,
                                                       quant_max=127,
                                                       dtype=torch.qint8,
                                                       reduce_range=False,
                                                       eps=2 ** -12),
    weight=fused_wt_fake_quant_range_neg_127_to_127)

default_per_channel_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                       quant_min=-128,
                                                       quant_max=127,
                                                       dtype=torch.qint8,
                                                       reduce_range=False,
                                                       eps=2 ** -12),
    weight=fused_per_channel_wt_fake_quant_range_neg_127_to_127)

def get_default_qconfig_dict(backend='x86', version=0):
    warnings.warn(
        "torch.ao.quantization.get_default_qconfig_dict is deprecated and will be removed in "
        "a future version. Please use torch.ao.quantization.get_default_qconfig_mapping instead.")
    return torch.ao.quantization.get_default_qconfig_mapping(backend, version).to_dict()

def get_default_qat_qconfig_dict(backend='x86', version=1):
    warnings.warn(
        "torch.ao.quantization.get_default_qat_qconfig_dict is deprecated and will be removed in "
        "a future version. Please use torch.ao.quantization.get_default_qat_qconfig_mapping instead.")
    return torch.ao.quantization.get_default_qat_qconfig_mapping(backend, version).to_dict()

def _assert_valid_qconfig(qconfig: Optional[QConfig],
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

QConfigAny = Optional[QConfig]
QConfigAny.__module__ = "torch.ao.quantization.qconfig"

def _add_module_to_qconfig_obs_ctr(
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

_ObserverOrFakeQuantizeConstructor = Union[_PartialWrapper, ObserverBase, FakeQuantizeBase]

def _obs_or_fq_ctr_equals(obs_or_fq1: _ObserverOrFakeQuantizeConstructor, obs_or_fq2: _ObserverOrFakeQuantizeConstructor):
    if isinstance(obs_or_fq1, _PartialWrapper) and isinstance(obs_or_fq2, _PartialWrapper):
        return _partial_wrapper_equals(obs_or_fq1, obs_or_fq2)
    return obs_or_fq1 == obs_or_fq2

def _partial_wrapper_equals(obs_or_fq1: _PartialWrapper, obs_or_fq2: _PartialWrapper):
    """
    Return whether the two partial wrappers are equal,
    """
    # functools.partial has no __eq__ operator defined so '==' defaults to 'is'
    obs_or_fq1_keywords = copy.copy(obs_or_fq1.p.keywords)
    obs_or_fq2_keywords = copy.copy(obs_or_fq2.p.keywords)
    keywords_equal = True
    # compare observer constructor with _obs_or_fq_ctr_equals since direct compare would fail
    if "observer" in obs_or_fq1_keywords and "observer" in obs_or_fq2_keywords:
        keywords_equal = keywords_equal and _obs_or_fq_ctr_equals(obs_or_fq1_keywords["observer"], obs_or_fq2_keywords["observer"])
        obs_or_fq1_keywords.pop("observer")
        obs_or_fq2_keywords.pop("observer")
    keywords_equal = keywords_equal and obs_or_fq1_keywords == obs_or_fq2_keywords
    return obs_or_fq1.p.func == obs_or_fq2.p.func and obs_or_fq1.p.args == obs_or_fq2.p.args and keywords_equal

def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
    """
    Returns `True` if `q1` equals `q2`, and `False` otherwise.
    """
    if q1 is None or q2 is None:
        return q1 == q2
    else:
        assert q1 is not None and q2 is not None
        try:
            # Qconfig weight and activation can be either a partial wrapper,
            # or an observer class. Special handling is required (above) for
            # comparing partial wrappers.
            activation_same = _obs_or_fq_ctr_equals(q1.activation, q2.activation)
            weight_same = _obs_or_fq_ctr_equals(q1.weight, q2.weight)
            return activation_same and weight_same
        except AttributeError:
            return q1 == q2

def _activation_is_memoryless(qconfig: QConfig):
    """
    Return whether the observer for activations defined in the given QConfig is memoryless.
    This means a MovingAverage observer with averaging constant equal to 1.
    """
    def _is_memoryless(observer):
        return hasattr(observer, "averaging_constant") and observer.averaging_constant == 1
    act = qconfig.activation()
    if isinstance(act, FakeQuantizeBase) and hasattr(act, "activation_post_process"):
        return _is_memoryless(act.activation_post_process)
    else:
        return _is_memoryless(act)

def _is_reuse_input_qconfig(qconfig: Optional[QConfig]):
    return qconfig is not None and \
        isinstance(qconfig.activation(), ReuseInputObserver) and \
        isinstance(qconfig.weight(), NoopObserver)
