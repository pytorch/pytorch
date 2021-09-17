from collections import namedtuple
from typing import Union, Optional, Any

import torch
import torch.nn as nn
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    default_fake_quant,
    default_per_channel_weight_fake_quant,
    default_weight_fake_quant,
    default_fused_act_fake_quant,
    default_fused_wt_fake_quant,
    FusedMovingAvgObsFakeQuantize,
    default_fused_per_channel_wt_fake_quant,
)

from .observer import (
    HistogramObserver,
    MovingAverageMinMaxObserver,
    PlaceholderObserver,
    default_debug_observer,
    default_dynamic_quant_observer,
    default_float_qparams_observer,
    default_observer,
    default_per_channel_weight_observer,
    default_placeholder_observer,
    default_weight_observer,
)

class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.


    Note that QConfig needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization preparation function will instantiate observers multiple times for each of the layers.


    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial):

      my_qconfig = QConfig(activation=MinMaxObserver.with_args(dtype=torch.qint8),
      weight=default_observer.with_args(dtype=torch.qint8))
    """
    def __new__(cls, activation, weight):
        # catch common mistakes
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("QConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        return super(QConfig, cls).__new__(cls, activation, weight)


default_qconfig = QConfig(activation=default_observer,
                          weight=default_weight_observer)

default_debug_qconfig = QConfig(weight=default_weight_observer,
                                activation=default_debug_observer)

default_per_channel_qconfig = QConfig(activation=default_observer,
                                      weight=default_per_channel_weight_observer)

class QConfigDynamic(namedtuple('QConfigDynamic', ['activation', 'weight'])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classes) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfigDynamic needs to contain observer **classes** (like MinMaxObserver) or a callable that returns
    instances on invocation, not the concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial):

      my_qconfig = QConfigDynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """
    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity):
        # catch common mistakes
        if isinstance(weight, nn.Module):
            raise ValueError("QConfigDynamic received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        return super(QConfigDynamic, cls).__new__(cls, activation, weight)

default_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
                                         weight=default_weight_observer)
float16_dynamic_qconfig = QConfigDynamic(activation=PlaceholderObserver.with_args(dtype=torch.float32),
                                         weight=PlaceholderObserver.with_args(dtype=torch.float16))
float16_static_qconfig = QConfigDynamic(activation=PlaceholderObserver.with_args(dtype=torch.float16),
                                        weight=PlaceholderObserver.with_args(dtype=torch.float16))
per_channel_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
                                             weight=default_per_channel_weight_observer)

# TODO: this is weight only quant, change this to QConfigWeightOnly
# or remove the QConfigDynamic later
float_qparams_weight_only_qconfig = QConfigDynamic(
    activation=default_placeholder_observer,
    weight=default_float_qparams_observer)

default_qat_qconfig = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant)

default_weight_only_qconfig = QConfig(activation=torch.nn.Identity,
                                      weight=default_weight_fake_quant)
default_activation_only_qconfig = QConfig(activation=default_fake_quant,
                                          weight=torch.nn.Identity)

# QAT config that uses a fused observer + fake quant modules for optimized training performance.
# to modify the activation/weight observers, the default entries in fake_quantize.py can be modified.
default_qat_qconfig_v2 = QConfig(activation=default_fused_act_fake_quant, weight=default_fused_wt_fake_quant)

def get_default_qconfig(backend='fbgemm'):
    if backend == 'fbgemm':
        qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=True),
                          weight=default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        qconfig = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                          weight=default_weight_observer)
    else:
        qconfig = default_qconfig
    return qconfig

def get_default_qat_qconfig(backend='fbgemm', version=1):
    # Histogram observer is too slow for quantization aware training
    if version is None:
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
    # Use the fused observer + fake_quant modules for doing QAT.
    if version == 1:
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
    return qconfig

def assert_valid_qconfig(qconfig: Optional[Union[QConfig, QConfigDynamic]],
                         mod: torch.nn.Module) -> None:
    if qconfig is None:
        return
    is_conv_transpose_mod = (
        isinstance(mod, torch.nn.ConvTranspose1d) or
        isinstance(mod, torch.nn.ConvTranspose2d) or
        isinstance(mod, torch.nn.ConvTranspose3d))
    if is_conv_transpose_mod:
        example_observer = qconfig.weight()
        is_per_channel = (
            isinstance(example_observer, torch.quantization.PerChannelMinMaxObserver) or
            isinstance(example_observer, torch.quantization.MovingAveragePerChannelMinMaxObserver)
        )
        assert not is_per_channel, \
            'Per channel weight observer is not supported yet for ConvTranspose{n}d.'

QConfigAny = Union[QConfig,
                   QConfigDynamic, None]


def add_module_to_qconfig_obs_ctr(
        qconfig: QConfigAny,
        module: Union[nn.Module, None]) -> Any:
    r"""This is a helper function for use in quantization prepare that updates a qconfig so that
    the constructors stored in the qconfig will create observers on the same device that
    'module' is on. This is intended to be used when the qconfigs are propagated to each
    module in order to avoid potential device alignment issues.

    Args:
        qconfig: QConfig or QConfigDynamic with obs constructors stored in activation and weight
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

    if isinstance(qconfig, QConfig):
        return QConfig(activation, weight)
    else:
        return QConfigDynamic(activation, weight)


def qconfig_equals(q1: QConfigAny, q2: QConfigAny):
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
            return partial_equals(q1.activation.p, q2.activation.p) and partial_equals(q1.weight.p, q2.weight.p)
        except AttributeError:
            return q1 == q2
