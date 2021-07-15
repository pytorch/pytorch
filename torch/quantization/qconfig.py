from collections import namedtuple
from .observer import (HistogramObserver, MovingAverageMinMaxObserver,
                       PlaceholderObserver, default_debug_observer,
                       default_dynamic_quant_observer,
                       default_float_qparams_observer, default_observer,
                       default_per_channel_weight_observer,
                       default_placeholder_observer, default_weight_observer)
from .fake_quantize import (FakeQuantize, default_fake_quant,
                            default_per_channel_weight_fake_quant,
                            default_weight_fake_quant)
import torch
import torch.nn as nn

from typing import Union, Optional, Any

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

def get_default_qat_qconfig(backend='fbgemm'):
    # Histogram observer is too slow for quantization aware training
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

QConfigAny = Union[QConfig, QConfigDynamic, None]

# should probably be in qconfig_utils but get circular reference issues TODO
def add_device_to_obs_ctr_in_qconfig(qconfig: QConfigAny, module: torch.nn.Module):
    if (module is None or qconfig is None or
            qconfig.activation.__module__ != 'torch.quantization.observer' or
            qconfig.weight.__module__ != 'torch.quantization.observer'):
        return qconfig

    def assert_and_get_unique_device(module: torch.nn.Module) -> Any:
        devices = {p.device for p in module.parameters()} | \
            {p.device for p in module.buffers()}
        assert len(devices) <= 1, (
            "prepare only works with cpu or single-device CUDA modules, "
            "but got devices {}".format(devices)
        )
        device = next(iter(devices)) if len(devices) > 0 else None
        return device

    def get_activation():
        obs_ctr = qconfig.activation.with_args(factory_kwargs={'device': None if module is None else assert_and_get_unique_device(module)})
        return obs_ctr()

    def get_weight():
        obs_ctr = qconfig.weight.with_args(factory_kwargs={'device': None if module is None else assert_and_get_unique_device(module)})
        return obs_ctr()

    if isinstance(qconfig, torch.quantization.QConfig):
        return torch.quantization.QConfig(
            activation=get_activation,
            weight=get_weight
        )
    elif isinstance(qconfig, torch.quantization.QConfigDynamic):
        return torch.quantization.QConfigDynamic(
            activation=get_activation,
            weight=get_weight
        )
    else:
        return qconfig
