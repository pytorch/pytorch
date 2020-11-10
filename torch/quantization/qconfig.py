from collections import namedtuple
import torch.nn as nn

from .fake_quantize import *
from .observer import *
from .quant_type import QuantType

class QConfig(namedtuple('QConfig', ['activation', 'weight', 'quant_type'])):
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


    QConfig can also take the QuantType enum for the `quant_type` argument.
    This is an optional argument to support proper custom module configuration.
    By default it is set to STATIC.
    """
    def __new__(cls, activation, weight, quant_type=None):
        # catch common mistakes
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError("QConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        if quant_type is None:
            quant_type = QuantType.STATIC
        return super(QConfig, cls).__new__(cls, activation, weight, quant_type)


default_qconfig = QConfig(activation=default_observer,
                          weight=default_weight_observer)

default_debug_qconfig = QConfig(weight=default_weight_observer,
                                activation=default_debug_observer)

default_per_channel_qconfig = QConfig(activation=default_observer,
                                      weight=default_per_channel_weight_observer)

class QConfigDynamic(namedtuple('QConfigDynamic', ['activation', 'weight', 'quant_type'])):
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

    QConfig can also take the QuantType enum for the `quant_type` argument.
    This is an optional argument to support proper custom module configuration.
    By default it is set to DYNAMIC
    """
    def __new__(cls, activation=torch.nn.Identity, weight=torch.nn.Identity,
                quant_type=None):
        # catch common mistakes
        if isinstance(weight, nn.Module):
            raise ValueError("QConfigDynamic received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        if quant_type is None:
            quant_type = QuantType.DYNAMIC
        return super(QConfigDynamic, cls).__new__(cls, activation, weight, quant_type)

default_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
                                         weight=default_weight_observer)
float16_dynamic_qconfig = QConfigDynamic(activation=PlaceholderObserver.with_args(dtype=torch.float16),
                                         weight=PlaceholderObserver.with_args(dtype=torch.float16))
per_channel_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
                                             weight=default_per_channel_weight_observer)

float_qparams_dynamic_qconfig = QConfigDynamic(activation=default_dynamic_quant_observer,
                                               weight=default_float_qparams_observer)

default_qat_qconfig = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant,
                              quant_type=QuantType.QAT)

default_weight_only_qconfig = QConfig(activation=torch.nn.Identity,
                                      weight=default_weight_fake_quant,
                                      quant_type=QuantType.WEIGHT_ONLY)
default_activation_only_qconfig = QConfig(activation=default_fake_quant,
                                          weight=torch.nn.Identity,
                                          quant_type=QuantType.ACTIVATION_ONLY)

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
                          weight=default_per_channel_weight_fake_quant,
                          quant_type=QuantType.QAT)
    elif backend == 'qnnpack':
        qconfig = QConfig(activation=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver,
                                                            quant_min=0,
                                                            quant_max=255,
                                                            reduce_range=False),
                          weight=default_weight_fake_quant,
                          quant_type=QuantType.QAT)
    else:
        qconfig = default_qat_qconfig
    return qconfig
