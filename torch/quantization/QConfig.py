from __future__ import absolute_import, division, print_function, unicode_literals
from collections import namedtuple
from .observer import *
from .fake_quantize import *

class QConfig(namedtuple('QConfig', ['activation', 'weight'])):
    """
    Describes how to quantize a layer or a part of the network by providing
    settings (observer classes) for activations and weights respectively.

    Note that QConfig needs to contain observer **classes** (like MinMaxObserver), not the concrete instances.
    Quantization preparation function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial):

      my_qconfig = QConfig(activation=MinMaxObserver.with_args(dtype=torch.qint8),
                           weight=default_observer.with_args(dtype=torch.qint8))
    """
    def __init__(self, activation, weight):
        # catch common mistakes
        if isinstance(activation, ObserverBase) or isinstance(weight, ObserverBase):
            raise ValueError("QConfig received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        super(QConfig, self).__init__(activation, weight)

default_qconfig = QConfig(activation=default_observer,
                          weight=default_weight_observer)

default_debug_qconfig = QConfig(weight=default_weight_observer,
                                activation=default_debug_observer)


class QConfig_dynamic(namedtuple('QConfig_dynamic', ['weight'])):
    """
    Describes how to dynamically quantize a layer or a part of the network by providing
    settings (observer classe) for weights.

    It's like QConfig, but for dynamic quantization.

    Note that QConfig_dynamic needs to contain observer **classes** (like MinMaxObserver), not the concrete instances.
    Quantization preparation function will instantiate observers multiple times for each of the layers.

    Observer classes have usually reasonable default arguments, but they can be overwritten with `with_args`
    method (that behaves like functools.partial):

      my_qconfig = QConfig_dynamic(weight=default_observer.with_args(dtype=torch.qint8))
    """
    def __init__(self, weight):
        # catch common mistakes
        if isinstance(weight, ObserverBase):
            raise ValueError("QConfig_dynamic received observer instance, please pass observer class instead. " +
                             "Use MyObserver.with_args(x=1) to override arguments to constructor if needed")
        super(QConfig_dynamic, self).__init__(weight)

default_dynamic_qconfig = QConfig_dynamic(weight=default_weight_observer)

default_qat_qconfig = QConfig(activation=default_fake_quant,
                              weight=default_weight_fake_quant)
