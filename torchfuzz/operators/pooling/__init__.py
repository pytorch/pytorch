"""Pooling operators module."""

from .max_pool1d import MaxPool1dOperator
from .max_pool2d import MaxPool2dOperator
from .max_pool3d import MaxPool3dOperator
from .avg_pool1d import AvgPool1dOperator
from .avg_pool2d import AvgPool2dOperator
from .avg_pool3d import AvgPool3dOperator
from .adaptive_avg_pool1d import AdaptiveAvgPool1dOperator
from .adaptive_avg_pool2d import AdaptiveAvgPool2dOperator
from .adaptive_avg_pool3d import AdaptiveAvgPool3dOperator
from .adaptive_max_pool1d import AdaptiveMaxPool1dOperator
from .adaptive_max_pool2d import AdaptiveMaxPool2dOperator
from .adaptive_max_pool3d import AdaptiveMaxPool3dOperator

__all__ = [
    'MaxPool1dOperator',
    'MaxPool2dOperator',
    'MaxPool3dOperator',
    'AvgPool1dOperator',
    'AvgPool2dOperator',
    'AvgPool3dOperator',
    'AdaptiveAvgPool1dOperator',
    'AdaptiveAvgPool2dOperator',
    'AdaptiveAvgPool3dOperator',
    'AdaptiveMaxPool1dOperator',
    'AdaptiveMaxPool2dOperator',
    'AdaptiveMaxPool3dOperator',
]
