## @package pooling
# Module caffe2.python.helpers.pooling
## @package fc
# Module caffe2.python.helpers.pooling
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = [
    'MaxPool',
    'AveragePool',
]


def MaxPool(model, blob_in, blob_out, use_cudnn=False, **kwargs):
    """Max pooling"""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    return model.net.MaxPool(blob_in, blob_out, order=model.order, **kwargs)


def AveragePool(model, blob_in, blob_out, use_cudnn=False, **kwargs):
    """Average pooling"""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    return model.net.AveragePool(
        blob_in,
        blob_out,
        order=model.order,
        **kwargs
    )
