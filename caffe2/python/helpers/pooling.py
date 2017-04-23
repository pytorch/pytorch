## @package fc
# Module caffe2.python.helpers.pooling
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def max_pool(model, blob_in, blob_out, use_cudnn=False, order="NCHW", **kwargs):
    """Max pooling"""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    return model.net.MaxPool(blob_in, blob_out, order=order, **kwargs)


def average_pool(model, blob_in, blob_out, use_cudnn=False, order="NCHW",
                 **kwargs):
    """Average pooling"""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    return model.net.AveragePool(
        blob_in,
        blob_out,
        order=order,
        **kwargs
    )
