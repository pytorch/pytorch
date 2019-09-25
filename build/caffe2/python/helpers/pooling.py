## @package pooling
# Module caffe2.python.helpers.pooling
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


def max_pool_with_index(model, blob_in, blob_out, order="NCHW", **kwargs):
    """Max pooling with an explicit index of max position"""
    return model.net.MaxPoolWithIndex(
        blob_in,
        [blob_out, blob_out + "_index"],
        order=order,
        **kwargs
    )[0]
