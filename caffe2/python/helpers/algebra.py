## @package algebra
# Module caffe2.python.helpers.algebra
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def transpose(model, blob_in, blob_out, use_cudnn=False, **kwargs):
    """Transpose."""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    return model.net.Transpose(blob_in, blob_out, **kwargs)


def sum(model, blob_in, blob_out, **kwargs):
    """Sum"""
    return model.net.Sum(blob_in, blob_out, **kwargs)


def batch_mat_mul(model, blob_in, blob_out,
                  enable_tensor_core=False, **kwargs):
    if enable_tensor_core:
        kwargs['engine'] = 'TENSORCORE'

    return model.net.BatchMatMul(blob_in, blob_out, **kwargs)
