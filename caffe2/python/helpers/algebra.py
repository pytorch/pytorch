## @package algebra
# Module caffe2.python.helpers.algebra
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = [
    'Transpose',
    'Sum',
]


def Transpose(model, blob_in, blob_out, **kwargs):
    """Transpose."""
    return model.net.Transpose(blob_in, blob_out, **kwargs)


def Sum(model, blob_in, blob_out, **kwargs):
    """Sum"""
    return model.net.Sum(blob_in, blob_out, **kwargs)
