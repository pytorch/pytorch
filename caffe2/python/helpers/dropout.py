## @package dropout
# Module caffe2.python.helpers.dropout
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__all__ = [
    'Dropout',
]


def Dropout(model, blob_in, blob_out, **kwargs):
    """Dropout"""
    return model.net.Dropout(
        blob_in, [blob_out, "_" + blob_out + "_mask"], **kwargs)[0]
