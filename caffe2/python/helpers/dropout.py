## @package dropout
# Module caffe2.python.helpers.dropout
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def dropout(model, blob_in, blob_out, use_cudnn=False, **kwargs):
    """dropout"""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    else:
        kwargs['engine'] = 'DEFAULT'
    assert 'is_test' in kwargs, "Argument 'is_test' is required"
    return model.net.Dropout(
        blob_in, [blob_out, "_" + blob_out + "_mask"], **kwargs)[0]
