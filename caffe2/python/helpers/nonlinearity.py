## @package nonlinearity
# Module caffe2.python.helpers.nonlinearity
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core

__all__ = [
    'Relu',
    'PRelu',
]


def PRelu(model, blob_in, blob_out, num_channels=1, slope_init=None,
          **kwargs):
    """PRelu"""
    slope_init = (
        slope_init if slope_init else ('ConstantFill', {'value': 0.25}))
    if model.init_params:
        slope = model.param_init_net.__getattr__(slope_init[0])(
            [],
            blob_out + '_slope',
            shape=[num_channels],
            **slope_init[1]
        )
    else:
        slope = core.ScopedBlobReference(
            blob_out + '_slope', model.param_init_net)

    model.params.extend([slope])

    return model.net.PRelu([blob_in, slope], [blob_out])


def Relu(model, blob_in, blob_out, **kwargs):
    """Relu."""
    return model.net.Relu(blob_in, blob_out, **kwargs)
