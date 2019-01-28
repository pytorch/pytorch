## @package elementwise_linear
# Module caffe2.python.helpers.elementwise_linear
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
from caffe2.python.modeling.parameter_info import ParameterTags


def _elementwise_linear(
    model, op_call, blob_in, blob_out, dim,
    weight_init=None, bias_init=None, **kwargs
):
    """Elementwise_Linear"""
    weight_init = weight_init or ('ConstantFill', {'value': 1.0})
    bias_init = bias_init or ('ConstantFill', {'value': 0.0})
    blob_out = blob_out or model.net.NextName()
    if model.init_params:
        weight = model.param_init_net.__getattr__(weight_init[0])(
            [],
            blob_out + '_w',
            shape=[dim],
            **weight_init[1]
        )
        bias = model.param_init_net.__getattr__(bias_init[0])(
            [],
            blob_out + '_b',
            shape=[dim],
            **bias_init[1]
        )
    else:
        weight = core.ScopedBlobReference(
            blob_out + '_w', model.param_init_net)
        bias = core.ScopedBlobReference(
            blob_out + '_b', model.param_init_net)

    model.AddParameter(weight, ParameterTags.WEIGHT)
    model.AddParameter(bias, ParameterTags.BIAS)
    return op_call([blob_in, weight, bias], blob_out, **kwargs)


def elementwise_linear(model, *args, **kwargs):
    return _elementwise_linear(
        model, model.net.ElementwiseLinear, *args, **kwargs)
