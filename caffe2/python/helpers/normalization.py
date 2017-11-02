# Copyright (c) 2016-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

## @package normalization
# Module caffe2.python.helpers.normalization
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import scope
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.proto import caffe2_pb2
from caffe2.python.modeling import initializers


def lrn(model, blob_in, blob_out, order="NCHW", use_cudnn=False, **kwargs):
    """LRN"""
    dev = kwargs['device_option'] if 'device_option' in kwargs \
        else scope.CurrentDeviceScope()
    is_cpu = dev is None or dev.device_type == caffe2_pb2.CPU
    if use_cudnn and (not is_cpu):
        kwargs['engine'] = 'CUDNN'
        blobs_out = blob_out
    else:
        blobs_out = [blob_out, "_" + blob_out + "_scale"]
    lrn = model.net.LRN(
        blob_in,
        blobs_out,
        order=order,
        **kwargs
    )

    if use_cudnn and (not is_cpu):
        return lrn
    else:
        return lrn[0]


def softmax(model, blob_in, blob_out=None, use_cudnn=False, **kwargs):
    """Softmax."""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    if blob_out is not None:
        return model.net.Softmax(blob_in, blob_out, **kwargs)
    else:
        return model.net.Softmax(blob_in, **kwargs)


def instance_norm(model, blob_in, blob_out, dim_in, order="NCHW", **kwargs):
    blob_out = blob_out or model.net.NextName()
    # Input: input, scale, bias
    # Output: output, saved_mean, saved_inv_std
    # scale: initialize with ones
    # bias: initialize with zeros

    def init_blob(value, suffix):
        return model.param_init_net.ConstantFill(
            [], blob_out + "_" + suffix, shape=[dim_in], value=value)
    scale, bias = init_blob(1.0, "s"), init_blob(0.0, "b")

    model.AddParameter(scale, ParameterTags.WEIGHT)
    model.AddParameter(bias, ParameterTags.BIAS)
    blob_outs = [blob_out, blob_out + "_sm", blob_out + "_siv"]
    if 'is_test' in kwargs and kwargs['is_test']:
        blob_outputs = model.net.InstanceNorm(
            [blob_in, scale, bias], [blob_out],
            order=order, **kwargs)
        return blob_outputs
    else:
        blob_outputs = model.net.InstanceNorm(
            [blob_in, scale, bias], blob_outs,
            order=order, **kwargs)
        # Return the output
        return blob_outputs[0]


def spatial_bn(model, blob_in, blob_out, dim_in,
               init_scale=1., init_bias=0.,
               ScaleInitializer=None, BiasInitializer=None,
               RunningMeanInitializer=None, RunningVarianceInitializer=None,
               order="NCHW", **kwargs):
    blob_out = blob_out or model.net.NextName()
    # Input: input, scale, bias, est_mean, est_inv_var
    # Output: output, running_mean, running_inv_var, saved_mean,
    #         saved_inv_var
    # scale: initialize with init_scale (default 1.)
    # bias: initialize with init_bias (default 0.)
    # est mean: zero
    # est var: ones

    if model.init_params:
        scale_init = ("ConstantFill", {'value': init_scale})
        bias_init = ("ConstantFill", {'value': init_bias})
        rm_init = ("ConstantFill", {'value': 0.0})
        riv_init = ("ConstantFill", {'value': 1.0})

        ScaleInitializer = initializers.update_initializer(
            ScaleInitializer, scale_init, ("ConstantFill", {})
        )
        BiasInitializer = initializers.update_initializer(
            BiasInitializer, bias_init, ("ConstantFill", {})
        )
        RunningMeanInitializer = initializers.update_initializer(
            RunningMeanInitializer, rm_init, ("ConstantFill", {})
        )
        RunningVarianceInitializer = initializers.update_initializer(
            RunningVarianceInitializer, riv_init, ("ConstantFill", {})
        )
    else:
        ScaleInitializer = initializers.ExternalInitializer()
        BiasInitializer = initializers.ExternalInitializer()
        RunningMeanInitializer = initializers.ExternalInitializer()
        RunningVarianceInitializer = initializers.ExternalInitializer()

    scale = model.create_param(
        param_name=blob_out + '_s',
        shape=[dim_in],
        initializer=ScaleInitializer,
        tags=ParameterTags.WEIGHT
    )

    bias = model.create_param(
        param_name=blob_out + '_b',
        shape=[dim_in],
        initializer=BiasInitializer,
        tags=ParameterTags.BIAS
    )

    running_mean = model.create_param(
        param_name=blob_out + '_rm',
        shape=[dim_in],
        initializer=RunningMeanInitializer,
        tags=ParameterTags.COMPUTED_PARAM
    )

    running_inv_var = model.create_param(
        param_name=blob_out + '_riv',
        shape=[dim_in],
        initializer=RunningVarianceInitializer,
        tags=ParameterTags.COMPUTED_PARAM
    )

    blob_outs = [blob_out, running_mean, running_inv_var,
                 blob_out + "_sm", blob_out + "_siv"]
    if 'is_test' in kwargs and kwargs['is_test']:
        blob_outputs = model.net.SpatialBN(
            [blob_in, scale, bias, blob_outs[1], blob_outs[2]], [blob_out],
            order=order, **kwargs)
        return blob_outputs
    else:
        blob_outputs = model.net.SpatialBN(
            [blob_in, scale, bias, blob_outs[1], blob_outs[2]], blob_outs,
            order=order, **kwargs)
        # Return the output
        return blob_outputs[0]


def layer_norm(model, blob_in, blob_out, **kwargs):
    def init_scalar_blob(value, suffix):
        return model.param_init_net.ConstantFill(
            [], blob_out + "_" + suffix, shape=[1], value=value)

    scale, bias = init_scalar_blob(1.0, "s"), init_scalar_blob(0.0, "b")

    model.AddParameter(scale, ParameterTags.WEIGHT)
    model.AddParameter(bias, ParameterTags.BIAS)

    normalized, mean, stdev = model.net.LayerNorm(
        [blob_in],
        [blob_out, blob_out + "_mean", blob_out + "_stdev"],
        **kwargs
    )

    weighted = model.net.Mul(
        [normalized, scale],
        [blob_out + "_weighted"],
        broadcast=1,
    )

    biased = model.net.Mul(
        [weighted, bias],
        [blob_out + "_biased"],
        broadcast=1,
    )

    return biased, mean, stdev
