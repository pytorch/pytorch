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

## @package nonlinearity
# Module caffe2.python.helpers.nonlinearity
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core


def prelu(model, blob_in, blob_out, num_channels=1, slope_init=None,
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

    model.AddParameter(slope)

    return model.net.PRelu([blob_in, slope], [blob_out])


def relu(model, blob_in, blob_out, use_cudnn=False, order="NCHW", **kwargs):
    """Relu."""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    return model.net.Relu(blob_in, blob_out, order=order, **kwargs)


def tanh(model, blob_in, blob_out, use_cudnn=False, order="NCHW", **kwargs):
    """Tanh."""
    if use_cudnn:
        kwargs['engine'] = 'CUDNN'
    return model.net.Tanh(blob_in, blob_out, order=order, **kwargs)
