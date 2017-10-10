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

## @package tools
# Module caffe2.python.helpers.tools
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def image_input(
    model, blob_in, blob_out, order="NCHW", use_gpu_transform=False, **kwargs
):
    assert 'is_test' in kwargs, "Argument 'is_test' is required"
    if order == "NCHW":
        if (use_gpu_transform):
            kwargs['use_gpu_transform'] = 1 if use_gpu_transform else 0
            # GPU transform will handle NHWC -> NCHW
            outputs = model.net.ImageInput(blob_in, blob_out, **kwargs)
            pass
        else:
            outputs = model.net.ImageInput(
                blob_in, [blob_out[0] + '_nhwc'] + blob_out[1:], **kwargs
            )
            outputs_list = list(outputs)
            outputs_list[0] = model.net.NHWC2NCHW(outputs_list[0], blob_out[0])
            outputs = tuple(outputs_list)
    else:
        outputs = model.net.ImageInput(blob_in, blob_out, **kwargs)
    return outputs


def video_input(model, blob_in, blob_out, **kwargs):
    data, label = model.net.VideoInput(blob_in, blob_out, **kwargs)
    return data, label
