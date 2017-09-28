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
