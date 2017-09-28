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

## @package arra_helpers
# Module caffe2.python.helpers.array_helpers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def concat(model, blobs_in, blob_out, **kwargs):
    """Depth Concat."""
    if kwargs.get('order') and kwargs.get('axis'):
        # The backend throws an error if both are given
        kwargs.pop('order')

    return model.net.Concat(
        blobs_in,
        [blob_out, "_" + blob_out + "_concat_dims"],
        **kwargs
    )[0]


def depth_concat(model, blobs_in, blob_out, **kwargs):
    """The old depth concat function - we should move to use concat."""
    print("DepthConcat is deprecated. use Concat instead.")
    return concat(blobs_in, blob_out, **kwargs)
