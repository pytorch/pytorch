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
