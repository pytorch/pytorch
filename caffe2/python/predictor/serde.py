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

## @package serde
# Module caffe2.python.predictor.serde
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def serialize_protobuf_struct(protobuf_struct):
    return protobuf_struct.SerializeToString()


def deserialize_protobuf_struct(serialized_protobuf, struct_type):
    deser = struct_type()
    deser.ParseFromString(serialized_protobuf)
    return deser
