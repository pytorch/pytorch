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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from caffe2.proto import caffe2_pb2


def _parseFile(filename):
    out_net = caffe2_pb2.NetDef()
    # TODO(bwasti): A more robust handler for pathnames.
    dir_path = os.path.dirname(__file__)
    with open('{dir_path}/{filename}'.format(dir_path=dir_path,
                                             filename=filename), 'rb') as f:
        out_net.ParseFromString(f.read())
    return out_net


init_net = _parseFile('init_net.pb')
predict_net = _parseFile('predict_net.pb')
