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

# @package constant_weight
# Module caffe2.fb.python.layers.constant_weight
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
import numpy as np


class ConstantWeight(ModelLayer):
    def __init__(
        self, model, input_record, weights, name='constant_weight', **kwargs
    ):
        super(ConstantWeight,
              self).__init__(model, name, input_record, **kwargs)
        self.output_schema = schema.Scalar(
            np.float32, self.get_next_blob_reference('adaptive_weight')
        )
        self.data = self.input_record.field_blobs()
        self.num = len(self.data)
        assert len(weights) == self.num
        self.weights = [
            self.model.add_global_constant(
                '%s_weight_%d' % (self.name, i), float(weights[i])
            ) for i in range(self.num)
        ]

    def add_ops(self, net):
        net.WeightedSum(
            [b for x_w_pair in zip(self.data, self.weights) for b in x_w_pair],
            self.output_schema()
        )
