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

# @package label_smooth
# Module caffe2.python.layers.label_smooth
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer
import numpy as np


class LabelSmooth(ModelLayer):
    def __init__(
        self, model, label, smooth_matrix, name='label_smooth', **kwargs
    ):
        super(LabelSmooth, self).__init__(model, name, label, **kwargs)
        self.label = label
        # shape as a list
        smooth_matrix = np.array(smooth_matrix).astype(np.float32).flatten()
        self.set_dim(smooth_matrix)
        self.set_smooth_matrix(smooth_matrix)
        self.output_schema = schema.Scalar(
            (np.float32, (self.dim, )),
            self.get_next_blob_reference('smoothed_label')
        )

    def set_dim(self, smooth_matrix):
        num_elements = smooth_matrix.size
        self.binary_prob_label = (num_elements == 2)
        if self.binary_prob_label:
            self.dim = 1
        else:
            assert np.sqrt(num_elements)**2 == num_elements
            self.dim = int(np.sqrt(num_elements))

    def set_smooth_matrix(self, smooth_matrix):
        if not self.binary_prob_label:
            self.smooth_matrix = self.model.add_global_constant(
                '%s_label_smooth_matrix' % self.name,
                array=smooth_matrix.reshape((self.dim, self.dim)),
                dtype=np.dtype(np.float32),
            )
            self.len = self.model.add_global_constant(
                '%s_label_dim' % self.name,
                array=self.dim,
                dtype=np.dtype(np.int64),
            )
        else:
            self.smooth_matrix = smooth_matrix

    def add_ops_for_binary_prob_label(self, net):
        if self.label.field_type().base != np.float32:
            float32_label = net.NextScopedBlob('float32_label')
            net.Cast([self.label()], [float32_label], to=core.DataType.FLOAT)
        else:
            float32_label = self.label()
        net.StumpFunc(
            float32_label,
            self.output_schema(),
            threshold=0.5,
            low_value=self.smooth_matrix[0],
            high_value=self.smooth_matrix[1],
        )

    def add_ops_for_categorical_label(self, net):
        if self.label.field_type().base != np.int64:
            int64_label = net.NextScopedBlob('int64_label')
            net.Cast([self.label()], [int64_label], to=core.DataType.INT64)
        else:
            int64_label = self.label()
        one_hot_label = net.NextScopedBlob('one_hot_label')
        net.OneHot([int64_label, self.len], [one_hot_label])
        net.MatMul([one_hot_label, self.smooth_matrix], self.output_schema())

    def add_ops(self, net):
        if self.binary_prob_label:
            self.add_ops_for_binary_prob_label(net)
        else:
            self.add_ops_for_categorical_label(net)
