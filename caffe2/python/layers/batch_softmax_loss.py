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

## @package batch_softmax_loss
# Module caffe2.python.layers.batch_softmax_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer
import numpy as np


class BatchSoftmaxLoss(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        name='batch_softmax_loss',
        **kwargs
    ):
        super(BatchSoftmaxLoss, self).__init__(
            model, name, input_record, **kwargs)

        assert schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('prediction', schema.Scalar()),
            ),
            input_record
        )

        self.output_schema = schema.Struct(
            (
                'softmax', schema.Scalar(
                    input_record.prediction.field_type(),
                    self.get_next_blob_reference('softmax')
                )
            ),
            (
                'loss', schema.Scalar(
                    np.float32, self.get_next_blob_reference('loss')
                )
            ),
        )

    def add_ops(self, net):
        label = self.input_record.label.field_blobs()
        if self.input_record.label.field_types()[0].base != np.int32:
            label = [
                net.Cast(label,
                         net.NextScopedBlob('int32_label'),
                         to=core.DataType.INT32)
            ]

        softmax_input = self.input_record.prediction.field_blobs() + label

        if 'weight' in self.input_record:
            weight_blob = self.input_record.weight()
            if self.input_record.weight.field_type().base != np.float32:
                weight_blob = net.Cast(
                    weight_blob,
                    weight_blob + '_float32',
                    to=core.DataType.FLOAT
                )

            softmax_input += [weight_blob]

        net.SoftmaxWithLoss(
            softmax_input,
            self.output_schema.field_blobs()
        )
