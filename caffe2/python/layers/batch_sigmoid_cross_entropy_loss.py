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

## @package batch_sigmoid_cross_entropy_loss
# Module caffe2.python.layers.batch_sigmoid_cross_entropy_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.layers.tags import Tags
import numpy as np


class BatchSigmoidCrossEntropyLoss(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        name='batch_sigmoid_cross_entropy_loss',
        **kwargs
    ):
        super(BatchSigmoidCrossEntropyLoss, self).__init__(
            model, name, input_record, **kwargs)

        assert schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar(np.float32)),
                ('prediction', schema.Scalar(np.float32)),
            ),
            input_record
        )
        assert input_record.prediction.field_type().shape == \
            input_record.label.field_type().shape, \
            "prediction and label must have the same shape"

        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])

        self.output_schema = schema.Scalar(
            (np.float32, tuple()), self.get_next_blob_reference('loss')
        )

    def add_ops(self, net):
        sigmoid_cross_entropy = net.SigmoidCrossEntropyWithLogits(
            [self.input_record.prediction(), self.input_record.label()],
            net.NextScopedBlob('sigmoid_cross_entropy')
        )

        net.AveragedLoss(
            sigmoid_cross_entropy, self.output_schema.field_blobs())
