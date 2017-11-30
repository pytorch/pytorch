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

## @package batch_lr_loss
# Module caffe2.python.layers.batch_lr_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    ModelLayer,
)
from caffe2.python.layers.tags import (
    Tags
)
import numpy as np


class BatchLRLoss(ModelLayer):

    def __init__(self, model, input_record, name='batch_lr_loss',
                 average_loss=True, **kwargs):
        super(BatchLRLoss, self).__init__(model, name, input_record, **kwargs)

        self.average_loss = average_loss

        assert (schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('prediction', schema.Scalar())
            ),
            input_record
        ) or schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('logit', schema.Scalar())
            ),
            input_record
        ))

        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])

        self.output_schema = schema.Scalar(
            np.float32,
            self.get_next_blob_reference('output')
        )

    def add_ops(self, net):
        # numerically stable log-softmax with crossentropy
        if schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('logit', schema.Scalar())
            ), self.input_record
        ):
            label = self.input_record.label()
            # mandatory cast to float32
            # self.input_record.label.field_type().base is np.float32 but
            # label type is actually int
            label = net.Cast(
                label,
                net.NextScopedBlob('label_float32'),
                to=core.DataType.FLOAT)
            label = net.ExpandDims(label, net.NextScopedBlob('expanded_label'),
                                    dims=[1])
            xent = net.SigmoidCrossEntropyWithLogits(
                [self.input_record.logit(), label],
                net.NextScopedBlob('cross_entropy'),
            )
        # TODO(T23937449): Change all the use cases of BatchLRLoss to the
        # numerically stable version
        else:
            class_probabilities = net.MakeTwoClass(
                self.input_record.prediction.field_blobs(),
                net.NextScopedBlob('two_class_predictions')
            )
            label = self.input_record.label.field_blobs()
            label = [net.Cast(
                label,
                net.NextScopedBlob('int32_label'),
                to=core.DataType.INT32)]
            xent = net.LabelCrossEntropy(
                [class_probabilities] + label,
                net.NextScopedBlob('cross_entropy'),
            )
        if 'weight' in self.input_record.fields:
            weight_blob = self.input_record.weight()
            if self.input_record.weight.field_type().base != np.float32:
                weight_blob = net.Cast(
                    weight_blob,
                    weight_blob + '_float32',
                    to=core.DataType.FLOAT
                )
            weight_blob = net.StopGradient(
                [weight_blob],
                [net.NextScopedBlob('weight_stop_gradient')],
            )
            xent = net.Mul(
                [xent, weight_blob],
                net.NextScopedBlob('weighted_cross_entropy'),
            )

        if self.average_loss:
            net.AveragedLoss(xent, self.output_schema.field_blobs())
        else:
            net.ReduceFrontSum(xent, self.output_schema.field_blobs())
