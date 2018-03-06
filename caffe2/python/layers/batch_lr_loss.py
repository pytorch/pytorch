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
                 average_loss=True, jsd_weight=0.0, **kwargs):
        super(BatchLRLoss, self).__init__(model, name, input_record, **kwargs)

        self.average_loss = average_loss

        assert (schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('logit', schema.Scalar())
            ),
            input_record
        ))

        assert jsd_weight >= 0 and jsd_weight <= 1
        self.jsd_weight = jsd_weight
        if self.jsd_weight > 0:
            assert 'prediction' in input_record
            self.jsd_weight_const = model.add_global_constant(
                'jsd_weight', self.jsd_weight
            )
            self.xent_weight_const = model.add_global_constant(
                'xent_weight', 1 - self.jsd_weight
            )

        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])

        self.output_schema = schema.Scalar(
            np.float32,
            self.get_next_blob_reference('output')
        )

    def add_ops(self, net):
        # numerically stable log-softmax with crossentropy
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
        # fuse with JSD
        if self.jsd_weight > 0:
            jsd = net.BernoulliJSD(
                [self.input_record.prediction(), label],
                net.NextScopedBlob('jsd'),
            )
            loss = net.WeightedSum(
                [xent, self.xent_weight_const, jsd, self.jsd_weight_const],
                net.NextScopedBlob('loss'),
            )
        else:
            loss = xent


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
            loss = net.Mul(
                [loss, weight_blob],
                net.NextScopedBlob('weighted_cross_entropy'),
            )

        if self.average_loss:
            net.AveragedLoss(loss, self.output_schema.field_blobs())
        else:
            net.ReduceFrontSum(loss, self.output_schema.field_blobs())
