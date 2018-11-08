## @package batch_mse_loss
# Module caffe2.python.layers.batch_mse_loss
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


class BatchMSELoss(ModelLayer):

    def __init__(self, model, input_record, name='batch_mse_loss', **kwargs):
        super(BatchMSELoss, self).__init__(model, name, input_record, **kwargs)

        assert schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('prediction', schema.Scalar())
            ),
            input_record
        )
        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])

        self.output_schema = schema.Scalar(
            np.float32,
            self.get_next_blob_reference('output'))

    def add_ops(self, net):
        prediction = net.Squeeze(
            self.input_record.prediction(),
            net.NextScopedBlob('squeezed_prediction'),
            dims=[1]
        )

        label = self.input_record.label.field_blobs()
        if self.input_record.label.field_type().base != (
                self.input_record.prediction.field_type().base):

            label = net.Cast(
                label,
                net.NextScopedBlob('cast_label'),
                to=schema.data_type_for_dtype(
                    self.input_record.prediction.field_type()
                )
            )

        label = net.StopGradient(
            label,
            net.NextScopedBlob('stopped_label')
        )

        l2dist = net.SquaredL2Distance(
            [label, prediction],
            net.NextScopedBlob('l2')
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
            l2dist = net.Mul(
                [l2dist, weight_blob],
                net.NextScopedBlob('weighted_l2_distance'),
            )

        net.AveragedLoss(l2dist, self.output_schema.field_blobs())
