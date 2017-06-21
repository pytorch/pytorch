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
                    model.net.NextScopedBlob(name + '_softmax')
                )
            ),
            (
                'loss', schema.Scalar(
                    np.float32, model.net.NextScopedBlob(name + '_loss')
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
