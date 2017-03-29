## @package batch_lr_loss
# Module caffe2.python.layers.batch_lr_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
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

        assert schema.is_schema_subset(
            schema.Struct(
                ('label', schema.Scalar()),
                ('prediction', schema.Scalar())
            ),
            input_record
        )
        self.tags.update({Tags.TRAIN_ONLY})

        self.output_schema = schema.Scalar(
            np.float32,
            model.net.NextScopedBlob(name + '_output'))

    # This should be a bit more complicated than it is right now
    def add_ops(self, net):
        class_probabilities = net.MakeTwoClass(
            self.input_record.prediction.field_blobs(),
            net.NextScopedBlob('two_class_predictions')
        )
        label = self.input_record.label.field_blobs()
        if self.input_record.label.field_types()[0] != np.int32:
            label = [
                net.Cast(label, net.NextScopedBlob('int32_label'), to='int32')]

        xent = net.LabelCrossEntropy(
            [class_probabilities] + label,
            net.NextScopedBlob('cross_entropy'),
        )
        if 'weight' in self.input_record.fields:
            xent = net.Mul(
                [xent, self.input_record.weight()],
                net.NextScopedBlob('weighted_scross_entropy'),
            )

        if self.average_loss:
            net.AveragedLoss(xent, self.output_schema.field_blobs())
        else:
            net.ReduceFrontSum(xent, self.output_schema.field_blobs())
