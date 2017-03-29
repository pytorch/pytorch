## @package batch_mse_loss
# Module caffe2.python.layers.batch_mse_loss
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
        self.tags.update({Tags.TRAIN_ONLY})

        self.output_schema = schema.Scalar(
            np.float32,
            model.net.NextScopedBlob(name + '_output'))

    def add_ops(self, net):
        prediction = net.Squeeze(
            self.input_record.prediction(),
            net.NextScopedBlob('squeezed_prediction'),
            dims=[1]
        )

        label = net.StopGradient(
            self.input_record.label(),
            net.NextScopedBlob('stopped_label')
        )

        l2dist = net.SquaredL2Distance(
            [label, prediction],
            net.NextScopedBlob('l2')
        )

        net.AveragedLoss(l2dist, self.output_schema.field_blobs())
