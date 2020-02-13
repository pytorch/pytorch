# @package batch_huber_loss
# Module caffe2.python.layers.batch_huber_loss
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


class BatchHuberLoss(ModelLayer):

    def __init__(self, model, input_record, name='batch_huber_loss', delta=1.0, **kwargs):
        super(BatchHuberLoss, self).__init__(model, name, input_record, **kwargs)

        assert delta > 0

        self._delta = delta

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

        const_delta = net.ConstantFill(
            label,
            net.NextScopedBlob("delta"),
            value=self._delta,
            dtype=core.DataType.FLOAT,
        )

        label = net.StopGradient(
            label,
            net.NextScopedBlob('stopped_label')
        )

        const_delta = net.StopGradient(
            const_delta,
            net.NextScopedBlob('stopped_delta')
        )

        # abs_error = np.abs(true - pred)
        abs_error = net.L1Distance(
            [label, prediction], net.NextScopedBlob("abs_error")
        )

        # quadratic = 0.5*min(abs_error, delta)^2, linear = delta*max(abs_error-delta, 0)
        min_error = net.Min(
            [abs_error, const_delta], net.NextScopedBlob("min_error_delta")
        )

        quadratic_term = net.Scale(
            net.Sqr(min_error), scale=float(0.5)
        )

        linear_term = net.Mul(
            [
                net.Sub([abs_error, min_error]),
                const_delta,
            ],
            net.NextScopedBlob("huber_linear_term")
        )

        # huber = 0.5 * min(abs_error, delta)^2 + delta * max(abs_error-delta, 0)
        huber_dist = net.Add(
            [quadratic_term, linear_term], net.NextScopedBlob("huber_dist")
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
            huber_dist = net.Mul(
                [huber_dist, weight_blob],
                net.NextScopedBlob("weighted_huber_distance"),
            )

        net.AveragedLoss(huber_dist, self.output_schema.field_blobs())
