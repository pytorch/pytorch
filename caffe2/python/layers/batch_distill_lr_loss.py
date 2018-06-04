## @package batch_distill_lr_loss
# Module caffe2.python.layers.batch_distill_lr_loss
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


class BatchDistillLRLoss(ModelLayer):

    def __init__(
            self, model, input_record,
            name='batch_distill_lr_loss', teacherWeight=0.0, **kwargs):

        super(BatchDistillLRLoss, self).__init__(model, name, input_record, **kwargs)

        assert teacherWeight >= 0 and teacherWeight <= 1, (
            'teacherWeight=%0.2f should be in [0, 1]' % teacherWeight
        )
        self._teacherWeight = teacherWeight

        assert schema.is_schema_subset(
            schema.Struct(
                ('teacher_label', schema.Scalar()),
                ('label', schema.Scalar()),
                ('logit', schema.Scalar()),
            ),
            input_record
        )
        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])

        self.output_schema = schema.Scalar(
            np.float32,
            self.get_next_blob_reference('output')
        )

    def add_ops(self, net):
        label = self.input_record.label()
        if self.input_record.label.field_type() != np.float32:
            label = net.Cast(
                label,
                net.NextScopedBlob('float_label'),
                to=core.DataType.FLOAT,
            )

        # Assuming 1-D input
        label = net.ExpandDims(label, net.NextScopedBlob('expanded_label'),
                               dims=[1])

        teacher_label = self.input_record.teacher_label()

        TEACHER_WEIGHT_THRESHOLD = 0.0

        threshold = net.ConstantFill(
            teacher_label,
            net.NextScopedBlob('threshold'),
            value=TEACHER_WEIGHT_THRESHOLD,
            dtype=core.DataType.FLOAT
        )

        keep_weights = net.ConstantFill(
            teacher_label,
            net.NextScopedBlob('keep_weights'),
            value=self._teacherWeight,
            dtype=core.DataType.FLOAT
        )

        zero_weights = net.ConstantFill(
            teacher_label,
            net.NextScopedBlob('zero_weights'),
            value=0.0,
            dtype=core.DataType.FLOAT
        )

        neg_ONE = net.ConstantFill(
            teacher_label,
            net.NextScopedBlob('neg_ONE'),
            value=-1.0,
            dtype=core.DataType.FLOAT
        )

        ONE = net.ConstantFill(
            teacher_label,
            net.NextScopedBlob('ONE'),
            value=1.0,
            dtype=core.DataType.FLOAT
        )

        judge = net.GT(
            [teacher_label, threshold],
            net.NextScopedBlob('judge'),
            broadcast=1
        )

        screened_teacher_weights = net.Conditional(
            [judge, keep_weights, zero_weights],
            net.NextScopedBlob('screened_teacher_weights')
        )

        neg_screened_teacher_weights = net.Mul(
            [neg_ONE, screened_teacher_weights],
            net.NextScopedBlob('neg_screened_teacher_weights'),
            broadcast=1
        )

        one_minus_screened_teacher_weights = net.Add(
            [neg_screened_teacher_weights, ONE],
            net.NextScopedBlob('one_minus_screened_teacher_weights'),
            broadcast=1
        )

        if self.input_record.teacher_label.field_type() != np.float32:
            teacher_label = net.Cast(
                teacher_label,
                net.NextScopedBlob('float_teacher_label'),
                to=core.DataType.FLOAT,
            )
        teacher_label = net.ExpandDims(
            teacher_label, net.NextScopedBlob('expanded_teacher_label'),
            dims=[1])

        true_xent = net.SigmoidCrossEntropyWithLogits(
            [self.input_record.logit(), label],
            net.NextScopedBlob('cross_entropy')
        )

        teacher_xent = net.SigmoidCrossEntropyWithLogits(
            [self.input_record.logit(), teacher_label],
            net.NextScopedBlob('teacher_cross_entropy')
        )

        scaled_true_xent = net.Mul(
            [true_xent, one_minus_screened_teacher_weights],
            net.NextScopedBlob('scaled_cross_entropy'),
            broadcast=1
        )
        scaled_teacher_xent = net.Mul(
            [teacher_xent, screened_teacher_weights],
            net.NextScopedBlob('scaled_teacher_cross_entropy')
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
            scaled_true_xent = net.Mul(
                [scaled_true_xent, weight_blob],
                net.NextScopedBlob('weighted_xent_label'),
            )
            scaled_teacher_xent = net.Mul(
                [scaled_teacher_xent, weight_blob],
                net.NextScopedBlob('weighted_xent_teacher'),
            )

        true_loss = net.AveragedLoss(
            scaled_true_xent,
            net.NextScopedBlob('true_loss')
        )
        teacher_loss = net.AveragedLoss(
            scaled_teacher_xent,
            net.NextScopedBlob('teacher_loss')
        )
        net.Add(
            [true_loss, teacher_loss],
            self.output_schema.field_blobs()
        )
