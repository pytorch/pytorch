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

        scaled_true_xent = net.Scale(
            true_xent,
            net.NextScopedBlob('scaled_cross_entropy'),
            scale=1.0 - self._teacherWeight,
        )
        scaled_teacher_xent = net.Scale(
            teacher_xent,
            net.NextScopedBlob('scaled_teacher_cross_entropy'),
            scale=self._teacherWeight,
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
