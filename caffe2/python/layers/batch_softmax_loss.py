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
        label_smoothing_matrix=None,
        label_prob=False,
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
        self.label_prob = label_prob

        # label smoothing matrix: a K * K matrix where K is the label
        # cardinality; (i, j) element is the value of for label i
        # treated/smoothed as label j
        self.label_smoothing_matrix = label_smoothing_matrix
        if self.label_smoothing_matrix is not None:
            self.initialize_label_smoothing_constants()

        self.output_schema = schema.Struct(
            (
                'softmax', schema.Scalar(
                    input_record.prediction.field_type(),
                    self.get_next_blob_reference('softmax')
                )
            ),
            (
                'loss', schema.Scalar(
                    np.float32, self.get_next_blob_reference('loss')
                )
            ),
        )

    def initialize_label_smoothing_constants(self):
        assert self.label_smoothing_matrix is not None
        self.label_smoothing_matrix = np.array(
            self.label_smoothing_matrix).astype(np.float32)
        assert len(self.label_smoothing_matrix.shape) == 2
        label_dim = self.label_smoothing_matrix.shape[0]
        assert label_dim == self.label_smoothing_matrix.shape[1]

        self.label_smoothing_matrix = self.model.add_global_constant(
            '%s_label_smoothing_matrix' % self.name,
            array=self.label_smoothing_matrix,
            dtype=np.dtype(np.float32),
        )
        self.label_dim = self.model.add_global_constant(
            '%s_label_dim' % self.name,
            array=label_dim,
            dtype=np.dtype(np.int64),
        )
        # default case: label is given NOT as target distribution
        # but when used in label smoothing, the label must be in probabilities
        self.label_prob = True

    def compute_smoothed_label(self, net):
        assert self.label_smoothing_matrix is not None
        label = self.input_record.label()
        original_label_type = self.input_record.label.field_type()
        if original_label_type.base != np.int64:
            int64_label = net.NextScopedBlob('int64_label')
            net.Cast([label], [int64_label], to=core.DataType.INT64)
        else:
            int64_label = label
        one_hot_label = net.NextScopedBlob('one_hot_label')
        smoothed_label = net.NextScopedBlob('smoothed_label')
        net.OneHot([int64_label, self.label_dim], [one_hot_label])
        net.MatMul([one_hot_label, self.label_smoothing_matrix], smoothed_label)
        return smoothed_label

    def add_ops(self, net):
        label = self.input_record.label.field_blobs()
        if self.label_smoothing_matrix is not None:
            label = [self.compute_smoothed_label(net)]
        elif not self.label_prob:
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
            self.output_schema.field_blobs(),
            label_prob=self.label_prob,
        )
