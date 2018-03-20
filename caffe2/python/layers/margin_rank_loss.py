## @package random_neg_rank_loss
# Module caffe2.python.layers.random_neg_rank_loss
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema, core
from caffe2.python.layers.layers import (
    ModelLayer,
)
from caffe2.python.layers.tags import (
    Tags
)
import numpy as np


class MarginRankLoss(ModelLayer):

    def __init__(self, model, input_record, name='margin_rank_loss',
                 margin=0.1, average_loss=False, **kwargs):
        super(MarginRankLoss, self).__init__(model, name, input_record, **kwargs)
        assert margin >= 0, ('For hinge loss, margin should be no less than 0')
        self._margin = margin
        self._average_loss = average_loss
        assert schema.is_schema_subset(
            schema.Struct(
                ('pos_prediction', schema.Scalar()),
                ('neg_prediction', schema.List(np.float32)),
            ),
            input_record
        )
        self.tags.update([Tags.EXCLUDE_FROM_PREDICTION])
        self.output_schema = schema.Scalar(
            np.float32,
            self.get_next_blob_reference('output'))

    def add_ops(self, net):
        neg_score = self.input_record.neg_prediction['values']()

        pos_score = net.LengthsTile(
            [
                self.input_record.pos_prediction(),
                self.input_record.neg_prediction['lengths']()
            ],
            net.NextScopedBlob('pos_score_repeated')
        )
        const_1 = net.ConstantFill(
            neg_score,
            net.NextScopedBlob('const_1'),
            value=1,
            dtype=core.DataType.INT32
        )
        rank_loss = net.MarginRankingCriterion(
            [pos_score, neg_score, const_1],
            net.NextScopedBlob('rank_loss'),
            margin=self._margin,
        )
        if self._average_loss:
            net.AveragedLoss(rank_loss, self.output_schema.field_blobs())
        else:
            net.ReduceFrontSum(rank_loss, self.output_schema.field_blobs())
