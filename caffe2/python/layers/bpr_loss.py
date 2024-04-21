## @package bpr_loss
# Module caffe2.python.layers.bpr_loss





from caffe2.python import schema
from caffe2.python.layers.layers import (
    ModelLayer,
)
from caffe2.python.layers.tags import (
    Tags
)
import numpy as np


# ref: https://arxiv.org/pdf/1205.2618.pdf
class BPRLoss(ModelLayer):

    def __init__(self, model, input_record, name='bpr_loss', **kwargs):
        super().__init__(model, name, input_record, **kwargs)
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
        # formula:
        # loss = - SUM(Ln(Sigmoid(Simlarity(u, pos) - Simlarity(u, neg))))
        neg_score = self.input_record.neg_prediction['values']()

        pos_score = net.LengthsTile(
            [
                self.input_record.pos_prediction(),
                self.input_record.neg_prediction['lengths']()
            ],
            net.NextScopedBlob('pos_score_repeated')
        )
        # https://www.tensorflow.org/api_docs/python/tf/math/log_sigmoid
        softplus = net.Softplus([net.Sub([neg_score, pos_score])])
        net.ReduceFrontSum(softplus, self.output_schema.field_blobs())
