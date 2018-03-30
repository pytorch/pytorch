# @package constant_weight
# Module caffe2.fb.python.layers.constant_weight
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
import numpy as np


class ConstantWeight(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        weights=None,
        name='constant_weight',
        **kwargs
    ):
        super(ConstantWeight,
              self).__init__(model, name, input_record, **kwargs)
        self.output_schema = schema.Scalar(
            np.float32, self.get_next_blob_reference('adaptive_weight')
        )
        self.data = self.input_record.field_blobs()
        self.num = len(self.data)
        weights = (
            weights if weights is not None else
            [1. / self.num for _ in range(self.num)]
        )
        assert len(weights) == self.num
        self.weights = [
            self.model.add_global_constant(
                '%s_weight_%d' % (self.name, i), float(weights[i])
            ) for i in range(self.num)
        ]

    def add_ops(self, net):
        net.WeightedSum(
            [b for x_w_pair in zip(self.data, self.weights) for b in x_w_pair],
            self.output_schema()
        )
