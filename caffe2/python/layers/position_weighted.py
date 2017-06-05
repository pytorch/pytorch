## @package position_weighted
# Module caffe2.python.layers.position_weighted
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    get_categorical_limit,
    LayerParameter,
    ModelLayer,
)

from caffe2.python.layers.tags import Tags
import numpy as np


class PositionWeighted(ModelLayer):
    def __init__(self, model, input_record, weight_optim=None,
                 name="position_weights"):
        super(PositionWeighted, self).__init__(model, name, input_record)

        self.shape = get_categorical_limit(input_record)

        self.pos_w = model.net.NextScopedBlob(name + "_pos_w")
        self.params.append(
            LayerParameter(
                parameter=self.pos_w,
                initializer=core.CreateOperator('ConstantFill',
                                                [],
                                                self.pos_w,
                                                shape=[self.shape, ],
                                                value=1.0
                                                ),
                optimizer=weight_optim
            ))

        self.output_schema = schema.Struct(
            ('position_weights',
                schema.Scalar((np.float32, self.shape),
                              model.net.NextScopedBlob(name + "_pos_w_gather")))
        )

        self.tags.update({Tags.HANDLE_AS_SPARSE_LAYER})

    def get_memory_usage(self):
        return self.shape

    def add_ops(self, net):
        inc_seq = net.LengthsRangeFill(
            [self.input_record.lengths()],
            self.input_record.lengths() + '_pos_w_seq'
        )

        net.Gather(
            [self.pos_w, inc_seq],
            self.output_schema.position_weights.field_blobs())
