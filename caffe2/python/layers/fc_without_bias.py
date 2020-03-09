## @package fc_without_bias
# Module caffe2.python.layers.fc_without_bias
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
from caffe2.python.layers.sampling_trainable_mixin import SamplingTrainableMixin

import math
import numpy as np


class FCWithoutBias(SamplingTrainableMixin, ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        output_dims,
        weight_init=None,
        weight_optim=None,
        name='fc_without_bias',
        uniform_weight_init_scale_numerator=1.0,
        **kwargs
    ):
        super(FCWithoutBias, self).__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"
        assert len(input_record.field_types()[0].shape) > 0, (
            "FCWithoutBias expects limited dimensions of the input tensor"
        )

        input_dims = input_record.field_types()[0].shape[0]
        assert input_dims > 0, (
            "FCWithoutBias expects input dimensions > 0, got {}".format(input_dims)
        )

        self.output_schema = schema.Scalar(
            (np.float32, (output_dims, )),
            self.get_next_blob_reference('output')
        )

        scale = math.sqrt(uniform_weight_init_scale_numerator / input_dims)
        weight_init = weight_init if weight_init else (
            'UniformFill', {'min': -scale,
                            'max': scale}
        )

        self.w = self.create_param(param_name='w',
                                   shape=[output_dims, input_dims],
                                   initializer=weight_init,
                                   optimizer=weight_optim)

    def _add_ops(self, net, params):
        net.MatMul(
            self.input_record.field_blobs() + params,
            self.output_schema.field_blobs(), trans_b=1, **self.kwargs
        )

    @property
    def param_blobs(self):
        return [self.w]
