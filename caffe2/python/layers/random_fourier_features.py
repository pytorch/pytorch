from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import (
    ModelLayer,
    LayerParameter
)

import numpy as np


class RandomFourierFeatures(ModelLayer):
    def __init__(
            self,
            model,
            input_record,
            output_dims,
            sigma,  # bandwidth
            w_init=None,
            b_init=None,
            name='random_fourier_features',
            **kwargs):

        super(RandomFourierFeatures, self).__init__(model, name, input_record,
                                                    **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"
        assert output_dims >= 1, "Expected output dimensions >= 1, got %s" \
                                    % output_dims

        input_dims = input_record.field_type().shape[0]
        assert input_dims >= 1, "Expected input dimensions >= 1, got %s" \
                                    % input_dims

        self.output_schema = schema.Scalar(
            (np.float32, (output_dims, )),
            model.net.NextScopedBlob(name + '_output')
        )

        self.output_dims = output_dims

        # Initialize train_init_net parameters
        w_init = w_init if w_init else (
            'GaussianFill', {'mean': 0.0, 'std': 1.0 / sigma}
        )

        b_init = b_init if b_init else (
            'UniformFill', {'min': 0.0, 'max': 2 * np.pi}
        )

        self.w = model.net.NextScopedBlob(name + "_w")
        self.b = model.net.NextScopedBlob(name + "_b")
        self.params.append(
            LayerParameter(
                parameter=self.w,
                initializer=core.CreateOperator(w_init[0],
                                                [],
                                                self.w,
                                                shape=(input_dims, output_dims),
                                                **w_init[1]
                                                ),
                optimizer=model.NoOptim))
        self.params.append(
            LayerParameter(
                parameter=self.b,
                initializer=core.CreateOperator(b_init[0],
                                                [],
                                                self.b,
                                                shape=[output_dims],
                                                **b_init[1]
                                                ),
                optimizer=model.NoOptim))

    def add_ops(self, net):
        # Matrix multiplication for input and w
        weighted_term = net.MatMul(self.input_record.field_blobs() + [self.w],
                                   net.NextScopedBlob('weighted_term'))
        # Add wx + b
        cosine_arg = net.Add([weighted_term, self.b],
                             net.NextScopedBlob('cosine_arg'),
                             broadcast=1, axis=1)

        # Apply cosine to new vectors
        new_feature_vec = net.Cos([cosine_arg],
                                  net.NextScopedBlob('new_feature_vec'))

        # Multiply each element in vector by sqrt(2/D)
        scale = np.sqrt(2.0 / self.output_dims)
        net.Scale([new_feature_vec],
                  self.output_schema.field_blobs(),
                  scale=scale)
