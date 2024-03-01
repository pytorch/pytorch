




from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer

import numpy as np


class RandomFourierFeatures(ModelLayer):
    """
    Implementation of random fourier feature map for feature processing.

    Applies sqrt(2 / output_dims) * cos(wx+b), where:
        output_dims is the output feature dimensions, and
        wx + b applies FC using randomized, fixed weight and bias parameters

    For more information, see the original paper:
        https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf

    Inputs:
        output_dims -- output feature dimensions
        sigma -- bandwidth for the Gaussian kernel estimator
        w_init -- initialization options for weight parameter
        b_init -- initialization options for bias parameter

    """
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

        super().__init__(model, name, input_record, **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"

        input_dims = input_record.field_type().shape[0]
        assert input_dims >= 1, "Expected input dimensions >= 1, got %s" \
                                % input_dims
        self.output_dims = output_dims
        assert self.output_dims >= 1, "Expected output dimensions >= 1, got %s" \
                                      % self.output_dims

        self.output_schema = schema.Scalar(
            (np.float32, (self.output_dims, )),
            self.get_next_blob_reference('output')
        )

        assert sigma > 0.0, "Expected bandwidth > 0, got %s" % sigma

        # Initialize train_init_net parameters
        w_init = w_init if w_init else (
            'GaussianFill', {'mean': 0.0, 'std': 1.0 / sigma}
        )

        b_init = b_init if b_init else (
            'UniformFill', {'min': 0.0, 'max': 2 * np.pi}
        )

        self.w = self.create_param(param_name='w',
                                   shape=[self.output_dims, input_dims],
                                   initializer=w_init,
                                   optimizer=model.NoOptim)

        self.b = self.create_param(param_name='b',
                                   shape=[self.output_dims],
                                   initializer=b_init,
                                   optimizer=model.NoOptim)

    def add_ops(self, net):
        # Random features: wx + b
        cosine_arg = net.FC(self.input_record.field_blobs() + [self.w, self.b],
                            net.NextScopedBlob("cosine_arg"))

        # Apply cosine to new vectors
        new_feature_vec = net.Cos([cosine_arg],
                                  net.NextScopedBlob('new_feature_vec'))

        # Multiply each element in vector by sqrt(2/D)
        scale = np.sqrt(2.0 / self.output_dims)
        net.Scale([new_feature_vec],
                  self.output_schema.field_blobs(),
                  scale=scale)
