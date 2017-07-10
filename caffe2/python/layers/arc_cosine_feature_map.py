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


class ArcCosineFeatureMap(ModelLayer):
    """
    A general version of the arc-cosine kernel feature map (s = 1 restores
    the original arc-cosine kernel feature map).

    Applies H(x) * x^s, where H is the Heaviside step function and x is the
    input after applying FC (such that x = w * x_orig + b).

    For more information, see the original paper:
        http://cseweb.ucsd.edu/~saul/papers/nips09_kernel.pdf

    Inputs :
        output_dims -- dimensions of the output vector
        s -- degree to raise transformed features
        scale -- amount to scale the standard deviation
        weight_optim -- optimizer for weight params; None for random features
        bias_optim -- optimizer for bias param; None for random features
    """
    def __init__(
            self,
            model,
            input_record,
            output_dims,
            s=0,
            scale=None,
            weight_optim=None,
            bias_optim=None,
            name='arc_cosine_feature_map',
            **kwargs):

        super(ArcCosineFeatureMap, self).__init__(model, name, input_record,
                                                  **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"

        self.params = []
        self.model = model
        self.name = name

        self.input_dims = input_record.field_type().shape[0]
        assert self.input_dims >= 1, "Expected input dimensions >= 1, got %s" \
                                     % self.input_dims

        self.output_schema = schema.Scalar(
            (np.float32, (output_dims, )),
            model.net.NextScopedBlob(name + '_output')
        )

        self.output_dims = output_dims
        assert self.output_dims >= 1, "Expected output dimensions >= 1, got %s" \
                                      % self.output_dims
        self.s = s
        assert (self.s >= 0), "Expected s >= 0, got %s" % self.s
        assert isinstance(self.s, int), "Expected s to be type int, got type %s" \
                                        % type(self.s)

        if scale:
            assert (scale > 0.0), "Expected scale > 0, got %s" % scale
            self.stddev = 1 / scale
        else:
            self.stddev = np.sqrt(1.0 / self.input_dims)

        # Initialize train_init_net parameters
        # Random Parameters
        self.random_w = self.model.net.NextScopedBlob(self.name + "_random_w")
        self.random_b = self.model.net.NextScopedBlob(self.name + "_random_b")
        self.params += self._initialize_params(self.random_w,
                                               self.random_b,
                                               w_optim=weight_optim,
                                               b_optim=bias_optim)

    def _initialize_params(self, w_blob, b_blob, w_optim=None, b_optim=None):
        """
        Initializes the Layer Parameters for weight and bias terms for features

        Inputs :
            w_blob -- blob to contain w values
            b_blob -- blob to contain b values
            w_optim -- optimizer to use for w; if None, then will use no optimizer
            b_optim -- optimizer to user for b; if None, then will use no optimizer
        """

        w_init = (
            'GaussianFill', {'mean': 0.0, 'std': self.stddev}
        )
        w_optim = w_optim if w_optim else self.model.NoOptim

        b_init = (
            'UniformFill', {'min': -0.5 * self.stddev, 'max': 0.5 * self.stddev}
        )
        b_optim = b_optim if b_optim else self.model.NoOptim

        w_param = LayerParameter(
            parameter=w_blob,
            initializer=core.CreateOperator(w_init[0],
                                            [],
                                            w_blob,
                                            shape=(self.output_dims, self.input_dims),
                                            **w_init[1]
                                            ),
            optimizer=w_optim)
        b_param = LayerParameter(
            parameter=b_blob,
            initializer=core.CreateOperator(b_init[0],
                                            [],
                                            b_blob,
                                            shape=[self.output_dims],
                                            **b_init[1]
                                            ),
            optimizer=b_optim)

        return [w_param, b_param]

    def _heaviside_with_power(self, net, input_features, output_blob, s):
        """
        Applies Heaviside step function and Relu / exponentiation to features
        depending on the value of s.

        Inputs:
            net -- net with operators
            input_features -- features to processes
            output_blob -- output blob reference
            s -- degree to raise the transformed features
        """
        if s == 0:
            # Apply Heaviside step function to random features
            ZEROS = self.model.global_constants['ZERO']
            bool_vec = net.GT([input_features, ZEROS],
                              net.NextScopedBlob('bool_vec'),
                              broadcast=1)
            return net.Cast(bool_vec,
                            output_blob,
                            to=core.DataType.FLOAT)
        elif s == 1:
            return net.Relu([input_features],
                            output_blob)
        else:
            relu_features = net.Relu([input_features],
                                     net.NextScopedBlob('relu_rand'))
            pow_features = net.Pow([input_features],
                                   net.NextScopedBlob('pow_rand'),
                                   exponent=float(s - 1))
            return net.Mul([relu_features, pow_features],
                           output_blob)

    def add_ops(self, net):
        input_blob = self.input_record.field_blobs()

        # Random features: wx + b
        random_features = net.FC(input_blob + [self.random_w, self.random_b],
                                 net.NextScopedBlob('random_features'))
        # Process random features
        self._heaviside_with_power(net,
                                   random_features,
                                   self.output_schema.field_blobs(),
                                   self.s)
