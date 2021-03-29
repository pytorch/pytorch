




from caffe2.python import schema
from caffe2.python.layers.layers import ModelLayer
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
        weight_init -- initialization distribution for weight parameter
        bias_init -- initialization distribution for bias pararmeter
        weight_optim -- optimizer for weight params; None for random features
        bias_optim -- optimizer for bias param; None for random features
        set_weight_as_global_constant -- if True, initialized random parameters
                                         will be constant across all distributed
                                         instances of the layer
        initialize_output_schema -- if True, initialize output schema as Scalar
                                    from Arc Cosine; else output schema is None
    """
    def __init__(
            self,
            model,
            input_record,
            output_dims,
            s=1,
            scale=1.0,
            weight_init=None,
            bias_init=None,
            weight_optim=None,
            bias_optim=None,
            set_weight_as_global_constant=False,
            initialize_output_schema=True,
            name='arc_cosine_feature_map',
            **kwargs):

        super(ArcCosineFeatureMap, self).__init__(model, name, input_record,
                                                  **kwargs)
        assert isinstance(input_record, schema.Scalar), "Incorrect input type"
        self.params = []
        self.model = model
        self.set_weight_as_global_constant = set_weight_as_global_constant

        self.input_dims = input_record.field_type().shape[0]
        assert self.input_dims >= 1, "Expected input dimensions >= 1, got %s" \
                                     % self.input_dims

        if initialize_output_schema:
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

        assert (scale > 0.0), "Expected scale > 0, got %s" % scale
        self.stddev = scale * np.sqrt(1.0 / self.input_dims)

        # Initialize train_init_net parameters
        # Random Parameters
        if set_weight_as_global_constant:
            w_init = np.random.normal(scale=self.stddev,
                                      size=(self.output_dims, self.input_dims))
            b_init = np.random.uniform(low=-0.5 * self.stddev,
                                       high=0.5 * self.stddev,
                                       size=self.output_dims)
            self.random_w = self.model.add_global_constant(
                name=self.name + "_fixed_rand_W",
                array=w_init
            )
            self.random_b = self.model.add_global_constant(
                name=self.name + "_fixed_rand_b",
                array=b_init
            )
        else:
            (self.random_w, self.random_b) = self._initialize_params(
                'random_w',
                'random_b',
                w_init=weight_init,
                b_init=bias_init,
                w_optim=weight_optim,
                b_optim=bias_optim
            )

    def _initialize_params(self, w_name, b_name, w_init=None, b_init=None,
                           w_optim=None, b_optim=None):
        """
        Initializes the Layer Parameters for weight and bias terms for features

        Inputs :
            w_blob -- blob to contain w values
            b_blob -- blob to contain b values
            w_init -- initialization distribution for weight parameter
            b_init -- initialization distribution for bias parameter
            w_optim -- optimizer to use for w; if None, then will use no optimizer
            b_optim -- optimizer to user for b; if None, then will use no optimizer
        """

        w_init = w_init if w_init else (
            'GaussianFill', {'mean': 0.0, 'std': self.stddev}
        )
        w_optim = w_optim if w_optim else self.model.NoOptim

        b_init = b_init if b_init else (
            'UniformFill', {'min': -0.5 * self.stddev, 'max': 0.5 * self.stddev}
        )
        b_optim = b_optim if b_optim else self.model.NoOptim

        w_param = self.create_param(param_name=w_name,
                                    shape=(self.output_dims, self.input_dims),
                                    initializer=w_init,
                                    optimizer=w_optim)

        b_param = self.create_param(param_name=b_name,
                                    shape=[self.output_dims],
                                    initializer=b_init,
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
            softsign_features = net.Softsign([input_features],
                                             net.NextScopedBlob('softsign'))
            return net.Relu(softsign_features, output_blob)
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
