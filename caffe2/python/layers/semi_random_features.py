from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import schema
from caffe2.python.layers.arc_cosine_feature_map import ArcCosineFeatureMap
import numpy as np


class SemiRandomFeatures(ArcCosineFeatureMap):
    """
    Implementation of the semi-random kernel feature map.

    Applies H(x_rand) * x_rand^s * x_learned, where
        H is the Heaviside step function,
        x_rand is the input after applying FC with randomized parameters,
        and x_learned is the input after applying FC with learnable parameters.

    If using multilayer model with semi-random layers, then input and output records
    should have a 'full' and 'random' Scalar. The random Scalar will be passed as
    input to process the random features.

    For more information, see the original paper:
        https://arxiv.org/pdf/1702.08882.pdf

    Inputs :
        output_dims -- dimensions of the output vector
        s -- if s == 0, will obtain linear semi-random features;
             else if s == 1, will obtain squared semi-random features;
             else s >= 2, will obtain higher order semi-random features
        scale_random -- amount to scale the standard deviation
                        (for random parameter initialization when weight_init or
                        bias_init hasn't been specified)
        scale_learned -- amount to scale the standard deviation
                        (for learned parameter initialization when weight_init or
                        bias_init hasn't been specified)

        weight_init_random -- initialization distribution for random weight parameter
                              (if None, will use Gaussian distribution)
        bias_init_random -- initialization distribution for random bias pararmeter
                            (if None, will use Uniform distribution)
        weight_init_learned -- initialization distribution for learned weight parameter
                               (if None, will use Gaussian distribution)
        bias_init_learned -- initialization distribution for learned bias pararmeter
                             (if None, will use Uniform distribution)
        weight_optim -- optimizer for weight params for learned features
        bias_optim -- optimizer for bias param for learned features

        set_weight_as_global_constant -- if True, initialized random parameters
                                         will be constant across all distributed
                                         instances of the layer
    """
    def __init__(
            self,
            model,
            input_record,
            output_dims,
            s=1,
            scale_random=1.0,
            scale_learned=1.0,
            weight_init_random=None,
            bias_init_random=None,
            weight_init_learned=None,
            bias_init_learned=None,
            weight_optim=None,
            bias_optim=None,
            set_weight_as_global_constant=False,
            name='semi_random_features',
            **kwargs):

        if isinstance(input_record, schema.Struct):
            schema.is_schema_subset(
                schema.Struct(
                    ('full', schema.Scalar()),
                    ('random', schema.Scalar()),
                ),
                input_record
            )
            self.input_record_full = input_record.full
            self.input_record_random = input_record.random

        elif isinstance(input_record, schema.Scalar):
            self.input_record_full = input_record
            self.input_record_random = input_record

        super(SemiRandomFeatures, self).__init__(
            model,
            self.input_record_full,
            output_dims,
            s=s,
            scale=scale_random,       # To initialize the random parameters
            weight_init=weight_init_random,
            bias_init=bias_init_random,
            weight_optim=None,
            bias_optim=None,
            set_weight_as_global_constant=set_weight_as_global_constant,
            initialize_output_schema=False,
            name=name,
            **kwargs)

        self.output_schema = schema.Struct(
            ('full', schema.Scalar(
                (np.float32, output_dims),
                model.net.NextScopedBlob(name + '_full_output')
            ),),
            ('random', schema.Scalar(
                (np.float32, output_dims),
                model.net.NextScopedBlob(name + '_random_output')
            ),),
        )

        # To initialize the learnable parameters
        assert (scale_learned > 0.0), \
            "Expected scale (learned) > 0, got %s" % scale_learned
        self.stddev = scale_learned * np.sqrt(1.0 / self.input_dims)

        # Learned Parameters
        (self.learned_w, self.learned_b) = self._initialize_params(
            'learned_w',
            'learned_b',
            w_init=weight_init_learned,
            b_init=bias_init_learned,
            w_optim=weight_optim,
            b_optim=bias_optim
        )

    def add_ops(self, net):
        # Learned features: wx + b
        learned_features = net.FC(self.input_record_full.field_blobs() +
                                  [self.learned_w, self.learned_b],
                                  net.NextScopedBlob('learned_features'))
        # Random features: wx + b
        random_features = net.FC(self.input_record_random.field_blobs() +
                                 [self.random_w, self.random_b],
                                 net.NextScopedBlob('random_features'))
        processed_random_features = self._heaviside_with_power(
            net,
            random_features,
            self.output_schema.random.field_blobs(),
            self.s
        )
        net.Mul([processed_random_features, learned_features],
                self.output_schema.full.field_blobs())
