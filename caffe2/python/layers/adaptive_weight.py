# @package adaptive_weight
# Module caffe2.fb.python.layers.adaptive_weight
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer
import numpy as np
'''
Implementation of adaptive weighting: https://arxiv.org/pdf/1705.07115.pdf
'''


class AdaptiveWeight(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        name='adaptive_weight',
        optimizer=None,
        weights=None,
        **kwargs
    ):
        super(AdaptiveWeight,
              self).__init__(model, name, input_record, **kwargs)
        self.output_schema = schema.Scalar(
            np.float32, self.get_next_blob_reference('adaptive_weight')
        )
        self.data = self.input_record.field_blobs()
        self.num = len(self.data)
        # mu_i = log(sigma_i^2)
        if weights is None:
            # mu_i is set such that all initial weights are 1. / num
            initializer = ('ConstantFill', {'value': np.log(self.num / 2.)})
        else:
            assert len(weights) == self.num
            weights = np.array(weights).astype(np.float32)
            values = np.log(1. / 2. / weights)
            initializer = (
                'GivenTensorFill', {
                    'values': values,
                    'dtype': core.DataType.FLOAT
                }
            )

        self.mu = self.create_param(
            param_name='mu',
            shape=[self.num],
            initializer=initializer,
            optimizer=optimizer,
        )

    def concat_data(self, net):
        reshaped = [
            net.NextScopedBlob('reshaped_data_%d' % i) for i in range(self.num)
        ]
        # coerce shape for single real values
        for i in range(self.num):
            net.Reshape(
                [self.data[i]],
                [reshaped[i], net.NextScopedBlob('new_shape_%d' % i)],
                shape=[1]
            )
        concated = net.NextScopedBlob('concated_data')
        net.Concat(
            reshaped, [concated, net.NextScopedBlob('concated_new_shape')],
            axis=0
        )
        return concated

    def compute_adaptive_sum(self, x, net):
        mu_exp = net.NextScopedBlob('mu_exp')
        net.Exp(self.mu, mu_exp)
        mu_exp_double = net.NextScopedBlob('mu_exp_double')
        net.Scale(mu_exp, mu_exp_double, scale=2.0)
        weighted_x = net.NextScopedBlob('weighted_x')
        net.Div([x, mu_exp_double], weighted_x)
        weighted_elements = net.NextScopedBlob('weighted_elements')
        net.Add([weighted_x, self.mu], weighted_elements)
        net.SumElements(weighted_elements, self.output_schema())

    def add_ops(self, net):
        data = self.concat_data(net)
        self.compute_adaptive_sum(data, net)
