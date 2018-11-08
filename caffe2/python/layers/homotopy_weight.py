# @package homotopy_weight
# Module caffe2.fb.python.layers.homotopy_weight

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, schema
from caffe2.python.layers.layers import ModelLayer
import numpy as np
import logging
logger = logging.getLogger(__name__)
'''
Homotopy Weighting between two weights x, y by doing:
    alpha x + beta y
where alpha is a decreasing scalar parameter ranging from [min, max] (default,
[0, 1]), and alpha + beta = max + min, which means that beta is increasing in
the range [min, max];

Homotopy methods first solves an "easy" problem (one to which the solution is
well known), and is gradually transformed into the target problem
'''


class HomotopyWeight(ModelLayer):
    def __init__(
        self,
        model,
        input_record,
        name='homotopy_weight',
        min_weight=0.,
        max_weight=1.,
        half_life=1e6,
        quad_life=3e6,
        atomic_iter=None,
        **kwargs
    ):
        super(HomotopyWeight,
              self).__init__(model, name, input_record, **kwargs)
        self.output_schema = schema.Scalar(
            np.float32, self.get_next_blob_reference('homotopy_weight')
        )
        data = self.input_record.field_blobs()
        assert len(data) == 2
        self.x = data[0]
        self.y = data[1]
        # TODO: currently model building does not have access to iter counter or
        # learning rate; it's added at optimization time;
        self.use_external_iter = (atomic_iter is not None)
        self.atomic_iter = (
            atomic_iter if self.use_external_iter else self.create_atomic_iter()
        )
        # to map lr to [min, max]; alpha = scale * lr + offset
        assert max_weight > min_weight
        self.scale = float(max_weight - min_weight)
        self.offset = self.model.add_global_constant(
            '%s_offset_1dfloat' % self.name, float(min_weight)
        )
        self.gamma, self.power = self.solve_inv_lr_params(half_life, quad_life)

    def solve_inv_lr_params(self, half_life, quad_life):
        # ensure that the gamma, power is solvable
        assert half_life > 0
        # convex monotonically decreasing
        assert quad_life > 2 * half_life
        t = float(quad_life) / float(half_life)
        x = t * (1.0 + np.sqrt(2.0)) / 2.0 - np.sqrt(2.0)
        gamma = (x - 1.0) / float(half_life)
        power = np.log(2.0) / np.log(x)
        logger.info(
            'homotopy_weighting: found lr param: gamma=%g, power=%g' %
            (gamma, power)
        )
        return gamma, power

    def create_atomic_iter(self):
        self.mutex = self.create_param(
            param_name=('%s_mutex' % self.name),
            shape=None,
            initializer=('CreateMutex', ),
            optimizer=self.model.NoOptim,
        )
        self.atomic_iter = self.create_param(
            param_name=('%s_atomic_iter' % self.name),
            shape=[1],
            initializer=(
                'ConstantFill', {
                    'value': 0,
                    'dtype': core.DataType.INT64
                }
            ),
            optimizer=self.model.NoOptim,
        )
        return self.atomic_iter

    def update_weight(self, net):
        alpha = net.NextScopedBlob('alpha')
        beta = net.NextScopedBlob('beta')
        lr = net.NextScopedBlob('lr')
        comp_lr = net.NextScopedBlob('complementary_lr')
        scaled_lr = net.NextScopedBlob('scaled_lr')
        scaled_comp_lr = net.NextScopedBlob('scaled_complementary_lr')
        if not self.use_external_iter:
            net.AtomicIter([self.mutex, self.atomic_iter], [self.atomic_iter])
        net.LearningRate(
            [self.atomic_iter],
            [lr],
            policy='inv',
            gamma=self.gamma,
            power=self.power,
            base_lr=1.0,
        )
        net.Sub([self.model.global_constants['ONE'], lr], [comp_lr])
        net.Scale([lr], [scaled_lr], scale=self.scale)
        net.Scale([comp_lr], [scaled_comp_lr], scale=self.scale)
        net.Add([scaled_lr, self.offset], [alpha])
        net.Add([scaled_comp_lr, self.offset], [beta])
        return alpha, beta

    def add_ops(self, net):
        alpha, beta = self.update_weight(net)
        # alpha x + beta y
        net.WeightedSum([self.x, alpha, self.y, beta], self.output_schema())
