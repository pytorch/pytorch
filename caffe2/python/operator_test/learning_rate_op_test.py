from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu

from hypothesis import given

import numpy as np
import math


class TestLearningRate(hu.HypothesisTestCase):
    @given(**hu.gcs_cpu_only)
    def test_alter_learning_rate_op(self, gc, dc):
        iter = np.random.randint(low=1, high=1e5, size=1)
        active_period = int(np.random.randint(low=1, high=1e3, size=1))
        inactive_period = int(np.random.randint(low=1, high=1e3, size=1))
        base_lr = float(np.random.random(1))

        def ref(iter):
            iter = float(iter)
            reminder = iter % (active_period + inactive_period)
            if reminder < active_period:
                return (np.array(base_lr), )
            else:
                return (np.array(0.), )

        op = core.CreateOperator(
            'LearningRate',
            'iter',
            'lr',
            policy="alter",
            active_first=True,
            base_lr=base_lr,
            active_period=active_period,
            inactive_period=inactive_period
        )

        self.assertReferenceChecks(gc, op, [iter], ref)

    @given(**hu.gcs_cpu_only)
    def test_hill_learning_rate_op(self, gc, dc):
        iter = np.random.randint(low=1, high=1e5, size=1)

        num_iter = int(np.random.randint(low=1e2, high=1e3, size=1))
        start_multiplier = 1e-4
        gamma = 1.0
        power = 0.5
        end_multiplier = 1e-2
        base_lr = float(np.random.random(1))

        def ref(iter):
            iter = float(iter)
            if iter < num_iter:
                lr = start_multiplier + (
                    1.0 - start_multiplier
                ) * iter / num_iter
            else:
                iter -= num_iter
                lr = math.pow(1.0 + gamma * iter, -power)
                lr = max(lr, end_multiplier)
            return (np.array(base_lr * lr), )

        op = core.CreateOperator(
            'LearningRate',
            'data',
            'out',
            policy="hill",
            base_lr=base_lr,
            num_iter=num_iter,
            start_multiplier=start_multiplier,
            gamma=gamma,
            power=power,
            end_multiplier=end_multiplier,
        )
        self.assertReferenceChecks(gc, op, [iter], ref)


if __name__ == "__main__":
    import unittest
    unittest.main()
