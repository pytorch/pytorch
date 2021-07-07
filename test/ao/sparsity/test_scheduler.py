# -*- coding: utf-8 -*-
from torch import nn
from torch.ao.sparsity import WeightNormSparsifier
from torch.ao.sparsity import BaseScheduler, LambdaSL

from torch.testing._internal.common_utils import TestCase

import warnings

class ImplementedScheduler(BaseScheduler):
    def get_sl(self):
        if self.last_epoch > 0:
            return [group['sparsity_level'] * 0.5
                    for group in self.sparsifier.module_groups]
        else:
            return list(self.base_sl)


class TestScheduler(TestCase):
    def test_constructor(self):
        model = nn.Sequential(
            nn.Linear(16, 16)
        )
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        scheduler = ImplementedScheduler(sparsifier)

        assert scheduler.sparsifier is sparsifier
        assert scheduler._step_count == 1
        assert scheduler.base_sl == [sparsifier.module_groups[0]['sparsity_level']]

    def test_order_of_steps(self):
        """Checks if the warning is thrown if the scheduler step is called
        before the sparsifier step"""

        model = nn.Sequential(
            nn.Linear(16, 16)
        )
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        scheduler = ImplementedScheduler(sparsifier)

        # Sparsifier step is not called
        with self.assertWarns(UserWarning):
            scheduler.step()

        # Correct order has no warnings
        # Note: This will trigger if other warnings are present.
        with warnings.catch_warnings(record=True) as w:
            sparsifier.step()
            scheduler.step()
            # Make sure there is no warning related to the base_scheduler
            for warning in w:
                fname = warning.filename
                fname = '/'.join(fname.split('/')[-5:])
                assert fname != 'torch/ao/sparsity/scheduler/base_scheduler.py'

    def test_step(self):
        model = nn.Sequential(
            nn.Linear(16, 16)
        )
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        assert sparsifier.module_groups[0]['sparsity_level'] == 0.5
        scheduler = ImplementedScheduler(sparsifier)
        assert sparsifier.module_groups[0]['sparsity_level'] == 0.5

        sparsifier.step()
        scheduler.step()
        assert sparsifier.module_groups[0]['sparsity_level'] == 0.25

    def test_lambda_scheduler(self):
        model = nn.Sequential(
            nn.Linear(16, 16)
        )
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        assert sparsifier.module_groups[0]['sparsity_level'] == 0.5
        scheduler = LambdaSL(sparsifier, lambda epoch: epoch * 10)
        assert sparsifier.module_groups[0]['sparsity_level'] == 0.0  # Epoch 0
        scheduler.step()
        assert sparsifier.module_groups[0]['sparsity_level'] == 5.0  # Epoch 1
