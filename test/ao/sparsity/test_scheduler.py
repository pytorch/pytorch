# Owner(s): ["module: unknown"]

from torch import nn

from torch.ao.pruning import WeightNormSparsifier
from torch.ao.pruning import BaseScheduler, LambdaSL, CubicSL

from torch.testing._internal.common_utils import TestCase

import warnings

class ImplementedScheduler(BaseScheduler):
    def get_sl(self):
        if self.last_epoch > 0:
            return [group['sparsity_level'] * 0.5
                    for group in self.sparsifier.groups]
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
        assert scheduler.base_sl == [sparsifier.groups[0]['sparsity_level']]

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
        assert sparsifier.groups[0]['sparsity_level'] == 0.5
        scheduler = ImplementedScheduler(sparsifier)
        assert sparsifier.groups[0]['sparsity_level'] == 0.5

        sparsifier.step()
        scheduler.step()
        assert sparsifier.groups[0]['sparsity_level'] == 0.25

    def test_lambda_scheduler(self):
        model = nn.Sequential(
            nn.Linear(16, 16)
        )
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        assert sparsifier.groups[0]['sparsity_level'] == 0.5
        scheduler = LambdaSL(sparsifier, lambda epoch: epoch * 10)
        assert sparsifier.groups[0]['sparsity_level'] == 0.0  # Epoch 0
        scheduler.step()
        assert sparsifier.groups[0]['sparsity_level'] == 5.0  # Epoch 1


class TestCubicScheduler(TestCase):
    def setUp(self):
        self.model_sparse_config = [
            {'tensor_fqn': '0.weight', 'sparsity_level': 0.8},
            {'tensor_fqn': '2.weight', 'sparsity_level': 0.4},
        ]
        self.sorted_sparse_levels = [conf['sparsity_level'] for conf in self.model_sparse_config]
        self.initial_sparsity = 0.1
        self.initial_step = 3

    def _make_model(self, **kwargs):
        model = nn.Sequential(
            nn.Linear(13, 17),
            nn.Dropout(0.5),
            nn.Linear(17, 3),
        )
        return model

    def _make_scheduler(self, model, **kwargs):
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=self.model_sparse_config)

        scheduler_args = {
            'init_sl': self.initial_sparsity,
            'init_t': self.initial_step,
        }
        scheduler_args.update(kwargs)

        scheduler = CubicSL(sparsifier, **scheduler_args)
        return sparsifier, scheduler

    @staticmethod
    def _get_sparsity_levels(sparsifier, precision=32):
        r"""Gets the current levels of sparsity in a sparsifier."""
        return [round(group['sparsity_level'], precision) for group in sparsifier.groups]

    def test_constructor(self):
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(model=model, initially_zero=True)
        self.assertIs(
            scheduler.sparsifier, sparsifier,
            msg="Sparsifier is not properly attached")
        self.assertEqual(
            scheduler._step_count, 1,
            msg="Scheduler is initialized with incorrect step count")
        self.assertEqual(
            scheduler.base_sl, self.sorted_sparse_levels,
            msg="Scheduler did not store the target sparsity levels correctly")

        # Value before t_0 is 0
        self.assertEqual(
            self._get_sparsity_levels(sparsifier), scheduler._make_sure_a_list(0.0),
            msg="Sparsifier is not reset correctly after attaching to the Scheduler")

        # Value before t_0 is s_0
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(model=model, initially_zero=False)
        self.assertEqual(
            self._get_sparsity_levels(sparsifier),
            scheduler._make_sure_a_list(self.initial_sparsity),
            msg="Sparsifier is not reset correctly after attaching to the Scheduler")

    def test_step(self):
        # For n=5, dt=2, there will be totally 10 steps between s_0 and s_f, starting from t_0
        model = self._make_model()
        sparsifier, scheduler = self._make_scheduler(
            model=model, initially_zero=True, init_t=3, delta_t=2, total_t=5)

        scheduler.step()
        scheduler.step()
        self.assertEqual(scheduler._step_count, 3, msg="Scheduler step_count is expected to increment")
        # Value before t_0 is supposed to be 0
        self.assertEqual(
            self._get_sparsity_levels(sparsifier), scheduler._make_sure_a_list(0.0),
            msg="Scheduler step updating the sparsity level before t_0")

        scheduler.step()  # Step = 3  =>  sparsity = initial_sparsity
        self.assertEqual(
            self._get_sparsity_levels(sparsifier), scheduler._make_sure_a_list(self.initial_sparsity),
            msg="Sparsifier is not reset to initial sparsity at the first step")

        scheduler.step()  # Step = 4  =>  sparsity ~ [0.3, 0.2]
        self.assertEqual(
            self._get_sparsity_levels(sparsifier, 1), [0.3, 0.2],
            msg="Sparsity level is not set correctly after the first step")

        current_step = scheduler._step_count - scheduler.init_t[0] - 1
        more_steps_needed = scheduler.delta_t[0] * scheduler.total_t[0] - current_step
        for _ in range(more_steps_needed):  # More steps needed to final sparsity level
            scheduler.step()
        self.assertEqual(
            self._get_sparsity_levels(sparsifier), self.sorted_sparse_levels,
            msg="Sparsity level is not reaching the target level afer delta_t * n steps ")
