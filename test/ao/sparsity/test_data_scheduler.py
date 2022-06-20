# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import logging
from torch.ao.sparsity import BaseDataScheduler, DataNormSparsifier
import warnings
from torch.testing._internal.common_utils import TestCase
from torch import nn
import torch
from typing import Tuple
import copy

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class ImplementedDataScheduler(BaseDataScheduler):
    def __init__(self, sparsifier, sparsifier_hyperparam, last_epoch=-1, verbose=False):
        super().__init__(sparsifier, sparsifier_hyperparam, last_epoch, verbose)

    def get_schedule_param(self):
        if self.last_epoch > 0:
            return {name: config['sparsity_level'] * 0.5
                    for name, config in self.data_sparsifier.data_groups.items()}
        else:
            return self.base_param


class TestBaseDataScheduler(TestCase):
    def _get_data(self):
        tensor1, param1, emb1 = torch.randn(5, 5), nn.Parameter(torch.randn(10, 10)), nn.Embedding(50, 5)
        data_list = [
            ('tensor1', tensor1), ('param1', param1), ('emb1', emb1)
        ]
        defaults = {
            'sparsity_level': 0.7,
            'sparse_block_shape': (1, 4),
            'zeros_per_block': 2
        }
        data_with_config = [
            {
                'name': 'tensor2', 'data': torch.randn(4, 4),
                'config': {'sparsity_level': 0.3}
            }
        ]
        return data_list, defaults, data_with_config

    def _get_sparsifier(self, return_data=False):
        data_list, defaults, data_with_config = self._get_data()
        sparsifier = DataNormSparsifier(data_list, **defaults)
        for data_config_dict in data_with_config:
            name, data, config = data_config_dict['name'], data_config_dict['data'], data_config_dict['config']
            sparsifier.add_data(name=name, data=data, **config)
        if return_data:
            return sparsifier, data_list, defaults, data_with_config
        return sparsifier

    def _get_scheduler(self, sparsifier):
        schedule_param = self._get_scheduler_param()
        scheduler = ImplementedDataScheduler(sparsifier, schedule_param)
        return scheduler

    def _get_scheduler_param(self):
        return 'sparsity_level'

    def _get_name_data_config(self, some_data, defaults):
        config = copy.deepcopy(defaults)
        if isinstance(some_data, Tuple):
            # dealing with data_list
            name, data = some_data
        else:
            # dealing with data_with_config
            name, data, new_config = some_data['name'], some_data['data'], some_data['config']
            config.update(new_config)
        return name, data, config

    def test_constructor(self):
        """Checks if the warning is thrown if the scheduler step is called
        before the sparsifier step"""
        sparsifier = self._get_sparsifier()
        scheduler = self._get_scheduler(sparsifier)
        scheduler_param = self._get_scheduler_param()

        assert scheduler.data_sparsifier == sparsifier
        assert scheduler._step_count == 1

        for name, config in sparsifier.data_groups.items():
            assert scheduler.base_param[name] == config.get(scheduler_param, None)

    def test_order_of_steps(self):
        sparsifier = self._get_sparsifier()
        scheduler = self._get_scheduler(sparsifier)

        # Sparsifier step is not called
        with self.assertWarns(UserWarning):
            scheduler.step()

        # Correct order has no warnings
        # Note: This will trigger if other warnings are present.
        with warnings.catch_warnings(record=True) as w:
            sparsifier.step()
            scheduler.step()
            # Make sure there is no warning related to the base_data_scheduler
            for warning in w:
                fname = warning.filename
                fname = '/'.join(fname.split('/')[-5:])
                assert fname != 'torch/ao/sparsity/experimental/scheduler/data_scheduler/base_data_scheduler.py'

    def test_step(self):
        sparsifier, data_list, defaults, data_with_config = self._get_sparsifier(return_data=True)
        scheduler = self._get_scheduler(sparsifier)
        scheduler_param = self._get_scheduler_param()

        all_data = data_list + data_with_config

        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data, defaults)
            assert sparsifier.data_groups[name][scheduler_param] == config[scheduler_param]

        sparsifier.step()
        scheduler.step()

        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data, defaults)
            assert sparsifier.data_groups[name][scheduler_param] == config[scheduler_param] * 0.5

    def test_state_dict(self):
        sparsifier, data_list, defaults, data_with_config = self._get_sparsifier(return_data=True)
        scheduler1 = self._get_scheduler(sparsifier)

        sparsifier.step()
        scheduler1.step()

        scheduler2 = self._get_scheduler(sparsifier)
        all_data = data_list + data_with_config
        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data, defaults)
            assert scheduler1.base_param[name] != scheduler2.base_param[name]
            assert scheduler1._last_param[name] == scheduler2.base_param[name]

        scheduler1_state = scheduler1.state_dict()
        scheduler2.load_state_dict(scheduler1_state)

        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data, defaults)
            assert scheduler1.base_param[name] == scheduler2.base_param[name]
            assert scheduler1._last_param[name] == scheduler2._last_param[name]
