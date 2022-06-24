# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import logging
from torch.ao.sparsity import BaseDataScheduler, DataNormSparsifier

from torch.testing._internal.common_utils import TestCase
from torch import nn
import torch

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class ImplementedDataScheduler(BaseDataScheduler):
    def __init__(self, sparsifier, sparsifier_hyperparam, last_epoch=-1, verbose=False):
        super().__init__(sparsifier, sparsifier_hyperparam, last_epoch, verbose)

    def get_schedule_param(self):
        pass


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
        return data_list, data_with_config, defaults

    def _get_sparsifier(self, data_list, data_with_config, defaults):
        sparsifier = DataNormSparsifier(data_list, **defaults)
        for data_config_dict in data_with_config:
            name, data, config = data_config_dict['name'], data_config_dict['data'], data_config_dict['config']
            sparsifier.add_data(name=name, data=data, **config)
        return sparsifier

    def _get_scheduler(self, sparsifier, schedule_param):
        scheduler = ImplementedDataScheduler(sparsifier, schedule_param)
        return scheduler

    def _get_schedule_param(self):
        return 'sparsity_level'

    def test_constructor(self):
        data_list, data_with_config, defaults = self._get_data()
        sparsifier = self._get_sparsifier(data_list, data_with_config, defaults)
        schedule_param = self._get_schedule_param()
        scheduler = self._get_scheduler(sparsifier, schedule_param)

        assert scheduler.data_sparsifier == sparsifier
        assert scheduler._step_count == 0  # 0 now as step() is not yet implemented

        for name, config in sparsifier.data_groups.items():
            assert scheduler.base_param[name] == config.get(schedule_param, None)
