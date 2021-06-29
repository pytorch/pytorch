# -*- coding: utf-8 -*-

import logging

from torch import nn
from torch.ao.sparsity import BasePruner, PruningParametrization
from torch.nn.utils import parametrize

from torch.testing._internal.common_utils import TestCase

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 16)
        )
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class ImplementedPruner(BasePruner):
    def update_mask(self):
        pass


class TestBasePruner(TestCase):
    def test_constructor(self):
        # Cannot instantiate the base
        self.assertRaisesRegex(TypeError, 'with abstract methods update_mask',
                               BasePruner)
        # Can instantiate the model with no configs
        model = Model()
        pruner = ImplementedPruner(model, None, None)
        assert len(pruner.module_groups) == 2
        pruner.step()
        # Can instantiate the model with configs
        pruner = ImplementedPruner(model, [model.linear], {'test': 3})
        assert len(pruner.module_groups) == 1
        assert pruner.module_groups[0]['path'] == 'linear'
        assert 'test' in pruner.module_groups[0]
        assert pruner.module_groups[0]['test'] == 3

    def test_prepare(self):
        model = Model()
        pruner = ImplementedPruner(model, None, None)
        pruner.prepare()
        for g in pruner.module_groups:
            module = g['module']
            # Check mask exists
            assert hasattr(module, 'mask')
            # Check parametrization exists and is correct
            assert parametrize.is_parametrized(module)
            assert hasattr(module, "parametrizations")
            assert type(module.parametrizations.weight[0]) == PruningParametrization

    def test_convert(self):
        model = Model()
        pruner = ImplementedPruner(model, None, None)
        pruner.prepare()
        pruner.convert()
        for g in pruner.module_groups:
            module = g['module']
            assert not hasattr(module, "parametrizations")
            assert not hasattr(module, 'mask')
