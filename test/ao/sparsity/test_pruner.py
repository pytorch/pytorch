# -*- coding: utf-8 -*-

import logging

import torch
from torch import nn
from torch.ao.sparsity import BasePruner, PruningParametrization
from torch.nn.utils import parametrize

from torch.testing._internal.common_utils import TestCase

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=False)
        )
        self.linear = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=True)
        )
        self.linear = nn.Linear(16, 16, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class MultipleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=False),
            nn.ReLU(),
            nn.Linear(5, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 6, bias=False)
        )
        self.linear = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class MultipleModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 6, bias=True)
        )
        self.linear = nn.Linear(6, 4, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class MultipleModelMixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 6, bias=True)
        )
        self.linear = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class SimplePruner(BasePruner):
    def update_mask(self, layer, **kwargs):
        layer.parametrizations.weight[0].pruned_outputs.add(1)


class MultiplePruner(BasePruner):
    def update_mask(self, layer, **kwargs):
        layer.parametrizations.weight[0].pruned_outputs.update([1, 2])


class TestBasePruner(TestCase):
    def test_constructor(self):
        # Cannot instantiate the base
        self.assertRaisesRegex(TypeError, 'with abstract methods update_mask',
                               BasePruner)
        # Can instantiate the model with no configs
        model = Model()
        pruner = SimplePruner(model, None, None)
        assert len(pruner.module_groups) == 2
        pruner.step()
        # Can instantiate the model with configs
        pruner = SimplePruner(model, [model.linear], {'test': 3})
        assert len(pruner.module_groups) == 1
        assert pruner.module_groups[0]['path'] == 'linear'
        assert 'test' in pruner.module_groups[0]
        assert pruner.module_groups[0]['test'] == 3

    def test_prepare(self):
        model = Model()
        x = torch.ones(128, 16)
        pruner = SimplePruner(model, None, None)
        pruner.prepare()
        for g in pruner.module_groups:
            module = g['module']
            # Check mask exists
            assert hasattr(module, 'mask')
            # Check parametrization exists and is correct
            assert parametrize.is_parametrized(module)
            assert hasattr(module, "parametrizations")
            # Assume that this is the 1st/only parametrization
            assert type(module.parametrizations.weight[0]) == PruningParametrization
        assert model(x).shape == (128, 16)

    def test_prepare_bias(self):
        model = ModelB()
        x = torch.ones(128, 16)
        pruner = SimplePruner(model, None, None)
        pruner.prepare()
        for g in pruner.module_groups:
            module = g['module']
            # Check mask exists
            assert hasattr(module, 'mask')
            # Check parametrization exists and is correct
            assert parametrize.is_parametrized(module)
            assert hasattr(module, "parametrizations")
            # Assume that this is the 1st/only parametrization
            assert type(module.parametrizations.weight[0]) == PruningParametrization
        assert model(x).shape == (128, 16)

    def test_convert(self):
        model = Model()
        x = torch.ones(128, 16)
        pruner = SimplePruner(model, None, None)
        pruner.prepare()
        pruner.convert()
        for g in pruner.module_groups:
            module = g['module']
            assert not hasattr(module, "parametrizations")
            assert not hasattr(module, 'mask')
        assert model(x).shape == (128, 16)

    def test_convert_bias(self):
        model = ModelB()
        x = torch.ones(128, 16)
        pruner = SimplePruner(model, None, None)
        pruner.prepare()
        pruner.convert()
        for g in pruner.module_groups:
            module = g['module']
            assert not hasattr(module, "parametrizations")
            assert not hasattr(module, 'mask')
        assert model(x).shape == (128, 16)

    def test_step(self):
        model = Model()
        x = torch.ones(16, 16)
        pruner = SimplePruner(model, None, None)
        pruner.prepare()
        pruner.enable_mask_update = True
        for g in pruner.module_groups:
            # Before step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set()
        pruner.step()
        for g in pruner.module_groups:
            # After step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set({1})
            assert not (False in (model(x)[:, 1] == 0))

        model = MultipleModel()
        x = torch.ones(7, 7)
        pruner = MultiplePruner(model, None, None)
        pruner.prepare()
        pruner.enable_mask_update = True
        for g in pruner.module_groups:
            # Before step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set()
        pruner.step()
        for g in pruner.module_groups:
            # After step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set({1, 2})
            assert not (False in (model(x)[:, 1] == 0))
            assert not (False in (model(x)[:, 2] == 0))

    def test_step_bias(self):
        model = ModelB()
        x = torch.ones(16, 16)
        pruner = SimplePruner(model, None, None)
        pruner.prepare()
        pruner.enable_mask_update = True
        for g in pruner.module_groups:
            # Before step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set()
        pruner.step()
        for g in pruner.module_groups:
            # After step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set({1})

        model = MultipleModelB()
        x = torch.ones(7, 7)
        pruner = MultiplePruner(model, None, None)
        pruner.prepare()
        pruner.enable_mask_update = True
        for g in pruner.module_groups:
            # Before step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set()
        pruner.step()
        for g in pruner.module_groups:
            # After step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set({1, 2})

        model = MultipleModelMixed()
        x = torch.ones(7, 7)
        pruner = MultiplePruner(model, None, None)
        pruner.prepare()
        pruner.enable_mask_update = True
        for g in pruner.module_groups:
            # Before step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set()
        pruner.step()
        for g in pruner.module_groups:
            # After step
            module = g['module']
            assert module.parametrizations.weight[0].pruned_outputs == set({1, 2})
