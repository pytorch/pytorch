# -*- coding: utf-8 -*-

import logging

import torch
from torch import nn
from torch.ao.sparsity import BaseSparsifier, WeightNormSparsifier
from torch.nn.utils.parametrize import is_parametrized

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


class ImplementedSparsifier(BaseSparsifier):
    def __init__(self, **kwargs):
        super().__init__(defaults=kwargs)

    def update_mask(self, layer, **kwargs):
        layer.mask[0] = 0


class TestBaseSparsifier(TestCase):
    def test_constructor(self):
        # Cannot instantiate the base
        self.assertRaisesRegex(TypeError, 'with abstract methods update_mask',
                               BaseSparsifier)
        # Can instantiate the model with no configs
        model = Model()
        sparsifier = ImplementedSparsifier(test=3)
        sparsifier.prepare(model, config=None)
        assert len(sparsifier.module_groups) == 2
        sparsifier.step()
        # Can instantiate the model with configs
        sparsifier = ImplementedSparsifier(test=3)
        sparsifier.prepare(model, [model.linear])
        assert len(sparsifier.module_groups) == 1
        assert sparsifier.module_groups[0]['path'] == 'linear'
        assert 'test' in sparsifier.module_groups[0]
        assert sparsifier.module_groups[0]['test'] == 3

    def test_step(self):
        model = Model()
        sparsifier = ImplementedSparsifier(test=3)
        sparsifier.enable_mask_update = True
        sparsifier.prepare(model, [model.linear])
        sparsifier.step()
        assert torch.all(model.linear.mask[0] == 0)

    def test_state_dict(self):
        model = Model()
        sparsifier0 = ImplementedSparsifier(test=3)
        sparsifier0.prepare(model, [model.linear])
        state_dict = sparsifier0.state_dict()
        sparsifier1 = ImplementedSparsifier()
        sparsifier1.prepare(model, None)
        assert sparsifier0.module_groups != sparsifier1.module_groups
        sparsifier1.load_state_dict(state_dict)
        assert sparsifier0.module_groups == sparsifier1.module_groups

    def test_mask_squash(self):
        model = Model()
        sparsifier = ImplementedSparsifier(test=3)
        sparsifier.prepare(model, [model.linear])
        assert hasattr(model.linear, 'mask')
        assert is_parametrized(model.linear, 'weight')
        assert not hasattr(model.seq[0], 'mask')
        assert not is_parametrized(model.seq[0], 'weight')

        sparsifier.squash_mask()
        assert not hasattr(model.seq[0], 'mask')
        assert not is_parametrized(model.seq[0], 'weight')
        assert not hasattr(model.linear, 'mask')
        assert not is_parametrized(model.linear, 'weight')


class TestWeightNormSparsifier(TestCase):
    def test_constructor(self):
        model = Model()
        sparsifier = WeightNormSparsifier()
        sparsifier.prepare(model, config=None)
        for g in sparsifier.module_groups:
            assert isinstance(g['module'], nn.Linear)
            # The module_groups are unordered
            assert g['path'] in ('seq.0', 'linear')

    def test_logic(self):
        model = Model()
        sparsifier = WeightNormSparsifier(sparsity_level=0.5)
        sparsifier.prepare(model, config=[model.linear])
        sparsifier.enable_mask_update = True
        sparsifier.step()
        self.assertAlmostEqual(model.linear.mask.mean().item(), 0.5, places=2)
