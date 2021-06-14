# -*- coding: utf-8 -*-

import logging

from torch import nn
from torch.ao.sparsity import BaseSparsifier, WeightNormSparsifier

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
    def update_mask(self):
        pass


class TestBaseSparsifier(TestCase):
    def test_constructor(self):
        # Cannot instantiate the base
        self.assertRaisesRegex(TypeError, 'with abstract methods update_mask',
                               BaseSparsifier)
        # Can instantiate the model with no configs
        model = Model()
        sparsifier = ImplementedSparsifier(model, None, None)
        assert len(sparsifier.module_groups) == 2
        sparsifier.step()
        # Can instantiate the model with configs
        sparsifier = ImplementedSparsifier(model, [model.linear], {'test': 3})
        assert len(sparsifier.module_groups) == 1
        assert sparsifier.module_groups[0]['path'] == 'linear'
        assert 'test' in sparsifier.module_groups[0]
        assert sparsifier.module_groups[0]['test'] == 3

    def test_state_dict(self):
        model = Model()
        sparsifier0 = ImplementedSparsifier(model, [model.linear], {'test': 3})
        state_dict = sparsifier0.state_dict()
        sparsifier1 = ImplementedSparsifier(model, None, None)
        assert sparsifier0.module_groups != sparsifier1.module_groups
        sparsifier1.load_state_dict(state_dict)
        assert sparsifier0.module_groups == sparsifier1.module_groups


class TestWeightNormSparsifier(TestCase):
    def test_constructor(self):
        model = Model()
        sparsifier = WeightNormSparsifier(model, config=None)
        for g in sparsifier.module_groups:
            assert isinstance(g['module'], nn.Linear)
            # The module_groups are unordered
            assert g['path'] in ('seq.0', 'linear')

    def test_logic(self):
        model = Model()
        sparsifier = WeightNormSparsifier(model, config=[model.linear],
                                          sparsity_level=0.5)
        sparsifier.enable_mask_update = True
        sparsifier.prepare()
        sparsifier.step()
        self.assertAlmostEqual(model.linear.mask.mean().item(), 0.5, places=2)
