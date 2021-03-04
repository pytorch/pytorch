from torch.testing._internal.common_utils import TestCase

from torch import mo
from torch import nn

import copy

class ModelUnderTest(nn.Module):
    def __init__(self, iC, oC):
        super().__init__()
        self.linear1 = nn.Linear(iC, 16)
        self.linear2 = nn.Linear(16, oC)

    def forward(self, x):
        x = self.linear1(x)
        x = self.lienar2(x)
        return x

class TestBaseSparsifierClass(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ModelUnderTest(8, 16)
        self.config = [
            {'params': self.model.linear1},
            {'params': self.model.linear2, 'target_ratio': 0.99, 'block_pattern': (1, 4)}
        ]
        self.kwargs = {
            'target_ratio': 0.7,
            'block_pattern': (1, 1),
            'scheme': 'inplace'
        }

    def test_getstate_setstate(self):
        r"""Tests the correctness of the __getstate__ and __setstate__."""
        sparsifier = mo.sparsity.Sparsifier(self.config)
        sparsifier.state['test_arg'] = 'abc.def'

        defaults = sparsifier.defaults
        state = copy.deepcopy(sparsifier.state)
        print(state)
        config = sparsifier.config_groups

        getstate = sparsifier.__getstate__()
        sparsifier.__setstate__(getstate)

        assert state == sparsifier.state
        assert config == sparsifier.config_groups
        assert defaults == sparsifier.defaults

        # TODO : Check this test -- it is failing
        # sparsifier.state['test_arg'] = 123.456
        # assert state != sparsifier.state, str(state)
        # sparsifier.__setstate__(state)
        # assert state == sparsifier.state, str(sparsifier.state) + ' ' + str(state)

    def test_add_group(self):
        r"""Tests if the default args are propagated throughout"""
        sparsifier = mo.sparsity.Sparsifier(self.config, **self.kwargs)

        for config in sparsifier.config_groups:
            for key in self.kwargs.keys():
                assert key in config
