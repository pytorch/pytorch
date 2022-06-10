# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import logging
import torch
from torch.nn.utils.parametrize import is_parametrized
from torch.testing._internal.common_utils import TestCase
from torch.ao.sparsity import BaseDataSparsifier

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class ImplementedSparsifier(BaseDataSparsifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_mask(self, name, data, **kwargs):
        mask = self.get_mask(name)
        mask[0] = 0
        linear_state = self.state[name]
        linear_state['step_count'] = linear_state.get('step_count', 0) + 1


class TestBaseDataSparsifier(TestCase):
    def test_constructor(self):
        # Test for torch tensors
        tensor1, tensor2, tensor3 = torch.randn(3, 3), torch.randn(4, 4), torch.randn(5, 5)
        data_list = [('tensor1', tensor1), ('tensor2', tensor2)]
        defaults = {'test': 2}

        sparsifier = ImplementedSparsifier(data_list=data_list, **defaults)
        assert len(sparsifier.data_groups) == 2

        sparsifier.add_data(name='tensor3', data=tensor3, test=3)
        assert len(sparsifier.data_groups) == 3  # should now be 3

        assert len(sparsifier.data_groups) == 3
        # tensor names that should be present in the sparsifier
        name_list = ['tensor1', 'tensor2', 'tensor3']
        assert all(name in sparsifier.data_groups for name in name_list)
        # check if the configs are loaded correctly
        assert sparsifier.data_groups['tensor1']['test'] == 2
        assert sparsifier.data_groups['tensor2']['test'] == 2
        assert sparsifier.data_groups['tensor3']['test'] == 3

    def test_step(self):
        # Test for torch tensors
        tensor1 = torch.randn(3, 3)
        sparsifier = ImplementedSparsifier()

        sparsifier.add_data(name='tensor1', data=tensor1, test=3)

        # Before step
        sparsified_data = sparsifier.get_data(name='tensor1', return_sparsified=True)
        original_data = sparsifier.get_data(name='tensor1', return_sparsified=False)
        mask = sparsifier.get_mask(name='tensor1')
        assert torch.all(sparsified_data == tensor1)
        assert torch.all(original_data == tensor1)
        assert torch.all(mask[0] == 1)
        step_count = 3
        for _ in range(0, step_count):
            sparsifier.step()
        # after step
        sparsified_data = sparsifier.get_data(name='tensor1', return_sparsified=True)
        original_data = sparsifier.get_data(name='tensor1', return_sparsified=False)
        assert torch.all(sparsified_data[0] == 0)
        assert torch.all(original_data == tensor1)
        assert torch.all(mask[0] == 0)
        assert 'step_count' in sparsifier.state['tensor1']
        assert sparsifier.state['tensor1']['step_count'] == 3

    def test_squash_mask(self):
        # Test for torch tensors
        tensor1 = torch.randn(3, 3)
        sparsifier = ImplementedSparsifier()

        sparsifier.add_data(name='tensor1', data=tensor1, test=3)
        assert hasattr(sparsifier._container, 'tensor1')
        assert is_parametrized(sparsifier._container, 'tensor1')

        sparsifier.step()
        sparsifier.squash_mask()

        assert not is_parametrized(sparsifier._container, 'tensor1')
        with self.assertRaises(ValueError):
            sparsifier.get_data('tensor1', return_sparsified=False)

    def test_add_data(self):
        # Test for torch tensors
        tensor1 = torch.randn(3, 3)
        sparsifier = ImplementedSparsifier()

        sparsifier.add_data(name='tensor1', data=tensor1, test=3)
        assert torch.all(sparsifier.get_data('tensor1') == tensor1)

        tensor1_new = torch.randn(5, 5)
        sparsifier.add_data(name='tensor1', data=tensor1_new, test=4)
        assert torch.all(sparsifier.get_data('tensor1') == tensor1_new)
        assert 'test' in sparsifier.data_groups['tensor1']
        assert sparsifier.data_groups['tensor1']['test'] == 4

    def test_state_dict(self):
        # Test for torch tensors
        tensor1, tensor2, tensor3 = torch.randn(3, 3), torch.randn(4, 4), torch.randn(5, 5)
        data_list = [('tensor1', tensor1)]
        defaults = {'test': 2}

        sparsifier = ImplementedSparsifier(data_list=data_list, **defaults)
        sparsifier.add_data(name='tensor3', data=tensor3, test=3)
        sparsifier.step()
        state_dict = sparsifier.state_dict()

        sparsifier_new = ImplementedSparsifier()
        sparsifier_new.add_data('tensor1', tensor1)
        sparsifier_new.add_data('tensor2', tensor2)

        assert sparsifier_new.state != sparsifier.state
        mask, mask_new = sparsifier.get_mask('tensor1'), sparsifier_new.get_mask('tensor1')
        self.assertNotEqual(mask, mask_new)

        sparsifier_new.load_state_dict(state_dict)
        assert len(sparsifier.state) == len(sparsifier_new.state)

        for name in sparsifier.state.keys():
            # compare mask and key names
            assert name in sparsifier_new.state
            assert 'mask' in sparsifier.state[name]
            assert 'mask' in sparsifier_new.state[name]
            mask = sparsifier.state[name]['mask']
            assert torch.all(mask == sparsifier_new.state[name]['mask'])

            # compare config
            dg, dg_new = sparsifier.data_groups, sparsifier_new.data_groups
            assert name in dg and name in dg_new
            assert dg[name] == dg_new[name]

            # compare container details
            container, container_new = sparsifier._container, sparsifier_new._container
            assert torch.all(getattr(container, name) == getattr(container_new, name))
            assert is_parametrized(container, name) == is_parametrized(container_new, name)
            if is_parametrized(container, name):
                param = getattr(container.parametrizations, name)[0]
                param_new = getattr(container_new.parametrizations, name)[0]
                assert hasattr(param, 'mask')
                assert hasattr(param_new, 'mask')
                self.assertEqual(param.__dict__, param_new.__dict__)
