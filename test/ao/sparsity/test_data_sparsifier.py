# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import logging
import torch
from torch.nn.utils.parametrize import is_parametrized
from torch.testing._internal.common_utils import TestCase
from torch.ao.sparsity import BaseDataSparsifier
from torch import nn

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

        # Test for nn.Parameters
        param1, param2 = nn.Parameter(torch.randn(4, 4)), nn.Parameter(torch.randn(5, 5))

        # Test for embeddings
        emb1, emb2 = nn.Embedding(10, 2), nn.Embedding(20, 2)
        emb1_bag, emb2_bag = nn.EmbeddingBag(10, 2), nn.EmbeddingBag(20, 2)
        data_list = [('tensor1', tensor1), ('tensor2', tensor2), ('param1', param1),
                     ('emb1', emb1), ('emb1_bag', emb1_bag)]
        defaults = {'test': 2}

        sparsifier = ImplementedSparsifier(data_list=data_list, **defaults)
        assert len(sparsifier.data_groups) == 5

        sparsifier.add_data(name='tensor3', data=tensor3, test=3)
        assert len(sparsifier.data_groups) == 6  # should now be 6

        sparsifier.add_data(name='param2', data=param2, test=4)
        assert len(sparsifier.data_groups) == 7  # should now be 7

        sparsifier.add_data(name='emb2', data=emb2, test=5)
        assert len(sparsifier.data_groups) == 8  # should now be 8

        sparsifier.add_data(name='emb2_bag', data=emb2_bag, test=5)
        assert len(sparsifier.data_groups) == 9  # should now be 9

        # data names that should be present in the sparsifier
        name_list = ['tensor1', 'tensor2', 'tensor3', 'param1', 'param2',
                     'emb1', 'emb2', 'emb1_bag', 'emb2_bag']
        assert all(name in sparsifier.data_groups for name in name_list)

        # check if the configs are loaded correctly
        assert sparsifier.data_groups['tensor1']['test'] == 2
        assert sparsifier.data_groups['tensor2']['test'] == 2
        assert sparsifier.data_groups['tensor3']['test'] == 3
        assert sparsifier.data_groups['param1']['test'] == 2
        assert sparsifier.data_groups['param2']['test'] == 4
        # For embeddings
        assert sparsifier.data_groups['emb1']['test'] == 2
        assert sparsifier.data_groups['emb1_bag']['test'] == 2
        assert sparsifier.data_groups['emb2']['test'] == 5
        assert sparsifier.data_groups['emb2_bag']['test'] == 5

    def test_step(self):
        # Test for torch tensors
        tensor1 = torch.randn(3, 3)
        # Test for nn.Parameters
        param1 = nn.Parameter(torch.randn(4, 4))
        # Test for embeddings
        emb1, emb1_bag = nn.Embedding(10, 2), nn.EmbeddingBag(10, 2)
        data_dict = {'tensor1': tensor1, 'param1': param1, 'emb1': emb1, 'emb1_bag': emb1_bag}

        sparsifier = ImplementedSparsifier()
        for name, data in data_dict.items():
            sparsifier.add_data(name=name, data=data, test=3)
        # Before step
        for name, data in data_dict.items():
            sparsified_data = sparsifier.get_data(name=name, return_sparsified=True)
            original_data = sparsifier.get_data(name=name, return_sparsified=False)
            mask = sparsifier.get_mask(name=name)
            if type(data) in [nn.Embedding, nn.EmbeddingBag]:
                data = data.weight
            assert original_data.requires_grad is False  # does not have the gradient
            assert sparsified_data.requires_grad is False  # does not track the gradient
            assert torch.all(sparsified_data == data)  # should not be sparsified before step
            assert torch.all(original_data == data)
            assert torch.all(mask[0] == 1)  # should not create the mask before step

        step_count = 3
        for _ in range(0, step_count):
            sparsifier.step()
        # after step
        for name, data in data_dict.items():
            sparsified_data = sparsifier.get_data(name=name, return_sparsified=True)
            original_data = sparsifier.get_data(name=name, return_sparsified=False)
            if type(data) in [nn.Embedding, nn.EmbeddingBag]:
                data = data.weight
            assert original_data.requires_grad is False  # does not have the gradient
            assert sparsified_data.requires_grad is False  # does not track the gradient
            assert torch.all(sparsified_data[0] == 0)
            assert torch.all(original_data == data)
            assert torch.all(mask[0] == 0)
            assert 'step_count' in sparsifier.state[name]
            assert sparsifier.state[name]['step_count'] == 3

    def test_squash_mask(self):
        # Test for torch tensors
        tensor1 = torch.randn(3, 3)
        # Test for nn.Parameters
        param1 = nn.Parameter(torch.randn(4, 4))
        # Test for embeddings
        emb1, emb1_bag = nn.Embedding(10, 2), nn.EmbeddingBag(10, 2)
        data_dict = {'tensor1': tensor1, 'param1': param1, 'emb1': emb1, 'emb1_bag': emb1_bag}

        sparsifier = ImplementedSparsifier()
        # adding data into the sparsifier
        for name, data in data_dict.items():
            sparsifier.add_data(name=name, data=data, test=3)

        for name, data in data_dict.items():
            assert hasattr(sparsifier._container, name)
            assert is_parametrized(sparsifier._container, name)

        sparsifier.step()
        sparsifier.squash_mask()

        for name, _ in data_dict.items():
            assert not is_parametrized(sparsifier._container, name)  # not parametrized anymore
            with self.assertRaises(ValueError):
                sparsifier.get_data(name, return_sparsified=False)

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
        # Test for nn params
        param1, param2 = nn.Parameter(torch.randn(6, 6)), nn.Parameter(torch.randn(7, 7))
        # Test for embeddings
        emb1, emb2 = nn.Embedding(10, 2), nn.Embedding(20, 2)
        emb1_bag, emb2_bag = nn.EmbeddingBag(10, 2), nn.EmbeddingBag(20, 2)
        data_list = [('tensor1', tensor1), ('param1', param1), ('emb1', emb1), ('emb1_bag', emb1_bag)]
        defaults = {'test': 2}

        sparsifier = ImplementedSparsifier(data_list=data_list, **defaults)
        sparsifier.add_data(name='tensor3', data=tensor3, test=3)
        sparsifier.step()
        state_dict = sparsifier.state_dict()

        sparsifier_new = ImplementedSparsifier()
        sparsifier_new.add_data('tensor1', tensor1)
        sparsifier_new.add_data('tensor2', tensor2)
        sparsifier_new.add_data('param1', param1)
        sparsifier_new.add_data('param2', param2)
        sparsifier_new.add_data('emb1', data=emb1)
        sparsifier_new.add_data('emb1_bag', data=emb1_bag)
        sparsifier_new.add_data('emb2', data=emb2)
        sparsifier_new.add_data('emb2_bag', data=emb2_bag)

        assert sparsifier_new.state != sparsifier.state
        mask_test_names = ['tensor1', 'param1', 'emb1', 'emb1_bag']
        for name in mask_test_names:
            mask, mask_new = sparsifier.get_mask(name), sparsifier_new.get_mask(name)
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
