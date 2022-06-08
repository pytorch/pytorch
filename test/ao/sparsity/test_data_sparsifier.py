# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import logging

import torch
from torch import nn
from torch.ao.sparsity import BaseDataSparsifier
from torch.nn.utils.parametrize import is_parametrized

from torch.testing._internal.common_utils import TestCase

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb1 = nn.Embedding(10, 2)
        self.emb_bag = nn.EmbeddingBag(20, 4)
        self.linear = nn.Linear(3,3)


class ImplementedSparsifier(BaseDataSparsifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_mask(self, name, data, **kwargs):
        mask = self.get_mask(name)
        mask[0] = 0
        linear_state = self.state['linear']
        linear_state['step_count'] = linear_state.get('step_count', 0) + 1


class TestBaseDataSparsifier(TestCase):
    def test_constructor(self):
        # Cannot instantiate the abstract base
        self.assertRaises(TypeError, BaseDataSparsifier)
        model = Model()

        # preparing some data to sparsify
        data_list = [('emb1', model.emb1), ('emb_bag', model.emb_bag)]
        defaults = {'test': 3}

        sparsifier = ImplementedSparsifier(data_list=data_list, **defaults)
        assert len(sparsifier.data_groups) == 2

        sparsifier.add_data(name='linear', data=model.linear.weight, test=4)
        assert len(sparsifier.data_groups) == 3 # should now be 3

        sparsifier = ImplementedSparsifier(**defaults)
        sparsifier.add_data(name='linear', data=model.linear.weight, test=4)

        assert len(sparsifier.data_groups) == 1
        assert 'linear' in sparsifier.data_groups
        assert torch.all(sparsifier.get_data('linear') == model.linear.weight)
        assert 'test' in sparsifier.data_groups['linear']
        assert sparsifier.data_groups['linear']['test'] == 4


    def test_step(self):
        model = Model()
        # preparing some data to sparsify
        sparsifier = ImplementedSparsifier()
        sparsifier.add_data(name='linear', data=model.linear.weight, test=4)

        sparsifier.step()
        # make sure that the mask is correct
        mask = sparsifier.get_mask(name='linear')
        sparsified_data = sparsifier.get_data(name='linear')
        original_data = sparsifier.get_data(name='linear', return_sparsified=False)
        assert torch.all(mask[0] == 0)
        assert torch.all(sparsified_data[0] == 0)
        assert torch.all(original_data == model.linear.weight)

    def test_state_dict(self):
        step_count = 3
        model = Model()

        # preparing some data to sparsify
        sparsifier0 = ImplementedSparsifier()
        sparsifier0.add_data(name='linear', data=model.linear.weight, test=4)
        mask = sparsifier0.get_mask(name='linear')
        mask.data = torch.arange(mask.shape[0] * mask.shape[1]).reshape(mask.shape)
        for _ in range(step_count):
            sparsifier0.step()
        state_dict = sparsifier0.state_dict()

        # Check the expected keys in the state_dict
        assert 'state' in state_dict
        assert 'linear' in state_dict['state']
        assert 'step_count' in state_dict['state']['linear']
        assert state_dict['state']['linear']['step_count'] == 3

        assert 'data_groups' in state_dict
        assert 'linear' in state_dict['data_groups']
        assert 'test' in state_dict['data_groups']['linear']
        
        # Check loading static_dict creates an equivalent model
        model1 = Model()
        data_list = [('emb1', model1.emb1), ('emb2', model1.emb_bag)]
        defaults = {'test': 1}
        sparsifier1 = ImplementedSparsifier(data_list=data_list, **defaults)
        sparsifier1.add_data(name='linear', data=model1.linear.weight, test=4)

        assert sparsifier0.state != sparsifier1.state
        mask0, mask1 = sparsifier0.get_mask('linear'), sparsifier1.get_mask('linear')       
        self.assertNotEqual(mask0, mask1)

        sparsifier1.load_state_dict(state_dict)

        # Make sure the states are loaded, and are correct
        assert sparsifier0.state == sparsifier1.state
        assert len(sparsifier0.data_groups) == len(sparsifier1.data_groups)

        for name in sparsifier0.data_groups.keys():
            assert name in sparsifier1.data_groups
            is_parametrized_s0 = is_parametrized(sparsifier0._container, name)
            is_parametrized_s1 = is_parametrized(sparsifier1._container, name)
            assert is_parametrized_s0 == is_parametrized_s1

            if is_parametrized_s0:
                param0 = getattr(sparsifier0._container.parametrizations, name)[0]
                param1 = getattr(sparsifier1._container.parametrizations, name)[0]
                assert hasattr(param0, 'mask')
                assert hasattr(param1, 'mask')
                self.assertEqual(param0.__dict__, param1.__dict__)          
            config0 = sparsifier0.data_groups[name]
            config1 = sparsifier1.data_groups[name]
            assert config0 == config1

    def test_squash_mask(self):
        model = Model()
        data_list = [('emb1', model.emb1), ('emb2', model.emb_bag)]
        defaults = {'test': 1}
        sparsifier = ImplementedSparsifier(data_list=data_list, **defaults)
        
        assert hasattr(sparsifier._container, 'emb1')
        assert hasattr(sparsifier._container, 'emb2')
        assert not hasattr(sparsifier._container, 'linear')
        assert is_parametrized(sparsifier._container, 'emb1')
        assert is_parametrized(sparsifier._container, 'emb2')

        sparsifier.squash_mask()

        assert not is_parametrized(sparsifier._container, 'emb1')
        assert not is_parametrized(sparsifier._container, 'emb2')
        with self.assertRaises(ValueError):
            sparsifier.get_data('emb1', return_sparsified=False)
            sparsifier.get_data('emb2', return_sparsified=False)
