# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import logging
import random
import torch
from torch.nn.utils.parametrize import is_parametrized
from torch.testing._internal.common_utils import TestCase
from torch.ao.sparsity import BaseDataSparsifier
from typing import Tuple
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


class TestBaseDataSparsiferType(TestCase):
    def __init__(self, data_list, defaults, data_with_config):
        self.data_list = data_list
        self.defaults = defaults
        self.data_with_config = data_with_config

    def get_name_data_config(self, some_data):
        if isinstance(some_data, Tuple):
            # dealing with data_list
            name, data = some_data
            config = self.defaults
        else:
            # dealing with data_with_config
            name, data, config = some_data['name'], some_data['data'], some_data['config']
        return name, data, config

    def get_sparsifier(self):
        sparsifier = ImplementedSparsifier(data_list=self.data_list, **self.defaults)
        assert len(sparsifier.data_groups) == len(self.data_list)
        for data_config_dict in self.data_with_config:
            name, data, config = data_config_dict['name'], data_config_dict['data'], data_config_dict['config']
            sparsifier.add_data(name=name, data=data, **config)
        return sparsifier

    def test_constructor(self):
        sparsifier = self.get_sparsifier()
        assert len(sparsifier.data_groups) == len(self.data_list) + len(self.data_with_config)

        all_data = self.data_list + self.data_with_config

        for some_data in all_data:
            name, _, config = self.get_name_data_config(some_data)
            assert name in sparsifier.data_groups
            assert sparsifier.data_groups[name] == config

    def test_step(self):
        sparsifier = self.get_sparsifier()
        all_data = self.data_list + self.data_with_config

        # Check data and mask before doing the step
        for some_data in all_data:
            name, data, _ = self.get_name_data_config(some_data)
            data = sparsifier._extract_weight(data)
            sparsified_data = sparsifier.get_data(name=name, return_original=False)
            original_data = sparsifier.get_data(name=name, return_original=True)
            mask = sparsifier.get_mask(name=name)
            assert torch.all(sparsified_data == data)
            assert torch.all(original_data == data)
            assert torch.all(mask[0] == 1)

        step_count = 3

        for _ in range(0, step_count):
            sparsifier.step()
        for some_data in all_data:
            name, data, _ = self.get_name_data_config(some_data)
            data = sparsifier._extract_weight(data)
            sparsified_data = sparsifier.get_data(name=name, return_original=False)
            original_data = sparsifier.get_data(name=name, return_original=True)
            mask = sparsifier.get_mask(name=name)
            assert torch.all(sparsified_data[0] == 0)
            assert torch.all(original_data == data)
            assert torch.all(mask[0] == 0)
            assert 'step_count' in sparsifier.state[name]
            assert sparsifier.state[name]['step_count'] == 3

    def test_squash_mask(self):
        sparsifier = self.get_sparsifier()
        all_data = self.data_list + self.data_with_config
        for some_data in all_data:
            name, _, _ = self.get_name_data_config(some_data)
            assert hasattr(sparsifier._container, name)
            assert is_parametrized(sparsifier._container, name)
        sparsifier.step()
        sparsifier.squash_mask()

        for some_data in all_data:
            name, _, _ = self.get_name_data_config(some_data)
            assert not is_parametrized(sparsifier._container, name)  # not parametrized anymore
            with self.assertRaises(ValueError):
                sparsifier.get_data(name, return_original=True)

    def test_add_data(self):
        sparsifier = self.get_sparsifier()
        all_data = self.data_list + self.data_with_config
        for some_data in all_data:
            name1, data1, _ = self.get_name_data_config(some_data)
            data1 = sparsifier._extract_weight(data1)
            assert torch.all(data1 == sparsifier.get_data(name=name1))
            # get some other data at random and with the same name
            rand_idx = random.randint(0, len(all_data) - 1)
            _, data2, _ = self.get_name_data_config(all_data[rand_idx])
            data2 = sparsifier._extract_weight(data2)
            sparsifier.add_data(name=name1, data=data2)
            assert torch.all(data2 == sparsifier.get_data(name=name1))

    def run_tests(self):
        self.test_constructor()
        self.test_squash_mask()
        self.test_add_data()
        self.test_step()
        self.test_state_dict()

    def test_state_dict(self):
        sparsifier1 = self.get_sparsifier()
        sparsifier2 = ImplementedSparsifier(data_list=[self.data_list[0]])
        sparsifier1.step()

        state_dict1 = sparsifier1.state_dict()

        assert sparsifier1.state != sparsifier2.state
        name, _, _ = self.get_name_data_config(self.data_list[0])
        self.assertNotEqual(sparsifier1.get_mask(name), sparsifier2.get_mask(name))

        sparsifier2.load_state_dict(state_dict1)
        assert len(sparsifier1.state) == len(sparsifier2.state)
        assert len(sparsifier1.data_groups) == len(sparsifier2.data_groups)

        for name in sparsifier1.state.keys():
            # compare mask
            assert name in sparsifier2.state
            assert 'mask' in sparsifier2.state[name]
            assert 'mask' in sparsifier1.state[name]
            mask1, mask2 = sparsifier1.state[name]['mask'], sparsifier2.state[name]['mask']
            assert torch.all(mask1 == mask2)

            # compare data_groups
            dg1, dg2 = sparsifier1.data_groups, sparsifier2.data_groups
            assert name in dg1 and name in dg2
            assert dg1[name] == dg2[name]

            # compare container
            container1, container2 = sparsifier1._container, sparsifier2._container
            assert torch.all(getattr(container1, name) == getattr(container2, name))
            assert is_parametrized(container1, name) == is_parametrized(container2, name)
            if is_parametrized(container1, name):
                param1 = getattr(container1.parametrizations, name)[0]
                param2 = getattr(container2.parametrizations, name)[0]
                assert hasattr(param1, 'mask')
                assert hasattr(param2, 'mask')
                self.assertEqual(param1.__dict__, param2.__dict__)


class TestBaseDataSparsifier(TestCase):
    """To add unit tests to support new data types for the BaseDataSparsifier, create the following
        data_list: List of tuples of name, data to be added to the constructor
        defaults: default config for the above data in data_list
        data_with_config: list of dictionaries defining name, data and config (look test_tensors())

    Once the above is done, create an instance of TestBaseDataSparsifierType and call all it's functions
    """
    def test_tensors(self):
        tensor1, tensor2, tensor3 = torch.randn(3, 3), torch.randn(4, 4), torch.randn(5, 5)
        tensor4, tensor5 = torch.randn(1, 1), torch.randn(4, 4)
        data_list = [('tensor1', tensor1), ('tensor2', tensor2), ('tensor3', tensor3)]
        defaults = {'test': 3}

        data_with_config = [
            {
                'name': 'tensor4', 'data': tensor4, 'config': {'test': 7}
            },
            {
                'name': 'tensor5', 'data': tensor5, 'config': {'test': 8}
            },
        ]
        tensor_test = TestBaseDataSparsiferType(data_list=data_list, defaults=defaults,
                                                data_with_config=data_with_config)
        tensor_test.run_tests()

    def test_nn_parameters(self):
        param1, param2, param3 = nn.Parameter(torch.randn(3, 3)), nn.Parameter(torch.randn(4, 4)), nn.Parameter(torch.randn(5, 5))
        param4, param5 = nn.Parameter(torch.randn(1, 1)), nn.Parameter(torch.randn(4, 4))
        data_list = [('param1', param1), ('param2', param2), ('param3', param3)]
        defaults = {'test': 3}

        data_with_config = [
            {
                'name': 'param4', 'data': param4, 'config': {'test': 7}
            },
            {
                'name': 'param5', 'data': param5, 'config': {'test': 8}
            },
        ]
        param_test = TestBaseDataSparsiferType(data_list=data_list, defaults=defaults,
                                               data_with_config=data_with_config)
        param_test.run_tests()

    def test_nn_embeddings(self):
        emb1, emb2, = nn.Embedding(10, 3), nn.Embedding(20, 3)
        emb1_bag, emb2_bag = nn.EmbeddingBag(10, 3), nn.EmbeddingBag(20, 3)

        emb3, emb3_bag = nn.Embedding(15, 3), nn.EmbeddingBag(20, 3)
        data_list = [('emb1', emb1), ('emb1_bag', emb1_bag), ('emb2', emb2), ('emb2_bag', emb2_bag)]
        defaults = {'test': 3}

        data_with_config = [
            {
                'name': 'emb3', 'data': emb3, 'config': {'test': 7}
            },
            {
                'name': 'emb3_bag', 'data': emb3_bag, 'config': {'test': 8}
            },
        ]
        emb_test = TestBaseDataSparsiferType(data_list=data_list, defaults=defaults,
                                             data_with_config=data_with_config)
        emb_test.run_tests()
