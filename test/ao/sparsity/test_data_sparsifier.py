# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import logging
import random
import torch
from torch.nn.utils.parametrize import is_parametrized
from torch.testing._internal.common_utils import TestCase
from torch.ao.sparsity import BaseDataSparsifier, DataNormSparsifier
from typing import Tuple
from torch import nn
import itertools
import math

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class ImplementedSparsifier(BaseDataSparsifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_mask(self, name, data, **kwargs):
        mask = self.get_mask(name)
        mask[0] = 0
        linear_state = self.state[name]
        linear_state['step_count'] = linear_state.get('step_count', 0) + 1


class _BaseDataSparsiferTestRunner:
    r"""This helper test class takes in any supported type of and runs some tests.
        The user is required to pass in the data that needs to sparsified and the
        runner will run some tests that needs to be passed in order for the data
        type to be supported.
        TODO: Change the structure by creating a separate test case class for each
              member function
    """
    def __init__(self, data_list, defaults, data_with_config):
        self.data_list = data_list
        self.defaults = defaults
        self.data_with_config = data_with_config

        # Temporary hack to quickly fix failing tests.
        # This will be rewritten as soon as possible
        self._test_case = TestCase()

    def _get_name_data_config(self, some_data):
        if isinstance(some_data, Tuple):
            # dealing with data_list
            name, data = some_data
            config = self.defaults
        else:
            # dealing with data_with_config
            name, data, config = some_data['name'], some_data['data'], some_data['config']
        return name, data, config

    def _get_sparsifier(self):
        sparsifier = ImplementedSparsifier(data_list=self.data_list, **self.defaults)
        assert len(sparsifier.data_groups) == len(self.data_list)
        for data_config_dict in self.data_with_config:
            name, data, config = data_config_dict['name'], data_config_dict['data'], data_config_dict['config']
            sparsifier.add_data(name=name, data=data, **config)
        return sparsifier

    def _run_constructor_test(self):
        sparsifier = self._get_sparsifier()
        assert len(sparsifier.data_groups) == len(self.data_list) + len(self.data_with_config)

        all_data = self.data_list + self.data_with_config

        for some_data in all_data:
            name, _, config = self._get_name_data_config(some_data)
            assert name in sparsifier.data_groups
            assert sparsifier.data_groups[name] == config

    def _run_step_test(self):
        sparsifier = self._get_sparsifier()
        all_data = self.data_list + self.data_with_config

        # Check data and mask before doing the step
        for some_data in all_data:
            name, data, _ = self._get_name_data_config(some_data)
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
            name, data, _ = self._get_name_data_config(some_data)
            data = sparsifier._extract_weight(data)
            sparsified_data = sparsifier.get_data(name=name, return_original=False)
            original_data = sparsifier.get_data(name=name, return_original=True)
            mask = sparsifier.get_mask(name=name)
            assert torch.all(sparsified_data[0] == 0)
            assert torch.all(original_data == data)
            assert torch.all(mask[0] == 0)
            assert 'step_count' in sparsifier.state[name]
            assert sparsifier.state[name]['step_count'] == 3

    def _run_squash_mask_test(self):
        sparsifier = self._get_sparsifier()
        all_data = self.data_list + self.data_with_config
        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            assert hasattr(sparsifier._container, name)
            assert is_parametrized(sparsifier._container, name)
        sparsifier.step()
        sparsifier.squash_mask()

        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            assert not is_parametrized(sparsifier._container, name)  # not parametrized anymore
            with self._test_case.assertRaises(ValueError):
                sparsifier.get_data(name, return_original=True)

    def _run_add_data_test(self):
        sparsifier = self._get_sparsifier()
        all_data = self.data_list + self.data_with_config
        for some_data in all_data:
            name1, data1, _ = self._get_name_data_config(some_data)
            data1 = sparsifier._extract_weight(data1)
            assert torch.all(data1 == sparsifier.get_data(name=name1))
            # get some other data at random and with the same name
            rand_idx = random.randint(0, len(all_data) - 1)
            _, data2, _ = self._get_name_data_config(all_data[rand_idx])
            data2 = sparsifier._extract_weight(data2)
            sparsifier.add_data(name=name1, data=data2)
            assert torch.all(data2 == sparsifier.get_data(name=name1))

    def run_tests(self):
        self._run_constructor_test()
        self._run_squash_mask_test()
        self._run_add_data_test()
        self._run_step_test()
        self._run_state_dict_test()

    def _run_state_dict_test(self):
        sparsifier1 = self._get_sparsifier()
        sparsifier2 = ImplementedSparsifier(data_list=[self.data_list[0]])
        sparsifier1.step()

        state_dict1 = sparsifier1.state_dict()

        assert sparsifier1.state != sparsifier2.state
        name, _, _ = self._get_name_data_config(self.data_list[0])
        self._test_case.assertNotEqual(sparsifier1.get_mask(name), sparsifier2.get_mask(name))

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
                self._test_case.assertEqual(param1.__dict__, param2.__dict__)


class TestBaseDataSparsifier(TestCase):
    """To add unit tests to support new data types for the BaseDataSparsifier, create the following
        data_list: List of tuples of name, data to be added to the constructor
        defaults: default config for the above data in data_list
        data_with_config: list of dictionaries defining name, data and config (look test_tensors())

    Once the above is done, create an instance of TestBaseDataSparsifierType and call all the run_tests()
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
        tensor_test = _BaseDataSparsiferTestRunner(data_list=data_list, defaults=defaults,
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
        param_test = _BaseDataSparsiferTestRunner(data_list=data_list, defaults=defaults,
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
        emb_test = _BaseDataSparsiferTestRunner(data_list=data_list, defaults=defaults,
                                                data_with_config=data_with_config)
        emb_test.run_tests()


class _NormDataSparsifierTestRunner(_BaseDataSparsiferTestRunner):
    r"""This helper test class takes in any supported type of and runs some tests.
        This inherits the TestBaseDataSparsifierRuner wherein some functions are
        over-ridden to take accomodate the specific sparsifier.
        TODO: Change the structure by creating a separate test case class for each
              member function
    """
    def __init__(self, data_list, defaults, data_with_config, norm_type='L1'):
        super().__init__(data_list=data_list, defaults=defaults, data_with_config=data_with_config)
        assert norm_type in ['L1', 'L2']
        self.norm_type = norm_type

    def _get_bounds_on_actual_sparsity(self, config, tensor_shape):
        r"""This function gets the bounds on actual sparsity.
            Note::
                Although we specify the sparsity_level parameter, this does not mean that
                the actual sparsity obtained after sparsification is the same as sparsity_level.
                The actual sparsity depends largely on the shape and the data itself.
        """
        sparsity_level = config['sparsity_level']
        zeros_per_block = config['zeros_per_block']
        sparse_block_shape = config['sparse_block_shape']

        height, width = tensor_shape[-2], tensor_shape[-1]
        block_height, block_width = sparse_block_shape
        number_blocks = math.ceil(height / block_height) * math.ceil(width / block_width)
        values_per_block = block_height * block_width

        if zeros_per_block == 0:
            return (1.0, 1.0)
        else:
            # min value assumes zeros_per_block is 1
            min_values_sparsified = number_blocks * sparsity_level
            # max value assumes actual zeros_per_block
            max_values_sparsified = min_values_sparsified * min(values_per_block, zeros_per_block)
            lower_bound = min_values_sparsified / (height * width)
            upper_bound = min(1.0, max_values_sparsified / (height * width))

            lower_bound, upper_bound = round(lower_bound, 3), round(upper_bound, 3)
            return lower_bound, upper_bound

    def _get_sparsifier(self):
        sparsifier = DataNormSparsifier(data_list=self.data_list, norm=self.norm_type, **self.defaults)
        assert len(sparsifier.data_groups) == len(self.data_list)
        for data_config_dict in self.data_with_config:
            name, data, config = data_config_dict['name'], data_config_dict['data'], data_config_dict['config']
            sparsifier.add_data(name=name, data=data, **config)
        return sparsifier

    def _run_step_test(self):
        sparsifier = self._get_sparsifier()
        all_data = self.data_list + self.data_with_config

        # mask before step() should not be sparsified
        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            mask = sparsifier.get_mask(name=name)
            assert (1.0 - mask.mean()) == 0  # checking sparsity level is 0

        sparsifier.step()

        for some_data in all_data:
            name, _, _ = self._get_name_data_config(some_data)
            mask = sparsifier.get_mask(name=name)
            config = sparsifier.data_groups[name]
            lb, ub = self._get_bounds_on_actual_sparsity(config, mask.shape)
            mask = mask.to(torch.float)
            actual_sparsity = round(1 - mask.mean().item(), 3)
            assert actual_sparsity >= lb and actual_sparsity <= ub
            assert actual_sparsity > 0.0  # exact sparsity level cannot be achieved due to size of tensor

        iters_before_collapse = 100

        test_sparsifier = DataNormSparsifier(sparsity_level=0.5, sparse_block_shape=(1, 4), zeros_per_block=4, norm=self.norm_type)

        for _ in range(iters_before_collapse):
            new_data = torch.randn(20, 20)
            test_sparsifier.add_data(name='test_data', data=new_data)
            test_sparsifier.step()
            mask = test_sparsifier.get_mask(name='test_data')
            mask = mask.to(torch.float)
            assert (1.0 - mask.mean().item()) > 0  # some sparsity achieved

    def _run_step_2_of_4_test(self):
        # overriding default config for test purposes
        default_config = {'sparsity_level': 1.0, 'zeros_per_block': 2, 'sparse_block_shape': (1, 4)}
        data_list = [('test_data', torch.randn(4, 4))]

        sparsifier = DataNormSparsifier(data_list=data_list, norm=self.norm_type, **default_config)
        sparsifier.step()

        for some_data in data_list:
            name, _ = some_data
            mask = sparsifier.get_mask(name=name)
            mask = mask.to(torch.float)
            self._test_case.assertAlmostEqual(1.0 - mask.mean().item(), 0.5, places=2)
            for row in mask:
                for idx in range(0, len(row), 4):
                    block = row[idx:idx + 4]
                    block, _ = block.sort()
                    assert (block[:2] == 0).all()
                    assert (block[2:] != 0).all()

    def _run_sparsity_level_test(self):
        sparsity_levels = [-1.0, 0.0, 0.5, 1.0, 2.0]
        sparse_block_shapes = [(1, 1), (1, 4), (2, 2), (4, 1)]
        zeros_per_blocks = [0, 1, 2, 3, 4]
        sparsifier = self._get_sparsifier()
        testcases = itertools.tee(itertools.product(sparsity_levels,
                                                    sparse_block_shapes,
                                                    zeros_per_blocks))

        # get some data
        name, data, _ = self.data_with_config['name'], self.data_with_config['data']
        for sl, sbs, zpb in testcases[0]:
            new_name = f'{name}_{sl}_{sbs}_{zpb}'
            current_config = {'sparsity_level': sl, 'sparse_block_shape': sbs, 'zeros_per_block': zpb}
            sparsifier.add_data(name=new_name, data=data, **current_config)
            if zpb > sbs[0] * sbs[1]:
                continue

        sparsifier.step()
        sparsifier.squash_mask()
        for sl, sbs, zpb in testcases[0]:
            new_name = f'{name}_{sl}_{sbs}_{zpb}'
            sparsified_data = sparsifier.get_data(name=new_name, original=False)
            # sparse mask
            sparse_mask = (sparsified_data == 0).float()
            if zpb == 0:
                assert sparse_mask.mean() == 0
            else:
                # Ratio of individual zeros in the tensor
                true_sl = min(max(sl, 0.0), 1.0)
                true_sl = true_sl * zpb / sbs[0] / sbs[1]
                assert sparse_mask.mean() == true_sl

    def run_tests(self):
        self._run_constructor_test()
        self._run_squash_mask_test()
        self._run_add_data_test()
        self._run_state_dict_test()
        self._run_step_test()
        self._run_step_2_of_4_test()


class TestNormDataSparsifiers(TestCase):
    """To add unit tests to support new data types for the NormDataSparsifier, create the following
        data_list: List of tuples of name, data to be added to the constructor
        defaults: default config for the above data in data_list
        data_with_config: list of dictionaries defining name, data and config (look test_tensors())

        Once the above is done, create an instance of _NormDataSparsifierTestRunner and call run_tests()
    """
    def test_tensors(self):
        tensor1, tensor2, tensor3 = torch.randn(3, 3), torch.randn(4, 4), torch.randn(5, 5)
        tensor4, tensor5 = torch.randn(10, 10), torch.randn(4, 4)
        data_list = [('tensor1', tensor1), ('tensor2', tensor2), ('tensor3', tensor3)]
        defaults = {'sparsity_level': 0.5, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}

        data_with_config = [
            {
                'name': 'tensor4', 'data': tensor4,
                'config': {'sparsity_level': 0.7, 'sparse_block_shape': (2, 3), 'zeros_per_block': 6}
            },
            {
                'name': 'tensor5', 'data': tensor5,
                'config': {'sparsity_level': 0.3, 'sparse_block_shape': (2, 3), 'zeros_per_block': 6}
            },
        ]
        tensor_test_l1 = _NormDataSparsifierTestRunner(data_list=data_list, defaults=defaults,
                                                       data_with_config=data_with_config, norm_type='L1')
        tensor_test_l1.run_tests()

        tensor_test_l2 = _NormDataSparsifierTestRunner(data_list=data_list, defaults=defaults,
                                                       data_with_config=data_with_config, norm_type='L2')
        tensor_test_l2.run_tests()

    def test_nn_parameters(self):
        param1, param2, param3 = nn.Parameter(torch.randn(3, 3)), nn.Parameter(torch.randn(4, 4)), nn.Parameter(torch.randn(5, 5))
        param4, param5 = nn.Parameter(torch.randn(10, 10)), nn.Parameter(torch.randn(4, 4))
        data_list = [('param1', param1), ('param2', param2), ('param3', param3)]
        defaults = {'sparsity_level': 0.5, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}

        data_with_config = [
            {
                'name': 'param4', 'data': param4,
                'config': {'sparsity_level': 0.7, 'sparse_block_shape': (2, 3), 'zeros_per_block': 6}
            },
            {
                'name': 'param5', 'data': param5,
                'config': {'sparsity_level': 0.3, 'sparse_block_shape': (2, 3), 'zeros_per_block': 6}
            },
        ]
        param_test_l1 = _NormDataSparsifierTestRunner(data_list=data_list, defaults=defaults,
                                                      data_with_config=data_with_config, norm_type='L1')
        param_test_l1.run_tests()

        param_test_l2 = _NormDataSparsifierTestRunner(data_list=data_list, defaults=defaults,
                                                      data_with_config=data_with_config, norm_type='L2')
        param_test_l2.run_tests()

    def test_nn_embeddings(self):
        emb1, emb2, = nn.Embedding(10, 3), nn.Embedding(20, 3)
        emb1_bag, emb2_bag = nn.EmbeddingBag(10, 3), nn.EmbeddingBag(20, 3)

        emb3, emb3_bag = nn.Embedding(15, 3), nn.EmbeddingBag(20, 3)
        data_list = [('emb1', emb1), ('emb1_bag', emb1_bag), ('emb2', emb2), ('emb2_bag', emb2_bag)]
        defaults = {'sparsity_level': 0.5, 'sparse_block_shape': (1, 4), 'zeros_per_block': 4}

        data_with_config = [
            {
                'name': 'emb3', 'data': emb3,
                'config': {'sparsity_level': 0.7, 'sparse_block_shape': (2, 3), 'zeros_per_block': 6}
            },
            {
                'name': 'emb3_bag', 'data': emb3_bag,
                'config': {'sparsity_level': 0.3, 'sparse_block_shape': (2, 3), 'zeros_per_block': 6}
            },
        ]
        emb_test_l1 = _NormDataSparsifierTestRunner(data_list=data_list, defaults=defaults,
                                                    data_with_config=data_with_config, norm_type='L1')
        emb_test_l1.run_tests()

        emb_test_l2 = _NormDataSparsifierTestRunner(data_list=data_list, defaults=defaults,
                                                    data_with_config=data_with_config, norm_type='L2')

        emb_test_l2.run_tests()
