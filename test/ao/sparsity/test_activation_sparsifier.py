# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

import copy
from torch.testing._internal.common_utils import TestCase
import logging
import torch
from torch.ao.sparsity.experimental.activation_sparsifier.activation_sparsifier import ActivationSparsifier
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.sparsity.sparsifier.utils import module_to_fqn

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.linear1 = nn.Linear(4608, 128)
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.max_pool1(out)

        batch_size = x.shape[0]
        out = out.reshape(batch_size, -1)

        out = F.relu(self.linear1(out))
        out = self.linear2(out)
        return out


class _ActivationSparsifierUtils:
    """
    Contains utility functions to build and test an activation sparsifier
    """
    @staticmethod
    def _get_defaults():
        """Returns the default config and sparse config dicts
        that will be passed while creating the sparsifier object
        """
        sparse_config = {
            'sparsity_level': 0.5
        }

        def agg_fn(x, y):
            return x + y

        def reduce_fn(x):
            return torch.mean(x, dim=0)

        return {
            'aggregate_fn': agg_fn,
            'reduce_fn': reduce_fn
        }, sparse_config

    @staticmethod
    def _get_register_layer_args(model):
        r"""Builds configuration for the layers that is going to be attached to the sparsifier.

        Returns:
            a list of tuples
                Each entry in the list corresponds to the layer arguments.
                First entry in the tuple corresponds to all the arguments other than sparse_config
                Second entry in the tuple corresponds to sparse_config
        """
        def _vanilla_norm_sparsifier(data, sparsity_level):
            r"""Similar to data norm spasifier but block_shape = (1,1).
            Simply, flatten the data, sort it and mask out the values less than threshold
            """
            data_norm = torch.abs(data).flatten()
            _, sorted_idx = torch.sort(data_norm)
            threshold_idx = round(sparsity_level * len(sorted_idx))
            sorted_idx = sorted_idx[:threshold_idx]

            mask = torch.ones_like(data_norm)
            mask.scatter_(dim=0, index=sorted_idx, value=0)
            mask = mask.reshape(data.shape)

            return mask

        register_layer1_args = {
            'layer': model.conv1,
            'mask_fn': _vanilla_norm_sparsifier
        }
        sparse_config_layer1 = {'sparsity_level': 0.3}

        register_layer2_args = {
            'layer': model.linear1,
            'features': [0, 10, 234],
            'feature_dim': 1,
            'mask_fn': _vanilla_norm_sparsifier
        }

        sparse_config_layer2 = {'sparsity_level': 0.1}

        return [(register_layer1_args, sparse_config_layer1), (register_layer2_args, sparse_config_layer2)]


class _ActivationSparsiferTestCase(TestCase):
    """Runner class for unit tests that comprises functions to test each functionality
    """
    def check_constructor(self, activation_sparsifier, model, defaults, sparse_config):
        """Check if the model, defaults and sparse_config are loaded correctly in the activation
        sparsifier
        """
        sparsifier_defaults = activation_sparsifier.defaults
        combined_defaults = {**defaults, 'sparse_config': sparse_config}

        # check model
        assert activation_sparsifier.model == model

        # more keys are populated in activation sparsifier (eventhough they may be None)
        assert len(combined_defaults) <= len(activation_sparsifier.defaults)

        for key, config in sparsifier_defaults.items():
            # all the keys in combined_defaults should be present in sparsifier defaults
            assert config == combined_defaults.get(key, None)

    def check_register_layer(self, activation_sparsifier, defaults, sparse_config, layer_args_list):
        """Checks if layers in the model are correctly mapped to it's arguments.

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            defaults (Dict)
                all default config (except sparse_config)

            sparse_config (Dict)
                default sparse config passed to the sparsifier

            layer_args_list (list of tuples)
                Each entry in the list corresponds to the layer arguments.
                First entry in the tuple corresponds to all the arguments other than sparse_config
                Second entry in the tuple corresponds to sparse_config
        """
        # check args
        data_groups = activation_sparsifier.data_groups
        assert len(data_groups) == len(layer_args_list)
        for layer_args in layer_args_list:
            layer_arg, sparse_config_layer = layer_args

            # check sparse config
            sparse_config_actual = copy.deepcopy(sparse_config)
            sparse_config_actual.update(sparse_config_layer)

            name = module_to_fqn(activation_sparsifier.model, layer_arg['layer'])
            assert data_groups[name]['layer'] == layer_arg['layer']

            assert data_groups[name]['sparse_config'] == sparse_config_actual

            # assert the rest
            other_config_actual = copy.deepcopy(defaults)
            other_config_actual.update(layer_arg)
            other_config_actual.pop('layer')

            for key, value in other_config_actual.items():
                assert key in data_groups[name]
                assert value == data_groups[name][key]

            # get_mask should raise error
            with self.assertRaises(ValueError):
                activation_sparsifier.get_mask(name=name)

    def check_pre_forward_hook(self, activation_sparsifier, data_list):
        """Registering a layer attaches a pre-forward hook to that layer. This function
        checks if the pre-forward hook works as expected. Specifically, checks if the
        input is aggregated correctly.

        Basically, asserts that the aggregate of input activations is the same as what was
        computed in the sparsifier.

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            data_list (list of torch tensors)
                data input to the model attached to the sparsifier

        """
        # can only check for the first layer
        data_agg_actual = data_list[0]
        model = activation_sparsifier.model
        layer_name = module_to_fqn(model, model.conv1)
        agg_fn = activation_sparsifier.data_groups[layer_name]['aggregate_fn']

        for i in range(1, len(data_list)):
            data_agg_actual = agg_fn(data_agg_actual, data_list[i])

        assert torch.all(activation_sparsifier.data_groups[layer_name]['data'] == data_agg_actual)

        return data_agg_actual

    def check_step(self, activation_sparsifier, data_agg_actual):
        """Checks if .step() works as expected. Specifically, checks if the mask is computed correctly.

        Args:
            activation_sparsifier (sparsifier object)
                activation sparsifier object that is being tested.

            data_agg_actual (torch tensor)
                aggregated torch tensor

        """
        model = activation_sparsifier.model
        layer_name = module_to_fqn(model, model.conv1)
        assert layer_name is not None

        reduce_fn = activation_sparsifier.data_groups[layer_name]['reduce_fn']

        data_reduce_actual = reduce_fn(data_agg_actual)
        mask_fn = activation_sparsifier.data_groups[layer_name]['mask_fn']
        sparse_config = activation_sparsifier.data_groups[layer_name]['sparse_config']
        mask_actual = mask_fn(data_reduce_actual, **sparse_config)

        mask_model = activation_sparsifier.get_mask(layer_name)

        assert torch.all(mask_model == mask_actual)

        for _, config in activation_sparsifier.data_groups.items():
            assert 'data' not in config


class TestActivationSparsifier(_ActivationSparsiferTestCase):
    def test_activation_sparsifier(self):
        """Simulates the workflow of the activation sparsifier, starting from object creation
        till squash_mask().
        The idea is to check that everything works as expected while in the workflow.
        """
        # simulate the workflow

        # STEP 1: make data and activation sparsifier object
        defaults, sparse_config = _ActivationSparsifierUtils._get_defaults()
        model = Model()  # create model
        activation_sparsifier = ActivationSparsifier(model, **defaults, **sparse_config)

        # Test Constructor
        self.check_constructor(activation_sparsifier, model, defaults, sparse_config)

        # STEP 2: Register some layers
        layer_args_list = _ActivationSparsifierUtils._get_register_layer_args(model)

        # Registering..
        for layer_args in layer_args_list:
            layer_arg, sparse_config_layer = layer_args
            activation_sparsifier.register_layer(**layer_arg, **sparse_config_layer)

        # check if things are registered correctly
        self.check_register_layer(activation_sparsifier, defaults, sparse_config, layer_args_list)

        # check if forward pre hooks actually work
        # create some dummy data and run model forward
        data_list = []
        num_data_points = 5
        for _ in range(0, num_data_points):
            rand_data = torch.randn(16, 1, 28, 28)
            model(rand_data)
            data_list.append(rand_data)

        data_agg_actual = self.check_pre_forward_hook(activation_sparsifier, data_list)

        # STEP 3: sparsifier step
        activation_sparsifier.step()

        # self.check_step()
        self.check_step(activation_sparsifier, data_agg_actual)
