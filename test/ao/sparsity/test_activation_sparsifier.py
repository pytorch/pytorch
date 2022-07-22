# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

from torch.testing._internal.common_utils import TestCase
import logging
import torch
from torch.ao.sparsity._experimental.activation_sparsifier.activation_sparsifier import ActivationSparsifier
import torch.nn as nn
import torch.nn.functional as F
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


class TestActivationSparsifier(TestCase):
    def _check_constructor(self, activation_sparsifier, model, defaults, sparse_config):
        """Helper function to check if the model, defaults and sparse_config are loaded correctly
        in the activation sparsifier
        """
        sparsifier_defaults = activation_sparsifier.defaults
        combined_defaults = {**defaults, 'sparse_config': sparse_config}

        # more keys are populated in activation sparsifier (eventhough they may be None)
        assert len(combined_defaults) <= len(activation_sparsifier.defaults)

        for key, config in sparsifier_defaults.items():
            # all the keys in combined_defaults should be present in sparsifier defaults
            assert config == combined_defaults.get(key, None)

    def test_activation_sparsifier(self):
        """Simulates the workflow of the activation sparsifier, starting from object creation
        till squash_mask().
        The idea is to check that everything works as expected while in the workflow.
        """
        # defining aggregate, reduce functions
        def agg_fn(x, y):
            return x + y

        def reduce_fn(x):
            return torch.mean(x, dim=0)

        # Creating default function and sparse configs
        # default sparse_config
        sparse_config = {
            'sparsity_level': 0.5
        }

        defaults = {
            'aggregate_fn': agg_fn,
            'reduce_fn': reduce_fn
        }

        # simulate the workflow
        # STEP 1: make data and activation sparsifier object
        model = Model()  # create model
        activation_sparsifier = ActivationSparsifier(model, **defaults, **sparse_config)

        # Test Constructor
        self._check_constructor(activation_sparsifier, model, defaults, sparse_config)
