# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]


import copy
import logging

import torch
from torch import nn
from torch.ao.pruning import BaseStructuredSparsifier, FakeStructuredSparsity
from torch.nn.utils import parametrize

from torch.testing._internal.common_utils import TestCase, skipIfTorchDynamo

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

DEVICES = {
    torch.device("cpu"),
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
}


class Linear(nn.Module):
    r"""Model with Linear layers, in Sequential and outside, without biases"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=False)
        )
        self.linear = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class LinearB(nn.Module):
    r"""Model with Linear layers, in Sequential and outside, with biases"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=True)
        )
        self.linear = nn.Linear(16, 16, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class MultipleLinear(nn.Module):
    r"""Model with multiple Linear layers, in Sequential and outside, without biases
    and with activation functions"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=False),
            nn.ReLU(),
            nn.Linear(5, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 6, bias=False)
        )
        self.linear = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class MultipleLinearB(nn.Module):
    r"""Model with multiple Linear layers, in Sequential and outside, with biases
    and with activation functions"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 8, bias=True),
            nn.ReLU(),
            nn.Linear(8, 6, bias=True)
        )
        self.linear = nn.Linear(6, 4, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class MultipleLinearMixed(nn.Module):
    r"""Model with multiple Linear layers, in Sequential and outside, some with biases
    and with activation functions"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 8, bias=False),
            nn.ReLU(),
            nn.Linear(8, 6, bias=True)
        )
        self.linear = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class Conv2dA(nn.Module):
    r"""Model with Conv2d layers, in Sequential and outside, without biases"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=False),
        )
        self.conv2d = nn.Conv2d(32, 64, 3, 1, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d(x)
        return x


class Conv2dB(nn.Module):
    r"""Model with Conv2d layers, in Sequential and outside, with biases"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),
        )
        self.conv2d = nn.Conv2d(32, 64, 3, 1, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d(x)
        return x


class Conv2dC(nn.Module):
    r"""Model with Conv2d layers, in Sequential and outside, with and without biases"""
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),
        )
        self.conv2d = nn.Conv2d(32, 64, 3, 1, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d(x)
        return x



class SimplePruner(BaseStructuredSparsifier):
    def update_mask(self, module, tensor_name, **kwargs):
        getattr(module.parametrizations, tensor_name)[0].mask[1] = False


class MultiplePruner(BaseStructuredSparsifier):
    def update_mask(self, module, tensor_name, **kwargs):
        getattr(module.parametrizations, tensor_name)[0].mask[1] = False
        getattr(module.parametrizations, tensor_name)[0].mask[2] = False


class TestBaseStructuredSparsifier(TestCase):
    def _check_pruner_prepared(self, model, pruner, device):
        for config in pruner.groups:
            module = config["module"]
            assert module.weight.device.type == device.type
            # Check mask exists
            assert config["tensor_fqn"] in pruner.state
            # Check parametrization exists and is correct
            assert parametrize.is_parametrized(module)
            assert hasattr(module, "parametrizations")
            # Assume that this is the 1st/only parametrization
            assert type(module.parametrizations.weight[0]) == FakeStructuredSparsity

    def _check_pruner_mask_squashed(self, model, pruner, device):
        for config in pruner.groups:
            modules = []
            if type(config['module']) is tuple:
                for module in config['module']:
                    modules.append(module)
            else:
                module = config['module']
                modules.append(module)
            for module in modules:
                assert module.weight.device.type == device.type
                assert not hasattr(module, "parametrizations")

    def _check_pruner_valid_before_step(self, model, pruner, device):
        for config in pruner.groups:
            modules = []
            if type(config['module']) is tuple:
                for module in config['module']:
                    modules.append(module)
            else:
                module = config['module']
                modules.append(module)
            for module in modules:
                assert module.weight.device.type == device.type
                assert module.parametrizations.weight[0].mask.dtype == torch.bool

    def _check_pruner_valid_after_step(self, model, pruner, mask, device):
        for config in pruner.groups:
            modules = []
            if type(config['module']) is tuple:
                for module in config['module']:
                    modules.append(module)
            else:
                module = config['module']
                modules.append(module)
            for module in modules:
                assert module.weight.device.type == device.type
                total = module.parametrizations.weight[0].mask.numel()
                assert module.parametrizations.weight[0].mask.count_nonzero() == total - mask

    def _test_constructor_on_device(self, model, device):
        self.assertRaisesRegex(TypeError, 'BaseStructuredSparsifier.* update_mask',
                               BaseStructuredSparsifier)
        model1 = copy.deepcopy(model).to(device)
        pruner = SimplePruner(None)
        pruner.prepare(model1, None)
        for g in pruner.groups:
            module = g['module']
            assert module.weight.device.type == device.type
        assert len(pruner.groups) == 2
        pruner.step()
        # Can instantiate the model with configs
        model2 = copy.deepcopy(model).to(device)
        pruner = SimplePruner({'test': 3})
        pruner.prepare(model2, [{"tensor_fqn": "linear.weight"}])
        assert len(pruner.groups) == 1
        assert pruner.groups[0]['module_fqn'] == 'linear'
        assert 'test' in pruner.groups[0]
        assert pruner.groups[0]['test'] == 3

    def test_constructor(self):
        model = Linear()
        for device in DEVICES:
            self._test_constructor_on_device(model, torch.device(device))

    def _test_prepare_linear_on_device(self, model, device):
        model = copy.deepcopy(model).to(device)
        x = torch.ones(128, 16, device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, None)
        self._check_pruner_prepared(model, pruner, device)
        assert model(x).shape == (128, 16)

    def test_prepare_linear(self):
        models = [Linear(), LinearB()]  # without and with bias
        for device in DEVICES:
            for model in models:
                self._test_prepare_linear_on_device(model, torch.device(device))

    def _test_prepare_conv2d_on_device(self, model, config, device):
        x = torch.ones((1, 1, 28, 28), device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, config)
        self._check_pruner_prepared(model, pruner, device)
        assert model(x).shape == (1, 64, 24, 24)

    def test_prepare_conv2d(self):

        models = [Conv2dA(), Conv2dB(), Conv2dC()]
        configs = [None, None, None]
        for device in DEVICES:
            for model, config in zip(models, configs):
                model = model.to(device)
                self._test_prepare_conv2d_on_device(model, config, torch.device(device))

    def _test_squash_mask_linear_on_device(self, model, device):
        model = copy.deepcopy(model).to(device)
        x = torch.ones(128, 16, device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, None)
        pruner.squash_mask()
        self._check_pruner_mask_squashed(model, pruner, device)
        assert model(x).shape == (128, 16)

    def test_squash_mask_linear(self):
        models = [Linear(), LinearB()]  # without and with bias
        for device in DEVICES:
            for model in models:
                self._test_squash_mask_linear_on_device(model, torch.device(device))

    def _test_squash_mask_conv2d_on_device(self, model, config, device):
        model = copy.deepcopy(model).to(device)
        x = torch.ones((1, 1, 28, 28), device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, config)
        pruner.squash_mask()
        self._check_pruner_mask_squashed(model, pruner, device)
        assert model(x).shape == (1, 64, 24, 24)

    def test_squash_mask_conv2d(self):

        models = [Conv2dA(), Conv2dB(), Conv2dC()]
        configs = [None, None, None]
        for device in DEVICES:
            for model, config in zip(models, configs):
                model = model.to(device)
                self._test_squash_mask_conv2d_on_device(model, config, torch.device(device))

    def _test_step_linear_on_device(self, model, is_basic, device):
        model = model.to(device)
        if is_basic:
            x = torch.ones(16, 16)
            pruner = SimplePruner(None)
            pruner.prepare(model, None)
            self._check_pruner_valid_before_step(model, pruner, device)
            pruner.step()
            self._check_pruner_valid_after_step(model, pruner, 1, device)
        else:
            x = torch.ones(7, 7)
            pruner = MultiplePruner(None)
            pruner.prepare(model, None)
            self._check_pruner_valid_before_step(model, pruner, device)
            pruner.step()
            self._check_pruner_valid_after_step(model, pruner, 2, device)

    def test_step_linear(self):
        basic_models = [Linear(), LinearB()]
        complex_models = [MultipleLinear(), MultipleLinearB(), MultipleLinearMixed()]
        for device in DEVICES:
            for model in basic_models:
                self._test_step_linear_on_device(model, True, torch.device(device))
            for model in complex_models:
                self._test_step_linear_on_device(model, False, torch.device(device))

    def _test_step_conv2d_on_device(self, model, config, device):
        model = model.to(device)
        x = torch.ones((1, 1, 28, 28)).to(device)
        pruner = SimplePruner(None)
        pruner.prepare(model, config)
        self._check_pruner_valid_before_step(model, pruner, device)
        pruner.step()
        self._check_pruner_valid_after_step(model, pruner, 1, device)
        assert model(x).shape == (1, 64, 24, 24)

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_step_conv2d(self):
        models = [Conv2dA(), Conv2dB(), Conv2dC()]
        configs = [None, None, None, None]
        for device in DEVICES:
            for model, config in zip(models, configs):
                self._test_step_conv2d_on_device(model, config, torch.device(device))
