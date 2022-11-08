# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]


import copy
import logging
import random


import torch
from torch import nn
from torch.ao.pruning import BaseStructuredSparsifier, FakeStructuredSparsity
from torch.nn.utils import parametrize

from torch.testing._internal.common_utils import TestCase, skipIfTorchDynamo

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

DEVICES = {
    torch.device("cpu"),
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
}


class Linear(nn.Module):
    r"""Model with Linear layers, in Sequential and outside, without biases"""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(16, 16, bias=False))
        self.linear = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class LinearB(nn.Module):
    r"""Model with Linear layers, in Sequential and outside, with biases"""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(16, 16, bias=True))
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
            nn.Linear(8, 6, bias=False),
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
            nn.Linear(8, 6, bias=True),
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
            nn.Linear(8, 6, bias=True),
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


<<<<<<< HEAD:test/ao/sparsity/test_structured_sparsifier.py

class SimplePruner(BaseStructuredSparsifier):
=======
class SimplePruner(BaseStructuredPruner):
>>>>>>> 83eb036eb8 (Add fx mode structured pruning):test/ao/sparsity/test_pruner.py
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
            if type(config["module"]) is tuple:
                for module in config["module"]:
                    modules.append(module)
            else:
                module = config["module"]
                modules.append(module)
            for module in modules:
                assert module.weight.device.type == device.type
                assert not hasattr(module, "parametrizations")

    def _check_pruner_valid_before_step(self, model, pruner, device):
        for config in pruner.groups:
            modules = []
            if type(config["module"]) is tuple:
                for module in config["module"]:
                    modules.append(module)
            else:
                module = config["module"]
                modules.append(module)
            for module in modules:
                assert module.weight.device.type == device.type
                assert module.parametrizations.weight[0].mask.dtype == torch.bool

    def _check_pruner_valid_after_step(self, model, pruner, mask, device):
        for config in pruner.groups:
            modules = []
            if type(config["module"]) is tuple:
                for module in config["module"]:
                    modules.append(module)
            else:
                module = config["module"]
                modules.append(module)
            for module in modules:
                assert module.weight.device.type == device.type
                total = module.parametrizations.weight[0].mask.numel()
                assert (
                    module.parametrizations.weight[0].mask.count_nonzero()
                    == total - mask
                )

    def _test_constructor_on_device(self, model, device):
        self.assertRaisesRegex(TypeError, 'BaseStructuredSparsifier.* update_mask',
                               BaseStructuredSparsifier)
        model1 = copy.deepcopy(model).to(device)
        pruner = SimplePruner(None)
        pruner.prepare(model1, None)
        for g in pruner.groups:
            module = g["module"]
            assert module.weight.device.type == device.type
        assert len(pruner.groups) == 2
        pruner.step()
        # Can instantiate the model with configs
        model2 = copy.deepcopy(model).to(device)
        pruner = SimplePruner({"test": 3})
        pruner.prepare(model2, [{"tensor_fqn": "linear.weight"}])
        assert len(pruner.groups) == 1
        assert pruner.groups[0]["module_fqn"] == "linear"
        assert "test" in pruner.groups[0]
        assert pruner.groups[0]["test"] == 3

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
                self._test_squash_mask_conv2d_on_device(
                    model, config, torch.device(device)
                )

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


class LinearA(nn.Module):
    r"""Model with only Linear layers without biases wrapped in a Sequential.
    Used to test basic pruned Linear-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=False),
            nn.Linear(500, 600, bias=False),
            nn.Linear(600, 10, bias=False),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class LinearB(nn.Module):
    r"""Model with only Linear layers without biases, some wrapped in a Sequential,
    some following the Sequential. Used to test basic pruned Linear-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=False),
            nn.Linear(500, 600, bias=False),
            nn.Linear(600, 400, bias=False),
        )
        self.linear1 = nn.Linear(400, 300, bias=False)
        self.linear2 = nn.Linear(300, 10, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class LinBiasA(nn.Module):
    r"""Model with only Linear layers, first one with bias, wrapped in a Sequential.
    Used to test pruned Linear-Bias-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=True),
            nn.Linear(500, 600, bias=False),
            nn.Linear(600, 10, bias=False),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class LinBiasB(nn.Module):
    r"""Model with only Linear layers, all with biases, wrapped in a Sequential.
    Used to test pruned Linear-Bias-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=True),
            nn.Linear(500, 600, bias=True),
            nn.Linear(600, 300, bias=True),
            nn.Linear(300, 10, bias=True),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class LinBiasC(nn.Module):
    r"""Model with only Linear layers, alternating layers with biases,
    wrapped in a Sequential. Used to test pruned Linear-Bias-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=True),
            nn.Linear(500, 600, bias=False),
            nn.Linear(600, 300, bias=True),
            nn.Linear(300, 10, bias=False),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class LinActA(nn.Module):
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, 600, bias=False),
            nn.Tanh(),
            nn.Linear(600, 400, bias=True),
        )
        self.linear1 = nn.Linear(400, 300, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(300, 10, bias=False)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.seq(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x


class LinActB(nn.Module):
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and functional
    activationals are called in between each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(700, 500, bias=True),
            nn.ReLU(),
            nn.Linear(500, 600, bias=False),
            nn.ReLU(),
            nn.Linear(600, 400, bias=True),
        )
        self.linear1 = nn.Linear(400, 300, bias=True)
        self.linear2 = nn.Linear(300, 800, bias=False)
        self.linear3 = nn.Linear(800, 10, bias=False)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.seq(x)
        x = self.linear1(x)
        # this seems to fail -> lin w/bias -> relu -> linear
        # x = self.act1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        return x


class Conv2dA(nn.Module):
    r"""Model with only Conv2d layers, all without bias, in a Sequential.
    Used to test pruned Conv2d-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=False),
            nn.Conv2d(320, 640, 3, 1, bias=False),
            nn.Conv2d(640, 480, 3, 1, bias=False),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Conv2dB(nn.Module):
    r"""Model with only Conv2d layers, all without bias, some in a Sequential and some following.
    Used to test pruned Conv2d-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=False),
            nn.Conv2d(320, 640, 3, 1, bias=False),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x


class Conv2dBiasA(nn.Module):
    r"""Model with only Conv2d layers, all with bias, wrapped in a Sequential.
    Used to test pruned Conv2d-Bias-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=True),
            nn.Conv2d(320, 640, 3, 1, bias=True),
            nn.Conv2d(640, 480, 3, 1, bias=True),
        )

    def forward(self, x):
        x = self.seq(x)
        return x


class Conv2dBiasB(nn.Module):
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some outside.
    Used to test pruned Conv2d-Bias-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=True),
            nn.Conv2d(320, 640, 3, 1, bias=False),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, bias=True)
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        return x


class Conv2dActA(nn.Module):
    r"""Model with only Conv2d layers, all without bias, some in a Sequential and some following.
    Activation function modules in between each layer.
    Used to test pruned Conv2d-Activation-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(320, 640, 3, 1, bias=False),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, bias=False)
        self.act1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, bias=False)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.act1(x)
        x = self.conv2d2(x)
        x = self.act2(x)
        return x


class Conv2dActB(nn.Module):
    r"""Model with only Conv2d layers, all without bias, some in a Sequential and some following.
    Activation function modules in between each Sequential layer, functional activations called
    in between outside Conv2d layers.
    Used to test pruned Conv2d-Activation-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(320, 640, 3, 1, bias=False),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, bias=False)
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.relu(x)
        x = self.conv2d2(x)
        x = F.hardtanh(x)
        return x


class Conv2dActC(nn.Module):
    r"""Model with only Conv2d layers, all with bias, some in a Sequential and some following.
    Activation function modules in between each layer.
    Used to test pruned Conv2d-Bias-Activation-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(320, 640, 3, 1, bias=True),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, bias=True)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.act1(x)
        x = self.conv2d2(x)
        x = self.act2(x)
        return x


class Conv2dActD(nn.Module):
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some following.
    Activation function modules in between each Sequential layer, functional activations called
    in-between each outside layer.
    Used to test pruned Conv2d-Bias-Activation-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(320, 640, 3, 1, bias=False),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, bias=True)
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, bias=False)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.relu(x)
        x = self.conv2d2(x)
        x = F.hardtanh(x)
        return x


class Conv2dPadBiasA(nn.Module):
    r"""Model with only Conv2d layers, all with bias and padding > 0,
    some in a Sequential and some following. Activation function modules in between each layer.
    Used to test special case of pruned Conv2d-Bias-(Activation)Conv2d fusion,
    when the second Conv2d layer has padding > 0."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(320, 640, 3, 1, padding=1, bias=True),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, padding=1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, padding=1, bias=True)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.act1(x)
        x = self.conv2d2(x)
        x = self.act2(x)
        return x


class Conv2dPadBiasB(nn.Module):
    r"""Model with only Conv2d layers, some with bias and padding > 0,
    some in a Sequential and some following. Activation function modules in between each layer.
    Used to test that bias is propagated correctly in the special case of
    pruned Conv2d-Bias-(Activation)Conv2d fusion, when the second Conv2d layer has padding > 0."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(320, 640, 3, 1, padding=1, bias=True),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, bias=False)
        self.act1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, padding=1, bias=True)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.act1(x)
        x = self.conv2d2(x)
        x = self.act2(x)
        return x


class Conv2dPadBiasC(nn.Module):
    r"""Model with only Conv2d layers, all with bias and some with padding > 0,
    some in a Sequential and some following. Activation function modules in between each layer.
    Used to test that bias is propagated correctly in the special case of
    pruned Conv2d-Bias-(Activation)Conv2d fusion, when the second Conv2d layer has padding > 0."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, 3, 1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(320, 640, 3, 1, bias=True),
            nn.Tanh(),
        )
        self.conv2d1 = nn.Conv2d(640, 480, 3, 1, padding=1, bias=True)
        self.act1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(480, 520, 3, 1, padding=1, bias=True)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.act1(x)
        x = self.conv2d2(x)
        x = self.act2(x)
        return x


class Conv2dPoolA(nn.Module):
    r"""Model with only Conv2d layers, all with bias, some in a Sequential and some following.
    Activation function modules in between each layer, Pool2d modules in between each layer.
    Used to test pruned Conv2d-Pool2d-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(320, 640, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(640, 480, kernel_size=3, padding=1, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(480, 520, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = self.maxpool(x)
        x = self.af1(x)
        x = self.conv2d2(x)
        return x


class Conv2dPoolB(nn.Module):
    r"""Model with only Conv2d layers, all with bias, some in a Sequential and some following.
    Activation function and Pool2d modules in between each Sequential layer, functional Pool2d between
    outside Conv2d layers.
    Used to test pruned Conv2d-Pool2d-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 320, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(320, 640, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(640, 480, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(480, 520, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=1)
        return x


class Conv2dPoolFlattenA(nn.Module):
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a functional Flatten followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(11, 13, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # test functional flatten
        x = self.fc(x)
        return x


class Conv2dPoolFlattenB(nn.Module):
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a Flatten module followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)
        self.af1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(44, 13, bias=True)

    def forward(self, x):
        x = self.seq(x)
        x = self.conv2d1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)
        x = self.af1(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class ImplementedPruner(BaseStructuredPruner):
    def update_mask(self, module, tensor_name, **kwargs):
        """Prunes 1/3 of the weight output channels, so resulting module has 33.3% pruning"""
        num_rows = len(module.parametrizations[tensor_name][0].mask)
        prune = random.sample(list(range(num_rows)), num_rows // 3)
        module.parametrizations[tensor_name][0].mask[prune] = False


class TestBaseStructuredPrunerConvert(TestCase):
    def _check_pruner_prepared(self, model, pruner, device):
        for config in pruner.groups:
            modules = []
            if type(config["module"]) is tuple:
                for module in config["module"]:
                    modules.append(module)
            else:
                module = config["module"]
                modules.append(module)
            for module in modules:
                assert module.weight.device == device
                # Check mask exists
                assert pruner.state.get(config["tensor_fqn"], None) is not None
                # Check parametrization exists and is correct
                assert parametrize.is_parametrized(module)
                assert hasattr(module, "parametrizations")
                # Assume that this is the 1st/only parametrization
                assert type(module.parametrizations.weight[0]) == FakeStructuredSparsity

    def _check_pruner_fused(self, model, pruner, device):
        for config in pruner.groups:
            modules = []
            if type(config["module"]) is tuple:
                for module in config["module"]:
                    modules.append(module)
            else:
                module = config["module"]
                modules.append(module)
            for module in modules:
                assert module.weight.device == device
                assert not hasattr(module, "parametrizations")
                assert not hasattr(module, "mask")

    def _test_linear_on_device(
        self, model, config, expected_shape, device, also_prune_bias
    ):
        model = model.to(device)
        model.eval()
        x = torch.ones(128, 700)

        pruner = ImplementedPruner({"prune_bias": also_prune_bias})
        pruner.prepare(model, config)
        pruner.step()

        y_expected = model(x)

        assert y_expected.shape == (128, 10)
        # assert y_traced.shape == (128, 10)
        self._check_pruner_prepared(model, pruner, device)

        # Fusion step
        fused = pruner.convert()
        y_fused = fused(x)

        assert y_fused.shape == expected_shape
        self._check_pruner_fused(model, pruner, device)
        if y_fused.shape == y_expected.shape:
            assert torch.isclose(y_expected, y_fused, rtol=1e-05, atol=1e-07).all()

    def test_linear(self):
        """Test fusion for models that only contain Linear modules.
        Currently support: Linear-Linear, Linear-Bias-Linear, Linear(Bias)-Activation-Linear"""
        for also_prune_bias in [True, False]:
            model_a = LinearA()
            config_a = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
            shape_a = (128, 7)

            model_b = LinearB()
            config_b = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "linear1.weight"},
                {"tensor_fqn": "linear2.weight"},
            ]
            shape_b = (128, 7)

            model_c = LinearB()
            config_c = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "linear2.weight"},
            ]
            shape_c = (128, 7)

            # Linear with bias
            model_d = LinBiasA()
            config_d = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
            shape_d = (128, 10)

            model_e = LinBiasB()
            config_e = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
            shape_e = (128, 10)

            model_f = LinBiasC()
            config_f = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
            shape_f = (128, 10)

            # Linear with activation
            model_g = LinActA()
            config_g = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "seq.4.weight"},
                {"tensor_fqn": "linear1.weight"},
            ]
            shape_g = (128, 10)

            model_h = LinActB()
            config_h = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.4.weight"},
                {"tensor_fqn": "linear1.weight"},
                {"tensor_fqn": "linear2.weight"},
            ]
            shape_h = (128, 10)

            test_models = [
                model_a,
                model_b,
                model_c,
                model_d,
                model_e,
                model_f,
                model_g,
                model_h,
            ]
            configs = [
                config_a,
                config_b,
                config_c,
                config_d,
                config_e,
                config_f,
                config_g,
                config_h,
            ]
            shapes = [
                shape_a,
                shape_b,
                shape_c,
                shape_d,
                shape_e,
                shape_f,
                shape_g,
                shape_h,
            ]

            for device in DEVICES:
                for model, config, expected_shape in zip(test_models, configs, shapes):
                    self._test_linear_on_device(
                        model,
                        config,
                        expected_shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def _test_conv2d_on_device(
        self, model, config, x, expected_shape, device, also_prune_bias
    ):
        model = model.to(device)
        model.eval()

        pruner = ImplementedPruner({"prune_bias": also_prune_bias})
        pruner.prepare(model, config)
        pruner.step()

        y_expected = model(x)

        assert y_expected.shape == expected_shape

        self._check_pruner_prepared(model, pruner, device)

        # Fusion step
        fused = pruner.convert()
        y_fused = fused(x)

        assert fused(x).shape == expected_shape
        self._check_pruner_fused(model, pruner, device)
        if y_fused.shape == y_expected.shape:
            assert torch.isclose(
                y_expected, y_fused, rtol=1e-05, atol=1e-06
            ).all(), f"fail for {type(model)}"

    def test_conv2d(self):
        """Test fusion for models that only contain Conv2d modules.
        Currently supports: Conv2d-Conv2d, Conv2d-Activation, Conv2d"""
        for also_prune_bias in [True, False]:
            model_a = Conv2dA()
            config_a = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
            shape_a = (1, 480, 22, 22)

            model_b = Conv2dB()
            config_b = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_b = (1, 520, 20, 20)

            # Conv2d with Activation and no Bias
            model_c = Conv2dActA()
            config_c = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_c = (1, 520, 20, 20)

            model_d = Conv2dActB()
            config_d = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_d = (1, 520, 20, 20)

            # Conv2d with Bias and no Activation
            model_e = Conv2dBiasA()
            config_e = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
            shape_e = (1, 480, 22, 22)

            model_f = Conv2dBiasB()
            config_f = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_f = (1, 520, 20, 20)

            # Conv2d with Activation and Bias
            model_g = Conv2dActC()
            config_g = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_g = (1, 520, 20, 20)

            model_h = Conv2dActD()
            config_h = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_h = (1, 520, 20, 20)

            # Conv2d with Padded layers after Bias layers
            model_i = Conv2dPadBiasA()
            config_i = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_i = (1, 520, 28, 28)

            model_j = Conv2dPadBiasB()
            config_j = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_j = (1, 520, 24, 24)

            model_k = Conv2dPadBiasC()
            config_k = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_k = (1, 520, 26, 26)

            # Conv2d with Pooling layers
            model_l = Conv2dPoolA()
            config_l = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.3.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_l = (1, 520, 5, 5)

            model_m = Conv2dPoolB()
            config_m = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.3.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
            shape_m = (1, 520, 3, 3)

            test_models = [
                model_a,
                model_b,
                model_c,
                model_d,
                model_e,
                model_f,
                model_g,
                model_h,
                model_i,
                model_j,
                model_k,
                model_l,
                model_m,
            ]
            configs = [
                config_a,
                config_b,
                config_c,
                config_d,
                config_e,
                config_f,
                config_g,
                config_h,
                config_i,
                config_j,
                config_k,
                config_l,
                config_m,
            ]
            expected_shapes = [
                shape_a,
                shape_b,
                shape_c,
                shape_d,
                shape_e,
                shape_f,
                shape_g,
                shape_h,
                shape_i,
                shape_j,
                shape_k,
                shape_l,
                shape_m,
            ]

            for device in DEVICES:
                for model, config, expected_shape in zip(
                    test_models, configs, expected_shapes
                ):
                    x = torch.ones((1, 1, 28, 28))
                    self._test_conv2d_on_device(
                        model,
                        config,
                        x,
                        expected_shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_complex_conv2d(self):
        """Test fusion for models that contain Conv2d & Linear modules.
        Currently supports: Conv2d-Pool2d-Flatten-Linear, Skip-add"""
        for also_prune_bias in [True, False]:
            # Conv2d Pool2d Flatten Linear
            model_a = Conv2dPoolFlattenA()
            config_a = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.3.weight"},
                {"tensor_fqn": "conv2d1.weight"},
                {"tensor_fqn": "conv2d2.weight"},
            ]
            shape_a = (1, 13)

            model_b = Conv2dPoolFlattenB()
            config_b = [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.3.weight"},
                {"tensor_fqn": "conv2d1.weight"},
                {"tensor_fqn": "conv2d2.weight"},
            ]
            shape_b = (1, 13)

            # Inputs
            x_1 = torch.ones((1, 1, 28, 28))
            x_3 = torch.ones((1, 3, 28, 28))

            test_models = [
                model_a,
                model_b,
            ]
            configs = [
                config_a,
                config_b,
            ]
            inputs = [x_1, x_1, x_3, x_3]
            expected_shapes = [
                shape_a,
                shape_b,
            ]

            for device in DEVICES:
                for model, config, input, expected_shape in zip(
                    test_models, configs, inputs, expected_shapes
                ):
                    self._test_conv2d_on_device(
                        model,
                        config,
                        input,
                        expected_shape,
                        torch.device(device),
                        also_prune_bias,
                    )
