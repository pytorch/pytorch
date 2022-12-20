# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]


import copy
import logging
import random

import torch
from torch.ao.pruning._experimental.pruner import (
    BaseStructuredSparsifier,
    FakeStructuredSparsity,
)
from torch.nn.utils import parametrize

from torch.testing._internal.common_utils import TestCase, skipIfTorchDynamo
from torch.testing._internal.common_pruning import (
    SimpleLinear,
    LinearBias,
    LinearActivation,
    LinearActivationFunctional,
    SimpleConv2d,
    Conv2dBias,
    Conv2dActivation,
    Conv2dPadBias,
    Conv2dPool,
    Conv2dPoolFlatten,
    Conv2dPoolFlattenFunctional,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

DEVICES = {
    torch.device("cpu"),
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
}


class SimplePruner(BaseStructuredSparsifier):
    def update_mask(self, module, tensor_name, **kwargs):
        getattr(module.parametrizations, tensor_name)[0].mask[1] = False


class ImplementedPruner(BaseStructuredSparsifier):
    def update_mask(self, module, tensor_name, **kwargs):
        """Prunes 1/3 of the weight output channels, so resulting module has 33.3% pruning"""
        num_rows = len(module.parametrizations[tensor_name][0].mask)
        prune = random.sample(list(range(num_rows)), num_rows // 3)
        module.parametrizations[tensor_name][0].mask[prune] = False


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
        self.assertRaisesRegex(
            TypeError,
            "BaseStructuredSparsifier.* update_mask",
            BaseStructuredSparsifier,
        )
        model1 = copy.deepcopy(model).to(device)
        pruner = SimplePruner(None)
        pruner.prepare(model1, None)
        pruner.enable_mask_update = True
        for g in pruner.groups:
            module = g["module"]
            assert module.weight.device.type == device.type
        assert len(pruner.groups) == 5
        pruner.step()
        # Can instantiate the model with configs
        model2 = copy.deepcopy(model).to(device)
        pruner = SimplePruner({"test": 3})
        pruner.prepare(model2, [{"tensor_fqn": "seq.0.weight"}])
        assert len(pruner.groups) == 1
        assert pruner.groups[0]["module_fqn"] == "seq.0"
        assert "test" in pruner.groups[0]
        assert pruner.groups[0]["test"] == 3

    def test_constructor(self):
        model = SimpleLinear()
        for device in DEVICES:
            self._test_constructor_on_device(model, torch.device(device))

    def _test_prepare_linear_on_device(self, model, device):
        model = copy.deepcopy(model).to(device)
        x = torch.ones(128, 7, device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, None)
        self._check_pruner_prepared(model, pruner, device)
        assert model(x).shape == (128, 10)

    def test_prepare_linear(self):
        models = [
            SimpleLinear(),
            LinearBias(),
            LinearActivation(),
            LinearActivationFunctional(),
        ]  # without and with bias
        for device in DEVICES:
            for model in models:
                self._test_prepare_linear_on_device(model, torch.device(device))

    def _test_prepare_conv2d_on_device(self, model, expected_shape, config, device):
        x = torch.ones((1, 1, 28, 28), device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, config)
        self._check_pruner_prepared(model, pruner, device)
        assert model(x).shape == expected_shape

    def test_prepare_conv2d(self):
        models = [
            SimpleConv2d(),
            Conv2dBias(),
            Conv2dActivation(),
            Conv2dPadBias(),
            Conv2dPool(),
        ]
        shapes = [
            (1, 52, 20, 20),
            (1, 52, 18, 18),
            (1, 52, 18, 18),
            (1, 52, 24, 24),
            (1, 52, 3, 3),
        ]
        configs = [None, None, None, None, None]
        for device in DEVICES:
            for model, shape, config in zip(models, shapes, configs):
                model = model.to(device)
                self._test_prepare_conv2d_on_device(
                    model, shape, config, torch.device(device)
                )

    def _test_step_linear_on_device(self, model, device):
        model = model.to(device)
        x = torch.ones(7, 7, device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, None)
        pruner.enable_mask_update = True
        self._check_pruner_valid_before_step(model, pruner, device)
        pruner.step()
        self._check_pruner_valid_after_step(model, pruner, 1, device)

    def test_step_linear(self):
        models = [
            SimpleLinear(),
            LinearBias(),
            LinearActivation(),
            LinearActivationFunctional(),
        ]
        for device in DEVICES:
            for model in models:
                self._test_step_linear_on_device(model, torch.device(device))

    def _test_step_conv2d_on_device(self, model, expected_shape, config, device):
        model = model.to(device)
        x = torch.ones((1, 1, 28, 28), device=device)
        pruner = SimplePruner(None)
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        self._check_pruner_valid_before_step(model, pruner, device)
        pruner.step()
        self._check_pruner_valid_after_step(model, pruner, 1, device)
        assert model(x).shape == expected_shape

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_step_conv2d(self):
        models = [
            SimpleConv2d(),
            Conv2dBias(),
            Conv2dActivation(),
            Conv2dPadBias(),
            Conv2dPool(),
        ]
        shapes = [
            (1, 52, 20, 20),
            (1, 52, 18, 18),
            (1, 52, 18, 18),
            (1, 52, 24, 24),
            (1, 52, 3, 3),
        ]
        configs = [None, None, None, None, None]
        for device in DEVICES:
            for model, shape, config in zip(models, shapes, configs):
                self._test_step_conv2d_on_device(
                    model, shape, config, torch.device(device)
                )

    def _check_pruner_pruned(self, model, pruner, device):
        for config in pruner.groups:
            module = config["module"]
            assert not hasattr(module, "parametrizations")
            assert not hasattr(module, "mask")

    def _test_linear_on_device(
        self, model, config, expected_shape, device, also_prune_bias
    ):
        model = model.to(device)
        model.eval()
        num_original_params = sum(p.numel() for p in model.parameters())
        x = torch.ones(128, 7, device=device)

        pruner = ImplementedPruner({"prune_bias": also_prune_bias})
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        pruner.step()

        y_expected = model(x)

        assert y_expected.shape == (128, 10)
        self._check_pruner_prepared(model, pruner, device)

        # Pruning step
        pruned = pruner.prune()
        y_pruned = pruned(x)
        num_pruned_params = sum(p.numel() for p in pruned.parameters())

        assert y_pruned.shape == expected_shape
        self._check_pruner_pruned(model, pruner, device)
        if y_pruned.shape == y_expected.shape:
            assert torch.isclose(y_expected, y_pruned, rtol=1e-05, atol=1e-07).all()
            assert num_pruned_params < num_original_params

    def test_prune_linear_linear(self):
        r"""test pruning linear-> linear modules"""
        configs, shapes = [], []
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))

        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "linear1.weight"},
            ]
        )
        shapes.append((128, 10))

        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))
        for device in DEVICES:
            for also_prune_bias in [True, False]:
                for config, shape in zip(configs, shapes):
                    self._test_linear_on_device(
                        SimpleLinear(),
                        config,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_prune_linear_bias_linear(self):
        # linear(bias) -> linear(no bias)
        configs, shapes = [], []
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
        )
        shapes.append((128, 10))

        # linear(bias) -> linear(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "seq.3.weight"},
            ]
        )
        shapes.append((128, 10))

        # linear(no bias) -> linear(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((128, 10))

        for device in DEVICES:
            for also_prune_bias in [True, False]:
                for config, shape in zip(configs, shapes):
                    self._test_linear_on_device(
                        LinearBias(),
                        config,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_prune_linear_activation_linear(self):
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.2.weight"},
            {"tensor_fqn": "seq.4.weight"},
            {"tensor_fqn": "linear1.weight"},
        ]
        shape = (128, 10)

        for device in DEVICES:
            for also_prune_bias in [True, False]:
                # test version with nn.Modules
                self._test_linear_on_device(
                    LinearActivation(),
                    config,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )
                # test functional version
                self._test_linear_on_device(
                    LinearActivationFunctional(),
                    config,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    def _test_conv2d_on_device(
        self, model, config, x, expected_shape, device, also_prune_bias
    ):
        model = model.to(device)
        num_original_params = sum(p.numel() for p in model.parameters())
        model.eval()

        pruner = ImplementedPruner({"prune_bias": also_prune_bias})
        pruner.prepare(model, config)
        pruner.enable_mask_update = True
        pruner.step()

        y_expected = model(x)
        assert y_expected.shape == expected_shape

        self._check_pruner_prepared(model, pruner, device)

        # Fusion step
        pruned = pruner.prune()
        y_pruned = pruned(x)
        num_pruned_params = sum(p.numel() for p in pruned.parameters())

        assert y_pruned.shape == expected_shape
        self._check_pruner_pruned(model, pruner, device)
        if y_pruned.shape == y_expected.shape:
            # TODO This rtol is a little high, need to double check if something specific is causing this to fail
            assert torch.isclose(
                y_expected, y_pruned, rtol=1e-1
            ).all(), f"fail for {type(model)}"
            # only time this should be equal is when all layers have padding and we can't prune
            assert num_pruned_params <= num_original_params

    def test_prune_conv2d_conv2d(self):
        configs, shapes = [], []
        # all within sequential blocks
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
            ]
        )
        shapes.append((1, 52, 20, 20))
        # prune across sequential blocks
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        shapes.append((1, 52, 20, 20))

        for device in DEVICES:
            x = torch.ones((1, 1, 28, 28), device=device)
            for also_prune_bias in [True, False]:
                for config, shape in zip(configs, shapes):
                    self._test_conv2d_on_device(
                        SimpleConv2d(),
                        config,
                        x,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_prune_conv2d_bias_conv2d(self):
        # Conv2d with Bias and no Activation
        configs, shapes = [], []
        # conv2d(bias) -> conv2d(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(no bias) -> conv2d(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(bias) -> conv2d(no bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.1.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        for device in DEVICES:
            x = torch.ones((1, 1, 28, 28), device=device)
            for also_prune_bias in [True, False]:
                for config, shape in zip(configs, shapes):
                    self._test_conv2d_on_device(
                        Conv2dBias(),
                        config,
                        x,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_prune_conv2d_activation_conv2d(self):
        # Conv2d with Activation and no Bias
        configs, shapes = [], []

        # conv2d(no bias) -> activatation -> conv2d(no bias)
        configs.append(
            [
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(bias) -> activatation -> conv2d(bias)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(bias) -> activation -> conv2d(no bias)
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        # conv2d(no bias) -> activation -> conv2d(bias)
        configs.append(
            [
                {"tensor_fqn": "conv2d1.weight"},
            ]
        )
        shapes.append((1, 52, 18, 18))

        for device in DEVICES:
            x = torch.ones((1, 1, 28, 28), device=device)
            for also_prune_bias in [True, False]:
                for config, shape in zip(configs, shapes):
                    self._test_conv2d_on_device(
                        Conv2dActivation(),
                        config,
                        x,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_prune_conv2d_padding_conv2d(self):
        # Conv2d with Padded layers after Bias layers
        configs, shapes = [], []

        # conv(padded, bias) -> conv(padded, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.4.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # conv(no bias, no pad) -> conv(padded, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.2.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        # conv(padded, bias) -> conv ( no bias ,no pad)
        configs.append(
            [
                {"tensor_fqn": "seq.0.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))
        # conv(pad, bias) -> conv(no pad, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.6.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))
        # conv(no pad, bias) -> conv(pad, bias)
        configs.append(
            [
                {"tensor_fqn": "seq.8.weight"},
            ]
        )
        shapes.append((1, 52, 24, 24))

        for device in DEVICES:
            x = torch.ones((1, 1, 28, 28), device=device)
            for also_prune_bias in [True, False]:
                for config, shape in zip(configs, shapes):
                    self._test_conv2d_on_device(
                        Conv2dPadBias(),
                        config,
                        x,
                        shape,
                        torch.device(device),
                        also_prune_bias,
                    )

    def test_prune_conv2d_pool_conv2d(self):
        # Conv2d with Pooling layers
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.3.weight"},
            {"tensor_fqn": "conv2d1.weight"},
            {"tensor_fqn": "conv2d2.weight"},
        ]
        shape = (1, 52, 3, 3)

        for device in DEVICES:
            x = torch.ones((1, 1, 28, 28), device=device)
            for also_prune_bias in [True, False]:
                self._test_conv2d_on_device(
                    Conv2dPool(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )

    @skipIfTorchDynamo("TorchDynamo fails with unknown reason")
    def test_complex_conv2d(self):
        """Test fusion for models that contain Conv2d & Linear modules.
        Currently supports: Conv2d-Pool2d-Flatten-Linear, Skip-add"""
        config = [
            {"tensor_fqn": "seq.0.weight"},
            {"tensor_fqn": "seq.3.weight"},
            {"tensor_fqn": "conv2d1.weight"},
            {"tensor_fqn": "conv2d2.weight"},
        ]
        shape = (1, 13)

        for device in DEVICES:
            x = torch.ones((1, 1, 28, 28), device=device)
            for also_prune_bias in [True, False]:
                self._test_conv2d_on_device(
                    Conv2dPoolFlattenFunctional(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )
                self._test_conv2d_on_device(
                    Conv2dPoolFlatten(),
                    config,
                    x,
                    shape,
                    torch.device(device),
                    also_prune_bias,
                )
