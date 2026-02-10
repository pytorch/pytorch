# Owner(s): ["module: sparse"]


import torch
from torch import nn
from torch.ao.pruning.sparsifier import utils
from torch.nn.utils import parametrize
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class ModelUnderTest(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=bias)
        self.seq = nn.Sequential(
            nn.Linear(16, 16, bias=bias), nn.Linear(16, 16, bias=bias)
        )

        # Make sure the weights are not random
        self.linear.weight = nn.Parameter(torch.zeros_like(self.linear.weight) + 1.0)
        self.seq[0].weight = nn.Parameter(torch.zeros_like(self.seq[0].weight) + 2.0)
        self.seq[1].weight = nn.Parameter(torch.zeros_like(self.seq[1].weight) + 3.0)
        if bias:
            self.linear = nn.Parameter(torch.zeros_like(self.linear.bias) + 10.0)
            self.seq[0] = nn.Parameter(torch.zeros_like(self.seq[0].bias) + 20.0)
            self.seq[0] = nn.Parameter(torch.zeros_like(self.seq[0].bias) + 30.0)

    def forward(self, x):
        x = self.linear(x)
        x = self.seq(x)
        return x


class TestFakeSparsity(TestCase):
    def test_masking_logic(self):
        model = nn.Linear(16, 16, bias=False)
        model.weight = nn.Parameter(torch.eye(16))
        x = torch.randn(3, 16)
        self.assertEqual(torch.mm(x, torch.eye(16)), model(x))

        mask = torch.zeros(16, 16)
        sparsity = utils.FakeSparsity(mask)
        parametrize.register_parametrization(model, "weight", sparsity)

        x = torch.randn(3, 16)
        self.assertEqual(torch.zeros(3, 16), model(x))

    def test_weights_parametrized(self):
        model = ModelUnderTest(bias=False)

        if hasattr(model.linear, "parametrizations"):
            raise AssertionError("model.linear should not have parametrizations")
        if hasattr(model.seq[0], "parametrizations"):
            raise AssertionError("model.seq[0] should not have parametrizations")
        if hasattr(model.seq[1], "parametrizations"):
            raise AssertionError("model.seq[1] should not have parametrizations")
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[1], "weight", utils.FakeSparsity(mask)
        )

        if not hasattr(model.linear, "parametrizations"):
            raise AssertionError("model.linear should have parametrizations")
        if not parametrize.is_parametrized(model.linear, "weight"):
            raise AssertionError("model.linear.weight should be parametrized")
        if not hasattr(model.seq[0], "parametrizations"):
            raise AssertionError("model.seq[0] should have parametrizations")
        if not parametrize.is_parametrized(model.linear, "weight"):
            raise AssertionError("model.linear.weight should be parametrized")
        if not hasattr(model.seq[1], "parametrizations"):
            raise AssertionError("model.seq[1] should have parametrizations")
        if not parametrize.is_parametrized(model.linear, "weight"):
            raise AssertionError("model.linear.weight should be parametrized")

    def test_state_dict_preserved(self):
        model_save = ModelUnderTest(bias=False)

        mask = torch.eye(16)
        parametrize.register_parametrization(
            model_save.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model_save.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model_save.seq[1], "weight", utils.FakeSparsity(mask)
        )
        state_dict = model_save.state_dict()

        model_load = ModelUnderTest(bias=False)
        mask = torch.zeros(model_load.linear.weight.shape)
        parametrize.register_parametrization(
            model_load.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.zeros(model_load.seq[0].weight.shape)
        parametrize.register_parametrization(
            model_load.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.zeros(model_load.seq[1].weight.shape)
        parametrize.register_parametrization(
            model_load.seq[1], "weight", utils.FakeSparsity(mask)
        )
        # Keep this strict, as we are not loading the 'mask'
        model_load.load_state_dict(state_dict, strict=False)

        # Check the parametrizations are preserved
        if not hasattr(model_load.linear, "parametrizations"):
            raise AssertionError("model_load.linear should have parametrizations")
        if not parametrize.is_parametrized(model_load.linear, "weight"):
            raise AssertionError("model_load.linear.weight should be parametrized")
        if not hasattr(model_load.seq[0], "parametrizations"):
            raise AssertionError("model_load.seq[0] should have parametrizations")
        if not parametrize.is_parametrized(model_load.linear, "weight"):
            raise AssertionError("model_load.linear.weight should be parametrized")
        if not hasattr(model_load.seq[1], "parametrizations"):
            raise AssertionError("model_load.seq[1] should have parametrizations")
        if not parametrize.is_parametrized(model_load.linear, "weight"):
            raise AssertionError("model_load.linear.weight should be parametrized")

        # Check the weights are preserved
        self.assertEqual(
            model_save.linear.parametrizations["weight"].original,
            model_load.linear.parametrizations["weight"].original,
        )
        self.assertEqual(
            model_save.seq[0].parametrizations["weight"].original,
            model_load.seq[0].parametrizations["weight"].original,
        )
        self.assertEqual(
            model_save.seq[1].parametrizations["weight"].original,
            model_load.seq[1].parametrizations["weight"].original,
        )

        # Check the masks are not preserved in the state_dict
        # We store the state_dicts in the sparsifier, not in the model itself.
        # TODO: Need to find a clean way of exporting the parametrized model
        self.assertNotEqual(
            model_save.linear.parametrizations["weight"][0].mask,
            model_load.linear.parametrizations["weight"][0].mask,
        )
        self.assertNotEqual(
            model_save.seq[0].parametrizations["weight"][0].mask,
            model_load.seq[0].parametrizations["weight"][0].mask,
        )
        self.assertNotEqual(
            model_save.seq[1].parametrizations["weight"][0].mask,
            model_load.seq[1].parametrizations["weight"][0].mask,
        )

    def test_jit_trace(self):
        model = ModelUnderTest(bias=False)

        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.linear, "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[0], "weight", utils.FakeSparsity(mask)
        )
        mask = torch.eye(16)
        parametrize.register_parametrization(
            model.seq[1], "weight", utils.FakeSparsity(mask)
        )

        # Tracing
        example_x = torch.ones(3, 16)
        model_trace = torch.jit.trace_module(model, {"forward": example_x})

        x = torch.randn(3, 16)
        y = model(x)
        y_hat = model_trace(x)
        self.assertEqual(y_hat, y)


if __name__ == "__main__":
    raise_on_run_directly("test/test_ao_sparsity.py")
