# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import torch
from torch.distributed.pipelining import pipeline, SplitPoint
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import run_tests, TestCase


d_hid = 16
n_layers = 8
microbatch_size = 4


class MLPModule(torch.nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class TransformerLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(*[MLPModule(d_hid) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerTests(TestCase):
    def test_ir(self, device):
        transformer = TransformerLike().to(device)
        x = torch.randn(microbatch_size, d_hid, device=device)

        # Split into 2 stages
        num_stages = 2
        split_spec = {f"layers.{n_layers // num_stages}": SplitPoint.BEGINNING}

        pipe = pipeline(
            transformer,
            (x,),
            split_spec=split_spec,
        )
        if pipe.num_stages != num_stages:
            raise AssertionError(f"{pipe.num_stages=}, expect {num_stages}")

        def get_layers(module):
            layers = [name for name, _ in module.layers.named_children()]
            return layers

        # Collect all layers in pipe
        layers = []
        for stage_idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(stage_idx)
            layers += get_layers(stage_mod)

        # Check layer completeness
        orig_layers = get_layers(transformer)
        if sorted(layers) != sorted(orig_layers):
            raise AssertionError(f"{layers} != {orig_layers}")
        print("Layers matched!")

        # Check equivalence
        ref = transformer(x)
        out = pipe(x)[0]
        torch.testing.assert_close(out, ref)
        print(f"Equivalence test passed {torch.sum(out)} ref {torch.sum(ref)}")


devices = ["cpu", "cuda", "hpu", "xpu"]
instantiate_device_type_tests(
    TransformerTests, globals(), only_for=devices, allow_xpu=True
)

if __name__ == "__main__":
    run_tests()
