# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
import torch

from model_registry import MLPModule
from torch.distributed.pipelining import pipe_split, pipeline
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


d_hid = 512
batch_size = 256

torch.manual_seed(0)


# Basic example
class ExampleCode(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        x = torch.mm(x, self.mm_param0)
        skip_connection = x
        x = x + y
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        pipe_split()
        x = torch.relu(x)
        x = x + skip_connection
        x = torch.mm(x, self.mm_param2)
        pipe_split()
        x = self.lin2(x)
        x = torch.relu(x)
        return x


class MultiMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp0 = MLPModule(d_hid)
        self.mlp1 = MLPModule(d_hid)
        self.mlp2 = MLPModule(d_hid)
        self.mlp3 = MLPModule(d_hid)

    def forward(self, x, y):
        x = self.mlp0(x)
        pipe_split()
        x = self.mlp1(x)
        pipe_split()
        x = self.mlp2(x)
        pipe_split()
        x = self.mlp3(x)
        return x - y


class PipeTests(TestCase):
    @parametrize("ModelClass", [ExampleCode, MultiMLP])
    def test_model_split(self, ModelClass):
        mod = ModelClass()
        x = torch.randn(batch_size, d_hid)
        y = torch.randn(batch_size, d_hid)

        pipe = pipeline(
            mod,
            num_chunks=4,
            example_args=(x, y),
        )

        assert pipe.num_stages == 4, f"nstages = {pipe.num_stages}, expect 4"

        ref_out = mod(x, y)
        out = pipe(x, y)[0]
        torch.testing.assert_close(out, ref_out)
        print(f"equivalence test passed {torch.sum(out)} ref {torch.sum(ref_out)}")

        # Check qualname
        # state_dict.keys include both parameters and persistent buffers
        old_names = set(mod.state_dict().keys())
        new_names = set()
        for idx in range(pipe.num_stages):
            stage_mod = pipe.get_stage_module(idx)
            new_names.update(stage_mod.state_dict().keys())

        assert (
            old_names == new_names
        ), f"""
        old names {old_names}
        new names {new_names}
        """
        print("Qualname check passed")


instantiate_parametrized_tests(PipeTests)

if __name__ == "__main__":
    run_tests()
