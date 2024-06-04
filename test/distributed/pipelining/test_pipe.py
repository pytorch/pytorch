# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
from model_registry import MLPModule, ModelWithParamAlias

import torch
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
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        x = torch.mm(x, self.mm_param1)  # mutli-use param
        skip_connection = x
        x = x + y
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)  # mutli-use param
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


EXPECTED_N_STAGES = {
    ExampleCode: 4,
    MultiMLP: 4,
    ModelWithParamAlias: 2,
}

# Currently, we don't enforce full set equality on the FQNs between the original
# and pipelined models, because in the multi-use param case, PP will deduplicate
# the FQNs from the state_dict.
# TODO
CHECK_FQN_SET_EQUALITY = False


class PipeTests(TestCase):
    @parametrize("ModelClass", [ExampleCode, MultiMLP, ModelWithParamAlias])
    def test_model_split(self, ModelClass):
        mod = ModelClass()
        x = torch.randn(batch_size, d_hid)
        y = torch.randn(batch_size, d_hid)

        pipe = pipeline(
            mod,
            num_chunks=4,
            example_args=(x, y),
        )

        assert (
            pipe.num_stages == EXPECTED_N_STAGES[ModelClass]
        ), f"nstages = {pipe.num_stages}, expect {EXPECTED_N_STAGES[ModelClass]}"

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
            stage_fqns = set(stage_mod.state_dict().keys())
            assert stage_fqns.issubset(old_names)
            new_names.update(stage_fqns)

        if CHECK_FQN_SET_EQUALITY:
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
