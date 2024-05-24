# Owner(s): ["oncall: distributed"]

import copy
import unittest

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule, fully_shard, share_comm_ctx
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, MLP
from torch.testing._internal.common_utils import run_tests


class TestFullyShardState(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 1

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fully_shard_state(self):
        """
        Tests the ability to get the state object from a fully sharded module.
        """
        num_mlps = 3
        model = nn.Sequential(*[MLP(8) for _ in range(num_mlps)])
        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        root_state = fully_shard.state(model)
        self.assertTrue(root_state is not None)
        all_states = [root_state] + [fully_shard.state(mlp) for mlp in model]
        # Check that each `fully_shard` call constructs a distinct state object
        self.assertEqual(len(set(all_states)), num_mlps + 1)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fully_shard_reapply(self):
        model = MLP(8)
        fully_shard(model)
        with self.assertRaisesRegex(
            AssertionError,
            "Each distinct composable distributed API can only be applied to a module once.",
        ):
            fully_shard(model)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fully_shard_cls(self):
        # Check that we only swap class for the module passed to `fully_shard`
        model = MLP(8)
        fully_shard(model)
        self.assertTrue(isinstance(model, MLP))
        self.assertTrue(isinstance(model, FSDPModule))
        self.assertEqual(model.__class__.__name__, "FSDPMLP")
        for module in model.modules():
            if module is model:
                continue
            self.assertFalse(isinstance(module, FSDPModule))

        # Check that slicing into a `Sequential` does not preserve FSDP
        model = nn.Sequential(*[MLP(8) for _ in range(3)])
        fully_shard(model)
        self.assertTrue(isinstance(model, nn.Sequential))
        self.assertTrue(isinstance(model, FSDPModule))
        self.assertEqual(model.__class__.__name__, "FSDPSequential")
        sliced_model = model[:2]
        self.assertTrue(isinstance(sliced_model, nn.Sequential))
        self.assertFalse(isinstance(sliced_model, FSDPModule))

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fully_shard_unsupported_module_cls(self):
        regex = (
            r"fully\_shard does not support containers that do not implement forward"
        )
        model = nn.ModuleList([MLP(8) for _ in range(3)])
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model)
        model = nn.ModuleDict({"1": MLP(8), "2": MLP(8)})
        with self.assertRaisesRegex(ValueError, regex):
            fully_shard(model)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_fully_shard_deepcopy(self):
        model = MLP(8)
        fully_shard(model)
        with self.assertRaisesRegex(AssertionError, "FSDP does not support deepcopy"):
            copy.deepcopy(model)

    @unittest.skipIf(not TEST_CUDA, "no cuda")
    def test_share_comm_ctx(self):
        # Check that passing in a non-FSDP-module raises an error
        model = nn.ModuleList([MLP(8) for _ in range(3)])
        with self.assertRaisesRegex(ValueError, "Expects list of FSDPModules but got"):
            share_comm_ctx([model])

        def check_same_comm_ctx(model: nn.Module):
            fsdp_states = [
                module._get_fsdp_state()
                for module in model.modules()
                if isinstance(module, FSDPModule)
            ]
            comm_ctx = fsdp_states[0]._fsdp_param_group.comm_ctx
            for fsdp_state in fsdp_states[1:]:
                self.assertEqual(comm_ctx, fsdp_state._fsdp_param_group.comm_ctx)

        # Check passing in a list of root FSDP modules works, where the root
        # FSDP modules have not yet been lazily initialized
        model = nn.ModuleList([MLP(8) for _ in range(3)])
        for mlp in model:
            fully_shard(mlp)
        share_comm_ctx(list(model.children()))
        x = torch.randn((1, 8), device="cuda")
        for mlp in model:
            x = mlp(x)  # run forward to lazy init
        check_same_comm_ctx(model)

        # Check passing in a list of all FSDP modules before lazy init
        model = nn.ModuleList([MLP(8) for _ in range(3)])
        for mlp in model:
            fully_shard(mlp.out_proj)
            fully_shard(mlp)
        fsdp_modules = [
            module for module in model.modules() if isinstance(module, FSDPModule)
        ]
        share_comm_ctx(fsdp_modules)
        for mlp in model:
            x = mlp(x)  # run forward to lazy init
        check_same_comm_ctx(model)

        # Check passing in a list of all FSDP modules after lazy init
        model = nn.ModuleList([MLP(8) for _ in range(3)])
        for mlp in model:
            fully_shard(mlp.out_proj)
            fully_shard(mlp)
        for mlp in model:
            x = mlp(x)  # run forward to lazy init
        fsdp_modules = [
            module for module in model.modules() if isinstance(module, FSDPModule)
        ]
        share_comm_ctx(fsdp_modules)
        check_same_comm_ctx(model)


if __name__ == "__main__":
    run_tests()
