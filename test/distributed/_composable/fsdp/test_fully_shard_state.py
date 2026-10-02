# Owner(s): ["oncall: distributed"]

import copy

import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.testing._internal.common_fsdp import FSDPTestMultiThread, MLP
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.two_tensor import TwoTensor


class TestFullyShardState(FSDPTestMultiThread):
    hw_classification = HardwareClassification.GENERIC

    @property
    def world_size(self) -> int:
        return 1

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

    def test_fully_shard_reapply(self):
        model = MLP(8)
        fully_shard(model)
        with self.assertRaisesRegex(
            AssertionError,
            "Each distinct composable distributed API can only be applied to a module once.",
        ):
            fully_shard(model)

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

    def test_fully_shard_deepcopy(self):
        model = MLP(8)
        fully_shard(model)
        with self.assertRaisesRegex(AssertionError, "FSDP does not support deepcopy"):
            copy.deepcopy(model)

    @parametrize("use_wrapper", [False, True])
    def test_sharded_param_storage(self, use_wrapper: bool):
        model = nn.Linear(8, 8, bias=False)
        if use_wrapper:
            model.weight = nn.Parameter(TwoTensor(model.weight, model.weight.clone()))
        fully_shard(model)
        param_group = model._get_fsdp_state()._fsdp_param_group
        self.assertIsNotNone(param_group)
        sharded_data = param_group.fsdp_params[0]._sharded_param_data
        if use_wrapper:
            if not isinstance(sharded_data, TwoTensor):
                raise AssertionError(f"Expected TwoTensor, got {type(sharded_data)}")
            tensors = (sharded_data.a, sharded_data.b)
        else:
            tensors = (sharded_data,)
        storage_sizes = [tensor.untyped_storage().size() for tensor in tensors]
        self.assertTrue(all(size > 0 for size in storage_sizes))

        model._free_sharded_params()
        self.assertEqual(
            [tensor.untyped_storage().size() for tensor in tensors],
            [0] * len(tensors),
        )
        model._restore_sharded_params()
        self.assertEqual(
            [tensor.untyped_storage().size() for tensor in tensors], storage_sizes
        )


instantiate_parametrized_tests(TestFullyShardState)

if __name__ == "__main__":
    run_tests()
