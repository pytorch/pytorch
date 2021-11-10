# Owner(s): ["oncall: distributed"]

import functools
import os
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed._fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed._fsdp.wrap import auto_wrap, default_auto_wrap_policy, enable_wrap, wrap
from torch.testing._internal.common_fsdp import (
    DummyProcessGroup
)
from torch.testing._internal.common_utils import (
    run_tests,
    find_free_port,
    TestCase
)


class TestAutoWrap(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # For all the tests here, we use a fake group
        self.process_group = DummyProcessGroup(rank=0, size=1)

    def test_wrap(self):
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5))
        self.assertTrue(isinstance(layer, FSDP))
        self.assertEqual(layer.rank, self.process_group.rank())
        self.assertEqual(layer.world_size, self.process_group.size())

    def test_wrap_disabled_outside_context(self):
        layer = wrap(nn.Linear(5, 5))
        self.assertTrue(isinstance(layer, nn.Linear))

    def test_wrap_override_defaults(self):
        new_process_group = DummyProcessGroup(rank=0, size=2)
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5), process_group=new_process_group)
        self.assertTrue(isinstance(layer, FSDP))
        self.assertEqual(layer.rank, 0)
        self.assertEqual(layer.world_size, 2)

    def test_auto_wrap(self):
        """
        Test to ensure with auto wrap, we wrap child modules correctly based on the min_num_params.
        ``nn.Linear(5, 5)`` does not exceed the bucket size, but combined they do.
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            sequential = nn.Sequential(
                nn.Linear(5, 5), nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
            )
            my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model.module[0], nn.Linear))
        self.assertTrue(isinstance(model.module[1], nn.Linear))
        self.assertTrue(isinstance(model.module[2], FSDP))
        self.assertTrue(isinstance(model.module[2].module[0], nn.Linear))
        self.assertTrue(isinstance(model.module[2].module[1], nn.Linear))

    def test_auto_wrap_preset_exclude_wrap(self):
        """
        Test to ensure excluded modules are not wrapped, regardless if the total param size is greater than the
        min_num_params. the default_auto_wrap_policy excludes wrapping for {nn.ModuleList, nn.ModuleDict}
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            sequential = nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])
            my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        self.assertTrue(isinstance(model, nn.ModuleList))
        self.assertTrue(isinstance(model[0], nn.Linear))
        self.assertTrue(isinstance(model[1], nn.Linear))

    def test_auto_wrap_preset_exclude_wrap_include_children(self):
        """
        Test to ensure excluded modules are not wrapped, but children are if param size is greater than
        min_num_params
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            sequential = nn.ModuleList([nn.Linear(10, 10)])
            my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        self.assertTrue(isinstance(model, nn.ModuleList))
        self.assertTrue(isinstance(model[0], FSDP))

    def test_auto_wrap_preset_force_leaf(self):
        """
        Test to ensure force-leaf modules are not wrapped, and children are not wrapped. The
        default_auto_wrap_policy forces leaf modules of type {nn.MultiheadAttention} to not be wrapped
        """
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            sequential = nn.Sequential(nn.Linear(10, 10), nn.MultiheadAttention(100, 1))
            my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
            model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
        self.assertTrue(isinstance(model.module[0], FSDP))
        # Assert children of multihead attention are not wrapped
        self.assertTrue(isinstance(model.module[1], nn.MultiheadAttention))
        self.assertTrue(isinstance(model.module[1].out_proj, nn.Linear))

    def test_auto_wrap_preset_force_leaf_custom(self):
        """
        Test to ensure force-leaf modules are not wrapped.
        """
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy,
            min_num_params=40,
            force_leaf_modules=default_auto_wrap_policy.FORCE_LEAF_MODULES.union({nn.Linear}),
        )
        with enable_wrap(
            auto_wrap_policy=my_auto_wrap_policy,
            wrapper_cls=FSDP,
            process_group=self.process_group,
        ):
            sequential = nn.Sequential(nn.Linear(10, 10), nn.ModuleList([nn.Linear(10, 10)]))
            model = auto_wrap(sequential)
        # Model was wrapped in FSDP as no inner modules were wrapped.
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model.module[0], nn.Linear))
        self.assertTrue(isinstance(model.module[1], nn.ModuleList))

    @unittest.skipIf(not torch.cuda.is_available(), "Test Requires CUDA")
    def test_auto_wrap_smoke_test(self):
        device = torch.device("cuda")
        torch.cuda.set_device(0)

        # Random port in case the next test run quickly, same port would cause conflict.
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)

        try:
            with enable_wrap(wrapper_cls=FSDP):
                sequential = nn.Sequential(
                    nn.Linear(5, 5), nn.Linear(5, 5), nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
                )
                my_auto_wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=40)
                model = auto_wrap(sequential, auto_wrap_policy=my_auto_wrap_policy)
            model.to(device)
            input = torch.rand((1, 5), dtype=torch.float).to(device)
            output = model(input)
            loss = F.mse_loss(input, output)
            loss.backward()
        finally:
            torch.distributed.destroy_process_group()
            del os.environ["MASTER_ADDR"]
            del os.environ["MASTER_PORT"]


if __name__ == "__main__":
    run_tests()
