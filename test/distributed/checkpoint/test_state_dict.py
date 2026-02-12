# Owner(s): ["oncall: distributed"]

import copy
import functools
import sys
from collections.abc import Callable
from itertools import chain, product
from typing import Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import replicate
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)
from torch.distributed.checkpoint import state_dict as ptd_state_dict
from torch.distributed.checkpoint.state_dict import (
    _patch_model_state_dict,
    _patch_optimizer_state_dict,
    get_model_state_dict,
    get_optimizer_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    fully_shard,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.testing._internal.common_dist_composable import (
    CompositeParamModel,
    UnitModule,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MultiProcessTestCase,
    with_comms,
)
from torch.testing._internal.distributed.common_state_dict import (
    FusionEmbedding,
    FusionEmbeddingWithHook,
    FusionEmbeddingWithModifier,
    VerifyStateDictMixin,
)
from torch.utils._pytree import tree_all, tree_all_only


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestStateDict(DTensorTestBase, VerifyStateDictMixin):
    """Tests state_dict and load_state_dict"""

    @property
    def world_size(self) -> int:
        return min(4, torch.accelerator.device_count())

    def _test_save_load(
        self,
        init_model_optim: Callable,
        test_frozen: bool = False,
        flatten_optimizer: bool = False,
    ) -> None:
        options = StateDictOptions(
            ignore_frozen_params=test_frozen,
            flatten_optimizer_state_dict=flatten_optimizer,
        )
        # Initialize original model and distributed model.
        model, optim, copy_optim, dist_model, dist_optim = init_model_optim()

        # Train 10 steps.
        _dist_optim = [dist_optim] if not isinstance(dist_optim, list) else dist_optim
        for _ in range(10):
            optim.zero_grad()
            for d_optim in _dist_optim:
                d_optim.zero_grad()

            batch = torch.rand(8, 100, device=device_type)
            model(batch).sum().backward()
            dist_model(batch).sum().backward()

            optim.step()
            for d_optim in _dist_optim:
                d_optim.step()

        # We need to ensure gradients don't exist, this the invariant of using DSD.
        optim.zero_grad()

        # Get the state_dict, and compare the result
        msd = model.state_dict()
        osd = optim.state_dict()
        dist_msd, dist_osd = get_state_dict(
            dist_model, optimizers=dist_optim, options=options
        )
        self._verify_msd(msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        if not flatten_optimizer:
            self._verify_osd(model, optim, osd, dist_osd)

        # Initialize a completely new model to simulate checkpoint load.
        _, _, _, dist_model, dist_optim = init_model_optim()

        # Simulate DCP distributed load. We need to first get the state_dict and
        # pass them to DCP to load the saved state_dict from the storage.
        # Then finally we can call set_state_dict().
        if not isinstance(dist_optim, list):
            dist_optim = [dist_optim]
        if test_frozen:
            # We won't be able to load the partial state_dict back.
            return
        # Since we already have the state_dict saved before, no need to call DCP.
        # We can directly load them back. This assert is to ensure that optimizer
        # state storage are initialized.
        # self.assertEqual(len(curr_dist_osd[STATE]), len(dist_osd[STATE]))
        set_model_state_dict(
            dist_model,
            model_state_dict=dist_msd,
            options=options,
        )
        set_optimizer_state_dict(
            dist_model,
            optimizers=dist_optim,
            optim_state_dict=dist_osd,
            options=options,
        )

        # Check if the new state_dict are the same
        dist_msd, dist_osd = get_state_dict(
            dist_model, optimizers=dist_optim, options=options
        )
        self._verify_msd(msd, dist_msd, options)
        # TODO: Ditto
        # self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        if not flatten_optimizer:
            self._verify_osd(model, optim, osd, dist_osd)

        # Test _patch_model_state_dict, and _patch_optimizer_state_dict
        _patch_model_state_dict(dist_model, options=options)
        _patch_optimizer_state_dict(dist_model, optimizers=dist_optim, options=options)
        dist_msd = dist_model.state_dict()
        dist_osd = dist_optim[0].state_dict()
        self._verify_msd(msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        if not flatten_optimizer:
            self._verify_osd(model, optim, osd, dist_osd)

    def _test_fsdp(
        self,
        *,
        use_orig_params: bool,
        use_dtensor: bool,
        wrapping: tuple[nn.Module] = (),
        compile_model: bool = False,
        optimizer_class: type[Optimizer],
    ) -> None:
        if not use_orig_params:
            return

        # TODO: remove this return after we complete the composable API side change for device_mesh
        if use_dtensor:
            return

        def init_model_optim():
            if use_dtensor:
                device_mesh = init_device_mesh(device_type, (self.world_size,))

            orig_model = CompositeParamModel(device=torch.device(device_type))
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-4, foreach=True)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-4, foreach=True)
            if wrapping:
                strategy = set(wrapping)
            else:
                strategy = {UnitModule}
            if use_dtensor:
                device_mesh = init_device_mesh(device_type, (self.world_size,))
                dist_model = FSDP(
                    copy.deepcopy(orig_model),
                    auto_wrap_policy=ModuleWrapPolicy(strategy),
                    use_orig_params=use_orig_params,
                    device_mesh=device_mesh,
                )
            else:
                dist_model = FSDP(
                    copy.deepcopy(orig_model),
                    auto_wrap_policy=ModuleWrapPolicy(strategy),
                    use_orig_params=use_orig_params,
                )

            if compile_model:
                dist_model = torch.compile(dist_model)
            dist_optim = optimizer_class(dist_model.parameters(), lr=1e-4, foreach=True)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp(self) -> None:
        self.run_subtests(
            {
                "use_orig_params": [True, False],
                "use_dtensor": [True, False],
                "wrapping": [(), (nn.Linear, UnitModule)],
                "optimizer_class": [
                    torch.optim.Adam,
                    torch.optim.AdamW,
                    torch.optim.SGD,
                ],
            },
            self._test_fsdp,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_compiled_fsdp(self) -> None:
        self.run_subtests(
            {
                "use_orig_params": [True],
                "use_dtensor": [False],
                "wrapping": [()],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            self._test_fsdp,
        )

    def _test_fsdp2(
        self,
        *,
        reshard_after_forward: Union[bool, int],
        optimizer_class: type[Optimizer],
        compile_model: bool,
        foreach: bool = True,
    ):
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device(device_type))
            orig_optim = optimizer_class(
                orig_model.parameters(), lr=1e-4, foreach=foreach
            )
            copy_optim = optimizer_class(
                orig_model.parameters(), lr=1e-4, foreach=foreach
            )

            dist_model = fully_shard(
                copy.deepcopy(orig_model),
                reshard_after_forward=reshard_after_forward,
            )

            if compile_model:
                dist_model = torch.compile(dist_model)
            dist_optim = optimizer_class(
                dist_model.parameters(), lr=1e-4, foreach=foreach
            )

            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp2(self) -> None:
        self.run_subtests(
            {
                "reshard_after_forward": [True, False],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
                "compile_model": [True, False],
            },
            self._test_fsdp2,
        )

    def _test_ddp(self, use_composable: bool, optimizer_class: type[Optimizer]) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device(device_type))
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-4)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-4)
            if use_composable:
                dist_model = replicate(copy.deepcopy(orig_model))
            else:
                dist_model = DDP(copy.deepcopy(orig_model))
            dist_optim = optimizer_class(dist_model.parameters(), lr=1e-4)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_ddp(self) -> None:
        self.run_subtests(
            {
                "use_composable": [True, False],
                "optimizer_class": [
                    torch.optim.Adam,
                    torch.optim.AdamW,
                    torch.optim.SGD,
                ],
            },
            self._test_ddp,
        )

    def _test_fsdp_ddp(
        self,
        optimizer_class: type[Optimizer],
        optim_in_backward: bool = False,
        test_frozen: bool = False,
    ) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device(device_type))
            if test_frozen:
                for param in chain(
                    orig_model.u1.parameters(), orig_model.u2.parameters()
                ):
                    param.requires_grad = False
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-4)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-4)
            dist_model = copy.deepcopy(orig_model)
            dist_model.l = DDP(dist_model.l)
            dist_model = FSDP(
                copy.deepcopy(orig_model),
                auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                use_orig_params=optim_in_backward,
                ignored_modules=[dist_model.l],
            )
            if optim_in_backward:
                _apply_optimizer_in_backward(
                    optimizer_class, dist_model.parameters(), {"lr": 1e-4}
                )
                dist_optim = [
                    p._in_backward_optimizers[0] for p in dist_model.parameters()
                ]
            else:
                dist_optim = optimizer_class(dist_model.parameters(), lr=1e-4)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim, test_frozen)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp_ddp(self) -> None:
        self.run_subtests(
            {
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            self._test_fsdp_ddp,
        )

    def _test_single_gpu(self, optimizer_class: type[Optimizer]) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device(device_type))
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-4)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-4)
            model_copy = copy.deepcopy(orig_model)
            optim_copy = optimizer_class(model_copy.parameters(), lr=1e-4)
            return orig_model, orig_optim, copy_optim, model_copy, optim_copy

        self._test_save_load(init_model_optim)

    @skip_if_lt_x_gpu(1)
    def test_single_gpu(self) -> None:
        self._test_single_gpu(torch.optim.Adam)
        self._test_single_gpu(torch.optim.AdamW)

    def _test_strict(self, parallelism: str) -> None:
        model = CompositeParamModel(device=torch.device(device_type))
        if parallelism == "DDP":
            model = DDP(model)
        else:
            model = fully_shard(model)

        model_state_dict = get_model_state_dict(model)
        model_state_dict["abc"] = torch.zeros(10)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            set_model_state_dict(model, model_state_dict=model_state_dict)
        key_iter = iter(model_state_dict.keys())
        for key in key_iter:
            if key != "abc":
                break
        model_state_dict.pop(key)
        incompatible_keys = set_model_state_dict(
            model,
            model_state_dict=model_state_dict,
            options=StateDictOptions(strict=False),
        )
        self.assertEqual(incompatible_keys.missing_keys, [key])
        self.assertEqual(incompatible_keys.unexpected_keys, ["abc"])
        model_state_dict.pop("abc")
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            set_model_state_dict(model, model_state_dict=model_state_dict)

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_strict(self) -> None:
        self.run_subtests(
            {"parallelism": ["DDP", "fully_shard"]},
            self._test_strict,
        )

    def _test_cpu_offload_full_state_dict(
        self, optimizer_class: type[Optimizer]
    ) -> None:
        orig_model = CompositeParamModel(device=torch.device(device_type))
        device_mesh = init_device_mesh(device_type, (self.world_size,))
        dist_model = FSDP(
            copy.deepcopy(orig_model),
            auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
            use_orig_params=True,
            device_mesh=device_mesh,
        )

        dist_optim = optimizer_class(dist_model.parameters(), lr=1e-4)

        mst, ost = get_state_dict(
            dist_model,
            dist_optim,
            options=StateDictOptions(cpu_offload=True),
        )

        cpu_device = torch.device("cpu")

        def is_cpu(v):
            if isinstance(v, DTensor):
                return v.device == cpu_device
            elif isinstance(v, ShardedTensor):
                shards = v.local_shards()
                if not shards:
                    return True
                return shards[0].tensor.device == cpu_device
            else:
                return v.device == cpu_device

        self.assertTrue(
            tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, mst)
        )
        self.assertTrue(
            tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, ost)
        )

        mst, ost = get_state_dict(
            dist_model, dist_optim, options=StateDictOptions(full_state_dict=True)
        )

        self.assertTrue(
            tree_all(lambda v: not isinstance(v, (DTensor, ShardedTensor)), mst)
        )
        self.assertTrue(
            tree_all(lambda v: not isinstance(v, (DTensor, ShardedTensor)), ost)
        )

        mst, ost = get_state_dict(
            dist_model,
            dist_optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

        if self.rank == 0:
            self.assertTrue(
                tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, mst)
            )
            self.assertTrue(
                tree_all_only((torch.Tensor, DTensor, ShardedTensor), is_cpu, ost)
            )
        else:
            self.assertEqual(mst, {})
            self.assertEqual(ost, {})

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_cpu_offload_full_state_dict(self) -> None:
        self.run_subtests(
            {"optimizer_class": [torch.optim.Adam, torch.optim.AdamW]},
            self._test_cpu_offload_full_state_dict,
        )

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_activation_ckpt_fqns_ddp(self) -> None:
        """Tests that activation checkpointing prefixes are removed from module names"""
        model = CompositeParamModel(device=torch.device(device_type))
        original_keys = get_model_state_dict(model).keys()

        apply_activation_checkpointing(model)
        model = DDP(model)
        new_keys = get_model_state_dict(model).keys()

        self.assertEqual(original_keys, new_keys)

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_activation_ckpt_fqns_fsdp1(self) -> None:
        self.run_subtests(
            {"use_orig_params": [True, False]},
            self._test_activation_ckpt_fqns_fsdp1,
        )

    def _test_activation_ckpt_fqns_fsdp1(self, use_orig_params: bool) -> None:
        """Tests that activation checkpointing prefixes are removed from module names"""
        model = CompositeParamModel(device=torch.device(device_type))
        original_keys = get_model_state_dict(model).keys()

        apply_activation_checkpointing(model)
        model = FSDP(model, use_orig_params=use_orig_params)
        new_keys = get_model_state_dict(model).keys()

        self.assertEqual(original_keys, new_keys)

    @skip_if_lt_x_gpu(1)
    def test_extra_state(self) -> None:
        model = CompositeParamModel(device=torch.device(device_type))

        def get_extra_state(self):
            return "MyState"

        def set_extra_state(self, state):
            return

        UnitModule.get_extra_state = get_extra_state
        UnitModule.set_extra_state = set_extra_state

        target_model = copy.deepcopy(model)
        set_model_state_dict(target_model, get_model_state_dict(target_model))
        self.assertEqual(model.state_dict()["u1._extra_state"], "MyState")
        self.assertEqual(model.state_dict(), get_model_state_dict(target_model))

    @skip_if_lt_x_gpu(1)
    def test_non_persistent_buffers(self) -> None:
        model = CompositeParamModel(device=torch.device(device_type))
        model.register_buffer(
            "dont_save_me", torch.rand(100, device=device_type), persistent=False
        )
        target_model = copy.deepcopy(model)
        set_model_state_dict(target_model, get_model_state_dict(target_model))
        self.assertEqual(model.state_dict(), get_model_state_dict(target_model))

    def _test_broadcast_from_rank0(self, wrapper) -> None:
        model = CompositeParamModel(device=torch.device(device_type))
        optim = torch.optim.Adam(model.parameters())
        fsdp_model = wrapper(copy.deepcopy(model))
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters())

        batch = torch.rand(8, 100, device=device_type)
        model(batch).sum().backward()
        optim.step()
        states, optim_states = get_state_dict(model, optim)

        fsdp_model(batch).sum().backward()
        fsdp_optim.step()

        def check(equal):
            fsdp_states = get_model_state_dict(
                fsdp_model,
                options=StateDictOptions(full_state_dict=True),
            )
            fsdp_optim_states = get_optimizer_state_dict(
                fsdp_model,
                fsdp_optim,
                options=StateDictOptions(full_state_dict=True),
            )
            if equal:
                self.assertEqual(states, fsdp_states)
                self.assertEqual(optim_states, fsdp_optim_states)
            else:
                self.assertNotEqual(states, fsdp_states)
                self.assertNotEqual(optim_states, fsdp_optim_states)

        check(equal=True)
        fsdp_model(batch).sum().backward()
        fsdp_optim.step()
        check(equal=False)

        # Drop the states to simulate loading from rank0
        if dist.get_rank() > 0:
            load_states = {}
            load_states2 = {}
            load_optim_states = {}
        else:
            load_states = copy.deepcopy(states)
            load_states2 = copy.deepcopy(states)
            load_optim_states = copy.deepcopy(optim_states)

        set_model_state_dict(
            fsdp_model,
            model_state_dict=load_states,
            options=StateDictOptions(broadcast_from_rank0=True, full_state_dict=True),
        )
        set_optimizer_state_dict(
            fsdp_model,
            fsdp_optim,
            optim_state_dict=load_optim_states,
            options=StateDictOptions(broadcast_from_rank0=True, full_state_dict=True),
        )

        check(equal=True)
        # Verify the `strict` flag.
        load_states = load_states2
        if load_states:
            key = next(iter(load_states.keys()))
            load_states.pop(key)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            set_model_state_dict(
                fsdp_model,
                model_state_dict=load_states,
                options=StateDictOptions(
                    broadcast_from_rank0=True, full_state_dict=True
                ),
            )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_broadcast_from_rank0(self) -> None:
        device_mesh = init_device_mesh(device_type, (self.world_size,))
        hsdp_device_mesh = init_device_mesh(device_type, (2, self.world_size // 2))
        self.run_subtests(
            {
                "wrapper": [
                    functools.partial(fully_shard, mesh=device_mesh),
                    functools.partial(FSDP, device_mesh=device_mesh),
                    functools.partial(
                        FSDP,
                        device_mesh=hsdp_device_mesh,
                        sharding_strategy=ShardingStrategy.HYBRID_SHARD,
                    ),
                ]
            },
            self._test_broadcast_from_rank0,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp_root_not_initialized(self) -> None:
        # This test verifies that FSDP root is not initialized but we should
        # still be able to  get the state_dict without errors because
        # fsdp_model.state_dict() will trigger the FSDP initialization.
        device_mesh = init_device_mesh(device_type, (self.world_size,))
        model = CompositeParamModel(device=torch.device(device_type))
        fsdp_model = FSDP(copy.deepcopy(model), device_mesh=device_mesh)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters())
        get_model_state_dict(fsdp_model)
        get_optimizer_state_dict(fsdp_model, fsdp_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_optim_state_dict_param_matching(self) -> None:
        # This test verifies parameters between optim and optim_state_dict
        # "initial_lr" is added to optim_state_dict, but not to the new optim
        # We test whether "initial_lr" appear in optim after
        # set_optimizer_state_dict.
        torch.manual_seed(0)
        model = nn.Sequential(
            *[nn.Linear(4, 4, device=device_type, bias=False) for _ in range(2)]
        )
        for layer in model:
            fully_shard(layer)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=[lambda epoch: 0.95**epoch])
        opt_state_dict = ptd_state_dict.get_optimizer_state_dict(
            model,
            optim,
            options=ptd_state_dict.StateDictOptions(
                full_state_dict=True, cpu_offload=True
            ),
        )
        if dist.get_rank() == 0:
            self.assertTrue("initial_lr" in opt_state_dict["param_groups"][0])

        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        self.assertTrue("initial_lr" not in optim.param_groups[0])

        ptd_state_dict.set_optimizer_state_dict(
            model,
            optim,
            optim_state_dict=opt_state_dict,
            options=ptd_state_dict.StateDictOptions(
                broadcast_from_rank0=True, full_state_dict=True
            ),
        )
        if dist.get_rank() == 0:
            self.assertTrue("initial_lr" in optim.param_groups[0])

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_flattened_osd(self) -> None:
        """
        Test flattened optimizer state dictionaries with different combinations of
        flatten_optimizer_state_dict flag for saving and loading.

        This test verifies that:
        1. We can save optimizer state dict with/without flattening
        2. We can load optimizer state dict with/without flattening
        3. The resulting optimizer state is equivalent regardless of flattening options
        """
        for flatten_to_save, flatten_to_load in product([True, False], repeat=2):
            device_mesh = init_device_mesh(device_type, (self.world_size,))
            model = CompositeParamModel(device=torch.device(device_type))
            fsdp_model = fully_shard(copy.deepcopy(model), mesh=device_mesh)
            fsdp_optim = torch.optim.AdamW(fsdp_model.parameters())
            batch = torch.rand(8, 100, device=device_type)
            fsdp_model(batch).sum().backward()
            fsdp_optim.step()
            fsdp_optim.zero_grad()

            # Get optimizer state dict with/without flattening option
            osd = get_optimizer_state_dict(
                fsdp_model,
                fsdp_optim,
                options=StateDictOptions(flatten_optimizer_state_dict=flatten_to_save),
            )

            # Create a new optimizer and load the state from osd
            fsdp_optim2 = torch.optim.AdamW(fsdp_model.parameters())
            set_optimizer_state_dict(
                fsdp_model,
                optimizers=fsdp_optim2,
                optim_state_dict=osd,
                options=StateDictOptions(flatten_optimizer_state_dict=flatten_to_load),
            )

            # Verify the loaded optimizer state matches the original
            self.assertEqual(fsdp_optim.state_dict(), fsdp_optim2.state_dict())

    def _test_deprecate_partial(self) -> None:
        model = CompositeParamModel(device=torch.device(device_type))

        model_state_dict1 = get_model_state_dict(model)
        model_state_dict1 = copy.deepcopy(model_state_dict1)
        with self.assertWarnsRegex(
            FutureWarning,
            "Getting submodules only model/optim state_dict is deprecated",
        ):
            model_state_dict2 = get_model_state_dict(model, submodules={model.l})
        model_state_dict2 = copy.deepcopy(model_state_dict2)
        with self.assertWarnsRegex(
            FutureWarning,
            "Getting submodules only model/optim state_dict is deprecated",
        ):
            model_state_dict3 = get_model_state_dict(
                model,
                submodules={model.l},
                options=StateDictOptions(keep_submodule_prefixes=False),
            )
        model_state_dict3 = copy.deepcopy(model_state_dict3)
        self.assertEqual(len(model_state_dict2), 2)
        self.assertEqual(len(model_state_dict3), 2)
        for key in model_state_dict3:
            full_fqn = f"l.{key}"
            value1 = model_state_dict1[full_fqn]
            value2 = model_state_dict2[full_fqn]
            value3 = model_state_dict3[key]
            self.assertEqual(value1, value2)
            self.assertEqual(value2, value3)

        zeros_state_dict = {
            k: torch.zeros_like(v) for k, v in model_state_dict1.items()
        }
        model.load_state_dict(zeros_state_dict)
        set_model_state_dict(
            model,
            model_state_dict=model_state_dict2,
            options=StateDictOptions(strict=False),
        )
        self.assertEqual(model.l.weight, model_state_dict1["l.weight"])
        self.assertEqual(model.l.bias, model_state_dict1["l.bias"])

        model.load_state_dict(zeros_state_dict)
        with self.assertWarnsRegex(FutureWarning, "Passing model_state_dict as a "):
            set_model_state_dict(
                model,
                model_state_dict={model.l: model_state_dict3},
                options=StateDictOptions(strict=False),
            )
        self.assertEqual(model.l.weight, model_state_dict1["l.weight"])
        self.assertEqual(model.l.bias, model_state_dict1["l.bias"])

    def _test_deprecate_fsdp_api(self) -> None:
        device_mesh = init_device_mesh(device_type, (self.world_size,))
        model = CompositeParamModel(device=torch.device(device_type))
        fsdp_model = FSDP(copy.deepcopy(model), device_mesh=device_mesh)
        with self.assertWarnsRegex(
            FutureWarning,
            r"FSDP.state_dict_type\(\) and FSDP.set_state_dict_type\(\) are being deprecated",
        ):
            with FSDP.state_dict_type(fsdp_model, StateDictType.FULL_STATE_DICT):
                fsdp_model.state_dict()

        with self.assertRaisesRegex(AssertionError, "FutureWarning not triggered"):
            with self.assertWarnsRegex(
                FutureWarning,
                r"FSDP.state_dict_type\(\) and FSDP.set_state_dict_type\(\) are being deprecated",
            ):
                get_model_state_dict(model)

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_deprecate_api(self) -> None:
        self._test_deprecate_partial()
        self._test_deprecate_fsdp_api()

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_shared_weight(self):
        class TiedEmbeddingModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.decoder = nn.Linear(embedding_dim, vocab_size)
                self.decoder.weight = self.embedding.weight  # Tying weights

            def forward(self, input):
                input = (input * 10).to(torch.int)
                embedded = self.embedding(input)
                output = self.decoder(embedded)
                return output

        def init_model_optim():
            device_mesh = init_device_mesh(device_type, (self.world_size,))
            orig_model = TiedEmbeddingModel(10000, 300).to(torch.device(device_type))
            orig_optim = torch.optim.AdamW(orig_model.parameters(), lr=1e-4)
            copy_optim = torch.optim.AdamW(orig_model.parameters(), lr=1e-4)
            dist_model = FSDP(copy.deepcopy(orig_model), device_mesh=device_mesh)
            dist_optim = torch.optim.AdamW(dist_model.parameters(), lr=1e-4)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)
        self.run_subtests(
            {
                "init_model_optim": [init_model_optim],
                "flatten_optimizer": [True, False],
            },
            self._test_save_load,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_setting_meta_device_model(self) -> None:
        # This test verifies that we can set model state dict by a meta device model
        torch.manual_seed(0)
        with torch.device("meta"):
            meta_model = nn.Sequential(*[nn.Linear(4, 4, bias=False) for _ in range(2)])
            for layer in meta_model:
                fully_shard(layer)
            fully_shard(meta_model)
        with torch.device("cpu"):
            cpu_model = nn.Sequential(*[nn.Linear(4, 4, bias=False) for _ in range(2)])
            full_sd = cpu_model.state_dict()
        set_model_state_dict(
            meta_model,
            model_state_dict=full_sd,
            options=StateDictOptions(full_state_dict=True, strict=False),
        )
        meta_model_state_dict = meta_model.state_dict()
        cpu_model_state_dict = get_model_state_dict(cpu_model)
        for cpu_model_key, cpu_model_value in cpu_model_state_dict.items():
            meta_model_value = (
                meta_model_state_dict[cpu_model_key]
                .full_tensor()
                .to(device=cpu_model_value.device)
            )
            self.assertEqual(cpu_model_value, meta_model_value)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_setting_meta_device_model_broadcasting_and_memory(self) -> None:
        # This test verifies that we can set model state dict by a meta device model
        # With the correlated changes in state_dict, meta device model should be accepted
        # in broadcasting and get copied successfully.
        torch.manual_seed(0)
        with torch.device("meta"):
            meta_model = nn.Sequential(
                *[nn.Linear(10000, 10000, bias=False) for _ in range(4)]
            )
            for layer in meta_model:
                fully_shard(layer)
            fully_shard(meta_model)
        with torch.device("cpu"):
            cpu_model = nn.Sequential(
                *[nn.Linear(10000, 10000, bias=False) for _ in range(4)]
            )
            full_sd = cpu_model.state_dict()
        set_model_state_dict(
            meta_model,
            model_state_dict=full_sd,
            options=StateDictOptions(
                broadcast_from_rank0=True, full_state_dict=True, strict=False
            ),
        )
        meta_model_state_dict = meta_model.state_dict()
        cpu_model_state_dict = get_model_state_dict(cpu_model)
        for cpu_model_key, cpu_model_value in cpu_model_state_dict.items():
            meta_model_value = (
                meta_model_state_dict[cpu_model_key]
                .full_tensor()
                .to(device=cpu_model_value.device)
            )
            self.assertEqual(cpu_model_value, meta_model_value)
        # Memory allocated and reserved are lower due to the change at _distribute_tensors
        # from view to clone. This test would fail if with view due to higher memory cost.
        memory_allocated = (
            torch.get_device_module(device_type).memory_allocated(0) / 1024 / 1024
        )
        memory_reserved = (
            torch.get_device_module(device_type).memory_reserved(0) / 1024 / 1024
        )
        self.assertTrue(memory_allocated <= 384)
        self.assertTrue(memory_reserved <= 768)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_set_cpu_model_state_dict_broadcast_from_rank0(self) -> None:
        torch.manual_seed(42)
        model = nn.Linear(2, 2)
        expected_state_dict = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        state_dict = expected_state_dict if torch.distributed.get_rank() == 0 else {}
        model._apply(lambda t: torch.zeros_like(t))

        set_model_state_dict(
            model,
            state_dict,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )

        for (actual_name, tensor), (expected_name, expected_tensor) in zip(
            model.state_dict().items(),
            expected_state_dict.items(),
        ):
            assert actual_name == expected_name
            torch.testing.assert_close(tensor, expected_tensor, msg=expected_name)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_multi_device_load_model_state_dict(self) -> None:
        torch.manual_seed(0)
        with torch.device("meta"):
            meta_submodel = nn.Linear(4, 4, bias=False)
        with torch.device("cpu"):
            cpu_submodel = nn.Linear(4, 4, bias=False)
        with torch.device(device_type):
            acc_submodel = nn.Linear(4, 4, bias=False)

        two_device_model_with_meta = nn.Sequential(meta_submodel, acc_submodel)
        two_device_model_without_meta = nn.Sequential(cpu_submodel, acc_submodel)

        with torch.device("cpu"):
            model_to_set = nn.Sequential(
                *[nn.Linear(4, 4, bias=False) for _ in range(2)]
            )
            full_sd = model_to_set.state_dict()
        set_model_state_dict(
            two_device_model_with_meta,
            model_state_dict=full_sd,
            options=StateDictOptions(
                broadcast_from_rank0=True, full_state_dict=True, strict=False
            ),
        )
        with self.assertRaisesRegex(ValueError, "Multiple devices found"):
            set_model_state_dict(
                two_device_model_without_meta,
                model_state_dict=full_sd,
                options=StateDictOptions(
                    broadcast_from_rank0=True, full_state_dict=True, strict=False
                ),
            )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_state_dict_with_hook_on_keys(self) -> None:
        with torch.device("meta"):
            metamodel = FusionEmbedding(4, 4, 4)
        with torch.device(device_type):
            gpumodel = FusionEmbeddingWithHook(4, 4, 4)
        gpumodel_state_dict = get_model_state_dict(gpumodel)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            set_model_state_dict(metamodel, gpumodel_state_dict)
        with torch.device("meta"):
            metamodel_modified = FusionEmbeddingWithModifier(4, 4, 4)
        set_model_state_dict(metamodel_modified, gpumodel_state_dict)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_multi_param_groups(self) -> None:
        class TestModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(64, 64)
                self.fc1 = torch.nn.Linear(64, 64)

            def forward(self, x):
                return self.fc1(self.fc(x))

        device_mesh = init_device_mesh(device_type, (self.world_size,))
        model = TestModel().to(device_type)
        parallelize_module(
            model,
            device_mesh,
            {
                "fc": ColwiseParallel(use_local_output=False),
                "fc1": RowwiseParallel(use_local_output=False),
            },
        )

        def _test_multi(
            optim_kwargs, full_state_dict, broadcast_from_rank0, cpu_offload
        ):
            if broadcast_from_rank0 and not full_state_dict:
                return

            optim = torch.optim.AdamW(**optim_kwargs)
            optim.zero_grad()
            model(torch.randn(64, 64, device=device_type)).sum().backward()
            optim.step()
            optim.zero_grad()

            options = torch.distributed.checkpoint.state_dict.StateDictOptions(
                full_state_dict=full_state_dict,
                broadcast_from_rank0=broadcast_from_rank0,
                cpu_offload=cpu_offload,
            )
            optim_state_dict = get_optimizer_state_dict(model, optim, options=options)

            new_optim = torch.optim.AdamW(**optim_kwargs)
            set_optimizer_state_dict(
                model, new_optim, optim_state_dict, options=options
            )
            self.assertEqual(optim.param_groups, new_optim.param_groups)
            self.assertEqual(optim.state, new_optim.state)

        _multi_optim_kwargs = {
            "params": [
                {"params": [model.fc.weight]},
                {"params": [model.fc1.weight], "lr": 0.2},
            ],
            "lr": 0.1,
        }
        _multi_optim_kwargs_empty_pg = {
            "params": [
                {"params": [model.fc.weight, model.fc1.weight]},
                {"params": [], "lr": 0.2},  # empty pg group here
            ],
            "lr": 0.1,
        }

        self.run_subtests(
            {
                "optim_kwargs": [_multi_optim_kwargs_empty_pg, _multi_optim_kwargs],
                "full_state_dict": [False, True],
                "broadcast_from_rank0": [False, True],
                # TODO: cpu_offload will cause get_optimizer_state_dict complain that
                # tensors are not on GPU.
                "cpu_offload": [False],
            },
            _test_multi,
        )


class TestNoComm(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(1)
    def test_no_dist(self) -> None:
        model = CompositeParamModel(device=torch.device(device_type))
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

        self.assertFalse(dist.is_initialized())
        msd = get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
        )
        for v in msd.values():
            self.assertFalse(v.is_cuda)
        self.assertEqual(model.state_dict(), msd)
        set_model_state_dict(model, model.state_dict())
        osd = get_optimizer_state_dict(
            model,
            optim,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        set_optimizer_state_dict(model, optim, osd)
        set_optimizer_state_dict(model, optim, optim.state_dict())


if __name__ == "__main__":
    run_tests()
