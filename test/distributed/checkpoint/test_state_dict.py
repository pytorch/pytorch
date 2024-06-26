# Owner(s): ["oncall: distributed"]

import copy
import functools
import sys
from itertools import chain
from typing import Callable, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable import fully_shard, replicate

# importing fully_shard as FSDP2 since the original fully_shard is used in this test.
# TODO: remove old composable fully_shard so that we don't have to import new fully_shard as FSDP2
from torch.distributed._composable.fsdp import (
    fully_shard as FSDP2,
    fully_shard as fsdp_fully_shard,
)
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor import DTensor, init_device_mesh
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
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.optim import _apply_optimizer_in_backward
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
from torch.testing._internal.distributed.common_state_dict import VerifyStateDictMixin
from torch.utils._pytree import tree_all, tree_all_only


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
        return min(4, torch.cuda.device_count())

    def _test_save_load(
        self,
        init_model_optim: Callable,
        test_frozen: bool = False,
    ) -> None:
        options = StateDictOptions(ignore_frozen_params=test_frozen)
        # Initialize original model and distributed model.
        model, optim, copy_optim, dist_model, dist_optim = init_model_optim()

        # Train 10 steps.
        for i in range(10):
            batch = torch.rand(8, 100, device="cuda")
            model(batch).sum().backward()
            optim.step()
            dist_model(batch).sum().backward()
            if not isinstance(dist_optim, list):
                dist_optim.step()
                dist_optim.zero_grad()
            else:
                for _dist_optim in dist_optim:
                    _dist_optim.zero_grad()
            optim.zero_grad()

        # Get the state_dict, and compare the result
        msd = model.state_dict()
        osd = optim.state_dict()
        dist_msd, dist_osd = get_state_dict(
            dist_model, optimizers=dist_optim, options=options
        )
        self._verify_msd(msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
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
        # We can directly load them back. This asser is to ensure that optimizer
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
        self._verify_osd(model, optim, osd, dist_osd)

        # Test _patch_model_state_dict, and _patch_optimizer_state_dict
        _patch_model_state_dict(dist_model, options=options)
        _patch_optimizer_state_dict(dist_model, optimizers=dist_optim, options=options)
        dist_msd = dist_model.state_dict()
        dist_osd = dist_optim[0].state_dict()
        self._verify_msd(msd, dist_msd, options)
        self._verify_osd_by_load(model, optim, copy_optim, dist_osd)
        self._verify_osd(model, optim, osd, dist_osd)

    def _test_fsdp(
        self,
        *,
        use_orig_params: bool,
        use_composable: bool,
        use_dtensor: bool,
        wrapping: Tuple[nn.Module] = (),
        compile_model: bool = False,
        optimizer_class: Type[Optimizer],
    ) -> None:
        if not use_orig_params and use_composable:
            return

        # TODO: remove this return after we complete the composable API side change for device_mesh
        if use_composable and use_dtensor:
            return

        def init_model_optim():
            if use_dtensor:
                device_mesh = init_device_mesh("cuda", (self.world_size,))

            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            if wrapping:
                strategy = set(wrapping)
            else:
                strategy = {UnitModule}
            if use_composable:
                dist_model = fully_shard(
                    copy.deepcopy(orig_model), policy=ModuleWrapPolicy(strategy)
                )
            else:
                if use_dtensor:
                    device_mesh = init_device_mesh("cuda", (self.world_size,))
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
            dist_optim = optimizer_class(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp(self) -> None:
        self.run_subtests(
            {
                "use_orig_params": [True, False],
                "use_composable": [True, False],
                "use_dtensor": [True, False],
                "wrapping": [tuple(), (nn.Linear, UnitModule)],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            self._test_fsdp,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_compiled_fsdp(self) -> None:
        self.run_subtests(
            {
                "use_orig_params": [True],
                "use_composable": [False],
                "use_dtensor": [False],
                "wrapping": [tuple()],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            self._test_fsdp,
        )

    def _test_fsdp2(
        self,
        *,
        reshard_after_forward: Union[bool, int],
        optimizer_class: Type[Optimizer],
        compile_model: bool,
        foreach: bool = True,
    ):
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = optimizer_class(
                orig_model.parameters(), lr=1e-3, foreach=foreach
            )
            copy_optim = optimizer_class(
                orig_model.parameters(), lr=1e-3, foreach=foreach
            )

            dist_model = FSDP2(
                copy.deepcopy(orig_model),
                reshard_after_forward=reshard_after_forward,
            )

            if compile_model:
                dist_model = torch.compile(dist_model)
            dist_optim = optimizer_class(
                dist_model.parameters(), lr=1e-3, foreach=foreach
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

    def _test_ddp(self, use_composable: bool, optimizer_class: Type[Optimizer]) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            if use_composable:
                dist_model = replicate(copy.deepcopy(orig_model))
            else:
                dist_model = DDP(copy.deepcopy(orig_model))
            dist_optim = optimizer_class(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_ddp(self) -> None:
        self.run_subtests(
            {
                "use_composable": [True, False],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            self._test_ddp,
        )

    def _test_fsdp_ddp(
        self,
        use_composable: bool,
        optimizer_class: Type[Optimizer],
        optim_in_backward: bool = False,
        test_frozen: bool = False,
    ) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            if test_frozen:
                for param in chain(
                    orig_model.u1.parameters(), orig_model.u2.parameters()
                ):
                    param.requires_grad = False
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            dist_model = copy.deepcopy(orig_model)
            if use_composable:
                replicate(dist_model.l)
                fully_shard(dist_model, policy=ModuleWrapPolicy({UnitModule}))
            else:
                dist_model.l = DDP(dist_model.l)
                dist_model = FSDP(
                    copy.deepcopy(orig_model),
                    auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
                    use_orig_params=optim_in_backward,
                    ignored_modules=[dist_model.l],
                )
            if optim_in_backward:
                _apply_optimizer_in_backward(
                    optimizer_class, dist_model.parameters(), {"lr": 1e-3}
                )
                dist_optim = [
                    p._in_backward_optimizers[0] for p in dist_model.parameters()
                ]
            else:
                dist_optim = optimizer_class(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim, test_frozen)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_fsdp_ddp(self) -> None:
        self.run_subtests(
            {
                "use_composable": [True, False],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
            },
            self._test_fsdp_ddp,
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_frozen_parameters(self) -> None:
        self.run_subtests(
            {
                "use_composable": [True],
                "optimizer_class": [torch.optim.Adam, torch.optim.AdamW],
                "test_frozen": [True],
            },
            self._test_fsdp_ddp,
        )

    # TODO: enable use_dtensor once 2D device_mesh support is fully landed.
    """
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_use_dtensor(self) -> None:
        self._test_fsdp_ddp(use_composable=False, use_dtensor=True)
    """

    # TODO: enable the test after FSDP + apply_optimizer_in_backward works.
    # Disable this test as it is broken after
    # https://github.com/pytorch/pytorch/pull/108298.
    """
    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_apply_optimizer_in_backward(self) -> None:
        self.run_subtests(
            {"use_composable": [True, False]},
            self._test_fsdp_ddp,
            optim_in_backward=True,
        )
    """

    def _test_single_gpu(self, optimizer_class: Type[Optimizer]) -> None:
        def init_model_optim():
            orig_model = CompositeParamModel(device=torch.device("cuda"))
            orig_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            copy_optim = optimizer_class(orig_model.parameters(), lr=1e-3)
            model_copy = copy.deepcopy(orig_model)
            optim_copy = optimizer_class(model_copy.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, model_copy, optim_copy

        self._test_save_load(init_model_optim)

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_single_gpu(self) -> None:
        self.run_subtests(
            {"optimizer_class": [torch.optim.Adam, torch.optim.AdamW]},
            self._test_single_gpu,
        )

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_strict(self) -> None:
        model = CompositeParamModel(device=torch.device("cuda"))

        model_state_dict = get_model_state_dict(model)
        key = next(iter(model_state_dict.keys()))
        model_state_dict["abc"] = torch.zeros(10)
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            set_model_state_dict(model, model_state_dict=model_state_dict)
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

    def _test_cpu_offload_full_state_dict(
        self, optimizer_class: Type[Optimizer]
    ) -> None:
        orig_model = CompositeParamModel(device=torch.device("cuda"))
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        dist_model = FSDP(
            copy.deepcopy(orig_model),
            auto_wrap_policy=ModuleWrapPolicy({UnitModule}),
            use_orig_params=True,
            device_mesh=device_mesh,
        )

        dist_optim = optimizer_class(dist_model.parameters(), lr=1e-3)

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
        model = CompositeParamModel(device=torch.device("cuda"))
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
        model = CompositeParamModel(device=torch.device("cuda"))
        original_keys = get_model_state_dict(model).keys()

        apply_activation_checkpointing(model)
        model = FSDP(model, use_orig_params=use_orig_params)
        new_keys = get_model_state_dict(model).keys()

        self.assertEqual(original_keys, new_keys)

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_extra_state(self) -> None:
        model = CompositeParamModel(device=torch.device("cuda"))

        def get_extra_state(self):
            return "MyState"

        def set_extra_state(self, state):
            return

        UnitModule.get_extra_state = get_extra_state
        UnitModule.set_extra_state = set_extra_state

        ddp_model = DDP(copy.deepcopy(model))
        set_model_state_dict(ddp_model, get_model_state_dict(ddp_model))
        self.assertEqual(model.state_dict()["u1._extra_state"], "MyState")
        self.assertEqual(model.state_dict(), get_model_state_dict(ddp_model))

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_non_persistent_buffers(self) -> None:
        model = CompositeParamModel(device=torch.device("cuda"))
        model.register_buffer(
            "dont_save_me", torch.rand(100, device="cuda"), persistent=False
        )
        ddp_model = DDP(copy.deepcopy(model))
        set_model_state_dict(ddp_model, get_model_state_dict(ddp_model))
        self.assertEqual(model.state_dict(), get_model_state_dict(ddp_model))

    def _test_broadcast_from_rank0(self, wrapper) -> None:
        model = CompositeParamModel(device=torch.device("cuda"))
        optim = torch.optim.Adam(model.parameters())
        fsdp_model = wrapper(copy.deepcopy(model))
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters())

        batch = torch.rand(8, 100, device="cuda")
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
    @skip_if_lt_x_gpu(2)
    def test_broadcast_from_rank0(self) -> None:
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        self.run_subtests(
            {
                "wrapper": [
                    functools.partial(FSDP2, mesh=device_mesh),
                    functools.partial(FSDP, device_mesh=device_mesh),
                ]
            },
            self._test_broadcast_from_rank0,
        )

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_broadcast_from_rank0_hsdp(self) -> None:
        device_mesh = init_device_mesh("cuda", (2, self.world_size // 2))
        self.run_subtests(
            {
                "wrapper": [
                    functools.partial(
                        FSDP,
                        device_mesh=device_mesh,
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
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        model = CompositeParamModel(device=torch.device("cuda"))
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
        device = "cuda"
        torch.manual_seed(0)
        model = nn.Sequential(
            *[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)]
        )
        for layer in model:
            fully_shard(layer)
        fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=[lambda epoch: 0.95**epoch]
        )
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
    def test_optim_state_dict_tensor_matching(self) -> None:
        device = "cuda"
        torch.manual_seed(0)
        model = nn.Sequential(
            *[nn.Linear(4, 4, device=device, bias=False) for _ in range(2)]
        )
        for layer in model:
            fsdp_fully_shard(layer)
        fsdp_fully_shard(model)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        x = torch.randn((4, 4), device=device)
        model(x).sum().backward()
        optim.step()
        optim.zero_grad()
        self.assertIsInstance(
            list(optim.state.values())[0]["exp_avg"], DTensor  # noqa: RUF015
        )
        opt_state_dict = ptd_state_dict.get_optimizer_state_dict(
            model,
            optim,
            options=ptd_state_dict.StateDictOptions(full_state_dict=True),
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        ptd_state_dict.set_optimizer_state_dict(
            model,
            optim,
            optim_state_dict=opt_state_dict,
            options=ptd_state_dict.StateDictOptions(full_state_dict=True),
        )
        self.assertIsInstance(
            list(optim.state.values())[0]["exp_avg"], DTensor  # noqa: RUF015
        )

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_flattened_osd(self) -> None:
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        model = CompositeParamModel(device=torch.device("cuda"))
        fsdp_model = FSDP2(copy.deepcopy(model), mesh=device_mesh)
        fsdp_optim = torch.optim.AdamW(fsdp_model.parameters())
        batch = torch.rand(8, 100, device="cuda")
        fsdp_model(batch).sum().backward()
        fsdp_optim.step()
        fsdp_optim.zero_grad()
        osd1 = get_optimizer_state_dict(fsdp_model, fsdp_optim)
        osd2 = get_optimizer_state_dict(
            fsdp_model,
            fsdp_optim,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        fsdp_optim2 = torch.optim.AdamW(fsdp_model.parameters())
        set_optimizer_state_dict(
            fsdp_model, optimizers=fsdp_optim2, optim_state_dict=osd2
        )
        self.assertEqual(fsdp_optim.state_dict(), fsdp_optim2.state_dict())
        set_optimizer_state_dict(
            fsdp_model, optimizers=fsdp_optim2, optim_state_dict=osd1
        )
        self.assertEqual(fsdp_optim.state_dict(), fsdp_optim2.state_dict())

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_deprecate_partial(self) -> None:
        model = CompositeParamModel(device=torch.device("cuda"))

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
        for key in model_state_dict3.keys():
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

    @with_comms
    @skip_if_lt_x_gpu(1)
    def test_deprecate_fsdp_api(self) -> None:
        device_mesh = init_device_mesh("cuda", (self.world_size,))
        model = CompositeParamModel(device=torch.device("cuda"))
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
            device_mesh = init_device_mesh("cuda", (self.world_size,))
            orig_model = TiedEmbeddingModel(10000, 300).to(torch.device("cuda"))
            orig_optim = torch.optim.AdamW(orig_model.parameters(), lr=1e-3)
            copy_optim = torch.optim.AdamW(orig_model.parameters(), lr=1e-3)
            dist_model = FSDP(copy.deepcopy(orig_model), device_mesh=device_mesh)
            dist_optim = torch.optim.AdamW(dist_model.parameters(), lr=1e-3)
            return orig_model, orig_optim, copy_optim, dist_model, dist_optim

        self._test_save_load(init_model_optim)


class TestNoComm(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(1)
    def test_no_dist(self) -> None:
        model = CompositeParamModel(device=torch.device("cuda"))
        optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

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
