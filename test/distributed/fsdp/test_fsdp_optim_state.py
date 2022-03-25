# Owner(s): ["oncall: distributed"]

import sys
from typing import Any, Dict, List, Type

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    OPTIM_TARGET_RANK,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class Bias(torch.nn.Module):
    """This module applies a 1D additive bias with dimension ``dim``."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim > 0
        torch.manual_seed(0)
        self.bias = torch.nn.Parameter(torch.randn((dim,)))

    def forward(self, x):
        return x + self.bias


class AffineA(torch.nn.Module):
    """
    This module applies an affine transformation (like ``nn.Linear``).
    AffineA
        Bias0
            bias
        weight
        Bias1
            bias
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        assert all(v > 0 for v in (in_dim, out_dim))
        torch.manual_seed(0)
        self.bias_module0 = Bias(out_dim)
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias_module1 = Bias(out_dim)

    def forward(self, x):
        x = x @ self.weight
        x = self.bias_module0(x)
        x = self.bias_module1(x)
        return x


class AffineB(torch.nn.Module):
    """
    This module applies an affine transformation (like ``nn.Linear``).
    AffineB
        weight
        Bias
            bias
        bias
    """
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        assert all(v > 0 for v in (in_dim, out_dim))
        torch.manual_seed(0)
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))
        self.bias_module = Bias(out_dim)
        self.bias = torch.nn.Parameter(torch.randn((out_dim,)))

    def forward(self, x):
        x = x @ self.weight
        x = self.bias_module(x)
        x = x + self.bias
        return x


class AffineModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.affine0 = AffineA(5, 7)
        self.affine1 = AffineB(7, 7)
        self.bias = torch.nn.Parameter(torch.randn((5,)))
        self.affine2 = torch.nn.Sequential(
            AffineB(7, 9),
            AffineA(9, 9),
            AffineB(9, 5),
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.affine0(x))
        x = self.relu(self.affine1(x))
        x = self.relu(self.affine2(x))
        x = x + self.bias
        return x

    def get_input(self, device):
        BATCH_SIZE = 8
        return (torch.randn((BATCH_SIZE, 5)).to(device),)

    def get_loss(self, inp, output):
        return output.sum()

    def run_backward(self, loss):
        loss.backward()

    @staticmethod
    def wrap(model, group=None) -> None:
        # Flatten Bias; then flatten weight and bias together into `affine1`
        model.affine1.bias_module = FSDP(
            model.affine1.bias_module, process_group=group,
        )
        model.affine1 = FSDP(model.affine1, process_group=group)
        # Flatten Bias0; flatten Bias1; then flatten weight into `affine2[1]`
        model.affine2[1].bias_module0 = FSDP(
            model.affine2[1].bias_module0, process_group=group,
        )
        model.affine2[1].bias_module1 = FSDP(
            model.affine2[1].bias_module1, process_group=group,
        )
        model.affine2[1] = FSDP(model.affine2[1], process_group=group)
        # Flatten weight, Bias, bias into `affine2[2]`
        model.affine2[2] = FSDP(model.affine2[2], process_group=group)

    def param_group0(self) -> List[torch.nn.Parameter]:
        # Use `affine1`'s parameters for the first parameter group to deviate
        # from the `model.parameters()` order
        return list(self.affine1.parameters())

    def param_group1(self) -> List[torch.nn.Parameter]:
        # Deviate from the `model.parameters()` order further by rearranging
        # `affine2`'s parameters to be before `affine0`'s parameters
        return list(self.affine2.parameters()) + \
            list(self.affine0.parameters())


class TestFSDPOptimState(FSDPTest):
    def _init_affine_model(
        self,
        wrap: bool,
        device: torch.device = torch.device("cuda"),
        group=None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        **optim_kwargs,
    ):
        model = AffineModel().to(device)
        if wrap:
            AffineModel.wrap(model, group)
        lr = optim_kwargs.pop("lr", 0.01)
        if not use_multiple_param_groups:
            optim_input = list(model.parameters())
        else:
            optim_input = [
                {"params": model.param_group0()},
                {"params": model.param_group1(), "weight_decay": 0.9}
            ]
        optim = optim_class(optim_input, lr=lr, **optim_kwargs)
        return model, optim, optim_input

    def _init_transformer_model(
        self,
        wrap: bool,
        device: torch.device = torch.device("cuda"),
        group=None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        **optim_kwargs,
    ):
        assert not use_multiple_param_groups, \
            "Multiple parameter groups for the transformer is not implemented"
        if group is None:
            group = dist.distributed_c10d._get_default_group()
        model = self._get_wrapped_model(group=group).to(device) if wrap \
            else self._get_nonwrapped_model(group=group).to(device)
        model.eval()  # disable dropout for determinism
        lr = optim_kwargs.pop("lr", 0.01)
        optim = optim_class(model.parameters(), lr=lr, **optim_kwargs)
        return model, optim, None

    def _step_model(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        device: torch.device = torch.device("cuda"),
        num_iters: int = 1,
    ) -> List[float]:
        """Performs a forward pass, backward pass, and optimizer step
        ``num_iters``-many times, and returns the per-iteration losses."""
        torch.manual_seed(0)  # set seed for determinism
        losses = []
        module = model.module if hasattr(model, "module") else model
        for _ in range(num_iters):
            inp = module.get_input(device)
            output = model(*inp)
            loss = module.get_loss(inp, output).to(device)
            losses.append(loss.item())
            module.run_backward(loss)
            optim.step()
        return losses

    def _broadcast_full_osd(self, full_osd: Dict[str, Any], group=None):
        """Broadcasts the full optimizer state dict in place of using
        ``torch.save()`` and ``torch.load()``."""
        obj_list = [full_osd]
        dist.broadcast_object_list(obj_list, src=OPTIM_TARGET_RANK, group=group)
        full_osd = obj_list[0]
        return full_osd

    def _are_equal_states(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any],
    ) -> bool:
        """Checks if ``state1`` and ``state2`` contain the same mappings."""
        if set(state1.keys()) != set(state2.keys()):
            return False
        for state_name, value1 in state1.items():
            value2 = state2[state_name]
            if type(value1) != type(value2):
                return False
            if torch.is_tensor(value1):  # tensor state
                assert torch.is_tensor(value2)
                # Check the values on CPU to be device-agnostic
                value1 = value1.cpu()
                value2 = value2.cpu()
                if value1.shape != value2.shape or \
                        not torch.all(torch.isclose(value1, value2)):
                    return False
            else:  # non-tensor state
                if value1 != value2:
                    return False
        return True

    def _check_same_state(self, full_osd, ref_osd, check_same_param_ids: bool):
        """Checks that ``full_osd`` and ``ref_osd`` have the same "state" part,
        allowing the parameter IDs to be different but still isomorphic."""
        assert "state" in ref_osd
        self.assertTrue("state" in full_osd)
        ref_osd_state = ref_osd["state"]
        full_osd_state = full_osd["state"]
        # Check parameter IDs are the same
        ref_osd_param_ids = set(ref_osd_state.keys())
        full_osd_param_ids = set(full_osd_state.keys())
        self.assertTrue(ref_osd_param_ids == full_osd_param_ids)
        # Perform strict check that accounts for parameter IDs matching
        if check_same_param_ids:
            for param_id, param_state in full_osd_state.items():
                for state_name, value in param_state.items():
                    ref_value = ref_osd_state[param_id][state_name]
                    self.assertEqual(value, ref_value)
            return
        # Otherwise, only require the parameter IDs to be isomorphic
        ref_osd_states = list(ref_osd["state"].values())
        full_osd_states = list(full_osd["state"].values())
        assert len(ref_osd_states) == len(full_osd_states)
        # Use brute-force quadratic-time comparison since it is hard to
        # hash a tensor by value instead of by object
        for full_osd_state in full_osd_states:
            # Check for at least one match (may be > 1 in toy edge cases, e.g.
            # multiple biases); nonetheless, each having >= 1 match and the two
            # lists have equal length imply that the list contents are equal
            self.assertTrue(any(
                self._are_equal_states(full_osd_state, ref_osd_state)
                for ref_osd_state in ref_osd_states
            ))

    def _check_same_param_groups(self, full_osd, ref_osd):
        """Checks that ``full_osd`` and ``ref_osd`` have the same
        "param_groups" part."""
        assert "param_groups" in ref_osd
        self.assertTrue("param_groups" in full_osd)
        ref_osd_param_groups = ref_osd["param_groups"]
        full_osd_param_groups = full_osd["param_groups"]
        self.assertTrue(len(full_osd_param_groups), len(ref_osd_param_groups))
        for full_osd_pg, ref_osd_pg in zip(full_osd_param_groups, ref_osd_param_groups):
            self.assertEqual(set(full_osd_pg.keys()), set(ref_osd_pg.keys()))
            for name, full_osd_value in full_osd_pg.items():
                # Even if the parameter IDs map differently to parameters,
                # "params" should still contain the same IDs and be in
                # increasing order; thus, the two values should equal
                self.assertEqual(full_osd_value, ref_osd_pg[name])

    @skip_if_lt_x_gpu(2)
    @parametrize("use_multiple_param_groups", [False, True])
    def test_full_optim_state_dict_affine(
        self,
        use_multiple_param_groups: bool,
    ) -> None:
        """
        Tests :meth:`full_optim_state_dict` by comparing the returned dict for
        an FSDP-wrapped model with that of an equivalent non-wrapped model.

        The parameter groups in the "param_groups" part and the values in the
        "state" part should be the same, but the parameter IDs (i.e. the keys
        in the "state" part) may be rearranged.
        """
        NUM_ITERS = 3
        model1, optim1, optim_input = self._init_affine_model(
            wrap=True, use_multiple_param_groups=use_multiple_param_groups,
        )
        losses1 = self._step_model(model1, optim1, num_iters=NUM_ITERS)
        full_osd = FSDP.full_optim_state_dict(model1, optim1, optim_input)
        if self.rank != OPTIM_TARGET_RANK:
            return

        model2, optim2, _ = self._init_affine_model(
            wrap=False, use_multiple_param_groups=use_multiple_param_groups,
        )
        losses2 = self._step_model(model2, optim2, num_iters=NUM_ITERS)
        ref_osd = optim2.state_dict()

        for i, (l1, l2) in enumerate(zip(losses1, losses2)):
            assert l1 == l2, f"Losses differ on iter {i}: {l1:.5f} {l2:.5f}"

        self._check_same_param_groups(full_osd, ref_osd)
        self._check_same_state(full_osd, ref_osd, check_same_param_ids=False)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_multiple_param_groups", [False, True])
    def test_shard_full_optim_state_dict_affine(
        self,
        use_multiple_param_groups: bool,
    ) -> None:
        """Tests :meth:`shard_full_optim_state_dict` for a non-FSDP-root model
        with nested FSDP instances."""
        self._test_shard_full_optim_state(
            model_class="affine",
            use_multiple_param_groups=use_multiple_param_groups,
        )

    @skip_if_lt_x_gpu(2)
    def test_shard_full_optim_state_dict_transformer(self) -> None:
        """Tests :meth:`shard_full_optim_state_dict` for an FSDP-root
        transformer model with shared parameters."""
        self._test_shard_full_optim_state(
            model_class="transformer", use_multiple_param_groups=False,
        )

    def _test_shard_full_optim_state(
        self,
        model_class: str,
        use_multiple_param_groups: bool,
    ):
        """
        (1) Runs a model with full world size for K iterations to generate a
        full optimizer state dict;
        (2) initializes a model with halved world size but the same FSDP
        wrapping scheme;
        (3) shards the full optimizer state dict from (1) according to the
        halved-world-size model;
        (4) runs the halved-world-size model for K iterations; and
        (5) checks that the sharded optimizer state dict from (3) matches the
        halved-world-size model's local optimizer state dict, meaning that the
        former could have equivalently been loaded into the local optimizer.
        """
        NUM_ITERS = 3
        initializer = self._init_affine_model if model_class == "affine" \
            else self._init_transformer_model if model_class == "transformer" \
            else None
        assert initializer is not None, f"Unsupported model: {model_class}"
        # Run a wrapped model with full world size for a few iterations
        model1, optim1, optim_input1 = initializer(
            wrap=True, use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model1, optim1, num_iters=NUM_ITERS)
        full_osd1 = FSDP.full_optim_state_dict(model1, optim1, optim_input1)
        # Broadcast instead of `torch.save()`/`torch.load()` so that all ranks
        # have the full state dict
        full_osd1 = self._broadcast_full_osd(full_osd1)
        # Create a new process group with halved world size
        new_group_ranks = [r for r in range(self.world_size) if r % 2 == 0]
        new_group = dist.new_group(ranks=new_group_ranks)
        if self.rank not in new_group_ranks:
            return
        # Run a wrapped model with halved world size (from scratch)
        model2, optim2, optim_input2 = initializer(
            wrap=True, group=new_group,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(model2, optim2, num_iters=NUM_ITERS)
        full_osd2 = FSDP.full_optim_state_dict(model2, optim2, optim_input2)
        full_osd2 = self._broadcast_full_osd(full_osd2, group=new_group)
        # As a sanity check, check that sharding the halved-world-size model's
        # full optimizer state dict according to itself is equivalent to its
        # local optimizer's state dict
        local_osd2 = optim2.state_dict()
        sharded_osd2 = FSDP.shard_full_optim_state_dict(
            full_osd2, model2, optim_input2,
        )
        self._check_same_param_groups(sharded_osd2, local_osd2)
        self._check_same_state(
            sharded_osd2, local_osd2, check_same_param_ids=True,
        )
        # Check that sharding the full-world-size model's full optimizer state
        # dict according to the halved-world-size model is equivalent to the
        # halved-world-size model's local optimizer state dict
        sharded_osd1 = FSDP.shard_full_optim_state_dict(
            full_osd1, model2, optim_input2,
        )
        self._check_same_param_groups(sharded_osd1, local_osd2)
        self._check_same_state(
            sharded_osd1, local_osd2, check_same_param_ids=True,
        )


instantiate_parametrized_tests(TestFSDPOptimState)

if __name__ == "__main__":
    run_tests()
