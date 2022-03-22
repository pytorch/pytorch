# Owner(s): ["oncall: distributed"]

import collections
import sys
from typing import Any, Dict, List, Type

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    OPTIM_TARGET_RANK,
)
from torch.distributed.fsdp.wrap import always_wrap_policy
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
    """This adds a 1D bias with dimension ``dim``."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"Invalid arg: `dim`={dim}")
        torch.manual_seed(0)
        self.bias = torch.nn.Parameter(torch.randn((dim,)))

    def forward(self, x):
        return x + self.bias


class LinearWithKBiases(torch.nn.Module):
    """
    A linear layer representing an affine transformation but using ``k``
    biases instead of just one. This ``k`` exactly counts the number of nested
    modules of this module.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_biases: int,
    ) -> None:
        if in_dim <= 0 or out_dim <= 0 or num_biases < 0:
            raise ValueError(
                f"Invalid args: `in_dim`={in_dim} `out_dim`={out_dim} "
                f"`num_biases`={num_biases}"
            )
        super().__init__()
        torch.manual_seed(0)
        self.weight = torch.nn.Parameter(torch.randn((in_dim, out_dim)))
        # NOTE: FSDP does not work if the biases are held in a
        # `torch.nn.ModuleList`, so we set each bias manually.
        for i in range(num_biases):
            setattr(self, f"bias{i}", Bias(out_dim))
        self.num_biases = num_biases

    def forward(self, x):
        z = x @ self.weight
        for bias_index in range(self.num_biases):
            bias = getattr(self, f"bias{bias_index}")
            z = bias(z)
        return z


class JaggedModel(torch.nn.Module):
    """
    A model with jagged nesting to exercise the optimizer state checkpointing.
    The structure and (unflattened) parameter IDs are as follows:
    JaggedModel
        linear0: LinearWithKBiases 0
            Bias 1
        linear1: Sequential
            LinearWithKBiases 2
                Bias 3
                Bias 4
                Bias 5
            LinearWithKBiases 6
                Bias 7
        linear2: LinearWithKBiases 8
        linear3: Sequential
            Bias 9
            Bias 10
            Bias 11
            Bias 12
            Bias 13
        linear4: LinearWithKBiases 14
            Bias 15
            Bias 16
    """
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = LinearWithKBiases(2, 3, 1)
        self.linear1 = torch.nn.Sequential(
            LinearWithKBiases(3, 5, 3),
            LinearWithKBiases(5, 5, 1),
        )
        self.linear2 = LinearWithKBiases(5, 4, 0)
        self.linear3 = torch.nn.Sequential(
            Bias(4), Bias(4), Bias(4), Bias(4), Bias(4),
        )
        self.linear4 = LinearWithKBiases(4, 3, 2)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        z = x
        for l in (
            self.linear0, self.linear1, self.linear2, self.linear3,
            self.linear4,
        ):
            z = l(z)
            z = self.relu(z)
        return z

    def get_input(self, device):
        BATCH_SIZE = 3
        return (torch.randn((BATCH_SIZE, 2)).to(device),)

    def get_loss(self, inp, output):
        return output.sum()

    def run_backward(self, loss):
        loss.backward()

    def get_params_copy(self):
        params = [
            self.linear0.weight, self.linear0.bias0.bias,
            self.linear1[0].weight, self.linear1[0].bias0.bias,
            self.linear1[0].bias1.bias, self.linear1[0].bias2.bias,
            self.linear1[1].weight, self.linear1[1].bias0.bias,
            self.linear2.weight,
            self.linear3[0].bias, self.linear3[1].bias, self.linear3[2].bias,
            self.linear3[3].bias, self.linear3[4].bias,
            self.linear4.weight, self.linear4.bias0.bias,
            self.linear4.bias1.bias,
        ]
        return [p.detach().clone() for p in params]

class WrappedJaggedModel(JaggedModel):
    """
    This model has the same underlying module structure as
    :class:`JaggedModel`, except some modules have now been wrapped using
    :class:`FullyShardedDataParallel`. The wrapping structure is as follows:
    WrappedJaggedModel
        linear0: LinearWithKBiases
            Bias
        linear1: FSDP(Sequential)
            FSDP(LinearWithKBiases)
                FSDP(Bias)
                FSDP(Bias)
                FSDP(Bias)
            FSDP(LinearWithKBiases)
                FSDP(Bias)
        linear2: LinearWithKBiases
        linear3: FSDP(Sequential)
            Bias
            Bias
            Bias
            Bias
            Bias
        linear4: FSDP(LinearWithKBiases)
            Bias
            FSDP(Bias)
    Note that ``linear3`` has a single flattened parameter consisting of 5
    unflattened parameters.
    """
    def __init__(self, group=None) -> None:
        super().__init__()
        self.linear1 = FSDP(
            self.linear1, process_group=group,
            auto_wrap_policy=always_wrap_policy,
        )
        self.linear3 = FSDP(self.linear3, process_group=group)

        assert self.linear4.num_biases == 2
        has_seen_first_bias = False

        def linear4_wrap_policy(module, recurse, unwrapped_params):
            is_leaf = len(list(module.children())) == 0
            if not is_leaf:
                return True  # wrap the parent
            # Do not wrap the first bias, only the second
            nonlocal has_seen_first_bias
            if not has_seen_first_bias:
                has_seen_first_bias = True
                return False
            return True

        self.linear4 = FSDP(
            self.linear4, process_group=group,
            auto_wrap_policy=linear4_wrap_policy,
        )

    def get_params_copy(self):
        params = [self.linear0.weight, self.linear0.bias0.bias]
        with self.linear1[0].summon_full_params():
            params.append(self.linear1[0].weight.detach().clone())
        with self.linear1[0].bias0.summon_full_params():
            params.append(self.linear1[0].bias0.bias.detach().clone())
        with self.linear1[0].bias1.summon_full_params():
            params.append(self.linear1[0].bias1.bias.detach().clone())
        with self.linear1[0].bias2.summon_full_params():
            params.append(self.linear1[0].bias2.bias.detach().clone())
        with self.linear1[1].summon_full_params():
            params.append(self.linear1[1].weight.detach().clone())
        with self.linear1[1].bias0.summon_full_params():
            params.append(self.linear1[1].bias0.bias.detach().clone())
        params.append(self.linear2.weight.detach().clone())
        with self.linear3.summon_full_params():
            params.append(self.linear3[0].bias.detach().clone())
            params.append(self.linear3[1].bias.detach().clone())
            params.append(self.linear3[2].bias.detach().clone())
            params.append(self.linear3[3].bias.detach().clone())
            params.append(self.linear3[4].bias.detach().clone())
        with self.linear4.summon_full_params():
            params.append(self.linear4.weight.detach().clone())
            params.append(self.linear4.bias0.bias.detach().clone())
            params.append(self.linear4.bias1.bias.detach().clone())
        return params


class AlternateWrappedJaggedModel(JaggedModel):
    """
    Like :class:`WrappedJaggedModel`, this model has the same underlying
    module structure as :class:`JaggedModel`, except some modules have now
    been wrapped using :class:`FullyShardedDataParallel`. Notably, this
    wrapping structure is different from :class:`WrappedJaggedModel` and is as
    follows:
    AlternateWrappedJaggedModel
        linear0: FSDP(LinearWithKBiases)
            FSDP(Bias)
        linear1: Sequential
            LinearWithKBiases
                Bias
                Bias
                Bias
            LinearWithKBiases
                Bias
        linear2: LinearWithKBiases
        linear3: Sequential
            Bias
            Bias
            Bias
            Bias
            Bias
        linear4: FSDP(LinearWithKBiases)
            Bias
            Bias
    We use an alternate wrapping to ensure that the optimizer state dict API is
    agnostic to the wrapping structure.
    """
    def __init__(self, group=None) -> None:
        super().__init__()
        self.linear0 = FSDP(
            self.linear0, process_group=group,
            auto_wrap_policy=always_wrap_policy,
        )
        self.linear4 = FSDP(self.linear4, process_group=group)

    def get_params_copy(self):
        params = []
        with self.linear0.summon_full_params():
            params.append(self.linear0.weight.detach().clone())
        with self.linear0.bias0.summon_full_params():
            params.append(self.linear0.bias0.bias.detach().clone())
        params.append(self.linear1[0].weight.detach().clone())
        params.append(self.linear1[0].bias0.bias.detach().clone())
        params.append(self.linear1[0].bias1.bias.detach().clone())
        params.append(self.linear1[0].bias2.bias.detach().clone())
        params.append(self.linear1[1].weight.detach().clone())
        params.append(self.linear1[1].bias0.bias.detach().clone())
        params.append(self.linear2.weight.detach().clone())
        params.append(self.linear3[0].bias.detach().clone())
        params.append(self.linear3[1].bias.detach().clone())
        params.append(self.linear3[2].bias.detach().clone())
        params.append(self.linear3[3].bias.detach().clone())
        params.append(self.linear3[4].bias.detach().clone())
        with self.linear4.summon_full_params():
            params.append(self.linear4.weight.detach().clone())
            params.append(self.linear4.bias0.bias.detach().clone())
            params.append(self.linear4.bias1.bias.detach().clone())
        return params


class TestFSDPOptimState(FSDPTest):
    def _init_wrapped(
        self,
        model_class: str,
        device: torch.device = torch.device("cuda"),
        group=None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        **optim_kwargs,
    ):
        if model_class == "jagged":
            model = WrappedJaggedModel(group=group).to(device)
        elif model_class == "alternate_jagged":
            model = AlternateWrappedJaggedModel(group=group).to(device)
        elif model_class == "transformer":
            assert group is not None, \
                "Transformer requires a group to be specified"
            model = self._get_wrapped_model(group=group)
            model.eval()  # disable dropout for determinism
        else:
            assert 0, f"Unsupported `model_class`: {model_class}"
        lr = optim_kwargs.pop("lr", 0.01)
        if not use_multiple_param_groups:
            optim = optim_class(model.parameters(), lr=lr, **optim_kwargs)
            optim_input = list(model.parameters())  # persist generator
        else:
            if model_class != "jagged" and model_class != "alternate_jagged":
                assert 0, f"Unsupported `model_class`: {model_class}"
            # Manually set the parameter groups to appease the constraint that
            # all of the unflattened parameters comprising a flattened
            # parameter must be in the same parameter group
            no_wd_params = list(model.linear0.parameters()) + \
                list(model.linear1.parameters()) + \
                list(model.linear4.parameters())
            wd_params = list(model.linear2.parameters()) + \
                list(model.linear3.parameters())
            optim_input = [
                {"params": no_wd_params},
                {"params": wd_params, "weight_decay": 0.9}
            ]
            optim = optim_class(optim_input, lr=lr, **optim_kwargs)
        return model, optim, optim_input

    def _init_nonwrapped(
        self,
        model_class: str,
        device: torch.device = torch.device("cuda"),
        group=None,
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        **optim_kwargs,
    ):
        # Both "jagged" and "alternate_jagged" share the same non-wrapped model
        if model_class == "jagged" or model_class == "alternate_jagged":
            model = JaggedModel().to(device)
        elif model_class == "transformer":
            assert group is not None, \
                "Transformer requires a group to be specified"
            model = self._get_nonwrapped_model(group)
            model.eval()  # disable dropout for determinism
        else:
            assert 0, f"Unsupported `model_class`: {model_class}"
        lr = optim_kwargs.pop("lr", 0.01)
        if not use_multiple_param_groups:
            optim = optim_class(model.parameters(), lr=lr, **optim_kwargs)
        else:
            if model_class != "jagged" and model_class != "alternate_jagged":
                assert 0, f"Unsupported `model_class`: {model_class}"
            # Manually set the parameter groups to appease the constraint that
            # all of the unflattened parameters comprising a flattened
            # parameter must be in the same parameter group
            no_wd_params = list(model.linear0.parameters()) + \
                list(model.linear1.parameters()) + \
                list(model.linear4.parameters())
            wd_params = list(model.linear2.parameters()) + \
                list(model.linear3.parameters())
            optim_input = [
                {"params": no_wd_params},
                {"params": wd_params, "weight_decay": 0.9}
            ]
            optim = optim_class(optim_input, lr=lr, **optim_kwargs)
        return model, optim

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
        for _ in range(num_iters):
            module = model.module if hasattr(model, "module") else model
            inp = module.get_input(device)
            output = model(*inp)
            loss = module.get_loss(inp, output).to(device)
            losses.append(loss.item())
            module.run_backward(loss)
            optim.step()
        return losses

    def _check_same_state(
        self,
        osd1: Dict[str, Any],
        osd2: Dict[str, Any],
    ) -> None:
        """Checks that the state of optimizer state dict ``osd1`` is the same
        as that of optimizer state dict ``osd2``."""
        self.assertTrue("state" in osd1)
        self.assertTrue("state" in osd2)
        state1 = osd1["state"]
        state2 = osd2["state"]
        self.assertEqual(len(state1), len(state2))
        for param_id, param_state in state1.items():
            for state_name, v1 in param_state.items():
                v2 = state2[param_id][state_name]
                self.assertEqual(type(v1), type(v2))
                if torch.is_tensor(v1):
                    v1 = v1.cpu()
                    v2 = v2.cpu()
                    if not torch.all(torch.isclose(v1, v2)) and dist.get_rank() == 0:
                        print(f"param_id={param_id} state={state_name}")
                        print(v1)
                        print(v2)
                        print()
                self.assertEqual(v1, v2)

    def _check_same_param_groups(
        self,
        osd1: Dict[str, Any],
        osd2: Dict[str, Any],
    ) -> None:
        """Checks that the parameter groups of optimizer state dict ``osd1``
        are the same as those of optimizer state dict ``osd2``."""
        self.assertTrue("param_groups" in osd1)
        self.assertTrue("param_groups" in osd2)
        pgs1 = osd1["param_groups"]
        pgs2 = osd2["param_groups"]
        self.assertEqual(len(pgs1), len(pgs2))
        for pg1, pg2 in zip(pgs1, pgs2):
            self.assertEqual(set(pg1.keys()), set(pg2.keys()))
            for hyperparam_name, hyperparam_value in pg1.items():
                self.assertEqual(hyperparam_value, pg2[hyperparam_name])

    def _broadcast_full_osd(self, full_osd: Dict[str, Any]):
        obj_list = [full_osd]
        dist.broadcast_object_list(obj_list, src=OPTIM_TARGET_RANK)
        full_osd = obj_list[0]
        return full_osd

    @skip_if_lt_x_gpu(2)
    @parametrize("model_class", ["jagged", "transformer"])
    @parametrize("use_multiple_param_groups", [False, True])
    def test_full_optim_state_dict(
        self,
        model_class: str,
        use_multiple_param_groups: bool,
    ):
        """Checks that the optimizer state dict returned from
        :meth:`full_optim_state_dict` for a model with FSDP instances matches
        the optimizer state dict from an equivalent local model without FSDP
        instances."""
        if model_class != "jagged" and use_multiple_param_groups:
            return  # skip since not supported
        NUM_ITERS = 3
        GROUP = dist.distributed_c10d._get_default_group()
        wrapped_model, wrapped_optim, optim_input = self._init_wrapped(
            model_class=model_class, group=GROUP,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        wrapped_losses = self._step_model(
            wrapped_model, wrapped_optim, num_iters=NUM_ITERS,
        )
        full_osd = FSDP.full_optim_state_dict(
            wrapped_model, wrapped_optim, optim_input=optim_input,
        )
        if self.rank != OPTIM_TARGET_RANK:
            return
        nonwrapped_model, nonwrapped_optim = self._init_nonwrapped(
            model_class=model_class, group=GROUP,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        nonwrapped_losses = self._step_model(
            nonwrapped_model, nonwrapped_optim, num_iters=NUM_ITERS,
        )
        for i, (l1, l2) in enumerate(zip(wrapped_losses, nonwrapped_losses)):
            assert l1 == l2, f"Losses differ on iter {i}: {l1} {l2}"
        local_osd = nonwrapped_optim.state_dict()
        self._check_same_state(full_osd, local_osd)
        self._check_same_param_groups(full_osd, local_osd)

    def _test_shard_full_optim_state_dict(
        self,
        model_class: str,
        use_new_process_group: bool,
        use_new_model_class: bool,
        use_multiple_param_groups: bool,
    ):
        if model_class != "jagged" and \
                (use_new_model_class or use_multiple_param_groups):
            return  # skip since not supported
        NUM_ITERS = 3
        default_group = dist.distributed_c10d._get_default_group()
        model1, optim1, optim_input1 = self._init_wrapped(
            model_class=model_class, group=default_group,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        losses1 = self._step_model(model1, optim1, num_iters=NUM_ITERS)
        full_osd = FSDP.full_optim_state_dict(model1, optim1, optim_input1)
        full_osd = _recursive_copy_to_device(
            self._broadcast_full_osd(full_osd), False, torch.device("cpu")
        )  # copy to avoid aliasing when we step the model again
        losses1.extend(self._step_model(model1, optim1, num_iters=NUM_ITERS))
        if use_new_process_group:
            new_group_ranks = [r for r in range(self.world_size) if r % 2 == 0]
            new_group = dist.new_group(ranks=new_group_ranks)
        else:
            new_group_ranks = list(range(self.world_size))
            new_group = default_group
        new_model_class = "alternate_jagged" if use_new_model_class \
            else model_class
        if self.rank not in new_group_ranks:
            return
        model2, optim2, optim_input2 = self._init_wrapped(
            model_class=new_model_class, group=new_group,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        losses2 = self._step_model(model2, optim2, num_iters=NUM_ITERS)
        for i, (l1, l2) in enumerate(zip(losses1[:NUM_ITERS], losses2)):
            assert l1 == l2, f"Losses differ on iter {i}: {l1} {l2}"
        sharded_osd1 = FSDP.shard_full_optim_state_dict(
            full_osd, model2, optim_input2,
        )
        sharded_osd2 = optim2.state_dict()
        self._check_same_state(sharded_osd1, sharded_osd2)
        self._check_same_param_groups(sharded_osd1, sharded_osd2)
        optim2.load_state_dict(sharded_osd2)
        losses2 = self._step_model(model2, optim2, num_iters=NUM_ITERS)
        for i, (l1, l2) in enumerate(zip(losses1[NUM_ITERS:], losses2)):
            self.assertEqual(l1, l2)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_new_process_group", [False, True])
    @parametrize("use_new_model_class", [False, True])
    @parametrize("use_multiple_param_groups", [False, True])
    def test_shard_full_optim_state_dict_jagged(
        self,
        use_new_process_group: bool,
        use_new_model_class: bool,
        use_multiple_param_groups: bool,
    ):
        """Checks that saving the full optimizer state dict of a model using
        :meth:`full_optim_state_dict` and sharding the dict according to a new
        model (with possibly different wrapping and/or smaller world size)
        using :meth:`shard_full_optim_state_dict` yields the same optimizer
        state dict as using that new model directly."""
        self._test_shard_full_optim_state_dict(
            "jagged", use_new_process_group, use_new_model_class,
            use_multiple_param_groups,
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_new_process_group", [False, True])
    def test_shard_full_optim_state_dict_transformer(
        self,
        use_new_process_group: bool,
    ):
        """Checks that saving the full optimizer state dict of an FSDP-wrapped
        transformer model using :meth:`full_optim_state_dict` and sharding the
        dict according to a new FSDP-wrapped version of the model (with
        possibly smaller world size) using :meth:`shard_full_optim_state_dict`
        yields the same optimizer state dict as using that new model
        directly."""
        self._test_shard_full_optim_state_dict(
            "transformer", use_new_process_group, use_new_model_class=False,
            use_multiple_param_groups=False,
        )


def _recursive_copy_to_device(
    value: Any,
    non_blocking: bool,
    device: torch.device,
) -> Any:
    """Recursively searches :class:`list` s, :class:`tuple` s, and
    :class:`dict` s and copies tensors to device if possible. Non-tensor values
    are passed as-is in the result."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone().to(device, non_blocking=non_blocking)
    if isinstance(value, (list, tuple)):
        values = [
            _recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device,
            ) for val in value
        ]
        return values if isinstance(value, list) else tuple(values)
    if isinstance(value, collections.abc.Mapping):
        return {
            key: _recursive_copy_to_device(
                val, non_blocking=non_blocking, device=device,
            ) for key, val in value.items()
        }
    return value


instantiate_parametrized_tests(TestFSDPOptimState)

if __name__ == "__main__":
    run_tests()
