# Owner(s): ["oncall: distributed"]

import sys
from contextlib import suppress
from typing import Any, Dict, List, Type

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
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
    """A model with jagged nesting to exercise the optimizer state
    checkpointing."""
    def __init__(self) -> None:
        super().__init__()
        self.linear0 = LinearWithKBiases(2, 3, 1)
        self.linear1 = torch.nn.Sequential(
            LinearWithKBiases(3, 5, 3),
            LinearWithKBiases(5, 5, 1),
        )
        self.linear2 = LinearWithKBiases(5, 4, 0)
        self.linear3 = torch.nn.Sequential(
            Bias(4), Bias(4), Bias(4), Bias(4), Bias(4), Bias(4),
        )
        self.linear4 = LinearWithKBiases(4, 1, 2)
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
    Note that `linear3` has a single flattened parameter consisting of 4
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
            FSDP(Bias)
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

        assert self.linear4.num_biases == 2
        has_wrapped_first_bias = False

        def linear4_wrap_policy(module, recurse, unwrapped_params):
            is_leaf = len(list(module.children())) == 0
            if not is_leaf:
                return True  # wrap the parent
            # Wrap the only first bias, not the second
            nonlocal has_wrapped_first_bias
            if not has_wrapped_first_bias:
                has_wrapped_first_bias = True
                return True
            return False

        self.linear4 = FSDP(
            self.linear4, process_group=group,
            auto_wrap_policy=linear4_wrap_policy,
        )

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
        else:
            assert 0, f"Unsupported model_class: {model_class}"
        lr = optim_kwargs.pop("lr", 0.01)
        if not use_multiple_param_groups:
            optim = optim_class(model.parameters(), lr=lr, **optim_kwargs)
            optim_input = list(model.parameters())  # persist generator
        else:
            if model_class != "jagged" and model_class != "alternate_jagged":
                assert 0, f"Unsupported model_class: {model_class}"
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
        optim_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        use_multiple_param_groups: bool = False,
        **optim_kwargs,
    ):
        # Both "jagged" and "alternate_jagged" share the same nonwrapped model
        if model_class == "jagged" or model_class == "alternate_jagged":
            model = JaggedModel().to(device)
        else:
            assert 0, f"Unsupported model_class: {model_class}"
        lr = optim_kwargs.pop("lr", 0.01)
        if not use_multiple_param_groups:
            optim = optim_class(model.parameters(), lr=lr, **optim_kwargs)
        else:
            if model_class != "jagged" and model_class != "alternate_jagged":
                assert 0, f"Unsupported model_class: {model_class}"
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
    ):
        """Performs a forward pass, backward pass, and optimizer step
        ``num_iters``-many times."""
        torch.manual_seed(0)  # set seed for parity checks
        for _ in range(num_iters):
            module = model.module if hasattr(model, "module") else model
            inp = module.get_input(device)
            output = model(*inp)
            loss = module.get_loss(inp, output).to(device)
            module.run_backward(loss)
            optim.step()

    def _check_same_state(
        self,
        state1: Dict[int, Dict[str, Any]],
        state2: Dict[int, Dict[str, Any]],
        check_tensors_on_cpu: bool = False,
    ) -> None:
        """
        Args:
            check_tensors_on_cpu (bool): If ``True``, then checks that all
                tensor state in ``state1`` is on CPU.
        """
        cpu_device = torch.device("cpu")
        for param_id, param_state in state1.items():
            for state_name, state_value in param_state.items():
                if check_tensors_on_cpu:
                    if torch.is_tensor(state_value):
                        self.assertEqual(state_value.device, cpu_device)
                    self.assertEqual(
                        state_value, state2[param_id][state_name].cpu(),
                    )
                else:
                    self.assertEqual(
                        state_value, state2[param_id][state_name],
                    )

    def _check_same_param_groups(
        self,
        pgs1: List[Dict[str, Any]],
        pgs2: List[Dict[str, Any]],
    ) -> None:
        """Checks that the parameter groups ``pgs1`` are the same as the
        parameter groups ``pgs2``."""
        self.assertEqual(len(pgs1), len(pgs2))
        for pg1, pg2 in zip(pgs1, pgs2):
            self.assertEqual(set(pg1.keys()), set(pg2.keys()))
            for hyperparam_name, hyperparam_value in pg1.items():
                self.assertEqual(hyperparam_value, pg2[hyperparam_name])

    def _check_same_model_params(
        self,
        model1: torch.nn.Module,
        model2: torch.nn.Module,
    ) -> None:
        """Checks that the model parameters are the same across ``model1`` and
        ``model2``, accounting for the possibility that ``model`` 's submodules
        may be FSDP instances (meaning that they need to be unsharded by
        summoning the full parameter)."""
        params = []
        for module in model1.children():
            context = module.summon_full_params() if isinstance(module, FSDP) \
                else suppress()
            with context:
                params.extend(p.detach().clone() for p in module.parameters())
        for p1, (name, p2) in zip(params, model2.named_parameters()):
            if self.rank == 0 and not torch.allclose(p1, p2):
                print(name)
                print("p2", p1)
                print("p2", p2)
            self.assertEqual(p1.shape, p2.shape)
            torch.testing.assert_close(p1, p2)


    @skip_if_lt_x_gpu(2)
    @parametrize("model_class", ["jagged"])
    @parametrize("use_multiple_param_groups", [False, True])
    def test_full_optim_state_dict(
        self,
        model_class: str,
        use_multiple_param_groups: bool,
    ):
        """Checks that state dict returned from :meth:`full_optim_state_dict`
        for a model with FSDP instances matches the optimizer state dict from
        an equivalent local model."""
        TARGET_RANK = 0
        # Get the full optim state dict
        wrapped_model, wrapped_optim, optim_input = self._init_wrapped(
            model_class=model_class,
            use_multiple_param_groups=use_multiple_param_groups,
        )

        self._step_model(wrapped_model, wrapped_optim)
        full_osd = FSDP.full_optim_state_dict(
            wrapped_model,
            wrapped_optim,
            optim_input=optim_input,
            target_rank=TARGET_RANK,
        )
        if self.rank != TARGET_RANK:
            self.assertEqual(len(full_osd), 0)
            return

        # Get the optim state dict for a local equivalent
        nonwrapped_model, nonwrapped_optim = self._init_nonwrapped(
            model_class=model_class,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(nonwrapped_model, nonwrapped_optim)
        local_osd = nonwrapped_optim.state_dict()

        # Check the "state" and "param_groups" parts
        self.assertTrue("state" in full_osd)
        self.assertTrue("param_groups" in full_osd)
        self._check_same_state(
            full_osd["state"], local_osd["state"], check_tensors_on_cpu=True,
        )
        self._check_same_param_groups(
            full_osd["param_groups"], local_osd["param_groups"],
        )

    @skip_if_lt_x_gpu(2)
    @parametrize("model_class", ["jagged"])
    @parametrize("use_multiple_param_groups", [False, True])
    def test_shard_full_optim_state_dict_basic(
        self,
        model_class: str,
        use_multiple_param_groups: bool,
    ):
        """Save and load the full optimizer state dict using the same model
        wrapping configuration."""
        TARGET_RANK = 0
        # Get the full optimizer state dict
        wrapped_model, wrapped_optim, optim_input = self._init_wrapped(
            model_class=model_class,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        self._step_model(wrapped_model, wrapped_optim)
        full_osd = FSDP.full_optim_state_dict(
            wrapped_model,
            wrapped_optim,
            target_rank=TARGET_RANK,
            optim_input=optim_input,
        )

        # Broadcast the full optim state dict so that each rank has it (since
        # normally it would be saved to and loaded from disk)
        obj_list = [full_osd]
        dist.broadcast_object_list(obj_list, src=TARGET_RANK)
        full_osd = obj_list[0]

        # Shard the full optimizer state dict
        sharded_osd = FSDP.shard_full_optim_state_dict(
            full_osd, wrapped_model, optim_input,
        )
        # Since we did not change the model wrapping configuration, this
        # sharded optimizer state dict should be the same as the original
        old_sharded_osd = wrapped_optim.state_dict()
        self.assertEqual(set(sharded_osd.keys()), set(old_sharded_osd.keys()))
        assert set(old_sharded_osd.keys()) == {"state", "param_groups"}
        self._check_same_state(sharded_osd["state"], old_sharded_osd["state"])
        self._check_same_param_groups(
            sharded_osd["param_groups"],
            old_sharded_osd["param_groups"],
        )
        # No need to check reloading since we already checked that
        # sharded optimizer state dict matches the original -- this would just
        # be testing `torch.optim.Optimizer.load_state_dict()`

    @skip_if_lt_x_gpu(2)
    @parametrize("wrapped_cls1", ["jagged"])
    @parametrize("wrapped_cls2", ["jagged", "alternate_jagged"])
    @parametrize("nonwrapped_cls", ["jagged"])
    @parametrize("use_multiple_param_groups", [False, True])
    @parametrize("use_new_process_group", [False, True])
    def test_shard_full_optim_state_dict_world_size(
        self,
        wrapped_cls1: str,
        wrapped_cls2: str,
        nonwrapped_cls: str,
        use_multiple_param_groups: bool,
        use_new_process_group: bool,
    ):
        """Save the full optimizer state dict and load it using a new
        process group with a smaller world size."""
        torch.backends.cuda.matmul.allow_tf32 = False
        TARGET_RANK = 0
        # Get the full optimizer state dict
        wrapped_model, wrapped_optim, optim_input = self._init_wrapped(
            model_class=wrapped_cls1,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        # Run a few steps to generate some optimizer state
        self._step_model(wrapped_model, wrapped_optim, num_iters=3)
        full_osd = FSDP.full_optim_state_dict(
            wrapped_model,
            wrapped_optim,
            optim_input=optim_input,
            target_rank=TARGET_RANK,
        )

        # Broadcast the full optim state dict so that each rank has it (since
        # normally it would be saved to and loaded from disk)
        obj_list = [full_osd]
        dist.broadcast_object_list(obj_list, src=TARGET_RANK)
        full_osd = obj_list[0]

        if use_new_process_group:
            new_group_ranks = [r for r in range(self.world_size) if r % 2 == 0]
            new_group = dist.new_group(ranks=new_group_ranks)
            if self.rank not in new_group_ranks:
                return
        else:
            new_group = None

        # Create a new wrapped model and local-only nonwrapped model both with
        # the same fresh parameters
        wrapped_model, wrapped_optim, optim_input = self._init_wrapped(
            model_class=wrapped_cls2, group=new_group,
            use_multiple_param_groups=use_multiple_param_groups,
        )
        nonwrapped_model, nonwrapped_optim = self._init_nonwrapped(
            model_class=nonwrapped_cls,
            use_multiple_param_groups=use_multiple_param_groups,
        )

        # Load the generated optimizer state from the original wrapped model
        sharded_osd = FSDP.shard_full_optim_state_dict(
            full_osd, wrapped_model, optim_input,
        )
        wrapped_optim.load_state_dict(sharded_osd)
        nonwrapped_optim.load_state_dict(full_osd)

        # Take a step with both models and check the parameters are the same
        self._step_model(wrapped_model, wrapped_optim)
        self._step_model(nonwrapped_model, nonwrapped_optim)
        self._check_same_model_params(wrapped_model, nonwrapped_model)


instantiate_parametrized_tests(TestFSDPOptimState)

if __name__ == "__main__":
    run_tests()
