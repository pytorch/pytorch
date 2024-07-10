# Owner(s): ["oncall: distributed"]

import collections
import copy

from typing import Any, List, Optional, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed._composable.fsdp import fully_shard
from torch.nn.parallel.scatter_gather import _is_namedtuple
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    check_sharded_parity,
    DoubleLinear,
    FSDPTest,
)
from torch.testing._internal.common_utils import run_tests


class TestFullyShardAutograd(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    def _reduce_1d_partial_grads(
        self, module: nn.Module, group: Optional[dist.ProcessGroup] = None
    ) -> None:
        group = group or dist.distributed_c10d._get_default_group()
        for param in module.parameters():
            if param.grad is not None:
                param.grad.div_(group.size())

    @skip_if_lt_x_gpu(2)
    def test_unused_forward_output(self):
        """
        Tests that gradients propagate when running a backward where some
        forward output is not used to compute the loss, motivated by:
        https://github.com/pytorch/pytorch/pull/83195
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_unused_forward_output,
        )

    def _test_unused_forward_output(self, reshard_after_forward: Union[bool, int]):
        torch.manual_seed(42)
        local_batch_size = 2
        global_batch_size, dim = (self.world_size * local_batch_size, 24)
        model = DoubleLinear(dim=dim, use_second_linear=True)
        ref_model = copy.deepcopy(model).cuda()
        fully_shard(model.lin1, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(1)  # same on all ranks
        for iter_idx in range(10):
            # Use all forward outputs in the loss/backward for the first half
            # of the iterations and only the 1st forward output for the rest
            global_inp = torch.rand((global_batch_size, dim), device="cuda")
            local_inp = global_inp[
                self.rank * local_batch_size : (self.rank + 1) * local_batch_size
            ].detach()
            out1, out2 = model(local_inp)
            loss = (out1 * out2).sum() if iter_idx < 3 else out1.sum()
            loss.backward()
            optim.step()
            ref_out1, ref_out2 = ref_model(global_inp)
            ref_loss = (ref_out1 * ref_out2).sum() if iter_idx < 3 else ref_out1.sum()
            ref_loss.backward()
            self._reduce_1d_partial_grads(ref_model)
            ref_optim.step()
            dist.all_reduce(loss)  # partial -> replicated
            self.assertEqual(loss, ref_loss)
            optim.zero_grad(set_to_none=(iter_idx % 2))
            ref_optim.zero_grad(set_to_none=(iter_idx % 2))
            check_sharded_parity(self, ref_model, model)

    @skip_if_lt_x_gpu(2)
    def test_unused_forward_module(self):
        """
        Tests that gradients propagate when running a backward where some
        forward module is not used to compute the loss, motivated by:
        https://github.com/pytorch/pytorch/pull/80245
        """
        self.run_subtests(
            {"reshard_after_forward": [True, False, 2]},
            self._test_unused_forward_module,
        )

    def _test_unused_forward_module(self, reshard_after_forward: Union[bool, int]):
        torch.manual_seed(42)
        local_batch_size, dim = (2, 24)
        global_batch_size = self.world_size * local_batch_size
        model = DoubleLinear(dim=dim, use_second_linear=False)
        ref_model = copy.deepcopy(model).cuda()
        fully_shard(model.lin1, reshard_after_forward=reshard_after_forward)
        fully_shard(model.lin2, reshard_after_forward=reshard_after_forward)
        fully_shard(model, reshard_after_forward=reshard_after_forward)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(1)  # same on all ranks
        for iter_idx in range(10):
            global_inp = torch.rand((global_batch_size, dim), device="cuda")
            local_inp = global_inp[
                self.rank * local_batch_size : (self.rank + 1) * local_batch_size
            ].detach()
            losses: List[torch.Tensor] = []
            for _model, inp in ((ref_model, global_inp), (model, local_inp)):
                losses.append(_model(inp).sum())
                losses[-1].backward()
            self._reduce_1d_partial_grads(ref_model)
            dist.all_reduce(losses[1])  # partial -> replicated
            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _optim in (optim, ref_optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2))

    @skip_if_lt_x_gpu(2)
    def test_nontensor_activations(self):
        """
        Tests that gradients propagate when running forward with nontensor
        data structures wrapping the activations. This is mainly to test the
        hook registration.
        """
        self.run_subtests(
            {"container_type": [list, collections.namedtuple, tuple, dict]},
            self._test_nontensor_activations,
        )

    def _test_nontensor_activations(self, container_type: Type):
        class Module(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.lin1 = nn.Linear(dim, dim)
                self.lin2 = nn.Linear(dim, dim)
                self.relu = nn.ReLU()

            def forward(self, inp: Any):
                # Assume that the "0th" element of `inp` is a tensor, run some
                # forward computation on it, and pack it back into the same
                # data structure type as `inp`
                if isinstance(inp, list):
                    return [self._forward(inp[0])]
                elif _is_namedtuple(inp):
                    return type(inp)(*([self._forward(inp[0])] + list(inp[1:])))
                elif isinstance(inp, tuple):
                    return (self._forward(inp[0]),)
                elif isinstance(inp, dict):
                    return {"x": self._forward(inp["x"])}
                else:
                    raise NotImplementedError(
                        f"Unsupported input type {type(inp)}: {inp}"
                    )

            def _forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.relu(self.lin2(self.relu(self.lin1(x))))

        class ToContainerType(nn.Module):
            def __init__(self, container_type: Type):
                super().__init__()
                self.container_type = container_type

            def forward(self, x: torch.Tensor):
                if self.container_type is list:
                    return [x]
                elif self.container_type is collections.namedtuple:
                    nt = collections.namedtuple("NT", "x y")
                    return nt(x, torch.ones_like(x))
                elif self.container_type is tuple:
                    return (x,)
                elif self.container_type is dict:
                    return {"x": x}
                else:
                    raise NotImplementedError(
                        f"Unsupported container type: {self.container_type}"
                    )

        class FromContainerType(nn.Module):
            def __init__(self, container_type: Type):
                super().__init__()
                self.container_type = container_type

            def forward(self, x: torch.Tensor):
                if self.container_type in (list, collections.namedtuple, tuple):
                    return x[0]
                elif self.container_type is dict:
                    return x["x"]
                else:
                    raise NotImplementedError(
                        f"Unsupported container type: {self.container_type}"
                    )

        torch.manual_seed(42)
        local_batch_size, dim = (2, 24)
        global_batch_size = self.world_size * local_batch_size
        model = nn.Sequential(
            ToContainerType(container_type),
            Module(dim),
            Module(dim),
            Module(dim),
            FromContainerType(container_type),
        )
        ref_model = copy.deepcopy(model).cuda()
        for module in model:
            fully_shard(module)
        fully_shard(model)
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)

        torch.manual_seed(1)  # same on all ranks
        for iter_idx in range(10):
            global_inp = torch.rand((global_batch_size, dim), device="cuda")
            local_inp = global_inp[
                self.rank * local_batch_size : (self.rank + 1) * local_batch_size
            ].detach()
            losses: List[torch.Tensor] = []
            for _model, inp in ((ref_model, global_inp), (model, local_inp)):
                losses.append(_model(inp).sum())
                losses[-1].backward()
            self._reduce_1d_partial_grads(ref_model)
            dist.all_reduce(losses[1])  # partial -> replicated
            self.assertEqual(losses[0], losses[1])
            check_sharded_parity(self, ref_model, model)
            for _optim in (optim, ref_optim):
                _optim.step()
                _optim.zero_grad(set_to_none=(iter_idx % 2))


if __name__ == "__main__":
    run_tests()
