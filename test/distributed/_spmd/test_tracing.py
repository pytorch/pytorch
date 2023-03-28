# Owner(s): ["oncall: distributed"]

from copy import deepcopy
from functools import wraps
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributed._spmd.api import Schema, SPMD
from torch.distributed._spmd.comm_tensor import CommTensor
from torch.distributed._tensor import DeviceMesh, Replicate
from torch.distributed.distributed_c10d import get_global_rank, get_world_size
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)


def with_comms(func):
    @base_with_comms
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # make sure we set different random seeds for each rank
        # otherwise we dont need DDP / SPMD
        # (we would have the same parameters and inputs everywhere)
        torch.manual_seed(torch.distributed.get_rank())
        return func(self, *args, **kwargs)

    return wrapper


class TraceDeviceMeshTestBase:
    def _test_tracing_all_reduce_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                tensor = mesh.all_reduce(tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return tensor * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            reduced_tensor = traced_fn(local_tensor.clone())
            res_num = sum(global_ranks)
            self.assertEqual(reduced_tensor, torch.ones(3, 3) * res_num)

    def _test_broadcast_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            def fn(tensor: torch.Tensor):
                received_tensor = CommTensor(tensor.clone())
                mesh.broadcast(received_tensor, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return received_tensor * 1

            local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank
            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)

            # execute traced DeviceMesh communication
            received_tensor = traced_fn(local_tensor)
            res_num = global_ranks[0]
            self.assertEqual(received_tensor, torch.ones(3, 3) * res_num)

    def _test_scatter_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)

        # check all dim groups
        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]
            scattered_tensors = [
                torch.ones(3, 3, device=self.device_type) * global_rank
                for global_rank in global_ranks
            ]

            def fn(to_receive: torch.Tensor, to_scatter: List[torch.Tensor]):
                to_scatter = [CommTensor(t) for t in to_scatter]
                to_receive = CommTensor(to_receive)
                mesh.scatter(to_receive, to_scatter, mesh_dim=dim)
                # multiply with 1 to trigger wait on read during tracing.
                return to_receive * 1

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            to_receive = torch.empty_like(
                scattered_tensors[mesh.get_coordinate()[dim]]
            )
            traced_fn = make_fx(fn)(to_receive, [t + 1 for t in scattered_tensors])

            received_tensor = traced_fn(to_receive, scattered_tensors)
            self.assertEqual(received_tensor, torch.ones(3, 3) * self.rank)

    def _test_all_gather_nd(self, mesh_tensor):
        mesh = DeviceMesh(self.device_type, mesh_tensor)
        # each rank have its own tensor, all_gather gives a big tensor
        local_tensor = torch.ones(3, 3, device=self.device_type) * self.rank

        dim_to_subgroups = mesh.get_dim_groups()
        for dim, dim_group in enumerate(dim_to_subgroups):
            dim_group_size = get_world_size(dim_group)
            global_ranks = [
                get_global_rank(dim_group, i) for i in range(dim_group_size)
            ]

            gathered_list = [
                torch.empty_like(local_tensor) for _ in range(dim_group_size)
            ]

            def fn(gathered_list: List[torch.Tensor], tensor: torch.Tensor):
                gathered_list = [CommTensor(t) for t in gathered_list]
                tensor = CommTensor(tensor)
                mesh.all_gather(gathered_list, tensor, mesh_dim=dim)
                return [t * 1 for t in gathered_list]

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(gathered_list, local_tensor + 1)
            gathered_list = traced_fn(gathered_list, local_tensor)

            self.assertEqual(len(gathered_list), dim_group_size)
            for idx, gathered_tensor in enumerate(gathered_list):
                self.assertEqual(gathered_tensor, torch.ones(3, 3) * global_ranks[idx])


class TraceDeviceMesh3DTest(DTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 8

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_broadcast_nd(self):
        self._test_broadcast_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_scatter_nd(self):
        self._test_scatter_nd(torch.arange(8).reshape(2, 2, 2))

    @with_comms
    def test_all_gather_nd(self):
        self._test_all_gather_nd(torch.arange(8).reshape(2, 2, 2))


class TraceDeviceMesh2DTest(DTensorTestBase, TraceDeviceMeshTestBase):
    @property
    def world_size(self):
        return 4

    @with_comms
    def test_tracing_all_reduce_nd(self):
        self._test_tracing_all_reduce_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_broadcast_nd(self):
        self._test_broadcast_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_scatter_nd(self):
        self._test_scatter_nd(torch.arange(4).reshape(2, 2))

    @with_comms
    def test_all_gather_nd(self):
        self._test_all_gather_nd(torch.arange(4).reshape(2, 2))


class TraceModuleTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _test_trace_replicate(self, model: nn.Module, x, *args, **kwargs):
        # if x.device.type == "cuda":
        ddp = DDP(deepcopy(model))
        spmd = SPMD(
            deepcopy(model),
            schema=Schema(
                mesh=DeviceMesh(self.device_type, torch.arange(self.world_size)),
                placements=[Replicate()],
            ),
            input_schemas=kwargs["inp_schemas"] if "inp_schemas" in kwargs else None,
        )
        if "inp_schemas" in kwargs:
            del kwargs["inp_schemas"]
        only_fw = False
        if "only_fw" in kwargs:
            only_fw = kwargs["only_fw"]
            del kwargs["only_fw"]
        if only_fw:
            output_ddp = ddp(x, *args, **kwargs)
            output_spmd = spmd(x, *args, **kwargs)
            self.assertTrue(output_ddp.size(), output_spmd.size())
            return
        ddp(x, *args, **kwargs).sum().backward()
        spmd(x, *args, **kwargs).sum().backward()
        for p1, p2 in zip(ddp.parameters(), spmd.parameters()):
            # DDP divides gradients by world size to compute average, but
            # _Partial tensor shouldn't do that automatically. Hence explicitly
            # do division here.
            self.assertTrue(
                p1.grad.allclose(p2.grad / self.world_size) or p1.grad.allclose(p2.grad)
            )

    @with_comms
    def test_torch_cat(self):
        x = torch.rand((2, 4)).to(self.device_type)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.rand((2, 4)))

            def forward(self, x):
                # TODO(anj): Using self.w and ignoring x results in an allgather call
                # that we have not yet supported.
                return torch.cat((self.w, self.w), 0)

        model = Model().to(self.device_type)
        inp_kwargs = {}
        inp_kwargs["inp_schemas"] = [
            Schema(
                mesh=DeviceMesh(self.device_type, torch.arange(self.world_size)),
                placements=[Replicate()],
            )
        ]
        self._test_trace_replicate(
            Model().to(self.device_type),
            torch.rand((2, 4)).to(self.device_type),
            **inp_kwargs,
        )

    @with_comms
    def test_layer_norm_fw(self):
        # This test is for get_item support. layer_norm contains
        # tuples in its output which means we need to support get_item.
        input_dims = []

        input = np.random.randn(4, 5).astype(np.float32)
        model = nn.LayerNorm(input.shape[1:]).to(self.device_type)
        pt_input = torch.tensor(input, dtype=torch.float).to(self.device_type)
        self._test_trace_replicate(model, pt_input)

    @with_comms
    def test_baked_in_shape(self):
        class LCE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                torch.manual_seed(5)
                self.w = torch.nn.Parameter(torch.rand((5, 10)))
                self.b = torch.nn.Parameter(torch.rand((5)))

            def forward(self, x, *args, **kwargs):
                # the code below will bake in the shape of x_t as arguments to expand
                x_t = x.permute(0, 2, 1)
                y_t = kwargs["dict_test"]["value"].expand(x_t.shape) + args[0][
                    0
                ].expand(x_t.shape)
                # code below triggers an "expand" with shape baked in.
                return torch.nn.functional.linear(y_t, self.w, self.b)

        model = LCE().to(self.device_type)
        x = torch.randn(2, 10, 80).to(self.device_type)
        y = torch.randn(2, 80, 10).to(self.device_type)
        z = torch.randn(2, 80, 10).to(self.device_type)
        self._test_trace_replicate(model, x, [y], dict_test={"value": z})

    @with_comms
    def test_sequential(self):
        model = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)]).to(
            self.device_type
        )
        x = torch.randn(2, 10).to(self.device_type)
        self._test_trace_replicate(model, x)

    @with_comms
    def test_parallel(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.module_list = nn.ModuleList([nn.Linear(10, 10) for _ in range(2)])

            def forward(self, x):
                return sum([m(x) for m in self.module_list])

        model = Model().to(self.device_type)
        x = torch.randn(2, 10).to(self.device_type)
        self._test_trace_replicate(model, x)

    @with_comms
    def test_hybrid(self):
        bottom_model = nn.Sequential(
            nn.Linear(4, 8),
            nn.Softmax(),
        ).to(self.device_type)

        top_model = nn.Sequential(
            nn.Linear(8, 2),
            nn.Softmax(),
        ).to(self.device_type)

        hybrid = nn.Sequential(
            DDP(deepcopy(bottom_model)),
            SPMD(
                deepcopy(top_model),
                schema=Schema(
                    mesh=DeviceMesh(self.device_type, torch.arange(self.world_size)),
                    placements=[Replicate()],
                ),
            ),
        )
        ddp = DDP(nn.Sequential(deepcopy(bottom_model), deepcopy(top_model)))
        input = torch.randn(12, 4).to(self.device_type)

        ddp(input).sum().backward()
        hybrid(input).sum().backward()
        for p1, p2 in zip(ddp.parameters(), hybrid.parameters()):
            # DDP divides gradients by world size to compute average, but
            # _Partial tensor shouldn't do that automatically. Hence explicitly
            # do division here.
            self.assertTrue(
                p1.grad.allclose(p2.grad / self.world_size) or p1.grad.allclose(p2.grad)
            )


if __name__ == "__main__":
    run_tests()
