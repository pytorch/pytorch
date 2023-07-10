# Owner(s): ["oncall: distributed"]

from copy import deepcopy
from functools import wraps
from typing import Any, List, Type

import torch
import torch.distributed as dist
import torch.fx as fx
import torch.nn as nn
from torch.distributed._spmd.api import compile, COMPILED_OBJECT_KEY, Override
from torch.distributed._spmd.comm_tensor import CommTensor
from torch.distributed._tensor import DeviceMesh
from torch.distributed._tensor.op_schema import OpSchema, OutputSharding
from torch.distributed._tensor.ops.utils import register_prop_rule
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.distributed_c10d import get_global_rank, get_world_size
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
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
        torch.manual_seed(self.rank)
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
            to_receive = torch.empty_like(scattered_tensors[mesh.get_coordinate()[dim]])
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

            def fn(tensor: torch.Tensor):
                big_tensor = mesh.all_gather(tensor, mesh_dim=dim)
                return list(torch.chunk(big_tensor, dim_group_size))

            # use a local_tensor + 1 for tracing to make sure that we are not
            # simply replaying recorded tensor value
            traced_fn = make_fx(fn)(local_tensor + 1)
            gathered_list = traced_fn(local_tensor)

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


class DataDependentModule(nn.Module):
    def __init__(self, world_size):
        super().__init__()
        self.world_size = world_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(
            "This eager implementation shouldn't be executed."
            "This implementation is just an example of how to get around "
            "data-dependant user-defined modules. "
        )
        shape = x.shape
        x = x.view(-1)
        positive = x[x >= 0]
        negative = x[x < 0]

        in_sizes = torch.tensor([positive.numel(), negative.numel()], dtype=torch.int32)
        out_sizes = torch.empty_like(in_sizes)
        dist.all_to_all_single(
            out_sizes,
            in_sizes,
            output_split_sizes=[1, 1],
            input_split_sizes=[1, 1],
        )

        xs = [positive, negative]
        ys = [torch.Tensor(out_sizes[i].item()) for i in range(out_sizes.numel())]
        dist.all_to_all(ys, xs)

        # some dummy compute
        for y in ys:
            y.add_(1)

        dist.all_to_all(xs, ys)

        return torch.cat(xs).reshape(shape)


class DummyModel(nn.Module):
    def __init__(self, world_size):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.ddm = DataDependentModule(world_size)
        self.l2 = nn.Linear(10, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        assert len(x.size()) == 2

        return self.relu(self.l2(self.ddm(self.l1(x))))


def ddm(x: torch.Tensor) -> torch.Tensor:
    return x


def ddm_backward(grad: torch.Tensor) -> torch.Tensor:
    return grad


dummy_lib = torch.library.Library("dummy", "DEF")
dummy_lib.define("ddm(Tensor x) -> Tensor")
dummy_lib.impl("ddm", ddm, "CompositeExplicitAutograd")
dummy_lib.define("ddm_backward(Tensor x) -> Tensor")
dummy_lib.impl("ddm_backward", ddm_backward, "CompositeExplicitAutograd")


def _identity_prop_rule(op_schema: OpSchema) -> OutputSharding:
    (x,) = op_schema.args_schema
    assert isinstance(x, DTensorSpec), f"expecting DTensorSpec but got {x}"

    return OutputSharding(output_spec=DTensorSpec(x.mesh, x.placements))


@register_prop_rule(torch.ops.dummy.ddm.default)
def _prop_ddm(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)


@register_prop_rule(torch.ops.dummy.ddm_backward.default)
def _prop_ddm_backward(op_schema: OpSchema) -> OutputSharding:
    return _identity_prop_rule(op_schema)


class DDMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.dummy.ddm(x)

    @staticmethod
    def backward(ctx: Any, grad_x: torch.Tensor) -> torch.Tensor:
        return torch.ops.dummy.ddm_backward(grad_x)


class DummyDDM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return DDMFunction.apply(x)


class TraceTrainStepTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_train_step_simple(self):
        @compile()
        def train_step(mod, inp):
            mod(inp).sum().backward()
            return [p.grad for p in mod.parameters()]

        inp = torch.randn(2, 10).cuda(self.rank)
        # FIXME(@mrshenli): remove manual seed once dist.compile can synchronize
        # module parameters.
        torch.manual_seed(0)
        mod = nn.Linear(10, 10).cuda(self.rank)

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_inp = deepcopy(inp)

        grads = train_step(mod, inp)
        ddp_mod(ddp_inp).sum().backward()

        for g1, p2 in zip(grads, ddp_mod.parameters()):
            # FIXME(@mrshenli): DDP by default divides gradients by world size.
            # Should we match that behavior?
            self.assertEqual(g1 / self.world_size, p2.grad)

    def _test_optimizer(self, mod, ddp_mod, opt, ddp_opt, inp, train_step):
        ddp_inp = deepcopy(inp)

        # materialize optimizer states
        mod(inp).sum().backward()
        opt.step()
        opt.zero_grad()
        # NB: In *normal* use cases you don't need to do this, but some
        # of the tests setup buffers which require_grad=True which is weird
        # and so work around it
        for buf in mod.buffers():
            buf.grad = None

        ddp_mod(ddp_inp).sum().backward()
        ddp_opt.step()
        ddp_opt.zero_grad()
        for buf in ddp_mod.buffers():
            buf.grad = None

        # test parameter parity
        train_step(mod, opt, inp)

        ddp_mod(ddp_inp).sum().backward()
        # FIXME(@mrshenli): DDP by default divides grads by world size, but
        # torch.distributed.compile does not do that yet.
        with torch.no_grad():
            for p in ddp_mod.parameters():
                p.grad *= self.world_size
        ddp_opt.step()

        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sgd(self):
        @compile()
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        # FIXME(@mrshenli): remove manual seed once dist.compile can synchronize
        # module parameters.
        torch.manual_seed(1)
        mod = nn.Linear(10, 10, bias=True).cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        inp = torch.randn(2, 10).cuda(self.rank)

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.01, foreach=True)
        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)

    def _test_adam(self, *, foreach: bool, fused: bool):
        class AssertOverride(Override):
            def __init__(self, outer):
                self.outer = outer

            def replacement(
                self, fqn: str, orig_submodule: torch.nn.Module
            ) -> torch.nn.Module:
                return orig_submodule

            def transform(
                self,
                gm: fx.GraphModule,
                flat_state: List[torch.Tensor],
            ) -> fx.Graph:
                # check dedup is successful, where there should only be 1 allreduce
                self.outer.assertEqual(
                    len(
                        [
                            n
                            for n in gm.graph.nodes
                            if n.target == torch.ops.c10d_functional.all_reduce.default
                        ]
                    ),
                    1,
                )

                return gm

        @compile(module_override=[AssertOverride(self)])
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        # FIXME(@mrshenli): remove manual seed once dist.compile can synchronize
        # module parameters.
        torch.manual_seed(0)
        # FIXME(@mrshenli): gradients for bias is missing
        mod = nn.Sequential(nn.Linear(10, 10, bias=False)).cuda(self.rank)
        opt = torch.optim.Adam(
            mod.parameters(),
            lr=0.01,
            foreach=foreach,
            fused=fused,
            capturable=True,
        )
        inp = torch.randn(2, 10).cuda(self.rank)

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = torch.optim.Adam(
            ddp_mod.parameters(), lr=0.01, foreach=foreach, fused=fused
        )
        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_adam_foreach(self):
        self._test_adam(foreach=True, fused=False)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_adam_fused(self):
        self._test_adam(foreach=False, fused=True)

    def _test_train_step_override(self):
        transform_targets = []

        class DDMOverride(Override):
            def replacement(
                self, fqn: str, orig_submodule: torch.nn.Module
            ) -> torch.nn.Module:
                return (
                    DummyDDM()
                    if isinstance(orig_submodule, DataDependentModule)
                    else orig_submodule
                )

            def transform(
                self,
                gm: fx.GraphModule,
                flat_state: List[torch.Tensor],
            ) -> fx.Graph:
                nonlocal transform_targets
                for node in gm.graph.nodes:
                    if node.target in [
                        torch.ops.dummy.ddm.default,
                        torch.ops.dummy.ddm_backward.default,
                    ]:
                        transform_targets.append(node.target)
                        # N.B.: this is not a complete subgraph representing
                        # original logic, as we are testing the ability to
                        # modify graph after DTensor expansion.
                        with gm.graph.inserting_before(node):
                            new_node = gm.graph.call_function(torch.add, args=node.args)
                        node.replace_all_uses_with(new_node)

                gm.graph.lint()
                gm.graph.eliminate_dead_code()

                return gm

        @compile(module_override=[DDMOverride()])
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        mod = DummyModel(self.world_size).cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=False)
        # FIXME: symbolic tracing treats bs=1 as constant, have to use bs > 1.
        inp = torch.randn(4, 10).cuda(self.rank)
        train_step(mod, opt, inp)

        # checking transforms are indeed invoked.
        self.assertEqual(
            transform_targets,
            [torch.ops.dummy.ddm.default, torch.ops.dummy.ddm_backward.default],
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_module_override(self):
        self._test_train_step_override()

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_module_multi_fqn_override(self):
        transform_targets = []

        class DDMOverride(Override):
            def replacement(
                self, fqn: str, orig_submodule: torch.nn.Module
            ) -> torch.nn.Module:
                return (
                    DummyDDM()
                    if isinstance(orig_submodule, DataDependentModule)
                    else orig_submodule
                )

            def transform(
                self,
                gm: fx.GraphModule,
                flat_state: List[torch.Tensor],
            ) -> fx.Graph:
                nonlocal transform_targets
                for node in gm.graph.nodes:
                    if node.target in [
                        torch.ops.dummy.ddm.default,
                        torch.ops.dummy.ddm_backward.default,
                    ]:
                        transform_targets.append(node.target)
                        # N.B.: this is not a complete subgraph representing
                        # original logic, as we are testing the ability to
                        # modify graph after DTensor expansion.
                        with gm.graph.inserting_before(node):
                            new_node = gm.graph.call_function(torch.add, args=node.args)
                        node.replace_all_uses_with(new_node)

                gm.graph.eliminate_dead_code()

                return gm

        class MultiDDM(nn.Module):
            def __init__(self, world_size):
                super().__init__()
                self.l1 = nn.Linear(10, 10)
                self.ddm1 = DataDependentModule(world_size)
                self.l2 = nn.Linear(10, 10)
                self.ddm2 = DataDependentModule(world_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                assert len(x.size()) == 2

                return self.relu(self.ddm2(self.l2(self.ddm1(self.l1(x)))))

        @compile(module_override=[DDMOverride()])
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        mod = MultiDDM(self.world_size).cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=False)
        inp = torch.randn(4, 10).cuda(self.rank)
        train_step(mod, opt, inp)

        # checking transforms are indeed invoked.
        self.assertEqual(
            transform_targets,
            [
                torch.ops.dummy.ddm.default,
                torch.ops.dummy.ddm.default,
                torch.ops.dummy.ddm_backward.default,
                torch.ops.dummy.ddm_backward.default,
            ],
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_gm_cache_and_transformation(self):
        class GraphOptimization:
            def __init__(self):
                self.call_count = 0

            def __call__(self, gm: fx.GraphModule) -> fx.GraphModule:
                self.call_count += 1
                return gm

        graph_optimization = GraphOptimization()

        @compile(gm_transformation=graph_optimization)
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        rank = torch.distributed.get_rank()
        torch.manual_seed(0)
        mod = nn.Linear(10, 10, bias=False).cuda(rank)
        opt = torch.optim.Adam(
            mod.parameters(), lr=0.01, foreach=False, capturable=True
        )
        inp = torch.randn(2, 10).cuda(rank)

        # materialize optimizer states
        mod(inp).sum().backward()
        opt.step()
        opt.zero_grad()

        train_step(mod, opt, inp)
        self.assertEqual(graph_optimization.call_count, 1)
        gm = train_step.__dict__[COMPILED_OBJECT_KEY].gm
        train_step(mod, opt, inp)
        self.assertEqual(id(gm), id(train_step.__dict__[COMPILED_OBJECT_KEY].gm))
        self.assertEqual(graph_optimization.call_count, 1)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_buffer(self):
        class BufferModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)
                self.register_buffer("dummy_buffer", torch.ones(10, 10))

            def forward(self, x):
                # N.B.: setting requires_grad in forward, as deepcopy does not
                # work for requires_grad=True buffers.
                self.dummy_buffer.requires_grad = True
                return torch.matmul(self.fc(x), self.dummy_buffer)

        class AssertOptimizer(torch.optim.Optimizer):
            def __init__(self, params, lr):
                super().__init__(params, dict(lr=lr))

            def step(self):
                assert len(self.param_groups[0]["params"]) == 2
                with torch.no_grad():
                    for p in self.param_groups[0]["params"]:
                        p += p.grad

        @compile()
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        torch.manual_seed(0)
        mod = BufferModule().cuda(self.rank)
        inp = torch.randn(2, 10).cuda(self.rank)
        opt = AssertOptimizer(mod.parameters(), lr=0.01)

        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])
        ddp_opt = AssertOptimizer(ddp_mod.parameters(), lr=0.01)

        self._test_optimizer(mod, ddp_mod, opt, ddp_opt, inp, train_step)
        self.assertEqual(mod.dummy_buffer, ddp_mod.module.dummy_buffer)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_expand_dimension(self):
        @compile()
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        mod = nn.Linear(10, 10, bias=True).cuda(self.rank)
        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        inp = torch.randn(2, 10).cuda(self.rank)
        train_step(mod, opt, inp)
        for node in train_step._compiled_obj.gm.graph.nodes:
            if node.target == torch.ops.aten.expand.default:
                # backward grad expandion op should match local batch size
                # instead of global batch size.
                self.assertEqual(node.args[1], [2, 10])


class CoverageTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _test_train_step(self, train_step, mod, *args):
        ddp_mod = DDP(deepcopy(mod), device_ids=[self.rank])

        opt = torch.optim.SGD(mod.parameters(), lr=0.01, foreach=True)
        ddp_opt = torch.optim.SGD(ddp_mod.parameters(), lr=0.01, foreach=True)

        ddp_args = deepcopy(args)

        # materialize optimizer states
        mod(*args).sum().backward()
        opt.step()
        opt.zero_grad()

        ddp_mod(*ddp_args).sum().backward()
        ddp_opt.step()
        ddp_opt.zero_grad()

        # test parameter parity
        train_step(mod, opt, *args)

        ddp_mod(*ddp_args).sum().backward()
        # FIXME(@mrshenli): DDP by default divides grads by world size, but
        # torch.distributed.compile does not do that yet.
        with torch.no_grad():
            for p in ddp_mod.parameters():
                p.grad *= self.world_size
        ddp_opt.step()

        for p1, p2 in zip(mod.parameters(), ddp_mod.parameters()):
            self.assertEqual(p1, p2)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_log_softmax(self):
        torch.manual_seed(0)

        @compile()
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        mod = nn.Sequential(
            nn.Linear(10, 10),
            nn.LogSoftmax(dim=1),
        ).cuda(self.rank)
        inp = torch.randn(2, 10).cuda(self.rank)
        self._test_train_step(train_step, mod, inp)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_nll_loss(self):
        class ModuleWithLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mod = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.LogSoftmax(dim=1),
                )
                self.lss = nn.NLLLoss()

            def forward(self, x, tgt):
                return self.lss(self.mod(x), tgt)

        torch.manual_seed(0)
        mod = ModuleWithLoss().cuda(self.rank)

        @compile()
        def train_step(mod, opt, inp, tgt):
            mod(inp, tgt).backward()
            opt.step()

        inp = torch.randn(2, 10).to(self.rank)
        tgt = torch.empty(2, dtype=torch.long).random_(0, 10).to(self.rank)

        self._test_train_step(train_step, mod, inp, tgt)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_replicated_embedding(self):
        N, D, B = 10, 8, 2

        class EmbeddingModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(N, D)
                self.norm = nn.LayerNorm(D, elementwise_affine=False)
                self.fc = nn.Linear(D, D)
                self.softmax = nn.Softmax(dim=1)
                self.lss = nn.NLLLoss()

            def forward(self, ids, tgt):
                return self.lss(self.softmax(self.fc(self.norm(self.emb(ids)))), tgt)

        torch.manual_seed(0)
        mod = EmbeddingModule().cuda(self.rank)

        @compile()
        def train_step(mod, opt, ids, tgt):
            mod(ids, tgt).sum().backward()
            opt.step()

        ids = torch.randint(0, N, (B,)).cuda(self.rank)
        tgt = torch.empty(B, dtype=torch.long).random_(0, D).to(self.rank)

        self._test_train_step(train_step, mod, ids, tgt)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_pos_embedding(self):
        N, D, B, Block = 10, 8, 2, 20

        class EmbeddingModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.wte = nn.Embedding(N, D)
                self.wpe = nn.Embedding(Block, D)
                self.norm = nn.LayerNorm(D, elementwise_affine=False)
                self.fc = nn.Linear(D, D)

            def forward(self, ids, tgt):
                _, t = ids.size()
                wte = self.wte(ids)
                wpe = self.wpe(
                    torch.arange(0, t, dtype=torch.long, device=ids.device).unsqueeze(0)
                )
                emb = wpe + wte
                norm = self.norm(emb)
                fc = self.fc(norm)
                log = F.softmax(fc, dim=-1)
                return F.cross_entropy(log.view(-1, log.size(-1)), tgt.view(-1))

        torch.manual_seed(0)
        mod = EmbeddingModule().cuda(self.rank)

        @compile()
        def train_step(mod, opt, ids, tgt):
            mod(ids, tgt).sum().backward()
            opt.step()

        ids = torch.randint(0, N, (B, Block)).cuda(self.rank)
        tgt = torch.empty((B, Block), dtype=torch.long).random_(0, D).to(self.rank)

        self._test_train_step(train_step, mod, ids, tgt)

    def _test_op_with_train_step(self, Model: Type[nn.Module]):
        torch.manual_seed(0)
        mod = Model().cuda(self.rank)

        @compile()
        def train_step(mod, opt, inp):
            mod(inp).sum().backward()
            opt.step()

        inp = torch.randn(2, 10).cuda(self.rank)
        self._test_train_step(train_step, mod, inp)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_factory_full(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                y = torch.full(x.shape, 7, device=x.device)
                return y + self.fc(x)

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_factory_arange(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                y = torch.arange(x.numel(), device=x.device).view(x.shape)
                z = torch.arange(0, x.numel(), device=x.device).view(x.shape)
                return self.fc(x) + y + z

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sym_numel(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                y = self.fc.weight.numel()
                return self.fc(x) + y

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_sym_stride(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                y = self.fc.weight.stride(0)
                return self.fc(x) + y

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_scalar(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                # FIXME: torch.tensor(x.numel()) is captured as a tensor constant
                y = torch.ops.aten.scalar_tensor.default(
                    7, dtype=x.dtype, device=x.device
                )
                return self.fc(x) + y

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_stack(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return torch.stack([x, self.fc(x)], dim=1)

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_arithmetic_ops_on_symint(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x) + x.shape[0] * x.numel() - x.shape[0] // 2

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_slice(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x)[:, :1]

        self._test_op_with_train_step(Model)

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_bulk_cat(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return torch.cat([self.fc(x) for _ in range(100)], dim=1)

        self._test_op_with_train_step(Model)


if __name__ == "__main__":
    if False:
        run_tests()
