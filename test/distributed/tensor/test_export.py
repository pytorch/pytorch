import torch 
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    with_comms,
)
import contextlib

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import (
    DeviceMesh,
    DTensor,
    Shard,
    Replicate,
    distribute_tensor,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    loss_parallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
)
from torch._functorch.aot_autograd import (
    aot_export_joint_with_descriptors,

)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.debug_mode import DebugMode


from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec


torch.utils._pytree.register_constant(DTensorSpec)
torch.utils._pytree.register_constant(DeviceMesh)


class Block(torch.nn.Module):
    def __init__(self, nheads, dim1, dim2):
        super().__init__()
        self.nheads = nheads
        bias = False
        self.wq = torch.nn.Linear(dim1, dim1, bias=bias)
        self.wk = torch.nn.Linear(dim1, dim1, bias=bias)
        self.wv = torch.nn.Linear(dim1, dim1, bias=bias)
        self.wo = torch.nn.Linear(dim1, dim1, bias=bias)
        self.w1 = torch.nn.Linear(dim1, dim2, bias=bias)
        self.w2 = torch.nn.Linear(dim2, dim1, bias=bias)

    def init_weights(self):
        for lin in [self.wq, self.wk, self.wv, self.wo, self.w1, self.w2]:
            torch.nn.init.normal_(lin.weight, std=0.02)

    def _compute_attention(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = q.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)  # (B, H, T, Dh)
        k = k.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)
        v = v.unflatten(-1, (self.nheads, -1)).permute(0, 2, 1, 3)

        o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        o = o.permute(0, 2, 1, 3).flatten(-2)  # (B, T, D)
        o = self.wo(o)
        return o

    def forward(self, x):
        o = self._compute_attention(x)
        o0 = o + x
        o = self.w1(o0)
        o = torch.relu(o)
        o = self.w2(o)
        o = o0 + o
        return o


class SimpleModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.mlp_0 = MLPModule(device)
        self.mlp_1 = MLPModule(device)

    def forward(self, input):
        return self.mlp_1(self.mlp_0(input))


def to_dtensor_params(module: torch.nn.Module, mesh: DeviceMesh):
    """
    Replace every registered Parameter with an nn.Parameter wrapping a DTensor
    placed as Replicate(). Broadcast once to ensure identical replicas.
    """
    for mod in module.modules():
        for pname, p in list(mod._parameters.items()):
            if p is None:
                continue
            torch.distributed.broadcast(p.data, src=0)

            local = p.data.to(torch.cuda.current_device())
            dt = DTensor.from_local(local, mesh, placements=[Replicate()])

            new_p = torch.nn.Parameter(dt, requires_grad=p.requires_grad)
            mod.register_parameter(pname, new_p)
    return module


class DTensorExportTest(DTensorTestBase):
    @with_comms
    def test_dtensor_constructor(self):
        mesh = self.build_device_mesh()
        nheads, dim1, dim2 = 8, 512, 2048
        model = Block(nheads, dim1, dim2).to(self.device_type)
        model.init_weights()
        model = to_dtensor_params(model, mesh)

        B_global, T = 32, 128
        assert B_global % self.world_size == 0, "B must be divisible by world_size"
        x_global = torch.randn(B_global, T, dim1, device=self.device_type)
        x = distribute_tensor(x_global, mesh, placements=[Shard(0)])  # Shard along batch dim

        ep = torch.export.export(model, (x,), strict=True)

        if self.rank == 0:
            print(ep)

    @with_comms
    def test_dtensor_module(self):
        dp_degree = 2
        tp_degree = self.world_size // dp_degree
        model = SimpleModel(self.device_type)

        # 2-D mesh is [dp, tp]
        mesh_2d = init_device_mesh(
            self.device_type,
            mesh_shape=(dp_degree, tp_degree),
            mesh_dim_names=["dp", "tp"],
        )

        inp = torch.rand(20, 10, device=self.device_type)
        parallelize_plan = {
            "mlp_0.net1": ColwiseParallel(),
            "mlp_0.net2": RowwiseParallel(),
            "mlp_1.net1": ColwiseParallel(),
            "mlp_1.net2": RowwiseParallel(),
        }
        tp_model = parallelize_module(model, mesh_2d["tp"], parallelize_plan)

        x = distribute_tensor(inp, mesh_2d["tp"], placements=[Replicate()]) 

        with DebugMode(record_torchfunction=False) as debug_mode:
            out = tp_model(x)

        ep = torch.export.export(tp_model, (x,), strict=True)

        if self.rank == 0:
            print(debug_mode.debug_string())
            print(ep)

        # this works! 
        # with contextlib.ExitStack() as stack:
        #     self.joint_with_descriptors = aot_export_joint_with_descriptors(
        #         stack,
        #         tp_model,
        #         (inp,),
        #     )
        #     gm = self.joint_with_descriptors.graph_module

        # if self.rank == 0:
        #     gm.print_readable()
