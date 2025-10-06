import torch
import sys
from torch.distributed.tensor.placement_types import Replicate, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.distributed.tensor import DTensor, distribute_tensor

torch._dynamo.config.assume_static_by_default = True
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

class Repro(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device_mesh):
        super().__init__()
        # Initialize the parameter as a distributed nn.Parameter using Shard(0), Shard(1)
        self.weight = torch.nn.Parameter(
            distribute_tensor(
                torch.rand(
                    [num_embeddings, embedding_dim],
                    dtype=torch.float32,
                    device="cuda",
                    requires_grad=True,
                ),
                device_mesh,
                [Shard(0), Shard(1)],
            )
        )
        # Add two parameters for matmul demonstration
        self.mat1 = torch.nn.Parameter(
            distribute_tensor(
                torch.randn(32, 64, device="cuda"),
                device_mesh,
                [Shard(0), Shard(1)],
            )
        )
        self.mat2 = torch.nn.Parameter(
            distribute_tensor(
                torch.randn(64, 16, device="cuda"),
                device_mesh,
                [Shard(0), Shard(1)],
            )
        )

    def forward(self, input_batch_inputs_):
        matmul_result = self.mat1 @ self.mat2

        torch._dynamo.graph_break()

        input_batch_inputs_ = input_batch_inputs_.redistribute(placements=[Shard(0), Shard(1)])
        to = self.weight.to(torch.float32)
        emb = torch.nn.functional.embedding(input_batch_inputs_, to)

        torch._dynamo.graph_break()

        emb = emb.redistribute(placements=[Shard(0), Shard(1)])
        return emb, matmul_result

world_size = 128
fake_store = FakeStore()
torch.distributed.init_process_group(
    "fake", store=fake_store, rank=0, world_size=world_size
)
mesh = torch.distributed.device_mesh.init_device_mesh(
    "cuda",
    (8, 2),
    mesh_dim_names=("dim1", "dim2"),
)
placements = (Shard(0), Shard(1))
arg0 = torch.randint(low=0, high=100, size=(2, 512), dtype=torch.int64, device="cuda")
arg0 = DTensor.from_local(arg0, mesh, placements)
num_embeddings = 202048
embedding_dim = 256
module = Repro(num_embeddings, embedding_dim, mesh)

out_eager_emb, out_eager_matmul = module(arg0)
(out_eager_emb.sum() + out_eager_matmul.sum()).backward()
print("Eager Success! ✅")
compiled_module = torch.compile(module, dynamic=True)
out_compiled_emb, out_compiled_matmul = compiled_module(arg0)
(out_compiled_emb.sum() + out_compiled_matmul.sum()).backward()
print("Compile Success! ✅")
# Compare outputs
out_eager_sum = out_eager_emb.sum() + out_eager_matmul.sum()
out_compiled_sum = out_compiled_emb.sum() + out_compiled_matmul.sum()
diff = (out_eager_sum - out_compiled_sum).abs().item()
rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
print(f"Relative diff (sum): {rel_diff:.6f}%")
if rel_diff > 5:
    print(f"❌ Forward output sums differ significantly (relative)!")
    print("out_eager_sum:", out_eager_sum.item())
    print("out_compiled_sum:", out_compiled_sum.item())
    print("Absolute diff:", diff)
    print("Relative diff (%):", rel_diff)
    sys.exit(1)

torch.distributed.destroy_process_group()
