import torch
from torch._decomp import register_decomposition

lib = torch.library.Library("fsdp_test", "DEF")

lib.define("chunk_cat_(Tensor(a!) out, Tensor[] tensors, int dim, int num_chunks) -> ()")

@torch.library.impl(lib, "chunk_cat_", "Meta")
def chunk_cat_(out, tensors, dim, num_chunks):
    torch._chunk_cat(
        tensors, dim, num_chunks, out=out
    )

@torch.library.impl(lib, "chunk_cat_", "CUDA")
def chunk_cat_(out, tensors, dim, num_chunks):
    torch._chunk_cat(tensors, dim, num_chunks, out=out)


def f(x, y, z):
    full_default_3: "f32[2, 524544]" = torch.ops.aten.full.default([2, 524544], 1.0, dtype = torch.float32, layout = torch.strided, device = "cuda", pin_memory = False)
    chunk_cat_default_1 = torch.ops.fsdp_test.chunk_cat_.default(full_default_3, [x, y, z], 0, 2)
    mul_out = torch.mul(full_default_3, full_default_3)
    sum_out = mul_out.sum()
    return sum_out


if __name__ == "__main__":
    x = torch.randn([1024, 512], device="cuda")
    y = torch.randn([512], device="cuda")
    z = torch.randn([1024, 512], device="cuda")
    eager_out = f(x, y, z)

    compiled_aot_eager_f = torch.compile(f, backend="aot_eager", fullgraph=True)
    compiled_aot_eager_out = compiled_aot_eager_f(x, y, z)
    assert torch.allclose(eager_out, compiled_aot_eager_out), f"eager_out: {eager_out}, compiled_aot_eager_out: {compiled_aot_eager_out}"

    compiled_inductor_f = torch.compile(f, backend="inductor", fullgraph=True)
    compiled_inductor_out = compiled_inductor_f(x, y, z)
    assert torch.allclose(eager_out, compiled_inductor_out), f"eager_out: {eager_out}, compiled_inductor_out: {compiled_inductor_out}"
