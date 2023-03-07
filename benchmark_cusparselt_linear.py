import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormSparsifier

def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

device = "cuda"
dtype = torch.float16
m, n, k = 64, 64, 64

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(m, k)

    def forward(self, x):
        return self.linear(x)

class cuSPARSELtLinear(nn.Linear):

    def forward(self, x):
        print("Here")
        self.cslt.masked_mm(x)
        return self.res

    @classmethod
    def from_dense(cls, mod, sample_input):
        """
        convert from nn.Linear
        """
        weight_tensor = mod.weight.data
        bias_tensor = mod.bias.data

        cusparselt = cls(mod.in_features,
                         mod.out_features)

        res = torch.empty(m, n, dtype=dtype, device=device)

        cslt = torch.classes.cusparselt.CusparseLtLinear(weight_tensor)
        cslt.init(res, input_tensor, bias_tensor)
        cslt.prune()
        cslt.compress()
        cslt.search_matmul_algo()
        cslt.masked_mm(input_tensor)

        cusparselt.cslt = cslt
        cusparselt.res = res
        return cusparselt


torch.set_printoptions(precision=1, threshold=None, edgeitems=4, linewidth=460, profile=None, sci_mode=False)

print("Running benchmark ...")

model = Model()
model.half()
model.cuda()

pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(1, 4), zeros_per_block=2)
pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
pruner.step()
pruner.squash_mask()


print("Creating tensors")

# weight_tensor, bias_tensor = model.linear.weight.data, model.linear.bias.data
input_tensor = torch.randn(k, n, device=device, dtype=dtype)

# c2 = torch.matmul(weight_tensor, input_tensor) + bias_tensor
# torch.testing.assert_close(c2, res, rtol=1e-3, atol=1e-3)
# print("Pass")


print("Creating module")
cslt_linear = cuSPARSELtLinear.from_dense(model.linear, input_tensor)

for i in range(5):
    input_tensor = torch.randn(k, n, device=device, dtype=dtype)
    c1 = cslt_linear(input_tensor)
    print(c1)
    c2 = torch.addmm(model.linear.bias.data, model.linear.weight.data, input_tensor) 
    print(c2)
    torch.testing.assert_close(c2, c1, rtol=1e-3, atol=1e-3)
    print(f"Result {i} are valid")


sparse_t = benchmark_torch_function_in_microseconds(cslt_linear, c)
dense_t = benchmark_torch_function_in_microseconds(torch.addmm, model.linear.weight.data, c)

print(f"sparse_t: {sparse_t:.0f}us dense_t: {dense_t:.0f}us speedup: {dense_t/sparse_t:.2f}x")
