from torch import nn
import torch
from torch.ao.pruning import WeightNormSparsifier

first_activation_dim =16
num_batches = 1
output_channel = 32
input_channel = 16
dtype = torch.float16

torch.set_printoptions(precision=1, threshold=None, edgeitems=4, linewidth=460, profile=None, sci_mode=False)


def run_benchmark(model, description, num_samples=100):
    # random_example = torch.randn(100, 700)
    device = next(model.parameters()).device
    random_example = torch.randn((num_batches, first_activation_dim, input_channel)).to(device)
    print("Latency Measurement Using PyTorch Benchmark...")
    num_threads = 1
    timer = benchmark.Timer(
        stmt="model(input_tensor)",
        globals={"model": model, "input_tensor": random_example},
        num_threads=num_threads,
        label="Latency Measurement",
        sub_label="torch.utils.benchmark.",
        description=description,
    )

    profile_result = timer.blocked_autorange(min_run_time=1)
    print(f"Mean Latency: {profile_result.mean * 1000:.5f} ms")
    print(f"Median Latency: {profile_result.median* 1000:.5f} ms")

    return profile_result

class cuSPARSELtLinear(nn.Linear):

    def __init__(self, batch_size, in_features, out_features):
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return x

    @classmethod
    def from_dense(cls, mod, batch_size):
        """
        convert from nn.Linear
        """
        cusparselt_linear = cls(mod.in_features,
                                mod.out_features)
        return linear


class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return self.linear(x)

m = Model()
m.linear.bias.data = torch.zeros_like(m.linear.bias.data)
m.cuda()

print("weight:")
print(m.linear.weight)
print("bias:")
print(m.linear.bias)

pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(4, 1), zeros_per_block=2)
pruner.prepare(m, [{"tensor_fqn": "linear.weight"}])
pruner.step()
pruner.squash_mask()

print("pruned weight:")
print(m.linear.weight)
m.half()

# activation is the input tensor
# 10, 10, 10
activation = torch.rand(num_batches, first_activation_dim, input_channel, device="cuda", dtype=dtype)
# res and bias 
res = torch.empty(num_batches, output_channel, first_activation_dim, device="cuda", dtype=dtype)
# print("inital res")
# print(res)

# benchmark cusparselt linear
weight_batched = m.linear.weight.data.unsqueeze(0).contiguous()
cusparse_linear = torch.classes.cusparselt.CusparseLtLinear(weight_batched)
temp = torch.permute(activation, (0, 2, 1)).contiguous()
cusparse_linear.init(temp, res, m.linear.bias.data.T)
cusparse_linear.prune()
cusparse_linear.compress()
cusparse_linear.search_matmul_algo()
cusparse_linear.masked_mm()
print("I am multiplying these two things:")
print(activation.shape)
print(weight_batched.shape)

print("final sparse res")
res = torch.permute(res, (0, 2, 1))
print(res)
print(res.shape)


dense_output = m(activation)
print("dense res")
print(dense_output)
print(dense_output.shape)
# print(torch.permute(dense_output, (0, 2, 1)))

print("Tensors are the same: ", torch.allclose(res, dense_output))

# print("sanity check")
# test = torch.permute(torch.matmul(weight_batched, temp), (0, 2, 1))
# print(test)
# print(test.shape)
