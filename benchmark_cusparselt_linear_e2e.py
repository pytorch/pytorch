import torch
import torch.utils.benchmark as benchmark
from torch import nn
from torch.ao.pruning import WeightNormSparsifier
from itertools import product

class Model(nn.Module):

    def __init__(self, m, k):
        super().__init__()
        # transposed so reversed
        self.linear = nn.Linear(k, m)

    def forward(self, x):
        return self.linear(x)

class cuSPARSELtLinear(nn.Linear):

    def forward(self, x):
        self.cslt.masked_mm(x.T)
        return self.res.T

    @classmethod
    def from_dense(cls, mod):
        """
        convert from nn.Linear
        """
        if getattr(mod, "sample_input", None) is not None:
            print("converting")
            sample_input = mod.sample_input
            weight_tensor = mod.weight.data
            bias_tensor = mod.bias.data.T.contiguous()

            cusparselt = cls(mod.in_features,
                             mod.out_features)

            # need to be separate otherwise numeric issues
            res = torch.zeros(mod.out_features, sample_input.size(1), dtype=dtype, device=device)
            offset = torch.zeros(mod.out_features, sample_input.size(1), dtype=dtype, device=device)

            # set up cusparselt
            cslt = torch.classes.cusparselt.CusparseLtLinear(weight_tensor)
            cslt.init(res, sample_input, bias_tensor, offset)
            cslt.prune()
            cslt.compress()
            cslt.search_matmul_algo()

            cusparselt.cslt = cslt
            cusparselt.res = res
            return cusparselt
        else:
            raise ValueError("No attribute sample_input set")

def get_linear(m, k, n):
    model = Model(m, k)
    model.half()
    model.cuda()

    pruner = WeightNormSparsifier(sparsity_level=1.0, sparse_block_shape=(4, 1), zeros_per_block=2)
    pruner.prepare(model, [{"tensor_fqn": "linear.weight"}])
    pruner.step()

    input_tensor = torch.randn(n, k, device=device, dtype=dtype)
    model.linear.sample_input = input_tensor.T

    # converted_model = pruner.convert(model, mapping={nn.Linear: cuSPARSELtLinear})
    # sparse_linear = converted_model
    # dense_linear = model

    sparse_linear = cuSPARSELtLinear.from_dense(model.linear)
    dense_linear = model.linear

    for i in range(5):
        input_tensor = torch.randn(n, k, device=device, dtype=dtype)
        dense_output = dense_linear(input_tensor)
        sparse_output = sparse_linear(input_tensor)
        print(dense_output, sparse_output)
        assert torch.allclose(sparse_output, dense_output, rtol=1e-3, atol=1e-3)

    return input_tensor, pruner, sparse_linear, dense_linear



results = []
device = "cuda"
dtype = torch.float16
torch.set_printoptions(precision=3, threshold=None, edgeitems=4, linewidth=460, profile=None, sci_mode=False)

shapes = [
    # distilbert shapes
    (768, 3072, 768),
    (3072, 768, 3072),
    # jiecao shapes
    (1024, 1536, 2048),
    (1024, 9408, 2048),
    (1024, 3200, 2048),
    (1024, 256, 9472),
    (1024, 10240, 256),
    (1024, 256, 12608),
    (1024, 2560, 1024),
    (1024, 512, 10240),
    (1024, 10240, 512),
    (1024, 2048, 1024),
    (1024, 512, 512),
    (1024, 1024, 1024),
    (1024, 2048, 2048),
    (2048, 1536, 2048),
    (2048, 9408, 2048),
    (2048, 3200, 2048),
    (2048, 256, 9472),
    (2048, 10240, 256),
    (2048, 256, 12608),
    (2048, 2560, 1024),
    (2048, 512, 10240),
    (2048, 10240, 512),
    (2048, 2048, 1024),
    (2048, 512, 512),
    (2048, 1024, 1024),
    (2048, 2048, 2048),
]
batch_sizes = [1] # [1, 4, 16, 64, 256]

for batch_size, (m, k, n) in product(batch_sizes, shapes):
    # label and sub_label are the rows
    # description is the column
    label = 'nn.ParameterizedLinear vs. nn.Linear vs. cuSPARSELt Linear'
    sub_label = f'batch_size: {batch_size:2d} | m:{m:6d} | k:{k:6d} | n:{n:6d}'

    input_tensor, pruner, sparse_linear, dense_linear = get_linear(m, k, n)

    # result = benchmark.Timer(
        # stmt='param_linear(input_tensor)',
        # globals={'input_tensor': input_tensor, 'param_linear': dense_linear},
        # label=label,
        # sub_label=sub_label,
        # description='prepare latency',
    # )
    # results.append(result.blocked_autorange(min_run_time=1))

    pruner.squash_mask()

    result = benchmark.Timer(
        stmt='dense_linear(input_tensor)',
        globals={'input_tensor': input_tensor, 'dense_linear': dense_linear},
        label=label,
        sub_label=sub_label,
        description='dense latency',
    )
    results.append(result.blocked_autorange(min_run_time=1))

    result = benchmark.Timer(
        stmt='sparse_linear(input_tensor)',
        globals={'input_tensor': input_tensor, 'sparse_linear': sparse_linear},
        label=label,
        sub_label=sub_label,
        description='sparse latency',
    )
    results.append(result.blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.colorize(rowwise=True)
compare.print()
