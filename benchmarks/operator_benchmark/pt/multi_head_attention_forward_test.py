
import operator_benchmark as op_bench
import torch


"""
Microbenchmarks for the masked_softmax operators on MHA.
"""


# Configs for masked_softmax ops
mha_configs_short = op_bench.config_list(
    attr_names=['L', 'N', 'E', 'S', 'num_heads', 'device'],
    attrs=[
        [256, 5, 16, 256, 4, 'cpu']
    ],
    tags=['short'],
    cross_product_configs={
        'sparsity': [0., 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1],
    },
)


mha_configs_long = op_bench.cross_product_configs(
    L=[256],
    N=[5],
    E=[16],
    S=[256],
    num_heads=[4],
    device=['cpu'],
    tags=['long'],
    sparsity=[0., 0.01, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1],
)


mha_ops_list = op_bench.op_list(
    attr_names=['op_name', 'op_func'],
    attrs=[
        ['multi_head_attention_forward', lambda x, q, k, v, attn_mask: x.forward(q, k, v, attn_mask=attn_mask)]
    ],
)


class MHABenchmark(op_bench.TorchBenchmarkBase):
    def init(self, L, N, E, S, device, num_heads, op_func, sparsity):
        self.query = torch.rand(L, N, E, device=device)
        self.key = torch.rand(S, N, E, device=device)
        self.value = torch.rand(S, N, E, device=device)

        self.attn_mask = torch.multinomial(
            torch.Tensor([sparsity, 1. - sparsity]),
            num_samples=L * S,
            replacement=True,
        ).view(S, L).to(torch.bool)

        self.mha = torch.nn.MultiheadAttention(embed_dim=E, num_heads=num_heads)

        self.op_func = op_func
        self.set_module_name("mha")

    def forward(self):
        with torch.no_grad():
            return self.op_func(self.mha, self.query, self.key, self.value, attn_mask=self.attn_mask)


op_bench.generate_pt_tests_from_op_list(mha_ops_list,
                                        mha_configs_short + mha_configs_long,
                                        MHABenchmark)


if __name__ == "__main__":
    op_bench.benchmark_runner.main()
