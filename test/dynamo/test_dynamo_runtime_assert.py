# Owner(s): ["module: dynamo"]
import time

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._dynamo.utils


class DenseBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = torch.nn.Linear(dim, dim)
        self.norm = torch.nn.LayerNorm(dim)
        self.gate = torch.nn.Linear(dim, dim)

    def forward(self, x):
        return self.norm(self.linear(x)) * torch.sigmoid(self.gate(x))


class DenseArch(torch.nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()
        self.blocks = torch.nn.ModuleList([DenseBlock(dim) for _ in range(num_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class RecModel(torch.nn.Module):
    """Simplified recommendation model with many nested submodules."""

    def __init__(self, num_events=10, num_layers=6, num_embeddings=12, dim=128):
        super().__init__()
        self.shared_arch = DenseArch(dim, num_layers)
        self.event_submodels = torch.nn.ModuleDict(
            {
                f"event_{i}": torch.nn.Sequential(
                    DenseArch(dim, num_layers),
                    torch.nn.Linear(dim, 1),
                )
                for i in range(num_events)
            }
        )
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(1000, dim) for _ in range(num_embeddings)]
        )

    def forward(self, x):
        x = self.shared_arch(x)
        outputs = []
        for submodel in self.event_submodels.values():
            outputs.append(submodel(x))
        return torch.cat(outputs, dim=-1)


class RuntimeAssertCompileTimeTests(torch._dynamo.test_case.TestCase):
    """Regression test for _set_node_metadata_hook overhead in
    insert_deferred_runtime_asserts (S603290).

    With inline_inbuilt_nn_modules=False, call_module nodes cause _copy_attr
    to install full module subtrees into the GraphModule, making gm.modules()
    large. If the pass wraps every loop iteration with _set_node_metadata_hook
    (which iterates gm.modules() on enter AND exit), the cost becomes
    O(nodes * modules) — catastrophic for large models.
    """

    @torch._dynamo.config.patch(inline_inbuilt_nn_modules=False)
    def test_insert_runtime_assert_pass_time(self):
        # Large model so overhead from O(nodes*modules) would be measurable.
        model = RecModel(num_events=30, num_layers=10, num_embeddings=20, dim=128)
        x = torch.randn(8, 128)

        torch._dynamo.reset()
        compiled = torch.compile(model, backend="eager")
        t0 = time.perf_counter()
        compiled(x)
        total_s = time.perf_counter() - t0

        pass_times = torch._dynamo.utils.compilation_time_metrics.get(
            "insert_deferred_runtime_asserts", []
        )
        pass_s = sum(pass_times)

        self.assertLess(
            pass_s / total_s,
            0.05,
            f"insert_deferred_runtime_asserts took {pass_s * 1000:.1f}ms out of "
            f"{total_s * 1000:.1f}ms ({pass_s / total_s * 100:.1f}% of compile time)",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
