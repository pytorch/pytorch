import sys

from benchmark_base import BenchmarkBase

import torch
import torch.export
from torch.fx.experimental._config import AggressiveGuardFreeMode


class UnbindSplitFlipCatModel(torch.nn.Module):
    """Model that performs unbind, split_with_sizes, flip, and cat operations."""

    def __init__(self, num_iterations: int = 40, batch_size: int = 128):
        super().__init__()
        self.num_iterations = num_iterations
        self.batch_size = batch_size

    def forward(self, sizes_tensors: list, data_tensors: list):
        results = []

        for i in range(self.num_iterations):
            sizes_tensor = sizes_tensors[i]
            data_tensor = data_tensors[i]

            to_result = sizes_tensor.to(torch.int64)
            unbind_result = torch.unbind(to_result)

            items = []
            for j in range(self.batch_size):
                getitem = unbind_result[j]
                item = getitem.item()
                torch._check(item >= 0)
                items.append(item)

            total = items[0]
            for j in range(1, self.batch_size):
                total = total + items[j]

            split_result = torch.split(data_tensor, items)

            flipped = []
            for j in range(self.batch_size):
                getitem = split_result[j]
                flip_result = torch.flip(getitem, [0])
                flipped.append(flip_result)

            cat_result = torch.cat(flipped)
            results.append(cat_result)

        return results


class Benchmark(BenchmarkBase):
    """Benchmark for unbind/split/flip/cat pattern with torch.export."""

    def __init__(self, num_iterations: int = 40, batch_size: int = 128):
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        super().__init__(
            category="unbind_split_flip_cat",
            backend="export",
            device="cpu",
        )

    def name(self):
        return f"{self.category()}_iter{self.num_iterations}_batch{self.batch_size}"

    def description(self):
        return "Benchmark unbind/split/flip/cat pattern with torch.export"

    def _prepare_once(self):
        torch._dynamo.config.capture_scalar_outputs = True
        torch.fx.experimental._config.aggressive_guard_free_semantics = (
            AggressiveGuardFreeMode.SKIP_RANGE_ANALYSIS
        )
        torch.fx.config.do_not_emit_stack_traces = True
        torch.manual_seed(0)

        self.model = UnbindSplitFlipCatModel(
            num_iterations=self.num_iterations,
            batch_size=self.batch_size,
        )

        self.sizes_list = []
        self.data_list = []

        for _ in range(self.num_iterations):
            sizes = torch.randint(1, 10, (self.batch_size,), dtype=torch.int64)
            total_size = int(sizes.sum().item())
            data = torch.arange(total_size, dtype=torch.int64)
            self.sizes_list.append(sizes)
            self.data_list.append(data)

    def _prepare(self):
        torch._dynamo.reset()

    def _work(self):
        torch.export.export(
            self.model,
            (self.sizes_list, self.data_list),
        )


def main():
    result_path = sys.argv[1]
    Benchmark().enable_instruction_count().collect_all().append_results(result_path)


if __name__ == "__main__":
    main()
