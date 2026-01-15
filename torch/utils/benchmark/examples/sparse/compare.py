# mypy: allow-untyped-defs
"""Example of Timer and Compare APIs:

$ python -m examples.sparse.compare
"""

import pickle
import sys
import time

import torch
import torch.utils.benchmark as benchmark_utils


class FauxTorch:
    """Emulate different versions of pytorch.

    In normal circumstances this would be done with multiple processes
    writing serialized measurements, but this simplifies that model to
    make the example clearer.
    """
    def __init__(self, real_torch, extra_ns_per_element) -> None:
        self._real_torch = real_torch
        self._extra_ns_per_element = extra_ns_per_element

    @property
    def sparse(self):
        return self.Sparse(self._real_torch, self._extra_ns_per_element)

    class Sparse:
        def __init__(self, real_torch, extra_ns_per_element) -> None:
            self._real_torch = real_torch
            self._extra_ns_per_element = extra_ns_per_element

        def extra_overhead(self, result):
            # time.sleep has a ~65 us overhead, so only fake a
            # per-element overhead if numel is large enough.
            size = sum(result.size())
            if size > 5000:
                time.sleep(size * self._extra_ns_per_element * 1e-9)
            return result

        def mm(self, *args, **kwargs):
            return self.extra_overhead(self._real_torch.sparse.mm(*args, **kwargs))

def generate_coo_data(size, sparse_dim, nnz, dtype, device):
    """
    Parameters
    ----------
    size : tuple
    sparse_dim : int
    nnz : int
    dtype : torch.dtype
    device : str
    Returns
    -------
    indices : torch.tensor
    values : torch.tensor
    """
    if dtype is None:
        dtype = 'float32'

    indices = torch.rand(sparse_dim, nnz, device=device)
    indices.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(indices))
    indices = indices.to(torch.long)

    values = torch.rand([nnz, ], dtype=dtype, device=device)
    return indices, values

def gen_sparse(size, density, dtype, device='cpu'):
    sparse_dim = len(size)
    nnz = int(size[0] * size[1] * density)
    indices, values = generate_coo_data(size, sparse_dim, nnz, dtype, device)
    return torch.sparse_coo_tensor(indices, values, size, dtype=dtype, device=device)

def main() -> None:
    tasks = [
        ("matmul", "x @ y", "torch.sparse.mm(x, y)"),
        ("matmul", "x @ y + 0", "torch.sparse.mm(x, y) + zero"),
    ]

    serialized_results = []
    repeats = 2
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "torch": torch if branch == "master" else FauxTorch(torch, overhead_ns),
                "x": gen_sparse(size=size, density=density, dtype=torch.float32),
                "y": torch.rand(size, dtype=torch.float32),
                "zero": torch.zeros(()),
            },
            label=label,
            sub_label=sub_label,
            description=f"size: {size}",
            env=branch,
            num_threads=num_threads,
        )
        for branch, overhead_ns in [("master", None), ("my_branch", 1), ("severe_regression", 10)]
        for label, sub_label, stmt in tasks
        for density in [0.05, 0.1]
        for size in [(8, 8), (32, 32), (64, 64), (128, 128)]
        for num_threads in [1, 4]
    ]

    for i, timer in enumerate(timers * repeats):
        serialized_results.append(pickle.dumps(
            timer.blocked_autorange(min_run_time=0.05)
        ))
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    comparison = benchmark_utils.Compare([
        pickle.loads(i) for i in serialized_results
    ])

    print("== Unformatted " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()

    print("== Formatted " + "=" * 80 + "\n" + "/" * 93 + "\n")
    comparison.trim_significant_figures()
    comparison.colorize()
    comparison.print()


if __name__ == "__main__":
    main()
