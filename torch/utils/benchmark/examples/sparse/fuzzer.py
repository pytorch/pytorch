"""Example of the Timer and Sparse Fuzzer APIs:

$ python -m examples.sparse.fuzzer
"""

import sys

import torch.utils.benchmark as benchmark_utils

def main():
    add_fuzzer = benchmark_utils.Fuzzer(
        parameters=[
            [
                benchmark_utils.FuzzedParameter(
                    name=f"k{i}",
                    minval=16,
                    maxval=16 * 1024,
                    distribution="loguniform",
                ) for i in range(3)
            ],
            benchmark_utils.FuzzedParameter(
                name="dim_parameter",
                distribution={2: 0.6, 3: 0.4},
            ),
            benchmark_utils.FuzzedParameter(
                name="sparse_dim",
                distribution={1: 0.3, 2: 0.4, 3: 0.3},
            ),
            benchmark_utils.FuzzedParameter(
                name="density",
                distribution={0.1: 0.4, 0.05: 0.3, 0.01: 0.3},
            ),
            benchmark_utils.FuzzedParameter(
                name="coalesced",
                distribution={True: 0.7, False: 0.3},
            )
        ],
        tensors=[
            [
                benchmark_utils.FuzzedSparseTensor(
                    name=name,
                    size=[f"k{i}" for i in range(3)],
                    min_elements=64 * 1024,
                    max_elements=128 * 1024,
                    sparse_dim="sparse_dim",
                    density="density",
                    dim_parameter="dim_parameter",
                    coalesced="coalesced"
                ) for name in ("x", "y")
            ],
        ],
        seed=0,
    )

    n = 100
    measurements = []

    for i, (tensors, tensor_properties, _) in enumerate(add_fuzzer.take(n=n)):
        x = tensors["x"]
        y = tensors["y"]
        shape = ", ".join(tuple(f'{i:>4}' for i in x.shape))
        x_tensor_properties = tensor_properties["x"]
        description = "".join([
            f"| {shape:<20} | ",
            f"{x_tensor_properties['sparsity']:>9.2f} | ",
            f"{x_tensor_properties['sparse_dim']:>9d} | ",
            f"{x_tensor_properties['dense_dim']:>9d} | ",
            f"{('True' if x_tensor_properties['is_hybrid'] else 'False'):>9} | ",
            f"{('True' if x.is_coalesced() else 'False'):>9} | "
        ])
        timer = benchmark_utils.Timer(
            stmt="torch.sparse.sum(x) + torch.sparse.sum(y)",
            globals=tensors,
            description=description,
        )
        measurements.append(timer.blocked_autorange(min_run_time=0.1))
        measurements[-1].metadata = {"nnz": x._nnz()}
        print(f"\r{i + 1} / {n}", end="")
        sys.stdout.flush()
    print()

    # More string munging to make pretty output.
    print(f"Average attemts per valid config: {1. / (1. - add_fuzzer.rejection_rate):.1f}")

    def time_fn(m):
        return m.mean / m.metadata["nnz"]

    measurements.sort(key=time_fn)

    template = f"{{:>6}}{' ' * 16} Shape{' ' * 17}\
    sparsity{' ' * 4}sparse_dim{' ' * 4}dense_dim{' ' * 4}hybrid{' ' * 4}coalesced\n{'-' * 108}"
    print(template.format("Best:"))
    for m in measurements[:10]:
        print(f"{time_fn(m) * 1e9:>5.2f} ns / element     {m.description}")

    print("\n" + template.format("Worst:"))
    for m in measurements[-10:]:
        print(f"{time_fn(m) * 1e9:>5.2f} ns / element     {m.description}")

if __name__ == "__main__":
    main()
