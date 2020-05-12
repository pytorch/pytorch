"""Example of the Timer and Fuzzer APIs:

$ python -m examples.fuzzer
"""

import sys

import utils as benchmark_utils


def main():
    add_fuzzer = benchmark_utils.Fuzzer(
        parameters=[
            benchmark_utils.FuzzedParameter("k0", 16, 16 * 1024, "loguniform"),
            benchmark_utils.FuzzedParameter("k1", 16, 16 * 1024, "loguniform"),
            benchmark_utils.FuzzedParameter("k2", 16, 16 * 1024, "loguniform"),
            benchmark_utils.FuzzedParameter("d", distribution={2: 0.6, 3: 0.4}),
        ],
        tensors=[
            benchmark_utils.FuzzedTensor(
                name="x", size=("k0", "k1", "k2"), dim_parameter="d",
                probability_contiguous=0.75, min_elements=64 * 1024,
                max_elements=128 * 1024,
            ),
            benchmark_utils.FuzzedTensor(
                name="y", size=("k0", "k1", "k2"), dim_parameter="d",
                probability_contiguous=0.75, min_elements=64 * 1024,
                max_elements=128 * 1024,
            ),
        ],
        seed=0,
    )

    n = 250
    measurements = []
    for i, (tensors, tensor_properties, _) in enumerate(add_fuzzer.take(n=n)):
        x, x_order = tensors["x"], str(tensor_properties["x"]["order"])
        y, y_order = tensors["y"], str(tensor_properties["y"]["order"])
        shape = ", ".join(tuple(f'{i:>4}' for i in x.shape))

        description = "".join([
            f"{x.numel():>7} | {shape:<16} | ",
            f"{'contiguous' if x.is_contiguous() else x_order:<12} | ",
            f"{'contiguous' if y.is_contiguous() else y_order:<12} | ",
        ])

        timer = benchmark_utils.Timer(
            stmt="x + y",
            globals=tensors,
            description=description,
        )

        measurements.append(timer.blocked_autorange(min_run_time=0.1))
        measurements[-1].metadata = {"numel": x.numel()}
        print(f"\r{i + 1} / {n}", end="")
        sys.stdout.flush()
    print()

    # More string munging to make pretty output.
    print(f"Average attemts per valid config: {1. / (1. - add_fuzzer.rejection_rate):.1f}")
    def time_fn(m):
        return m.median / m.metadata["numel"]
    measurements.sort(key=time_fn)

    template = f"{{:>6}}{' ' * 19}Size    Shape{' ' * 13}X order        Y order\n{'-' * 80}"
    print(template.format("Best:"))
    for m in measurements[:15]:
        print(f"{time_fn(m) * 1e9:>4.1f} ns / element     {m.description}")

    print("\n" + template.format("Worst:"))
    for m in measurements[-15:]:
        print(f"{time_fn(m) * 1e9:>4.1f} ns / element     {m.description}")


if __name__ == "__main__":
    main()
