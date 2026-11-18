"""Trivial use of Timer API:

$ python -m examples.simple_timeit
"""

import torch

import torch.utils.benchmark as benchmark_utils


def main() -> None:
    timer = benchmark_utils.Timer(
        stmt="x + y",
        globals={"x": torch.ones((4, 8)), "y": torch.ones((1, 8))},
        label="Broadcasting add (4x8)",
    )

    for i in range(3):
        print(f"Run: {i}\n{'-' * 40}")
        print(f"timeit:\n{timer.timeit(10000)}\n")
        print(f"autorange:\n{timer.blocked_autorange()}\n\n")


if __name__ == "__main__":
    main()
