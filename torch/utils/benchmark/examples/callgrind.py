"""Demonstrate Callgrind collection in both C++ and Python."""

import torch
import torch.utils.benchmark as benchmark_utils


def main():
    python_timer = benchmark_utils.Timer(
        stmt="y = x + torch.ones((1,))",
        setup="x = torch.ones((1,))",
    )

    cpp_timer = benchmark_utils.Timer(
        stmt="auto y = torch::add(x, torch::ones({1}));",
        setup="torch::Tensor x = torch::ones({1});",
        language="C++",
    )

    python_times = python_timer.blocked_autorange(min_run_time=5)
    cpp_times = cpp_timer.blocked_autorange(min_run_time=5)
    print(f"{python_times}\n\n{cpp_times}\n")

    callgrind_number = 10000
    python_stats = python_timer.collect_callgrind(
        number=callgrind_number, collect_baseline=False)
    cpp_stats = cpp_timer.collect_callgrind(number=callgrind_number)
    print(f"{python_stats}\n\n{cpp_stats}\n")

    torch.set_printoptions(linewidth=160)
    delta = python_stats.as_standardized().delta(cpp_stats.as_standardized())
    print(f"{delta}\n")
    print(delta.filter(lambda l: "py" in l.lower()))


if __name__ == "__main__":
    main()
