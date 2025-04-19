import argparse

import benchmark_core

import benchmark_utils

import torch


"""Performance microbenchmarks's main binary.

This is the main function for running performance microbenchmark tests.
It also registers existing benchmark tests via Python module imports.
"""
parser = argparse.ArgumentParser(
    description="Run microbenchmarks.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


def parse_args():
    parser.add_argument(
        "--tag-filter",
        "--tag_filter",
        help="tag_filter can be used to run the shapes which matches the tag. (all is used to run all the shapes)",
        default="short",
    )

    # This option is used to filter test cases to run.
    parser.add_argument(
        "--operators",
        help="Filter tests based on comma-delimited list of operators to test",
        default=None,
    )

    parser.add_argument(
        "--operator-range",
        "--operator_range",
        help="Filter tests based on operator_range(e.g. a-c or b,c-d)",
        default=None,
    )

    parser.add_argument(
        "--test-name",
        "--test_name",
        help="Run tests that have the provided test_name",
        default=None,
    )

    parser.add_argument(
        "--list-ops",
        "--list_ops",
        help="List operators without running them",
        action="store_true",
    )

    parser.add_argument(
        "--output-json",
        "--output_json",
        help="JSON file path to write the results to",
        default=None,
    )

    parser.add_argument(
        "--list-tests",
        "--list_tests",
        help="List all test cases without running them",
        action="store_true",
    )

    parser.add_argument(
        "--iterations",
        help="Repeat each operator for the number of iterations",
        type=int,
    )

    parser.add_argument(
        "--num-runs",
        "--num_runs",
        help="Run each test for num_runs. Each run executes an operator for number of <--iterations>",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--min-time-per-test",
        "--min_time_per_test",
        help="Set the minimum time (unit: seconds) to run each test",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--warmup-iterations",
        "--warmup_iterations",
        help="Number of iterations to ignore before measuring performance",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--omp-num-threads",
        "--omp_num_threads",
        help="Number of OpenMP threads used in PyTorch runtime",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--mkl-num-threads",
        "--mkl_num_threads",
        help="Number of MKL threads used in PyTorch runtime",
        default=None,
        type=int,
    )

    parser.add_argument(
        "--report-aibench",
        "--report_aibench",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Print result when running on AIBench",
    )

    parser.add_argument(
        "--use-jit",
        "--use_jit",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Run operators with PyTorch JIT mode",
    )

    parser.add_argument(
        "--forward-only",
        "--forward_only",
        type=benchmark_utils.str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Only run the forward path of operators",
    )

    parser.add_argument(
        "--device",
        help="Run tests on the provided architecture (cpu, cuda)",
        default="None",
    )

    args, _ = parser.parse_known_args()

    if args.omp_num_threads:
        # benchmark_utils.set_omp_threads sets the env variable OMP_NUM_THREADS
        # which doesn't have any impact as C2 init logic has already been called
        # before setting the env var.

        # In general, OMP_NUM_THREADS (and other OMP env variables) needs to be set
        # before the program is started.
        # From Chapter 4 in OMP standard: https://www.openmp.org/wp-content/uploads/openmp-4.5.pdf
        # "Modifications to the environment variables after the program has started,
        # even if modified by the program itself, are ignored by the OpenMP implementation"
        benchmark_utils.set_omp_threads(args.omp_num_threads)
        torch.set_num_threads(args.omp_num_threads)
    if args.mkl_num_threads:
        benchmark_utils.set_mkl_threads(args.mkl_num_threads)

    return args


def main():
    args = parse_args()
    benchmark_core.BenchmarkRunner(args).run()


if __name__ == "__main__":
    main()
