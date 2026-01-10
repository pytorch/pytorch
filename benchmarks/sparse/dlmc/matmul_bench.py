# Sparse benchmarks

# This benchmark is for  sparse matmul performance test.
# They exist for comparing the performance of sparse matrix routines
# `sparse @ vector`, `sparse @ sparse` and `sparse @ dense` with different backends (CPU/CUDA)
# and with other frameworks such as scipy.

import argparse
import os
import sys

from scipy.sparse import isspmatrix

import torch
import torch.utils.benchmark as benchmark_utils
from .utils import load_dlmc_dataset


def scipy_matmul(mat1, mat2):
    if isspmatrix(mat1) and isspmatrix(mat2):
        return mat1.dot(mat2).tocoo()
    return mat1.dot(mat2)


def matmul_backward(a_dense, b_dense, grad_output):
    r1 = a_dense.matmul(b_dense)
    r1.backward(grad_output)


def sparse_matmul_backward(a, b, grad_output):
    c = torch.sparse.mm(a, b)
    c.backward(grad_output)


OPS_MAP = {
    "sparse@sparse": "torch.sparse.mm",
    "sparse@dense": "torch.matmul",
    "sparse@vector": "torch.matmul",
}


# also get the arguments as input from the user using `argparse`
def parse_args():
    parser = argparse.ArgumentParser(description="matmul benchmark")
    parser.add_argument("--path", type=str, help="DLMC dataset path")
    parser.add_argument("--dataset", type=str, default="magnitude_pruning")
    parser.add_argument("--hidden-size", "--hidden_size", default=2048, type=int)
    parser.add_argument("--backward-test", "--backward_test", action="store_true")
    parser.add_argument(
        "--operation",
        type=str,
        help="|".join(OPS_MAP.keys()),
        default=next(iter(OPS_MAP)),
    )
    parser.add_argument("--with-cuda", "--with_cuda", action="store_true")
    parser.add_argument(
        "--timer-min-run-time", "--timer_min_run_time", default=1, type=float
    )
    return parser


def get_tasks(op, backward_test, device):
    def filter_ops(operation):
        if backward_test:
            test_name = device + ":matmul-backward"
            return [
                (
                    test_name,
                    device,
                    "torch:" + operation.replace("sparse", "dense"),
                    "matmul_backward(dx, dy, grad_output)",
                ),
                (
                    test_name,
                    device,
                    "torch:" + operation,
                    "sparse_matmul_backward(x, y, sparse_grad_output)",
                ),
            ]
        else:
            test_name = device + ":matmul-forward"
            return list(
                filter(
                    None,
                    [
                        (
                            test_name,
                            device,
                            "torch:" + operation.replace("sparse", "dense"),
                            f"{OPS_MAP[operation]}(dx, dy)",
                        ),
                        (
                            test_name,
                            device,
                            "torch:" + operation,
                            f"{OPS_MAP[operation]}(x, y)",
                        ),
                        (
                            test_name,
                            device,
                            "scipy:" + operation,
                            "scipy_matmul(sx, sy)",
                        )
                        if device == "cpu"
                        else None,
                    ],
                )
            )

    all_operations = {
        "sparse@sparse": filter_ops("sparse@sparse"),
        "sparse@dense": filter_ops("sparse@dense"),
        "sparse@vector": filter_ops("sparse@vector"),
    }
    return all_operations[op]


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    if args.with_cuda and not torch.cuda.is_available():
        raise RuntimeError("No CUDA available")

    dataset_path = args.path
    dataset_name = args.dataset
    dataset_path = os.path.join(dataset_path, dataset_name)
    device = "cuda" if args.with_cuda else "cpu"

    tasks = get_tasks(args.operation, args.backward_test, device)
    repeats = 3
    timers = [
        benchmark_utils.Timer(
            stmt=stmt,
            globals={
                "scipy_matmul": scipy_matmul,
                "matmul_backward": matmul_backward,
                "sparse_matmul_backward": sparse_matmul_backward,
                **variables,
            },
            label=label,
            sub_label=sub_label,
            description=f"{sparsity}",
            env=device,
        )
        for sparsity in [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
        for label, device, sub_label, stmt in tasks
        for variables in load_dlmc_dataset(
            dataset_path,
            args.operation,
            args.hidden_size,
            sparsity,
            device,
            args.backward_test,
        )
    ]
    measurements = []

    for i, timer in enumerate(timers * repeats):
        m = timer.blocked_autorange(min_run_time=args.timer_min_run_time)
        m.metadata = {"device": "cuda" if m.task_spec.env.find("cuda") >= 0 else "cpu"}
        measurements.append(m)
        print(f"\r{i + 1} / {len(timers) * repeats}", end="")
        sys.stdout.flush()
    print()

    comparison = benchmark_utils.Compare(measurements)

    print("== Results " + "=" * 80 + "\n" + "/" * 95 + "\n")
    comparison.print()
