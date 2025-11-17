# mypy: allow-untyped-defs
from typing import Any, cast, Optional, Union
from collections.abc import Callable

import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch.utils.benchmark import Timer


__all__ = ["bench_all", "benchmark_compile"]


_warned_tensor_cores = False
_default_float_32_precision = torch.get_float32_matmul_precision()

try:

    from tabulate import tabulate

    HAS_TABULATE = True
except ModuleNotFoundError:
    HAS_TABULATE = False
    tabulate = None  # type: ignore[assignment]
    print("tabulate is not installed, please pip install tabulate to use this utility")

if HAS_TABULATE:
    def _enable_tensor_cores():
        global _warned_tensor_cores

        if torch.cuda.is_available():
            if torch.backends.cuda.matmul.allow_tf32 is False and torch.cuda.get_device_capability() >= (8, 0):
                torch.set_float32_matmul_precision("high")
                if not _warned_tensor_cores:
                    print("Your GPU supports tensor cores")
                    print("we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`")
                    _warned_tensor_cores = True

    def _disable_tensor_cores():
        torch.set_float32_matmul_precision(_default_float_32_precision)

    def bench_loop(
        model: Union[torch.nn.Module, Callable],
        sample_input: Union[torch.Tensor, Any],
        num_iters: int = 5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn: Optional[Callable] = None,
    ):
        # Define the statement and setup for the benchmark
        if optimizer and loss_fn:
            # Training mode
            stmt = """
    output = model(sample_input)
    loss = loss_fn(output) if loss_fn else output.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
            """
        else:
            # Inference mode
            stmt = "model(sample_input)"

        # Create the Timer object
        timer = Timer(
            stmt=stmt,
            globals={"model": model, "sample_input": sample_input, "optimizer": optimizer, "loss_fn": loss_fn},
        )


        result = timer.timeit(number=num_iters)

        # Get the average time per iteration in milliseconds
        avg_time = result.mean * 1000
        return round(avg_time, 2)

    def benchmark_compile(
        model: Union[torch.nn.Module, Callable],
        sample_input: Union[torch.Tensor, Any],
        num_iters: int = 5,
        backend: Optional[str] = None,
        mode: Optional[str] = "default",
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn : Union[torch.nn.Module, Callable, None] = None,
    ):
        """
        Use this utility to benchmark torch.compile
        """
        if backend:
            try:
                torch._dynamo.reset()
                compile_counter_with_backend = CompileCounterWithBackend(backend)
                opt_model = torch.compile(model, backend=compile_counter_with_backend, mode=mode)

                # Compilation only happens after the first inference
                compilation_time = bench_loop(opt_model, sample_input, 1, optimizer, loss_fn)

                running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

                if compile_counter_with_backend.frame_count == 0:
                    raise RuntimeError("No compilation occurred during benchmarking.")

                if compile_counter_with_backend.frame_count > 1:
                    raise RuntimeError("Recompilation occurred during benchmarking.")

            except Exception as e:
                print(e)
                print(f"Failed to compile {backend} with mode {mode}")
                return None, None
        else:
            opt_model = model
            compilation_time = None
            running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

        compilation_time = round(compilation_time, 2) if compilation_time else None
        running_time = round(running_time, 2) if running_time else None


        return compilation_time, running_time


    def bench_all(
        model : Union[torch.nn.Module, Callable],
        sample_input: Union[torch.Tensor, Any],
        num_iters : int = 5,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_fn : Union[torch.nn.Module, Callable, None] = None,
    ):
        """
        This is a simple utility that can be used to benchmark torch.compile
        In particular it ensures that your GPU is setup to use tensor cores if it supports its
        It also tries out all the main backends and prints a table of results so you can easily compare them all
        Many of the backendds have their own optional dependencies so please pip install them separately

        You will get one table for inference and another for training
        If you'd like to leverage this utility for training make sure to pass in a torch.optim.Optimizer

        The important warnings are
        Your GPU supports tensor cores
        we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`

        If a compilation fails for any reason including the dependency not being included
        then we will print Failed to compile {backend} with mode {mode}
        """
        field_names = ["Train/Inference", "Backend", "Mode", "Compilation Time", "Average Running Time"]
        table = []


        eager_time = None
        torch._dynamo.reset()
        _, eager_time = benchmark_compile(model, sample_input, num_iters, None, None, optimizer)
        table.append(
            [("Training" if optimizer else "Inference"), "Eager", "-", "-", f"{eager_time} ms"]
        )

        for backend in torch._dynamo.list_backends():

            if backend == "inductor":
                mode_options = cast(list[Optional[str]], list(torch._inductor.list_mode_options().keys())) + [None]
                for mode in mode_options:
                    if mode == "default":
                        continue
                    torch._dynamo.reset()
                    try:
                        if torch.cuda.is_available():
                            _enable_tensor_cores()
                        compilation_time, running_time = benchmark_compile(
                            model, sample_input, num_iters, backend, mode, optimizer, loss_fn)
                    finally:
                        if torch.cuda.is_available():
                            _disable_tensor_cores()
                            table.append([
                                ("Training" if optimizer else "Inference"),
                                # pyrefly: ignore  # redundant-condition
                                backend if backend else "-",
                                mode if mode is not None else "-",
                                f"{compilation_time} ms " if compilation_time else "-",
                                f"{running_time} ms " if running_time else "-",
                            ])

            else:
                torch._dynamo.reset()
                compilation_time, running_time = benchmark_compile(
                    model, sample_input, num_iters, backend, None, optimizer, loss_fn)

                if running_time is not None:
                    table.append([
                        ("Training" if optimizer else "Inference"),
                        backend, "-",
                        f"{compilation_time} ms " or "-",
                        f"{running_time} ms ",
                    ])


        # pyrefly: ignore  # not-callable
        return tabulate(table, headers=field_names, tablefmt="github")
