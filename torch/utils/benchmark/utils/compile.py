import torch

__all__ = ["bench_all", "benchmark_compile"]

import torch._dynamo
import time
from typing import Optional, List, Callable, Union, Any, cast
from tabulate import tabulate

_warned_tensor_cores = False


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
    torch.set_float32_matmul_precision("highest")

def bench_loop(
    model: Union[torch.nn.Module, Callable],
    sample_input: Union[torch.Tensor, Any],
    num_iters: int,
    optimizer: torch.optim.Optimizer = None,
    loss_fn: Callable = None,
):
    """
    This is a simple loop that can be used to benchmark a model for either training or inference
    It takes care of taking several measurements and averaging them
    It also takes care of calling `cuda.synchronize()` if the model is on GPU
    """
    durations = []
    for _ in range(num_iters):
        start = time.time()

        if optimizer:
            optimizer.zero_grad()
            output = model(sample_input)
            loss = loss_fn(output) if loss_fn else output.sum()
            loss.backward()
            optimizer.step()
        else:
            model(sample_input)

        # Synchronize CUDA operations before measuring the end time
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()
        durations.append(end - start)

    return sum(durations) / num_iters

def benchmark_compile(
    model: Union[torch.nn.Module, Callable],
    sample_input: Union[torch.Tensor, Any],
    num_iters: int = 5,
    backend: Optional[str] = None,
    mode: Optional[str] = "default",
    optimizer: torch.optim.Optimizer = None,
    loss_fn : Union[torch.nn.Module, Callable] = None,
):
    """
    Use this utility to benchmark torch.compile
    """
    if backend:
        try:
            opt_model = torch.compile(model, backend=backend, mode=mode)

            # Compilation only happens after the first inference
            compilation_time = bench_loop(opt_model, sample_input, 1, optimizer, loss_fn)

        except BaseException as e:
            print(e)
            print(f"Failed to compile {backend} with mode {mode}")
            return None, None
    else:
        opt_model = model
        compilation_time = None

    # Benchmark
    running_time = bench_loop(opt_model, sample_input, num_iters, optimizer, loss_fn)

    return compilation_time, running_time


def bench_all(
    model : Union[torch.nn.Module, Callable],
    sample_input: Union[torch.Tensor, Any],
    num_iters : int = 5,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_fn : Union[torch.nn.Module, Callable] = None,
):
    """
    This is a simple utility that can be used to benchmark torch.compile
    In particular it ensures that your GPU is setup to use tensor cores if it supports its
    It also tries out all the main backends and prints a table of results so you can easily compare them all
    Many of the backendds have their own optional dependencies so please pip install them seperately

    The tables will look like the below, one for inference and one for training
    If you'd like to leverage this utility for training make sure to pass in a torch.optim.Optimizer

    | Train/Inference   | Backend         | Mode            | Compilation Time      |   Average Running Time | Speedup            |
    |-------------------|-----------------|-----------------|-----------------------|------------------------|--------------------|
    | Inference         | Eager           | -               | -                     |            0.000146246 | -                  |
    | Inference         | aot_ts_nvfuser  | -               | 0.014810323715209961  |            0.000166035 | 0.8808156232050546 |
    | Inference         | cudagraphs      | -               | 0.013611078262329102  |            0.000135565 | 1.0787900105522334 |
    | Inference         | inductor        | default         | 0.0026526451110839844 |            0.000137377 | 1.06456091634849   |
    | Inference         | inductor        | reduce-overhead | 0.002624988555908203  |            0.000135946 | 1.07576289021396   |
    | Inference         | inductor        | max-autotune    | 0.0026438236236572266 |            0.000134993 | 1.0833627693394559 |
    | Inference         | inductor        |                 | 0.0025985240936279297 |            0.00013566  | 1.0780316344463972 |
    | Inference         | ipex            | -               | 0.0026051998138427734 |            0.000132751 | 1.1016522988505748 |
    | Inference         | nvprims_nvfuser | -               | 0.002631664276123047  |            0.000132751 | 1.1016522988505748 |
    | Inference         | onnxrt          | -               | 0.0026395320892333984 |            0.000138378 | 1.0568573397656789 |
    | Inference         | tvm             | -               | 0.0026519298553466797 |            0.000141668 | 1.032312352743184  |

    The important warnings are
    Your GPU supports tensor cores
    we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`

    If a compilation fails for any reason including the dependency not being included
    then we will print Failed to compile {backend} with mode {mode}


    """
    field_names = ["Train/Inference", "Backend", "Mode", "Compilation Time", "Average Running Time", "Speedup"]
    table = []


    eager_time = None
    torch._dynamo.reset()
    _, eager_time = benchmark_compile(model, sample_input, num_iters, None, None, optimizer)
    table.append(
        [("Training" if optimizer else "Inference"), "Eager", "-", "-", eager_time, "-"])

    for backend in torch._dynamo.list_backends():

        if backend == "inductor":
            mode_options = cast(List[Optional[str]], list(torch._inductor.list_mode_options().keys())) + [None]
            for mode in mode_options:
                torch._dynamo.reset()
                try:
                    if torch.cuda.is_available():
                        _enable_tensor_cores()
                    compilation_time, running_time = benchmark_compile(
                        model, sample_input, num_iters, backend, mode, optimizer, loss_fn)
                finally:
                    if torch.cuda.is_available():
                        _disable_tensor_cores()
                        if running_time is not None:
                            speedup = eager_time / running_time
                            table.append([
                                ("Training" if optimizer else "Inference"),
                                backend, mode, compilation_time or "-", running_time, speedup or "-"
                            ])

        else:
            torch._dynamo.reset()
            compilation_time, running_time = benchmark_compile(model, sample_input, num_iters, backend, None, optimizer, loss_fn)

            if running_time is not None:
                speedup = eager_time / running_time
                table.append([
                    ("Training" if optimizer else "Inference"),
                    backend, "-", compilation_time or "-", running_time, speedup or "-"
                ])


    return tabulate(table, headers=field_names, tablefmt="github")
