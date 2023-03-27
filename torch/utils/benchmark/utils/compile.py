import torch

__all__ = ["bench_all", "benchmark_compile"]

import torch
import torch._dynamo as torchdynamo
import time
import warnings
from typing import Optional, List

# Added this otherwise all the warnings from each backend will be printed
# Not sure if this warning will propgate if user is not using this config
warnings.filterwarnings("ignore")



class SimpleTable:
    """
    This is a simple table class that can be used to print a table of data.
    The primary reason it's here is to avoid taking an external dependency.
    If it's useful, it can be moved to a more general location.
    """
    def __init__(self, field_names : List[str]):
        self.field_names = field_names
        self.rows = []

    def add_row(self, row):
        self.rows.append(row)

    def __str__(self):
        if not self.field_names or not self.rows:
            return ""

        formatted_rows = []
        for row in self.rows:
            formatted_row = []
            for cell in row:
                if isinstance(cell, float):
                    formatted_cell = f"{cell:10.4f}"
                else:
                    formatted_cell = str(cell)
                formatted_row.append(formatted_cell)
            formatted_rows.append(formatted_row)

        # Find the maximum width for each column
        column_widths = [max([len(cell) for cell in column] + [len(header)]) for header, column in zip(self.field_names, zip(*formatted_rows))]

        header = " | ".join([f"{str(field_name).ljust(width)}" for field_name, width in zip(self.field_names, column_widths)])
        separator = "-+-".join(["-" * width for width in column_widths])

        rows = []
        for row in formatted_rows:
            rows.append(" | ".join([f"{cell.ljust(width)}" for cell, width in zip(row, column_widths)]))

        return f"{header}\n{separator}\n" + "\n".join(rows)

def bench_loop(model : torch.nn.Module, sample_input : torch.Tensor, num_iters : int, is_training : bool =False, optimizer : torch.optim.Optimizer = None):
    """
    This is a simple loop that can be used to benchmark a model for either training or inference
    It takes care of taking several measurements and averaging them
    It also takes care of calling `cuda.synchronize()` if the model is on GPU
    """
    durations = []
    for _ in range(num_iters):
        start = time.time()
        
        if is_training and optimizer:
            optimizer.zero_grad()
            output = model(sample_input)
            loss = output.sum()
            loss.backward()
            optimizer.step()
        else:
            model(sample_input)
        
        end = time.time()

        if sample_input.get_device() >= 0:
            torch.cuda.synchronize()

        durations.append(end - start)
    
    return sum(durations) / num_iters

def benchmark_compile(model: torch.nn.Module, sample_input: torch.Tensor, num_iters: int = 5, backend: Optional[str] = None, mode="default", is_training=False, optimizer=None):
    """
    Use this utility to benchmark torch.compile
    """
    if backend:
        try:
            opt_model = torch.compile(model, backend=backend, mode=mode)
            
            # Compilation only happens after the first inference
            compilation_time = bench_loop(opt_model, sample_input, 1, is_training, optimizer)

        except:
            print(f"Failed to compile {backend} with mode {mode}")
            return None, None
    else:
        opt_model = model
        compilation_time = None

    # Benchmark
    running_time = bench_loop(opt_model, sample_input, num_iters, is_training, optimizer)
    
    return compilation_time, running_time

    
def bench_all(model : torch.nn.Module, sample_input : torch.Tensor, num_iters : int=2, is_training : bool =False, optimizer: Optional[torch.optim.Optimizer]=None):
    """
    This is a simple utility that can be used to benchmark torch.compile
    In particular it ensures that your GPU is setup to use tensor cores if it supports its
    It also tries out all the main backends and prints a table of results so you can easily compare them all
    Many of the backendds have their own optional dependencies so please pip install them seperately
    
    The tables will look like
    Train/Inference | Backend        | Mode | Compilation Time | Average Running Time | Speedup   
    ----------------+----------------+------+------------------+----------------------+-----------
    Training        | Eager          | -    | -                |     0.0005           | -         
    Training        | aot_ts_nvfuser | -    |     0.5023       |     0.0006           |    -0.3070
    Training        | cudagraphs     | -    |     0.0242       |     0.0259           |   -51.4658

    The important warnings are
    Your model is loaded on GPU
    Your GPU supports tensor cores, we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`

    If a compilation fails for any reason including the dependency not being included then we will print Failed to compile {backend} with mode {mode}


    """
    if next(model.parameters()).is_cuda:
        print("Your model is loaded on GPU")
        if torch.backends.cuda.matmul.allow_tf32 is False and torch.cuda.get_device_capability() >= (8, 0):
            print("Your GPU supports tensor cores, we will enable it automatically by setting `torch.set_float32_matmul_precision('high')`")
            torch.set_float32_matmul_precision("high")


    table = SimpleTable(field_names = ["Train/Inference", "Backend", "Mode", "Compilation Time", "Average Running Time", "Speedup"])


    eager_time = None
    torchdynamo.reset()
    _, eager_time = benchmark_compile(model, sample_input, num_iters, None, None, is_training, optimizer)
    table.add_row([("Training" if is_training else "Inference"), "Eager", "-", "-", eager_time, "-"])

    for backend in torchdynamo.list_backends():
        if backend == "ipex":  # ipex has an annoying import error it prints
            continue

        if backend == "inductor":
            for mode in list(torch._inductor.list_mode_options().keys()) + [None]:
                if mode == "default":
                    continue
                torchdynamo.reset()
                compilation_time, running_time = benchmark_compile(model, sample_input, num_iters, backend, mode, is_training, optimizer)
                if running_time is not None:
                    speedup = (eager_time - running_time) / eager_time if eager_time else None
                    table.add_row([("Training" if is_training else "Inference"), backend, mode or "-", compilation_time or "-", running_time, speedup or "-"])
        else:
            torchdynamo.reset()
            compilation_time, running_time = benchmark_compile(model, sample_input, num_iters, backend, None, is_training, optimizer)
            if running_time is not None:
                speedup = (eager_time - running_time) / eager_time if eager_time else None
                table.add_row([("Training" if is_training else "Inference"), backend, "-", compilation_time or "-", running_time, speedup or "-"])

    return table

if __name__ == "__main__":
    torchdynamo.reset()
    
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super(ToyModel, self).__init__()
            self.weight = torch.nn.Parameter(torch.Tensor(2, 2))

        def forward(self, x):
            return x * self.weight

    model = ToyModel().cuda()

    print("===== Inference =====")
    inference_table = bench_all(model, torch.ones(1024, 2, 2).cuda(), 5)
    assert(inference_table.rows[0][0] == "Inference")
    assert(inference_table.rows[0][1] == "Eager")
    assert(inference_table.rows[0][2] == "-")
    print(inference_table)
    print("\n===== Training =====")
    training_table = bench_all(model, torch.ones(1024, 2, 2).cuda(), 5, is_training=True, optimizer=torch.optim.SGD(model.parameters(), lr=0.01))
    assert(training_table.rows[0][0] == "Training")
    assert(training_table.rows[0][1] == "Eager")
    assert(training_table.rows[0][2] == "-")
    print(training_table)