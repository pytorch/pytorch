```{eval-rst}
.. currentmodule:: torch.profiler
```

# torch.profiler

## Overview

```{eval-rst}
.. automodule:: torch.profiler
```

Performance profiling is the process of analyzing a program's execution to
understand its behavior and identify bottlenecks. It involves measuring various
aspects of the program, such as the time taken by different functions, memory
usage, and hardware utilization. The goal is to pinpoint inefficiencies and
areas for optimization, ultimately leading to a faster, more efficient model.

In the context of deep learning, profiling helps answer questions like:

* Which operations in my model are the most time-consuming?
* Is my GPU being fully utilized, or is it sitting idle waiting for data?
* How much memory is my model using, and are there any unexpected spikes?
* Is the data loading pipeline a bottleneck?

### Common Performance Bottlenecks

Deep learning models often exhibit a set of common performance issues.
Understanding these can help you focus your profiling efforts.

| Bottleneck | Description | Potential Solutions |
| --- | --- | --- |
| **Data Loading** | The data pipeline is too slow, causing the GPU to wait for data ("input-bound"). | Increase `DataLoader` workers, use pinned memory, or optimize augmentations. |
| **GPU Idle Time** | The GPU is not being kept busy with computation. | Overlap data transfers with computation, use larger batch sizes, or fuse operations. |
| **Inefficient Operators** | Certain operations are slow or not using the most efficient implementation. | Replace with efficient alternatives, use mixed-precision, or leverage ``torch.compile``. |
| **Memory Usage** | Too much memory consumed, leading to OOM errors or limited batch size. | Use gradient checkpointing, reduce model size, or use memory-efficient optimizers. |
| **Communication Overhead** | In distributed training, gradient synchronization time is too high. | Use gradient compression, overlap communication with computation, or tune NCCL. |

```{seealso}
For a deep dive into the profiler's internal architecture (Kineto, CUPTI, and
the data flow pipeline), see {ref}`profiler-architecture`.

For memory profiling workflows using memory snapshots, see
{ref}`memory-profiling`.
```

## Getting Started

This section walks through a practical first profiling session using a ResNet18
model.

### Setting Up

```python
import torch
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18().to(device)
inputs = torch.randn(5, 3, 224, 224).to(device)
```

### Using the Profile Context Manager

The ``torch.profiler.profile`` context manager is the primary entry point. It
accepts arguments to configure the profiling session:

```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    with torch.profiler.record_function("model_inference"):
        model(inputs)
```

* ``activities`` specifies which device events to record (CPU, CUDA, or both).
* ``record_shapes=True`` captures input shapes for each operator.
* ``profile_memory=True`` tracks tensor memory allocation and deallocation.
* ``with_stack=True`` records the Python source location for each operator.
* ``record_function`` adds a custom label to a code block, making it easy to
  find in profiling output.

### Analyzing Results

The ``key_averages()`` method aggregates results by operator name:

```python
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
```

This prints a summary table with the following columns:

| Column | Description |
| --- | --- |
| **Name** | The operator name or ``record_function`` label |
| **Self CPU** | Time in the operator itself, excluding child operators |
| **CPU total** | Total time including child operators |
| **CPU time avg** | Average time per call |
| **# of Calls** | Number of times the operator was called |
| **Self CUDA** | GPU time in the operator itself (when CUDA profiling is enabled) |
| **CUDA total** | Total GPU time including child operators |

### Exporting a Trace

For a detailed timeline visualization, export the trace to a JSON file:

```python
prof.export_chrome_trace("trace.json")
```

This creates a Chrome-compatible trace file that can be loaded into Perfetto or
``chrome://tracing``.

## Visualizing Traces with Perfetto

The recommended tool for visualizing PyTorch Profiler traces is
`Perfetto <https://ui.perfetto.dev>`_, an open-source trace analysis tool.

```{note}
The TensorBoard integration with the PyTorch profiler
(``torch.profiler.tensorboard_trace_handler``) is deprecated. Use Perfetto
or ``chrome://tracing`` to view ``trace.json`` files instead.
```

### Loading a Trace

After exporting a trace with ``prof.export_chrome_trace("trace.json")``, open
`ui.perfetto.dev <https://ui.perfetto.dev>`_ in your browser and either click
"Open trace file" or drag and drop the ``trace.json`` file onto the page.

### Navigating the UI

| Action | Control |
| --- | --- |
| Zoom in/out | ``W`` / ``S`` keys or scroll wheel |
| Pan left/right | Click and drag the timeline |
| Select an event | Click on it; details appear in the bottom pane |

### Interpreting Tracks

The Perfetto timeline is organized into tracks, each representing a different
source of events:

| Track | Description |
| --- | --- |
| **Processes** | The processes running during the profiling session |
| **GPU** | GPU activity, with sub-tracks for each CUDA stream |
| **CPU** | Activity on each CPU core |
| **PyTorch Profiler** | Metadata including ``record_function`` labels |

### Identifying Bottlenecks

The most common bottleneck visible in a Perfetto trace is **GPU idle time** —
large gaps in the GPU track where no kernels are running. By zooming into these
gaps and examining the CPU track, you can often see that the CPU is busy with
data loading or preprocessing, starving the GPU of work.

## Advanced Profiling Techniques

### Scheduling Profiling Runs

Profiling every iteration adds overhead. The ``torch.profiler.schedule``
function controls which iterations are profiled:

```python
my_schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)

with torch.profiler.profile(schedule=my_schedule) as prof:
    for step in range(10):
        # ... training code ...
        prof.step()
```

| Parameter | Description |
| --- | --- |
| ``wait`` | Steps to skip at the start of each cycle |
| ``warmup`` | Steps to run the profiler but discard results (reduces initial overhead) |
| ``active`` | Steps to actively record |
| ``repeat`` | Times to repeat the cycle (0 = indefinite) |

The ``prof.step()`` call is essential — it signals the profiler to advance to
the next state in the schedule.

### Customizing Traces with ``record_function``

``record_function`` adds human-readable labels to any block of code. These
labels appear in both the ``key_averages()`` table and the Perfetto trace:

```python
with torch.profiler.profile(...) as prof:
    with torch.profiler.record_function("data_loading"):
        data = next(iter(data_loader))

    with torch.profiler.record_function("model_forward"):
        output = model(data)

    with torch.profiler.record_function("loss_and_backward"):
        loss = criterion(output, target)
        loss.backward()
```

### Stack Trace Collection

Setting ``with_stack=True`` records the Python call stack for each operator.
This allows you to trace a slow operator back to the exact line of source code
that triggered it:

```python
with torch.profiler.profile(with_stack=True) as prof:
    model(inputs)
```

### Estimating Operator FLOPS

The profiler can estimate floating-point operations for operators like matrix
multiplications and convolutions:

```python
with torch.profiler.profile(with_flops=True) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="flops", row_limit=10))
```

## API Quick Reference

| API | Description |
| --- | --- |
| ``torch.profiler.profile(...)`` | Main context manager for profiling sessions |
| ``torch.profiler.ProfilerActivity.CPU`` | Activity type for CPU events |
| ``torch.profiler.ProfilerActivity.CUDA`` | Activity type for CUDA GPU events |
| ``torch.profiler.ProfilerActivity.XPU`` | Activity type for Intel XPU events |
| ``torch.profiler.schedule(wait, warmup, active, repeat)`` | Returns a callable schedule for controlling profiling steps |
| ``torch.profiler.record_function(name)`` | Context manager to label a code block in the trace |
| ``prof.key_averages()`` | Returns aggregated profiler events grouped by operator name |
| ``prof.export_chrome_trace(path)`` | Exports the trace to a Chrome/Perfetto-compatible JSON file |
| ``prof.export_stacks(path)`` | Exports stack traces for flame graph visualization |
| ``prof.step()`` | Advances the profiler to the next step in the schedule |

## API Reference
```{eval-rst}

.. autoclass:: torch.profiler.profile
  :members:
  :inherited-members:

.. autoclass:: torch.profiler.ProfilerAction
  :members:

.. autoclass:: torch.profiler.ProfilerActivity
  :members:

.. autofunction:: torch.profiler.schedule

.. autofunction:: torch.profiler.tensorboard_trace_handler

.. autofunction:: torch.profiler.supported_activities
```

## Intel Instrumentation and Tracing Technology APIs

```{eval-rst}
.. autofunction:: torch.profiler.itt.is_available

.. autofunction:: torch.profiler.itt.mark

.. autofunction:: torch.profiler.itt.range_push

.. autofunction:: torch.profiler.itt.range_pop

.. autofunction:: torch.profiler.itt.range
```

<!-- This module needs to be documented. Adding here in the meantime
for tracking purposes -->
```{eval-rst}
.. py:module:: torch.profiler.itt
.. py:module:: torch.profiler.profiler
.. py:module:: torch.profiler.python_tracer
```
