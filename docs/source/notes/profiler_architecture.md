(profiler-architecture)=
# PyTorch Profiler Architecture: Kineto and CUPTI

This note describes the internal architecture of the PyTorch Profiler and
the components that work together to collect and process performance data.

## Architecture Overview

The profiler's architecture has three main layers:

1. **The User-Facing API (``torch.profiler``)**: The context managers and
   functions for starting, stopping, and configuring profiling sessions.
2. **Kineto**: The core C++ library responsible for trace collection. It
   orchestrates performance data gathering from the CPU and hardware
   accelerators. See the [Kineto repository](https://github.com/pytorch/kineto)
   for more details.
3. **Hardware-Specific Backends**: Low-level interfaces that Kineto uses to
   communicate with specific hardware, such as NVIDIA's CUPTI for CUDA GPUs.

## The ``torch.profiler`` API

The ``torch.profiler`` module is the entry point for all profiling activities.
The most important component is the ``torch.profiler.profile`` context manager,
which wraps the code to be analyzed.

When you enter the ``profile`` context, the profiler activates Kineto to start
collecting data. The ``activities`` argument specifies which event types to
record (CPU, CUDA, etc.). You can use ``record_function`` to add custom labels,
making profiling output easier to interpret.

## Kineto: The Trace Collection Engine

Kineto is the C++ library that handles the heavy lifting of trace collection.
When active, Kineto receives notifications about events in the PyTorch runtime
— operator dispatches, kernel launches, memory allocations — and records them
with timestamps and metadata into a trace.

Kineto is designed to be extensible and integrates with various hardware
backends. This allows the PyTorch Profiler to support a wide range of devices
beyond NVIDIA GPUs.

## Hardware Backends: CUPTI and Beyond

To collect data from a hardware accelerator, Kineto relies on a backend that
communicates with that hardware's performance monitoring tools. For NVIDIA GPUs,
this backend is the **CUDA Profiling Tools Interface (CUPTI)**, a library
provided by NVIDIA for instrumenting and profiling CUDA applications. Kineto
uses CUPTI to subscribe to GPU events such as kernel launches and memory copies.

This modular architecture keeps PyTorch device-agnostic: Python brokers the
session, the profiler translates requests into backend runtime calls, and the
runtime interacts with the accelerator. See
{doc}`/accelerator/profiler` for details on integrating
profilers for custom accelerator backends.

## Data Flow

The following table summarizes the data flow during a profiling session:

| Step | Component | Action |
| --- | --- | --- |
| 1 | User Code | Enters the ``torch.profiler.profile`` context manager |
| 2 | ``torch.profiler`` | Activates Kineto and configures the session |
| 3 | PyTorch Runtime | Dispatches operators; Kineto records CPU-side events |
| 4 | CUDA Runtime | Launches GPU kernels |
| 5 | CUPTI | Detects kernel launches and notifies Kineto |
| 6 | Kineto | Records GPU-side events in the trace |
| 7 | ``torch.profiler`` | Finalizes the trace when the context exits |
| 8 | User Code | Uses the ``prof`` object to analyze or export data |

```{seealso}
- {doc}`/profiler` — ``torch.profiler`` API reference and usage guide
- [Kineto GitHub repository](https://github.com/pytorch/kineto) — source code
  for the trace collection engine
- {doc}`/accelerator/profiler` — integrating the profiler with custom
  accelerator backends
```
