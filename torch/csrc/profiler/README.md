# Profiler Overview

This README describes the details of how the profiler is implemented.

The profiler instruments PyTorch to collect information about the model's execution. Its main features are:
* Instrumenting op calls on the CPU side
* Interfacing with [Kineto](https://github.com/pytorch/kineto/) to collect information from the GPU (or other accelerators)
* Collecting python stack traces
* Exporting this information, e.g. in a chrome trace, or to be processed by downstream tools like [HTA](https://github.com/facebookresearch/HolisticTraceAnalysis)

## Table of Contents

- [Codebase Structure](#codebase-structure)
- [`RecordFunction`](#recordfunction)
- [Autograd Integration](#autograd-integration)
- [Collection and Post-Processing](#collection-and-post-processing)
- [Kineto Integration](#kineto-integration)
- [Python Tracing](#python-tracing)

## Codebase Structure ##

TODO

## `RecordFunction` ##

[/aten/src/ATen/record_function.h](/aten/src/ATen/record_function.h)

`RecordFunction` is used by the profiler to instrument CPU-side events.

`RecordFunction` is a general method of instrumenting function calls in PyTorch. It can be used for other general applications, e.g. see [Features for Large-Scale Deployments](https://pytorch.org/docs/stable/notes/large_scale_deployments.html). In PyTorch, it is already included at some important locations; notably, in the [dispatcher](https://github.com/pytorch/pytorch/blob/247c603da9b780534e25fb1d90b6e5a528b625b1/aten/src/ATen/core/dispatch/Dispatcher.h#L650), surrounding every op.

Users (or PyTorch itself) can register callbacks that will be executed whenever a `RecordFunction` guard is encountered. The profiler uses this mechanism to record the start and end times for each op call, as well as user-provided `RecordFunction` annotations. The `RecordFunction` machinery is designed to have relatively low overhead, especially when there are no callbacks registered. Nevertheless, there can still be some overhead.

There is also a python binding for `RecordFunction` in python (`with torch.profiler.record_function`); this is often used by users to annotate events corresponding to module-level events.

## Autograd Integration ##

The autograd engine is responsible for automatically computing gradients.

The profiler records two pieces of information from the autograd engine:
* [Sequence number](/aten/src/ATen/SequenceNumber.h): this is a unique-per-thread index assigned to each op call(\*) in the forward pass. When a backward op is triggered, it is also assigned a sequence number matching the sequence number of the forward op that caused that backward op to be executed. Using this information, the profiler is able to match forward and backward ops; in chrome traces, this feature can be enabled with the "fwd_bwd" flow events
* [Forward thread id](https://github.com/pytorch/pytorch/blob/2e3fce54506ba82eee2c890410bf7a1405a64ec6/aten/src/ATen/record_function.h#L357): Autograd can be used in multi-threaded environments. The forward thread ID indicates the ID of the thread on which the forward op was executed on. This information is needed because the sequence number, mentioned above, is only unique within a thread; the forward thread ID is used for differentiating different ops with the same sequence number.

(\*) Note that only op invocations whose inputs require gradients are assigned a sequence number

## Collection and Post-Processing ##

TODO

## Kineto Integration ##

TODO

## Python Tracing ##

TODO
