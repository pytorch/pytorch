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
- [Torch Operation Collection](#torch-operation-collection)
- [Allocation Event Collection](#allocation-event-collection)
- [Kineto Integration](#kineto-integration)
- [Python Tracing](#python-tracing)
- [Clock Alignment](#clock-alignment)

## Codebase Structure ##

This section highlights directories an files that are significant to the profiler. Lesser relevant files, directories, and modules are omitted.
```
torch/
│
├── profiler/                # Main package containing the core frontend logic
│   ├── __init__.py          # Initialization file for profiler package
│   ├── profiler.py          # Main profiler frontend class
│   └── _utils.py            # FunctionEvent utils
│
├── autograd/               # Autograd package
│   ├── __init__.py          # Initialization file for autograd package
│   ├── profiler.py          # Main profiler backend class
│   └── profiler_utils.py    # FunctionEvent utils
│
├── csrc/                   # C and C++ source code
│   └── profiler/            # Profiler C++ source code
│       ├── collection.cpp                 # Main collection logic
│       ├── collection.h                   # Collection definitions
│       ├── kineto_client_interface.cpp   # Interface to call Profiler from kineto (on-demand only)
│       ├── kineto_client_interface.h     # Client interface definitions
│       ├── kineto_shim.cpp                # Shim to call kineto from profiler
│       ├── kineto_shim.h                  # Shim definitions
│       ├── util.cpp                       # utils for handling args in profiler events
│       ├── util.h                         # util definitions
│       └── README.md                      # This file
│   └── autograd/            # Autograd C++ source code
│       ├── profiler_python.cpp          # Main python stack collection logic
│       ├── profiler_python.h            # Python stack collection definitions
│       ├── profiler_kineto.cpp          # Profiler backend logic for starting collection/kineto
│       └── profiler_kineto.h            # Profiler backend definitions for starting collection/kineto
│   └── ATen/                # ATen C++ source code
│       ├── record_function.cpp          # RecordFunction collection logic
│       └── record_function.h            # RecordFunction definitions
└── LICENSE                  # License information
```
## `RecordFunction` ##

[aten/src/ATen/record_function.h](../../../aten/src/ATen/record_function.h)

`RecordFunction` is used by the profiler to instrument CPU-side events.

`RecordFunction` is a general method of instrumenting function calls in PyTorch. It can be used for other general applications, e.g. see [Features for Large-Scale Deployments](https://pytorch.org/docs/stable/notes/large_scale_deployments.html). In PyTorch, it is already included at some important locations; notably, in the [dispatcher](https://github.com/pytorch/pytorch/blob/247c603da9b780534e25fb1d90b6e5a528b625b1/aten/src/ATen/core/dispatch/Dispatcher.h#L650), surrounding every op.

Users (or PyTorch itself) can register callbacks that will be executed whenever a `RecordFunction` guard is encountered. The profiler uses this mechanism to record the start and end times for each op call, as well as user-provided `RecordFunction` annotations. The `RecordFunction` machinery is designed to have relatively low overhead, especially when there are no callbacks registered. Nevertheless, there can still be some overhead.

There is also a python binding for `RecordFunction` in python (`with torch.profiler.record_function`); this is often used by users to annotate events corresponding to module-level events.

## Autograd Integration ##

The autograd engine is responsible for automatically computing gradients.

The profiler records two pieces of information from the autograd engine:
* [Sequence number](../../../aten/src/ATen/SequenceNumber.h): this is a unique-per-thread index assigned to each op call(\*) in the forward pass. When a backward op is triggered, it is also assigned a sequence number matching the sequence number of the forward op that caused that backward op to be executed. Using this information, the profiler is able to match forward and backward ops; in chrome traces, this feature can be enabled with the "fwd_bwd" flow events
* [Forward thread id](https://github.com/pytorch/pytorch/blob/2e3fce54506ba82eee2c890410bf7a1405a64ec6/aten/src/ATen/record_function.h#L357): Autograd can be used in multi-threaded environments. The forward thread ID indicates the ID of the thread on which the forward op was executed on. This information is needed because the sequence number, mentioned above, is only unique within a thread; the forward thread ID is used for differentiating different ops with the same sequence number.

(\*) Note that only op invocations whose inputs require gradients are assigned a sequence number

## Torch Operation Collection ##
This section describes the general flow for collecting torch operations during auto-trace (in-process, synchronous tracing). For details on on-demand tracing (out-of-process, asynchronous), please refer to the Libkineto README.

When a trace begins, the autograd/profiler backend calls into `profiler_kineto.cpp` to prepare, start, or stop collection. At the start of tracing, the `onFunctionEnter` and `onFunctionExit` callbacks defined in `profiler_kineto.cpp` are registered.

Callback registration can be either global or local, depending on the `ExperimentalConfig` used:
- **Global:** The callback is registered to all threads throughout execution.
- **Local:** The callback is registered only to threads present *at the start* of tracing.
Within `onFunctionEnter`, the profiler creates a `ThreadLocalSubqueue` instance for each thread, ensuring that each CPU operation is associated with the thread on which it was executed. When a torch operation is entered, the profiler calls `begin_op` (defined in `collection.cpp`) to record the necessary information. The `begin_op` routine is intentionally lightweight, as it is on the "hot path" during profiling. Excessive overhead here would distort the profile and reduce its usefulness. Therefore, only minimal information is collected during the callback; most logic occurs during post-processing.

## Allocation Event Collection ##

Unlike torch operations, which have a start and stop, allocation events are represented as `cpu_instant_event` (zero duration). As a result, `RecordFunction` is bypassed for these events. Instead, `emplace_allocation_event` is called directly to enqueue the event into the appropriate `ThreadLocalSubqueue`.

## Kineto Integration ##

Kineto serves as an abstraction layer for collecting events across multiple architectures. It interacts with libraries such as CUPTI to receive GPU and accelerator events, which are then forwarded to the frontend profiler. Kineto requires time to "prepare" (also referred to as "warmup") these third-party modules to avoid distorting the profile with initialization routines. While this could theoretically be done at job startup, keeping a heavy library like CUPTI running unnecessarily introduces significant overhead.
As previously mentioned, `profiler_kineto.cpp` is used in the backend to invoke the appropriate profiler stage. It also calls into `kineto_shim.cpp`, which triggers the corresponding routines in Kineto. Once a trace is complete, all events collected by Kineto are forwarded to the profiler for two main reasons:
1. To coalesce all data and complete any post-processing between profiler and Kineto events.
2. To forward these events to the Python frontend as `FunctionEvents`.
The final step in integration is file export. After all events have been collected and post-processed, they can be exported to a JSON file for visualization in Perfetto or Chrome Tracer. This is done by calling Kineto's `ActivityTraceInterface::save`, which writes all event information to disk.

## Python Tracing ##

When `with_stack=True` is set in the profiler, the Python stack tracer is generated using the `make` function defined in `PythonTracerBase`. The implementation resides in `profiler_python.cpp`.
To profile the stack, `PyEval_SetProfile` is used to trace and handle various execution events within a Python program. This enables comprehensive profiling by monitoring and responding to specific cases:
- **Python Function Calls (`PyTrace_CALL`):** The `recordPyCall` method logs each Python function call, capturing essential details for later analysis.
- **C Function Calls (`PyTrace_C_CALL`):** The `recordCCall` method documents calls to C functions, including relevant arguments, providing a complete view of the program's execution flow.
- **Python Function Returns (`PyTrace_RETURN`):** Exit times of Python functions are recorded, enabling precise measurement of function execution durations.
- **C Function Returns and Exceptions (`PyTrace_C_RETURN` and `PyTrace_C_EXCEPTION`):** Exit times for C functions are tracked, whether they conclude normally or due to an exception, ensuring all execution paths are accounted for.
This setup allows for detailed and accurate data collection on both Python and C function executions, facilitating thorough post-processing and analysis. After profiling, the accumulated event stacks are processed to match entrances and exits, constructing complete events for further analysis by the profiler.
**Note:** For Python 3.12.0–3.12.4, a bug in CPython requires the use of `sys.monitoring` as a workaround.

## Clock Alignment ##

Depending on the system environment, the profiler will use the most efficient clock when creating a timestamp. The default for most Linux systems is TSC, which records time in the form of CPU cycles. To convert from this time to the unix time in nanoseconds, we create a clock converter. If Kineto is included in the profiler, this converter will also be passed into Kineto as well to ensure alignment.
