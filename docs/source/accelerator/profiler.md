# Profiler Integration

## Background

PyTorch ships a device-agnostic profiler that instruments CPU-side operator dispatch, coordinates with accelerator collectors, captures Python stacks, and exports aggregated statistics or Chrome/Perfetto traces. For core architecture, see [`torch/csrc/profiler/README.md`][PyTorch Profiler README].

There are two primary integration paths for accelerators:

1. Legacy autograd profiler:
    - Can attach backend-specific hooks via `ProfilerStubs` to record device events and compute elapsed times.
    - Works without Kineto; suitable for PrivateUse1 backends that want a minimal, self-contained path.

2. Kineto `IActivityProfiler` plugin:
    - Registers a full activity profiler with Kineto via `REGISTER_PRIVATEUSE1_PROFILER`.
    - Wires Kineto sessions and correlation-ID plumbing; vendors extend this to emit kernel events, flow links, and Chrome/Perfetto trace compatibility.
    - Requires Kineto at backend build time (`kineto_LIBRARY` from `find_package(Torch)`, guarded by `USE_KINETO`).

| Path | Python API | Profiler State | What it provides |
| ---- | ---------- | -------------- | ---------------- |
| Legacy (1) | `torch.autograd.profiler.profile(use_device="openreg")` (default `use_kineto=False`) | `KINETO_PRIVATEUSE1_FALLBACK` | Operator-level timing via `ProfilerStubs` device events |
| Kineto plugin (2) | `profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1])` | `KINETO_PRIVATEUSE1` | Kineto session + correlation-ID plumbing; vendors add kernel events and flow links |

Both paths can coexist when the backend extension is built with Kineto available (`kineto_LIBRARY` from `find_package(Torch)`). The legacy stubs path always works; the Kineto plugin path requires `USE_KINETO` at backend build time. PyTorch core already exposes `REGISTER_PRIVATEUSE1_PROFILER`; vendors implement and register their `IActivityProfiler` in the backend extension.

## Design

### Architecture overview

| Layer                | Responsibility                                                                                                                                      | Source                          |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| Python control plane | Owns profiler lifecycle (`prepare → start → stop → step`) and exposes user APIs such as `torch.autograd.profiler.profile`.                          | `torch/autograd/profiler.py`    |
| Profiler stubs       | Implements `torch::profiler::impl::ProfilerStubs` so the profiler can record device events, synchronize, iterate devices, and compute elapsed time. | `torch/csrc/profiler/stubs/`    |
| Device runtime       | Provides streams, events, and device guards used by the stubs; implementation is backend-specific.                                                  | Backend extension (vendor code) |

This layering keeps PyTorch device-agnostic: Python brokers the session, `ProfilerStubs` translate profiler requests into backend runtime calls, and the runtime interacts with the accelerator.

### Key contracts

* **Record hooks**: `record()` must capture (optional) device index, allocate a backend event, optionally stash a CPU timestamp, and enqueue the event on the active stream.
* **Elapsed time**: `elapsed()` is responsible for synchronizing individual events and returning durations in microseconds.
* **Synchronization hooks**: `synchronize()` and `onEachDevice()` guarantee phase transitions (e.g., warmup → active) are aligned across devices.
* **Annotations**: `mark`, `rangePush`, and `rangePop` can be implemented to enrich traces; otherwise they may be left as no-ops.

## Implementation (Legacy way)

Here we use OpenReg (Open Registration) to illustrate the minimal set of hooks a `PrivateUse1` accelerator needs to expose so the profiler can attribute ATen ops, `record_function` ranges, and user code to device activity. OpenReg keeps upstream code untouched by translating profiler requests into its runtime calls, mirroring what a production accelerator would implement inside an out-of-tree extension.

OpenReg supports both paths: the legacy autograd profiler (`use_kineto=False`, the default) for operator-level timing via stubs, and the modern `torch.profiler.profile` API (`use_kineto=True`) for the Kineto plugin path described below.

### Profiler stubs (C++)

[`torch::profiler::impl::OpenRegMethods`][openreg-stubs] inherits from `ProfilerStubs` and wires the hooks described above:

| Method                         | Purpose                                                                                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `record`                       | Grabs the current `OpenRegStream`, creates an `orEvent`, captures an optional CPU timestamp via `c10::getTime()`, and records the event on the stream. |
| `elapsed`                      | Synchronizes both events, calls `orEventElapsedTime`, and converts milliseconds to microseconds for the profiler.                                      |
| `onEachDevice`                 | Uses `c10::DeviceGuard(DeviceType::PrivateUse1)` to iterate over `torch.openreg.device_count()` so schedulers can run per-device setup or teardown.    |
| `synchronize`                  | Calls `orDeviceSynchronize()` to align device work with CPU scheduling phases.                                                                         |
| `enabled` and annotation shims | Report availability and provide placeholder implementations for mark/push/pop.                                                                         |

The constructor registers the methods once via `registerPrivateUse1Methods(&methods);`, making them discoverable whenever the profiler is enabled with `use_device="openreg"`.

### Python control plane

On the Python side, no new entrypoint is required—developers use the standard autograd profiler:

```python
from torch.autograd.profiler import profile as autograd_profile
from torch.profiler import record_function

with autograd_profile(use_device="openreg", record_shapes=True) as prof:
    with record_function("matmul"):
        x = torch.randn(512, 512, device="openreg")
        y = torch.randn(512, 512, device="openreg")
        z = x @ y

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof.export_chrome_trace("openreg_trace.json")
```

### Data capture flow

1. User code enters `autograd_profile(use_device="openreg")`.
2. The profiler transitions to `ProfilerState.KINETO_PRIVATEUSE1_FALLBACK`.
3. The profiler asks the active backend to `record()` an event.
4. The OpenReg stubs allocate `orEvent` objects, attach them to the current stream, and stash CPU timestamps.
5. When events end, the profiler calls `elapsed()` to compute durations.

## Implementation (Kineto Plugin)

```{note}
This section covers the Kineto `IActivityProfiler` plugin path for kernel-level tracing. It requires `USE_KINETO` at build time. All Kineto-dependent code must be guarded with `#ifdef USE_KINETO`.
```

The plugin path has two layers: a **device library** component (the CUPTI analog) and the **PyTorch integration** layer. OpenReg keeps these clearly separated.

### Device library: correlation tracking

The device library provides `openreg::profiler::OpenRegTracer` (`third_party/openreg/csrc/tracer.h/.cpp`) — a singleton with a thread-local correlation-ID stack and an atomic enable/disable flag that the profiler session uses to control the recording window.

Kineto pushes/pops correlation IDs through the session. The session calls C-style activity APIs in `openreg.h` (mirroring CUPTI):

* `orActivityEnableTracing()` / `orActivityDisableTracing()` — control the recording window
* `orActivityPushExternalCorrelationId()` / `orActivityPopExternalCorrelationId()` — maintain the correlation stack

A real vendor's equivalent would be their device tracing SDK (e.g., CUPTI for CUDA).

### PyTorch integration: IActivityProfiler and IActivityProfilerSession

Implement the two Kineto interfaces from `third_party/kineto/libkineto/include/IActivityProfiler.h`. In OpenReg, these live in `torch_openreg/csrc/profiler/` — the backend extension integration layer.

* **`IActivityProfiler`** — stateless factory. Two `configure()` overloads both create and return a session:
  - `configure(activity_types, config)` — synchronous overload; required by the interface. The OpenReg stub implements this as the core session-creation path.
  - `configure(ts_ms, duration_ms, activity_types, config)` — Kineto's child-profiler path calls this overload for all traces (including on-demand), passing `profileStartTime()` epoch ms and `profileDuration()` ms. The OpenReg stub ignores scheduling and delegates to the first overload; vendors use `ts_ms`/`duration_ms` to defer device-SDK activation.
* **`IActivityProfilerSession`** — per-trace session. `start()`/`stop()` manage the profiling window and toggle activity tracing via `orActivityEnableTracing()`/`orActivityDisableTracing()`; `getTraceBuffer()` returns the buffer to Kineto.
  - **Reference stub**: `processTrace()` only sets the trace span (`traceBuffer_.span = TraceSpan(startTs_, endTs_, "openreg")`); it emits no kernel records.
  - **Vendor extension**: replace `processTrace()` to flush records from your device tracing SDK, emit `GenericTraceActivity` entries with timestamps (µs), correlation IDs, and flow links (`flow.id = correlationId`, `flow.type = kLinkAsyncCpuGpu`, `flow.start = 0`).

### Registration and build

Register with one line: `REGISTER_PRIVATEUSE1_PROFILER(OpenRegActivityProfiler)`. The macro (defined in `torch/csrc/profiler/standalone/privateuse1_profiler.h`) creates a static registration object that forwards a factory to Kineto at profiler init time.

Kineto is on by default in PyTorch; no special build flags are needed unless explicitly disabled it with `USE_KINETO=0`. In backend extension, `find_package(Torch)` sets `kineto_LIBRARY`; link against `kineto` and `torch_cpu_library`, and guard Kineto code with `#ifdef USE_KINETO`. Without Kineto, the plugin compiles as a no-op and only the legacy stubs path is available.

### Usage

```python
import torch
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1]) as prof:
    x = torch.randn(512, 512, device="openreg")
    y = torch.randn(512, 512, device="openreg")
    z = x @ y

prof.export_chrome_trace("kernel_trace.json")
```

[PyTorch Profiler README]: https://github.com/pytorch/pytorch/blob/main/torch/csrc/profiler/README.md "PyTorch Profiler README"
[openreg-stubs]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/profiler/stubs/openreg.cpp "OpenReg profiler stubs"
