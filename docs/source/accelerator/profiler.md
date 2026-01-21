# Profiler Integration

## Background

PyTorch ships a device-agnostic profiler that instruments CPU-side operator dispatch, coordinates with accelerator collectors, captures Python stacks, and exports aggregated statistics or Chrome/Perfetto traces. For core architecture, see [`torch/csrc/profiler/README.md`][PyTorch Profiler README].

There are two primary integration paths for accelerators:

1. Legacy autograd profiler:
    - Can attach backend-specific hooks via `ProfilerStubs` to record device events and compute elapsed times.
    - Works without Kineto; suitable for PrivateUse1 backends that want a minimal, self-contained path.

2. Kineto-based timeline:
    - Bridges to Kineto, which aggregates device timelines via vendor libraries (e.g., CUPTI for CUDA).
    - Provides rich activity traces and advanced export/visualization, but requires a Kineto-capable backend.

This document focuses on path (1): how a `PrivateUse1` accelerator exposes the minimal hooks to plug into the legacy autograd profiler so ATen ops and `record_function` ranges are correctly attributed to device activity.

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

OpenReg currently relies on the legacy profiler (`torch.autograd.profiler.profile`) interface rather than the modern one (`torch.profiler.profile`) because the latter enforces `use_kineto=True`.

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


[PyTorch Profiler README]: https://github.com/pytorch/pytorch/blob/main/torch/csrc/profiler/README.md "PyTorch Profiler README"
[openreg-stubs]: https://github.com/pytorch/pytorch/blob/main/test/cpp_extensions/open_registration_extension/torch_openreg/csrc/profiler/stubs/openreg.cpp "OpenReg profiler stubs"
