---
myst:
  html_meta:
    description: A guide to torch.xpu, a PyTorch module to run on Intel GPU (XPU)
    keywords: XPU, Intel GPU, torch.xpu
---

(xpu-guide)=

# XPU guide

## Runtime

### Device

PyTorch XPU device management APIs are primarily provided through the {mod}torch.xpu module.

#### Device Visibility

To restrict the set of visible XPU devices, set the `ZE_AFFINITY_MASK` environment variable before launching your application. Only the devices specified by this variable will be visible to the application.

If ZE_AFFINITY_MASK is not set, all available XPU devices are visible by default.

For example, on a system with eight XPU devices:

| ZE_AFFINITY_MASK | Device 0 | Device 1 | Device 2 | Device 3 | Device 4 | Device 5 | Device 6 | Device 7 |
|------------------|----------|----------|----------|----------|----------|----------|----------|----------|
| (not set)        | Visible  | Visible  | Visible  | Visible  | Visible  | Visible  | Visible  | Visible  |
| 0,1,2,3,4,5,6,7  | Visible  | Visible  | Visible  | Visible  | Visible  | Visible  | Visible  | Visible  |
| 0,2,4,6          | Visible  | Hidden   | Visible  | Hidden   | Visible  | Hidden   | Visible  | Hidden   |

For example:
```bash
# Make only devices 0 and 1 visible.
export ZE_AFFINITY_MASK=0,1

python train.py
```

#### Device Synchronization
Starting with PyTorch 2.14, {func}`torch.xpu.synchronize` performs device-wide synchronization. It waits for all work submitted to the specified XPU device to complete, including work enqueued on streams created outside of PyTorch.

#### Device Telemetry

PyTorch provides a comprehensive set of GPU telemetry APIs for XPU under the {mod}torch.xpu module. These APIs enable applications to query real-time hardware metrics directly from Python, including:
- {func}`GPU core temperature <torch.xpu.temperature>`
- {func}`GPU clock frequency <torch.xpu.clock_rate>`
- {func}`Power consumption <torch.xpu.power_draw>`
- {func}`Engine utilization <torch.xpu.utilization>`
- {func}`Memory bandwidth utilization <torch.xpu.memory_usage>`
- {func}`Global memory usage<torch.xpu.device_memory_used>`
- {func}`Memory usage per process<torch.xpu.list_gpu_processes>`

For the best telemetry experience and compatibility, we recommend using Intel® Arc™ B-Series GPUs or newer Intel GPU platforms (Xe2 architecture or later).

### Stream

{class}`torch.xpu.Stream` is a wrapper around a `sycl::queue` (analogous to `cudaStream_t`) that represents an in-order asynchronous execution queue on an XPU device. Operations submitted to the same stream are executed sequentially in submission order, while operations on different streams may execute concurrently, subject to hardware and runtime scheduling. This API is fully compatible with the device-agnostic {class}`torch.Stream` interface, allowing to write portable stream management code across different accelerator backends.

Use {func}`torch.xpu.get_stream_from_external` to create a {class}`torch.xpu.Stream` from an externally managed `sycl::queue`. This is useful when integrating PyTorch with existing SYCL applications or libraries that already manage their own queues. This API provides functionality similar to {class}`torch.cuda.ExternalStream`.

### Event

{class}`torch.xpu.Event` is a synchronization primitive that can be used to monitor XPU execution progress, measure execution time, and orchestrate dependencies betwween between different {class}`torch.xpu.Stream`. For backend-independent stream management, PyTorch provides the device-agnostic {class}`torch.Event` interface, enabling portable synchronization code across different accelerator backends.
Currently, {class}`torch.xpu.Event` does not support inter-process communication (IPC). Support for IPC will be added in a future release.

## XPU vs CUDA: key differences

When porting code from CUDA to XPU, there are a few behavioral differences to be aware of.
If you are writing portable code that is intended to run across different accelerator backends, we strongly recommend using the device-agnostic APIs in {mod}torch.accelerator whenever possible.

| Aspect | CUDA | XPU | ACCELERATOR |
|---|---|---|---|
| Device visibility | `CUDA_VISIBLE_DEVICES` | `ZE_AFFINITY_MASK` | NA |
| External stream | `torch.cuda.ExternalStream` | `torch.xpu.get_stream_from_external(ptr)` | NA |
