# CUDA stream ordering for transports: design and prototype

## Goal

Queue `producer -> transfer -> consumer` without explicitly synchronizing the
submitting CPU thread. Registration, connection setup, allocation, and teardown
remain outside the steady-state path. This does not promise that all CUDA driver
calls are nonblocking or eliminate host work.

## Existing approaches

TorchStore's NIXL transport submits operations and awaits native status polling.
It does not itself add CUDA producer/consumer stream dependencies.

Mooncake exposes `transfer_write_on_cuda`, `transfer_read_on_cuda`, and batch
variants. They enqueue `cudaLaunchHostFunc`. The CUDA callback submits the transfer
and polls completion before returning; later work on that stream waits for the
callback. Its callback busy-polls and exits the process on failure.

Sources:
- [TorchStore NIXL](https://github.com/meta-pytorch/torchstore/blob/main/torchstore/transport/nixl.py)
- [Mooncake CUDA callback](https://github.com/kvcache-ai/Mooncake/blob/main/mooncake-integration/transfer_engine/transfer_engine_py.cpp)

## Critical constraint: callbacks cannot call CUDA

CUDA prohibits CUDA API calls inside host functions, including indirect calls
through a transport plugin. Host functions on independent streams may also be
serialized. They must not wait for CUDA work that is not ordered before them.

A direct NIXL/UCX VRAM-transfer experiment aborted inside the callback: UCX called
`cuPointerGetAttributes` and `cuDevicePrimaryCtxGetState`, received “operation not
permitted,” and failed memory packing. Prewarming a transfer did not fix it.
This is not a supported direct-VRAM implementation.

## Implemented prototype: pinned-host staging

The private `nixl._cuda_host.CudaHostTransport` adapts Mooncake's callback approach
for NIXL's tested UCX **host-memory** path. It rejects local or remote VRAM.

1. Connect and register pinned CPU buffers before enqueueing.
2. Queue a nonblocking GPU-to-pinned-CPU producer copy on the stream.
3. Enqueue a host function. After prior stream work finishes, submit the NIXL
   transfer, poll native Work completion with a 1 ms backoff, and check its error.
4. Return from the callback only after completion. Subsequent pinned-CPU-to-GPU
   copies and consumer kernels on that stream may execute.

The returned `threading.Event` supports optional host completion observation;
there are no arbitrary Future callbacks that could accidentally call CUDA from
the CUDA callback thread. The bridge retains the transport, memory views,
descriptors, and ctypes callback until a CUDA event proves the callback has
returned. Explicit close synchronizes for safe reclamation. Do not unregister
memory or close the underlying transport until the bridge closes.

A write's local completion does **not** establish remote-consumer readiness.
Receivers must exchange a completion notification before using incoming data.
Reads require the remote source to be ready and retained before submission.
The prototype does not provide a remote notification protocol.

### Failure and limitations

Callback errors and polling timeouts terminate the process with `_exit(1)` so
GPU consumers cannot run after failed DMA. Native submission itself can block
beyond the timeout, so use an external watchdog. Timeout starts when the callback
runs; it does not bound producer execution or callback scheduling delay.

Only the UCX pinned-host path was exercised. Other plugin/configuration paths need
an audit for indirect CUDA calls. Host callbacks may serialize across streams;
this is not a general-purpose overlap scheduler. There is no graph capture,
tracing, cancellation, recoverable asynchronous failure, or automatic protection
against explicit unregister/close while queued.

Prewarm kernels and allocate buffers before measurement. Lazy CUDA module loading
and allocation can introduce synchronization even when enqueue contains no explicit
CPU wait. `CUDA_MODULE_LOADING=EAGER` helps but is not a universal guarantee.

## Direct-VRAM alternative

A separate progress thread could query a producer event, submit NIXL outside a
CUDA callback, and signal a GPU-side gate after completion. A mapped-host word
with `cuStreamWaitValue32` is one possible gate. CUDA-visible producer/gate/consumer
event edges are needed; memory-operation ordering alone is invisible to CUDA's
scheduler. A scratch mapped-word experiment worked after warmup but reproduced a
cold-kernel lazy-loading deadlock. This alternative is not shipped by this PR.

Promotion requires explicit remote-write visibility capability checks/flushes,
bounded completion slots, queued-operation lifetime integration, device-visible
failure semantics, and multihost validation. Prefer a backend's native CUDA stream
API where available and proven safe over a generic host bridge.

## Validation

Opt-in tests run in bounded subprocesses:
- Real two-process NIXL/UCX host-memory writes and reads between pinned buffers,
  ordered with CUDA staging copies on two GPUs.
- Enqueue returns while a deliberately blocked callback remains incomplete.
- An injected callback failure terminates only the test child with exit code 1.

Run with `TORCH_TEST_CUDA_TRANSPORT=1` and `CUDA_MODULE_LOADING=EAGER`, with NIXL,
its CUDA/UCX dependencies, and `cuda-bindings` installed. The native test does not
establish multihost RDMA performance or direct-VRAM support. Report host enqueue
latency separately from end-to-end transfer throughput; no speedup is claimed.
