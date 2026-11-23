# torch.utils.component_benchmark

Component Benchmark captures real inputs at module boundaries and replays each
component independently. Use it to compare the behavior of model components in
eager execution, compilation, inference and training. It implements the core
workflow described in [Component Benchmark: Hierarchical Model Profiling for
Large-scale Recommendation Systems](https://arxiv.org/abs/2609.30656).

## Basic usage

```python
import torch
from torch.utils.component_benchmark import run_cb, ReportPlugin

model = torch.nn.Sequential(
    torch.nn.Linear(32, 64), torch.nn.ReLU(), torch.nn.Linear(64, 8)
).eval()
inputs = torch.randn(16, 32)
result = run_cb(
    model, inputs, max_depth=2,
    result_plugins=[ReportPlugin("benchmark_results")],
)
```

Open `benchmark_results/report.html` for an interactive module hierarchy,
metric selection, zoom controls and a searchable result table. JSON and CSV
exports live alongside it. Reports and profiler traces work locally without
an upload service or a network connection.

One tensor is one positional argument. Pass multiple positional arguments as a
tuple and use `model_kwargs` for keyword arguments. A tuple-valued single
argument must be wrapped in a one-element tuple. Use `PyTorchProvider` to
explicitly describe an argument-free model or a custom provider to supply
representative data from another framework.

## Capture and replay

Traversal is breadth-first and depth is relative to `submodule_path`. The capture
forward executes a copy of the parent model and snapshots inputs before each
module invocation. Returned components reference the provider's original
modules; execution plugins copy them before benchmarking. The provider's model,
inputs, gradients, buffers and RNG state are preserved by the built-in plugins.

Repeated invocations are an ordered call sequence, including each invocation's
autocast settings. One benchmark iteration replays that entire sequence. The
default `copy_input=True` protects input snapshots from in-place operations;
input-copy overhead is included in latency. Disable copying only for inputs
that are safe to reuse. Custom containers must be registered as PyTorch pytrees.
Tensor snapshots retain strides, shared storage, repeated references and
`requires_grad` while severing the parent autograd graph.

With `capture_output_grad=True`, CB backpropagates through the parent using
`loss_and_bwd` and records gradients at each component's outputs. Replay then
uses those boundary gradients. Otherwise a profiler plugin can supply its own
backward callback. `component_loss_and_bwd` backpropagates unit gradients through
all differentiable output leaves; `default_loss_and_bwd` uses the first leaf.
Pass a callable, `"component"`, `"default"` or `True` to `run_cb(loss_and_bwd=...)`
to enable backward in the default profiler plugins.

`Component.data` ordinarily represents bootstrap batches: the first batch is
measured, and later batches warm up additional shapes. Set
`data_is_call_sequence=True` only when the list describes invocations within
one parent iteration. These populations are kept separate in results.

Inactive modules and containers without a forward method are reported without
synthetic measurements. One failed profiler plugin does not stop other plugins
or child modules. The bootstrap forward must fit on the available device;
providers can supply a smaller representative model/batch before extraction.
Modules with stateful Python forward logic, uncopyable resources or model/input
storage ties may require a workload-specific provider or plugin.

## Plugins and preprocessing

The built-in plugins cover:

- `CoreForwardBenchmarkPlugin`: synchronized wall latency with warmup, median,
  mean, IQR and raw block samples; optional compilation, backward, activation
  checkpointing and optimizer execution. An optimizer is created once and
  stepped once per replayed sequence, after accumulating its gradients.
- `FlopsCounterPlugin`: formula-covered eager FLOPs and per-operator counts.
- `MemoryPlugin`: CUDA allocator peaks and optional local memory snapshots.
  CPU peak allocation is unavailable and is represented as `None`.
- `ProfilerPlugin`: local Kineto traces and operator summaries with CPU/device
  timings, shapes and FLOPs.
- `TorchDispatchModesPlugin`: an explicit estimate of eager tensor input/output
  bytes and bandwidth, with optional caller-supplied peak bandwidth. Alias-only
  view operations are excluded. This is not a measurement of physical device
  traffic and does not model caches or fused-kernel accesses.
- `MFUPlugin`: utilization against an explicit `peak_flops_per_second` matching
  the workload's hardware, dtype and sparsity.
- `AccuracyCheckPlugin`: eager versus transformed outputs, parameter gradients,
  parameters, buffers and optional optimizer states.
- `GPUBoundSimulationPlugin`: CUDA graph replay with event timing, excluding
  Python input-copy and launch costs. It requires graph-safe inputs and
  `copy_input=False`.
- `ComponentCapturePlugin`: local component serialization for trusted replay.
- `ReportPlugin`: local JSON, CSV and interactive HTML reports.

Use `to_dtype`, `repeat_batch` and `compose` as `preproc_func` values. Batch
dimensions are explicit: a selector returns an axis for batch tensors and
`None` for static masks or other non-batch data. Supply a custom preprocessor
for jagged offsets, lengths and values. Recapture output gradients after
changing the batch size. This avoids silently applying dense batch replication
to ragged data.

Custom `IPlugin` implementations return `ComponentResult`. The optional
`plugin_name` prefixes metric keys to prevent collisions; duplicate keys are
reported as errors. `IResultPlugin` implementations consume final results, and
`IProvider` implementations supply `Component` objects. An empty custom plugin
list performs input capture and module discovery without profiling. Optional
`use_multiprocess=True` executes plugins in one spawned worker, with no
simultaneous benchmarks competing for the device. Components and plugins must
be picklable.

## Interpreting measurements

Parent and child latency numbers overlap in scope and must not be added as
though they were an exclusive end-to-end breakdown. Components are independently
replayed, so caches, compiler fusion boundaries and launch behavior can differ
from the full model. Raw samples, their block size, thread count, device,
PyTorch version and UTC run timestamps accompany the metrics.

FLOP counts include only registered `FlopCounterMode` formulas. They are not
hardware instruction counts. MFU uses those FLOPs and synchronized wall latency,
so it includes CPU overhead and optional copying. CUDA graph event measurements
are a separate metric and are not substituted for wall latency. Profiler device
times may overlap across streams and are not a latency substitute.

Component files contain Python pickles. Only load trusted files with
`Component.load`. Custom module classes must remain importable when replaying
saved components or using multiprocessing.

## Examples

Small DeepFM and transformer workloads are provided under
`benchmarks/component_benchmark`:

```sh
python benchmarks/component_benchmark/run.py --model deepfm --device cpu --output /tmp/deepfm_cb && python benchmarks/component_benchmark/run.py --model transformer --device cuda --backward --compile --output /tmp/transformer_cb
```

They demonstrate hierarchical profiling of ordinary PyTorch models. Workload
data loading and distributed-model preparation belong in custom providers.
Additional measurement tools and artifact storage can be supplied through
plugins without adding dependencies to the core package.

## API

```{eval-rst}
.. automodule:: torch.utils.component_benchmark
   :members:
   :imported-members:
```
