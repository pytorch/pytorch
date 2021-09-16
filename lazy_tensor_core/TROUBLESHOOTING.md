# Troubleshooting

Note that the information in this section is subject to be removed in future releases,
since many of them are peculiar to a given internal implementation which might change.

To diagnose issues, we can use execution metrics and counters provided by lazy
tensors framework. The **first thing** to check when model is slow is to generate
a metrics report.

Metrics report is extremely helpful in diagnosing issues. Please try to include it in your bug
report sent to us if you have it.

## Get A Metrics Report

Put the following line in your program to generate a report:

```Python
import lazy_tensor_core.debug.metrics as met

print(met.metrics_report())
```

## Understand The Metrics Report

The report includes things like:
- how many time we issue compilations and time spent on issuing.
- how many times we execute and time spent on execution
- how many device data handles we create/destroy etc.

This information is reported in terms of percentiles of the samples. An example is:

```
Metric: CompileTime
  TotalSamples: 202
  Counter: 06m09s401ms746.001us
  ValueRate: 778ms572.062us / second
  Rate: 0.425201 / second
  Percentiles: 1%=001ms32.778us; 5%=001ms61.283us; 10%=001ms79.236us; 20%=001ms110.973us; 50%=001ms228.773us; 80%=001ms339.183us; 90%=001ms434.305us; 95%=002ms921.063us; 99%=21s102ms853.173us
```

We also provide counters, which are named integer variables which track internal software status. For example:

```
Counter: CachedSyncTensors
  Value: 395
```

In this report, any counter that starts with `aten::`
indicates a context switch between the lazy device and CPU, which can be a
potential performance optimization area in the model code.

Counters are useful to understand which operations are routed back to the CPU engine of _PyTorch_.
They are fully qualified with their C++ namespace:

```
Counter: aten::nonzero
  Value: 33
```

If you see `aten::` ops other than `nonzero` and `_local_scalar_dense`, that usually means a missing
lowering in the accelerator plugin.

## Performance Profiling and Auto-Metrics Analysis
In addition, to manually inspecting the above metrics we provide ways to automatically analyze the above metrics report and provide a summary. Simply run your workload with `PT_XLA_DEBUG=1`.

To profile your workload in depth to undertand bottlenecks please check the following resources:
* [Official tutorial](https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling)
* [Colab notebook](https://colab.research.google.com/github/pytorch/xla/blob/master/contrib/colab/pytorch-xla-profiling-colab.ipynb)
* [Sample MNIST training script with profiling](https://github.com/pytorch/xla/blob/master/test/test_profile_mp_mnist.py)

## Known Performance Caveats

Lazy tensors behaves semantically like regular tensors and share the full tensor
interface with CPU & GPU tensors.
However, constraints in hardware, accelerator compiler backends and the lazy evaluation
model suggest certain patterns might result in bad performance.

If your model shows bad performance, keep in mind the following caveats:

1.  **Degraded performance with too many recompilations.**

    Compilation is expensive. For accelerators which don't support dynamic shapes,
    the lazy tensors framework automatically recompiles the graph every time new
    shapes are encountered.
    Usually models should stabilize within a few steps and you can see huge speedup for the rest of training.

    In order to avoid recompilations, not only must shapes be constant, but computations across lazy devices in all hosts should also be constant.

    _Possible sources_:
    * Direct or indirect uses of `nonzero` introduce dynamic shapes; for example, masked indexing `base[index]` where `index` is a mask tensor.
    * Loops with a different number of iterations between steps can result in different execution graphs, thus require recompilations.

    _Solution_:
    * Tensor shapes should be the same between iterations, or a low number of shape variations should be used.
    * Pad tensors to fixed sizes when possible.

1.  **Certain operations don't have native translations.**

    For these operations the lazy tensor framework automatically transfers to the
    CPU memory, evaluates on CPU, and transfers the result back to the lazy device.
    Doing too many such operations during the training step can lead to significant slowdowns.

    _Possible sources_:

    - The `item()` operation explicitly asks to evaluate the result. Don't use it unless it's necessary.

    _Solution_:

    - Check [metrics report section](#metrics-report) to find out the missing ops
    and request a lowering from the accelerator vendor.
    - Even when a PyTorch tensor is known as a scalar, avoid using `tensor.item()`. Keep it as a tensor and use tensor operations on it.
    - Use `torch.where` to substitute control flow when applicable.
      E.g. The control flow with `item()` used in [clip_grad_norm_](https://github.com/pytorch/pytorch/blob/de19eeee99a2a282fc441f637b23d8e50c75ecd1/torch/nn/utils/clip_grad.py#L33) is problematic and impacts performance.
      The current version in master gives us a dramatic performance improvement.
      ```python
      ...
      else:
        device = parameters[0].device
        total_norm = torch.zeros([], device=device if parameters else None)
        for p in parameters:
          param_norm = p.grad.data.norm(norm_type) ** norm_type
          total_norm.add_(param_norm)
        total_norm = (total_norm ** (1. / norm_type))
      clip_coef = torch.tensor(max_norm, device=device) / (total_norm + 1e-6)
      for p in parameters:
        p.grad.data.mul_(torch.where(clip_coef < 1, clip_coef, torch.tensor(1., device=device)))
      ```

1. **Data parallel iterators may drop the last few batches in the input iterator.**

   This is to make sure we do the same amount of work on all accelerator devices.

   _Solution_:

   * When dataset is small, and there are too few steps, this may result in a no-op epoch. Therefore, it is better to use
   small batch sizes in those cases.

## Lazy Tensor Quirks

1. **Lazy tensor internals are opaque.** Lazy tensors always appear to be
contiguous and without storage. Networks should not try to check the strides
of lazy tensors.

1. **Lazy tensors should be moved to the CPU before saving them.** Saving
lazy tensors directly causes them to be loaded back on the device(s) they were
saved from. If a device is unavailable at load time then the load will fail.
Moving lazy tensors to the CPU before saving them lets you decide which
device(s) to put the loaded tensors on. This is necessary if you want to
load the tensors on a machine without lazy devices. Care should be taken
moving the lazy tensors to the CPU before saving them, however, as moving
tensors across device types does not preserve view relationships. Instead,
views should be reconstructed as necessary after the tensors are loaded.

1. **Copying an lazy Tensor with Python's copy.copy returns a deep copy, not a
shallow copy.** Use a view of a lazy tensor to get a shallow copy of it.

1. **Handling shared weights.** Modules can share weights by setting the
Parameters of one module to another. This "tying" of module weights should
be done **AFTER** the modules are moved to a lazy device. Otherwise two
independent copies of the shared tensor will be made on the lazy device.

## More Debugging Tools

We don't expect users to use tools in this section to debug their models. But we might ask for
them when you submit a bug report since they provide additional information that metrics report
doesn't have.

### Environment Variables

There are also a number of environment variables which control the behavior of the
lazy tensors framework.

Setting such variables will cause different degrees of performance degradation, so they should
only be enabled for debugging.

* ```LTC_IR_DEBUG```: Enables the _Python_ stack trace to be captured where creating IR nodes,
  hence allowing to understand which _PyTorch_ operation was responsible for generating the IR.

* ```LTC_SAVE_TENSORS_FILE```: The path to a file which will be used to dump the IR graphs during
  execution. Note that the file can become really big if the option is left enabled and the
  _PyTorch_ program let run for long time. The graphs are appended to the file, so to have a clean
  sheet from run to run, the file should be explicitly removed.

* ```LTC_SAVE_TENSORS_FMT```: The format of the graphs stored within the _LTC_SAVE_TENSORS_FILE_
  file. Can be ```text``` (the default), ```dot``` (the _Graphviz_ format) or ```backend```.

* ```LTC_METRICS_FILE```: If set, the path to a local file where the internal metrics will be
  saved at every step. Metrics will be appended to the file, if already existing.

* ```SYNC_TENSORS_OPBYOP```: The same as _GET_TENSORS_OPBYOP_ but for "sync tensors"
  operation (the operation used at the end of a step, to flush pending IR computations and
  materialize them into _TPU_ device data).

* ```LTC_SYNC_WAIT```: Forces the lazy tensor sync operation to wait for its completion, before
  moving to the next step.

* ```LTC_USE_BF16```: If set to 1, tranforms all the _PyTorch_ _Float_ values into _BiFloat16_
  when sending to the _TPU_ device. Note that when using `LTC_USE_BF16=1` tensor arithmetic will
  be done in reduced precision and so tensors will not be accurate if accumulated over time.
  For example:

  ```
  # In reduced bfloat16 precision
  >>> torch.tensor(4096, dtype=torch.bfloat16) + torch.tensor(1, dtype=torch.bfloat16)
  tensor(4096., dtype=torch.bfloat16)
  # Whereas in full float32 precision
  >>> torch.tensor(4096) + torch.tensor(1)
  tensor(4097)
  ```
  So to get accurate metrics such as average loss value over many steps, use manual mixed
  precision where metrics stay in FP32.

* ```LTC_USE_32BIT_LONG```: If set to 1, maps _PyTorch_ _Long_ types to 32bit type.
  On some accelerators, 64-bit integer computations are expensive, so setting
  this flag might help. It should be verified by the user that truncating to 32-bit
  values is a valid operation according to the use of _PyTorch_ _Long_ values in it.

### Retrieving Stack Traces

In the event that the _PyTorch_ process is hanging, it might be useful to include the stack
traces together with the GitHub issue.

First thing is to find out which PID the _PyTorch_ process is associated with. Using the ```ps```
command it is possible to find that information. It will be a _python_ process running your
main _python_ file.

In order to allow _GDB_ to attach a user process the following command should be run as root:

```Shell
echo 0 > /proc/sys/kernel/yama/ptrace_scope
```

The above command remains active until the machine is rebooted.

The, given the PID, it is possible to grab the stack traces with the following command:

```Shell
./scripts/dump_stacks.py PID > /tmp/stack-traces.log
```

### Logging
To enable logging from your python code, try adding the following to your code:
```
from caffe2.python import workspace
workspace.GlobalInit(['caffe2', '--caffe2_log_level=-4'])
```

To be noted, it is '-4' not '4' because of [logging_is_not_google_glog.h](https://github.com/pytorch/pytorch/blob/0c3904d18061ea31c9fe1bded5893ffb07f0a4b5/c10/util/logging_is_not_google_glog.h#L106).
For internal FB users, you can search caffe2_common_gflags to learn more.
