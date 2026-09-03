(torch.compiler_overview)=

# torch.compiler

`torch.compiler` is a namespace through which some of the internal compiler
methods are surfaced for user consumption. The main function and the feature in
this namespace is `torch.compile`.

`torch.compile` is a PyTorch function introduced in PyTorch 2.x that aims to
solve the problem of accurate graph capturing in PyTorch and ultimately enable
software engineers to run their PyTorch programs faster. `torch.compile` is
written in Python and it marks the transition of PyTorch from C++ to Python.

`torch.compile` leverages the following underlying technologies:

- **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython
  feature called the Frame Evaluation API to safely capture PyTorch graphs.
  Methods that are available externally for PyTorch users are surfaced
  through the `torch.compiler` namespace.
- **TorchInductor** is the default `torch.compile` deep learning compiler
  that generates fast code for multiple accelerators and backends. You
  need to use a backend compiler to make speedups through `torch.compile`
  possible. For NVIDIA, AMD and Intel GPUs, it leverages OpenAI Triton as the key
  building block.
- **AOT Autograd** captures not only the user-level code, but also backpropagation,
  which results in capturing the backwards pass "ahead-of-time". This enables
  acceleration of both forwards and backwards pass using TorchInductor.

To better understand how `torch.compile` tracing behavior on your code, or to
learn more about the internals of `torch.compile`, please refer to the [`torch.compile` programming model](compile/programming_model.md).

:::{note}
In some cases, the terms `torch.compile`, TorchDynamo, `torch.compiler`
might be used interchangeably in this documentation.
:::

`torch.compiler` also includes an ahead-of-time API, `torch.compiler.precompile`. It
captures a whole computation from positional-argument tuples in `example_inputs` -- with
the model(s) included in each tuple, e.g.
`precompile(lambda model, x: model(x), example_inputs=[(model, x)])` -- and lowers it to
a self-contained, runnable Python source string plus an acceleration cache. Reload the
artifact with `torch.compiler.precompile.load`; since no weights are baked in, you pass
the model again at runtime. The `tracer` argument selects the capture front-end and
defaults to `"make_fx"`: a non-strict trace of a single `example_inputs` tuple, so
control flow and shapes are specialized to that one example. The optional
`tracer="dynamo"` path instead accepts several example
tuples and retains the guarded recompilations they trigger, including automatically
dynamic graphs. It retains guards derived from explicit inputs and treats the Python
environment as an unchecked invariant between capture and runtime; changing that
environment can silently run a specialization captured for the old state. Initial
support is for Python functions with positional tensor/scalar arguments and containers
of those values; graph breaks, closures, `nn.Module`, and numpy array/scalar arguments
are not supported yet.
Globals whose object graph contains a tensor are rejected, as are mutations of
state reachable from a module global (a helper's `global`, an attribute, dict or
list of an object a module holds, a default argument left at its default) and
distinct tensor inputs sharing or overlapping storage (the same tensor object may
be passed more than once). Mutations of the call's own argument objects are
captured and replayed on the caller's objects when the artifact is served.
Each captured Dynamo graph's differentiability is inferred from its inputs, mirroring
`torch.compile`: `requires_grad` inputs yield differentiable graphs whose served outputs
retain a `grad_fn` and can be passed to `backward()`; no-grad inputs yield inference
graphs. On the Inductor backend the backward is precompiled (readable forward and
backward code, working across captured recompilations, rejecting output-tangent
patterns not observed during capture -- the ordinary all-tangents-defined pattern is
always covered); on the eager backend the backward is live eager autograd through the
emitted ops, neither captured nor specialized. The sibling entry point
`torch.compiler.precompile.stateful` captures incrementally: each call runs
its example tuples for real inside a loop the caller owns, returns a list of that call's
per-example results plus an opaque `state` to pass back in, and rewrites an
always-loadable artifact on disk; call `state.close()` when done capturing. See
the {ref}`API reference <torch.compiler_api>` for details.

:::{warning}
`torch.compile` may not support recently released major versions of Python.

If you attempt to use `@torch.compile` in an unsupported Python
environment, you may encounter an error similar to:

```
RuntimeError: torch.compile is not supported on Python 3.xx.0+

```

Please ensure that your current Python version is within the range
supported by PyTorch for `torch.compile`.

If you have installed PyTorch on a Python version that is too new,
you will need to switch to an earlier Python version in order to use `torch.compile`.
:::

As mentioned above, to run your workflows faster, `torch.compile` through
TorchDynamo requires a backend that converts the captured graphs into a fast
machine code. Different backends can result in various optimization gains.
The default backend is called TorchInductor, also known as *inductor*,
TorchDynamo has a list of supported backends developed by our partners,
which can be seen by running `torch.compiler.list_backends()` each of which
with its optional dependencies.

Some of the most commonly used backends include:

**Training & inference backends**

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Backend
     - Description
   * - ``torch.compile(m, backend="inductor")``
     - Uses the TorchInductor backend. `Read more <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
   * - ``torch.compile(m, backend="cudagraphs")``
     - CUDA graphs with AOT Autograd. `Read more <https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html>`__
   * - ``torch.compile(m, backend="ipex")``
     - Uses IPEX on CPU. `Read more <https://github.com/intel/intel-extension-for-pytorch>`__
```

**Inference-only backends**

```{eval-rst}
.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Backend
     - Description
   * - ``torch.compile(m, backend="tensorrt")``
     - Uses Torch-TensorRT for inference optimizations. Requires ``import torch_tensorrt`` in the calling script to register backend. `Read more <https://github.com/pytorch/TensorRT>`__
   * - ``torch.compile(m, backend="ipex")``
     - Uses IPEX for inference on CPU. `Read more <https://github.com/intel/intel-extension-for-pytorch>`__
   * - ``torch.compile(m, backend="tvm")``
     - Uses Apache TVM for inference optimizations. `Read more <https://tvm.apache.org/>`__
   * - ``torch.compile(m, backend="openvino")``
     - Uses OpenVINO for inference optimizations. `Read more <https://docs.openvino.ai/torchcompile>`__
```




```{toctree}
:maxdepth: 1
:hidden:

torch.compiler_get_started.md
```

```{toctree}
:maxdepth: 1
:hidden:

core_concepts
```

```{toctree}
:maxdepth: 1
:hidden:

performance
```

```{toctree}
:maxdepth: 1
:hidden:

advanced
```

```{toctree}
:maxdepth: 1
:hidden:


troubleshooting_faqs
```

```{toctree}
:maxdepth: 1
:hidden:

api_reference
```
