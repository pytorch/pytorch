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
a runnable Python source string plus an acceleration cache. Make-fx artifacts are
self-contained. Dynamo artifacts may import modules referenced by transformed globals;
installed artifacts also import the defining Python modules.
For compatibility, positional arguments after the callable still describe one example
call; they cannot be combined with `example_inputs`.
Reload the artifact with `torch.compiler.precompile.load`; since no weights are baked in,
you pass the model again at runtime. The optional `tracer="dynamo"` path accepts several
example calls and retains the guarded recompilations they trigger, including
automatically dynamic graphs. Use `torch.compiler.ExampleInput(args=..., kwargs=...)`
for a call with keyword arguments. Its serialized guard records are filtered while
preserving how every example dispatches. The contract requires the Python environment,
including globals and context-manager state, to remain semantically unchanged and allows
recompilation-causing variation only through explicit inputs. Environment-only guards may
therefore be omitted, while every portable input-derived guard is retained by default.
Distinct tensor inputs must not share or overlap storage at capture or runtime.
An explicit input must not also be reachable through globals or other environment state;
dynamic native indirection that hides such an identity relation is unsupported. Dynamo
input pytree structures must be serializable so runtime safety checks can be reconstructed.
User-defined code must access Python module objects (`types.ModuleType`) through static
attribute paths rather than pass or alias them. Python functions that assign or delete
globals are rejected.
Explicitly filtering an input guard is a risky drop and requires opting out of the default
safety gate. Guards are rebuilt from frozen capture state and checked for predicate drift
before the artifact is emitted. Breaking an unchecked environment assumption can silently
miscompute. Tensor, scalar, Python-container, and `nn.Module` arguments are supported.
Function defaults must be recursive immutable literals; mutable or user-defined values
must be passed explicitly rather than used as defaults. Tensor-valued globals are also
rejected when referenced by user-defined code because user-owned tensors must be explicit
inputs. Graph breaks and closure-free
`torch._dynamo.disable` functions are preserved; top-level closures and
nested functions that capture locals are not yet supported. The eager backend also
preserves higher-order graphs such as `torch.cond`, `torch.while_loop`, non-reentrant
checkpointing, `vmap`, autocast, and grad-mode regions without symbolically retracing
them at load. Captured nested frames that cannot be reached by a source-only dispatcher
use an isolated installed artifact. Loading prepares its backends and guards, its first
call installs them, and `unload()` removes them; it can also be scoped with `with`.
An uncovered call raises instead of compiling a new variant, just as it does for a
standalone artifact. With
`training=True`, both eager and Inductor artifacts retain autograd history; Inductor
graphs include readable compiled forward and backward code. Serving pins grad mode to
this option: inference artifacts disable gradients even inside `torch.enable_grad()`,
while training artifacts enable them even inside `torch.no_grad()`. This training mode
works across captured recompilations and graph breaks. `PrecompileSummary` reports
coverage and dropped guards, including which example first produced each frame variant,
while the `require_*` options let callers reject incomplete or insufficiently guarded captures.
See the {ref}`API reference <torch.compiler_api>` for details.

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
