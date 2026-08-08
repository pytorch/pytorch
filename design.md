# Yet Another Nonstrict API

May 14, 2026

This document sketches a small public API family for full-graph graph capture,
JIT compilation, and target-specific compiled artifact management:

- `make_fx`
- `jit`
- `compile_artifact`

The intended programming model is close to JAX's `make_jaxpr`, `jax.jit`, and
low-level executable serialization APIs. The APIs should stay simple: no special
handling for `nn.Module`, optimizers, parameter discovery, or other implicit
state. Users pass all state explicitly through function arguments.

Prototype: https://github.com/pytorch/pytorch/pull/183726

## Rationale

Frontier users and compiler-oriented users often want a tracing model that does
not infer too much or get in the way. JAX's APIs are a good reference point: the
user writes a pure function, marks static inputs explicitly, and gets a clear
split between graph capture, lazy execution, and compiled artifacts.

Torch APIs such as `torch.compile` and `torch.export` carry substantial
complexity around Python programs with implicit state: modules, parameters,
optimizers, mutation, object identity, and graph breaks. Those are important
APIs, but they are not the programming model here. This API family is for
functions whose dynamic inputs are tensors or pytrees of tensors, with explicit
static arguments.

Any divergence from JAX should be deliberate and documented.

## API Roles

### `make_fx`

Graph capture and inspection API.

```python
gm = make_fx(f, static_argnames=())(*example_args)
```

Semantics:

- Captures `f` into an FX-style graph.
- Always performs nonstrict full-graph capture.
- Intended for debugging, inspection, graph transforms, and compiler
  development.
- Does not imply compilation or artifact creation.
- Is not a portable IR boundary. In PyTorch, lower-level ops, decompositions,
  dispatch behavior, fake tensor metadata, and device-specific operator choices
  can depend on the target/device represented by the example inputs.
- Does not produce a deployment artifact.
- Static arguments are explicitly named with `static_argnames`; these values are
  baked into the graph as constants.
- Non-static arguments must be tensors or pytrees of tensors.

The prototype's `make_fx` always uses `tracing_mode="fake"` internally.

Example:

```python
from nonstrict import make_fx

def f(xs, *, scale):
    return {"y": xs["x"] * scale}

gm = make_fx(f, static_argnames="scale")({"x": torch.ones(2)}, scale=2)
out = gm(torch.ones(2))
```

### `jit`

Execution-facing API.

```python
jf = jit(f, static_argnames=(), backend="inductor")
out = jf(*args)
```

Semantics:

- Returns a callable wrapper.
- Lazily captures, specializes, compiles, caches, and executes.
- Owns specialization and compilation policy.
- May internally manage a cache of compiled callables or artifacts.
- Does not produce a persistent artifact unless `compile_artifact` is requested.

The `jit` cache key is intentionally based only on call inputs:

- The pytree structure of non-static arguments.
- Tensor properties for every dynamic tensor leaf, e.g. shape, stride, dtype,
  device, layout, requires-grad, and other metadata needed by the compiler.
- Values of all static arguments named by `static_argnames`.

No global module state, optimizer state, or hidden object state should enter the
cache key unless the user explicitly passes it as an argument.

With `backend="inductor"`, the conceptual lowering pipeline is:

```text
Python function
  -> make_fx
  -> AOTAutograd
  -> Inductor
  -> compiled callable
```

The exact backend entrypoint should be the same one used by
`torch.compile(..., backend="inductor")` where possible.

Options owned by `jit` may include:

- dynamic/static shape policy
- guard policy
- decomposition table
- backend/compiler selection
- optimization mode
- autotuning policy
- cache policy

Example:

```python
from nonstrict import jit

def f(xs, *, scale):
    return {"y": xs["x"] * scale}

jf = jit(f, static_argnames="scale")
out = jf({"x": torch.ones(2)}, scale=2)
```

#### Explicit Module State

This API should not special-case `nn.Module`. To trace a module-like program,
pass state explicitly. A future reparameterization helper could make this
ergonomic:

```python
state_dict = mod.state_dict()

def forward_backward(state_dict, x, target):
    with reparametrize(mod, state_dict):
        y = mod(x)
        loss = loss_fn(y, target)
        grads = torch.autograd.grad(loss, mod.parameters())
    return y, grads

y, grads = jit(forward_backward)(state_dict, x, target)
```

This keeps the core API pure-function-shaped. If a module's state affects the
graph, that state should be explicit in the argument list.

### `compile_artifact`

Artifact-facing API.

Canonical form:

```python
artifact = compile_artifact(
    f,
    static_argnames=(),
    backend="inductor",
    target=Target.current(),
    **artifact_options,
)(
    *args_or_specs,
)
```

Semantics:

- Returns a callable artifact materializer.
- Materializes one specialization of a function as a serialized compiled
  artifact, without requiring the user to construct a `jit` wrapper first.
- Eagerly compiles if the specialization is not already cached by that
  materializer.
- Produces a target-specific executable package, not a portable IR.
- The artifact is intended to be loaded and reused on machines compatible with
  its target/runtime metadata.
- Users manually manage artifact files and compatibility expectations.

This is not `export`. In PyTorch, a portable export-like API for this layer is
probably not a productive direction: lower-level ops, decompositions, and
runtime behavior are already target-sensitive. A compiled artifact should be
described honestly as backend-, runtime-, device-, and version-specific.

Example:

```python
from nonstrict import CompiledArtifact, compile_artifact

artifact = compile_artifact(f)(torch.ones(2))
out = artifact.run(torch.ones(2))

artifact.save("f.compiled")
loaded = CompiledArtifact.load("f.compiled")
out = loaded.run(torch.ones(2))
```

The artifact API should support:

- target machine/runtime metadata
- artifact format
- runtime ABI compatibility
- debug info
- bundled constants
- bundled autotune data
- compatibility strictness
- save/load metadata

The backend should provide the implementation hook that produces a
`CompiledArtifact`. For Inductor, this can reuse the megacache /
`standalone_compile` path.

## JAX Alignment

The intended alignment is:

| PyTorch API | JAX analogue | Role |
| --- | --- | --- |
| `make_fx` | `jax.make_jaxpr` | show me the graph |
| `jit` | `jax.jit` | make this callable fast |
| `compile_artifact` | low-level executable serialization | materialize target-specific executable |

The `make_fx` comparison is conceptual, not a portability claim. `jax.make_jaxpr`
prints a JAX-level staged representation. PyTorch `make_fx` captures an
FX-style graph of lower-level PyTorch ops, and that graph can already reflect
target/device-specific lowering choices.

`jax.export` is not the intended analogue for `compile_artifact`. JAX export
serializes StableHLO and calling-convention metadata; it may still require
backend compilation after loading. A PyTorch export-like portable IR API at this
layer is probably a lost cause for this proposal. `compile_artifact` is closer
to serializing a compiled executable/cache artifact and should be described as
target-specific.

Known prototype divergence:

- JAX accepts dynamic Python numeric scalars (`int`, `float`, `bool`) and
  abstracts them as scalar array-like values. The current prototype requires
  non-static arguments to be tensors or pytrees of tensors. Numeric scalars must
  be passed as tensors or marked static. This is a deliberate simplification for
  the prototype and should be revisited if we want closer JAX compatibility.

## Static Arguments

Static arguments are named with `static_argnames`.

Rules:

- Static arguments are compile-time constants.
- Static values participate in cache keys.
- Static values must be hashable and effectively immutable.
- Static values must not contain tensors in the current prototype.
- Changing a static value causes a separate specialization.

This mirrors JAX's `static_argnames` model, with the tensor-containing-static
restriction added to keep tensor metadata handling unambiguous.

## Pytrees

Dynamic inputs and outputs may be pytrees.

For dynamic inputs:

- The pytree structure is part of the cache key.
- Leaves must be tensors in the current prototype.
- Flattened tensor leaves form the graph/backend calling convention.

For outputs:

- The graph returns flattened tensor leaves.
- The wrapper records the output pytree spec and reconstructs the original
  output structure for `jit` and compiled artifact calls.

PyTorch `TreeSpec` is hashable and equality-checked, so it can be used directly
as a cache-key component for prototype purposes.

## Non-Goals

- `compile_artifact` is not a replacement for `make_fx`.
- `compile_artifact` is not a portable graph or IR export API.
- `jit` should not primarily be described as an artifact API, even if it
  internally manages compiled artifacts.
- `export` should not be used for this feature name.
