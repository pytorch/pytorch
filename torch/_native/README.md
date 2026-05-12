# Native Ops (and DSLs)

The `torch._native` directory provides a place for PyTorch native ops written in python and DSLs, along with utilities to help facilitate this.

# Creating & Registering a native op

**All native ops must be registered to the dispatcher**

When writing native ops, they are required to interact meaningfully with torch's dispatcher, and thus must be registered correctly.

As a further clarification, ops cannot be labelled as `CompositeImplicitAutograd` in `native_functions.yaml`, as-in the op must have an explicit autograd function registered, or at minimum an explicit implementation registered for the same backend as being overridden/added.

Each override is described by **two callables**: a predicate (`cond`) and an implementation (`impl`). At call time the registry's router evaluates the cond against the op's arguments and, if it returns `True`, routes the call to `impl`. If no registered cond matches, the router transparently falls through to the original aten kernel — there's no need to thread a fallback kernel or a `DispatchKeySet` through your code.

For overrides that always apply (no predicate), pass `unconditional_override=True` and omit `cond` (or pass `None`).

## A Note on Imports

All registrations will happen at the end of `import torch`. It is expected at that point that **no DSL runtime library is loaded by registration code** - this means that the runtime(s) must only be imported lazily. We can still check the presence of a module, and get its version without importing, but special care must be taken when writing op kernels to not import DSLs too early. An illustrative example is below, using `triton`:

First, we're going to write the registration function, and a top-level call, being very careful to not pull in the `triton` package early:

```
# torch/_native/ops/test_op/triton_impl.py

# NOTE: no triton import in the file

from ... import triton_utils as tu

def my_cond(*args, **kwargs) -> bool:
    # Cheap predicate: e.g. dtype / ndim checks. Avoid expensive work here
    # if possible; this runs on every call.
    return True

def my_impl(*args, **kwargs):
    # Lazily import the kernels (and triton) on first call
    from .triton_kernels import outer_fn
    return torch.library.wrap_triton(outer_fn)(*args, **kwargs)

def register_to_dispatch():
    tu.register_op_override(
        "aten",
        "_scaled_mm_v2",
        "CUDA",
        cond=my_cond,
        impl=my_impl,
    )

```

Note the lazy import of kernels inside `my_impl` - this function isn't called during registration (it just needs to be defined), and on first call it'll import the `triton_kernels` module, with all its dependencies. This kernels module can import anything it wants internally, our restriction on no DSL imports during registration has been met. Quick example of `triton_kernels.py` below:

```
# torch/_native/ops/test_op/triton_kernels.py
import triton

@triton.jit
def inner_fn(...) -> ...:
    # depends internally on triton, triton.language
    pass

@triton.jit
def outer_fn(...) -> ...:
    # depends internally on triton, triton.language
    inner_fn(...)
```


## Registering Implementations to Existing Operators

There are 2 options when interacting with an existing operator:
1. Replace the operator for **all** cases with a new implementation
2. Replace **some subset of functionality** of a given operator with a new implementation, falling-back to the original implementation otherwise.

Both cases are very similar, with 2) only requiring an extra step to obtain the original implementation, and logic to determine which implementation should be run for a given case.

### Replacing an Operator

This follows a simple and standard path, with a good example being the implementation of [FlashAttention v4 (FAv4)](https://github.com/pytorch/pytorch/blob/1f66f34cda5b5ad02d231b90fa0c0de2cb4e02d1/torch/nn/attention/_fa4.py#L67) in torch.

The following example replaces the implementation of `aten._scaled_grouped_mm_v2` on `CUDA` devices for **every** call:

```
from ... import cutedsl_utils as cu

def my_impl(*args, **kwargs) -> ...:
    """
    Replacement implementation — signature matches aten::_scaled_grouped_mm_v2.
    """
    pass

# Pass `unconditional_override=True` and `cond=None` to apply the impl to
# every call (a trivially-true predicate is substituted internally).
def register_kernel_override():
    cu.register_op_override(
        "aten",
        "_scaled_grouped_mm_v2",
        "CUDA",
        cond=None,
        impl=my_impl,
        unconditional_override=True,
    )
```

### Replacing a Subset of Calls

This time we only want to override the behavior of a subset of `aten._scaled_grouped_mm_v2` calls, based on some input property. Write a `cond` that returns `True` when the override should fire and `False` otherwise — when it returns `False`, the registry's router transparently falls through to the original aten kernel. You never need to call the fallback yourself.

```
from ... import cutedsl_utils as cu

def my_cond(arg1, arg2, *args, **kwargs) -> bool:
    """
    Decide whether to invoke my_impl. Keep this cheap — it runs on every call.
    Avoid APIs that aren't defined on FakeTensors (e.g. `_is_cow_tensor`) if
    you want the override to apply under torch.export / torch.compile too.
    """
    return arg1 == ... and arg2 == ...


def my_impl(arg1, arg2, *args, **kwargs) -> ...:
    """
    Replacement implementation — lazy-import the actual kernel and run it.
    """
    from .my_impl_kernel import my_kernel
    return my_kernel(arg1, arg2, *args, **kwargs)


# Override `aten::_scaled_grouped_mm_v2` on CUDA only when `my_cond` returns
# True. For other inputs the native aten kernel runs — the registry captures
# it automatically before installing the override and wires the fallback.
def register_kernel_override():
    cu.register_op_override(
        "aten",
        "_scaled_grouped_mm_v2",
        "CUDA",
        cond=my_cond,
        impl=my_impl,
    )
```

Notes:
* `cond` and `impl` have the **same signature as the aten op** — no extra
  `dispatch_keys` or `fallback_kernel` parameters.
* Multiple overrides can be registered for the same op at the same
  dispatch key; the router tries them in registration order and stops at
  the first `cond` that returns `True`. See § "Registration Orders and You"
  below for how to control ordering.

## Registering a New Operator

We currently don't have a use for this functionality, please come talk to us if you want it!

# Adding a new DSL

Adding a new DSL is as simple as adding a single helper utils file, then writing your op.

Some universal utilities are provided in `common_utils.py`:
* `check_native_jit_disabled() -> bool` : returns `True` if native DSL ops have been globally disabled.
* `_available_version(package: str) -> tuple[int,int,int] | None` : Gets the installed version of `package` without importing it
* `_unavailable_reasons(deps: list[tuple[str,str]]) -> str | None` : For a list of `(package_name, module_name)` pairs, check (without importing) if the module is available, returning a string describing the mitigation step if it isn't.

## DSL utils file spec

A DSL utils file, named `$dsl_utils.py` (i.e. `cutedsl_utils.py` for `$dsl=cutedsl`) requires three methods to be implemented.

`runtime_available() -> bool` : tell the user if the runtime is available - note that this needs to be available during init, and must be fork-safe. Packages should also not be imported at this time - rely on `importlib.util.find_spec(package_name)` or similar to get the necessary information without importing.

`runtime_version() -> tuple[int, int, int]` : return the `(major, minor, update)` version of the installed package.


```
register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    cond: _OpCondFn | None,
    impl: _OpImplFn,
    *,
    allow_multiple_override: bool = False,
    unconditional_override: bool = False,
) -> None
```
Register a given implementation to a library.

* `lib_symbol`: namespace you're overriding — `"aten"` for most cases.
* `op_symbol`: the op to override, either a bare name (`"bmm"`, resolving to `aten.bmm.default`) or overload-qualified (`"add_.Tensor"` → `aten.add_.Tensor`).
* `dispatch_key`: typically `"CPU"` or `"CUDA"` (or any other backend key).
* `cond`: predicate choosing when `impl` applies. May be `None` if `unconditional_override=True`, in which case a trivially-true predicate is substituted.
* `impl`: implementation callable; its signature matches the aten op.
* `unconditional_override`: apply `impl` to every call, bypassing the cond check.

For full argument details see the docstring for `register_op_override` in [registry.py](registry.py).

`deregister_op_overrides() -> None` : De-register all operators that are currently registered by this DSL. Note that `torch._native.registry` has a `deregister_op_overrides` method to enable this in a centralized fashion.

An example of an implementation of this spec can be found in [cutedsl_utils.py](cutedsl_utils.py), but please talk to us if you're planning on adding a new DSL.

## Registration Orders and You

Currently the registration order (both in general and per-op) is set by the order of imports in `torch/_native/ops/__init__.py`, noting that registration acts as a stack, in that **the last registered override for an op is the first that will be called**. If you wish to exercise control of the override ordering, please utilize one of the methods below.

### User-Ordering Functions

We allow for user-defined ordering functions of the form:

```
from torch._native.registry import _OverrideNode

def ordering_fn(
    op_symbol: str,
    dispatch_key: str,
    graph: list[_OverrideNode],
) -> list[_OverrideNode]
```

In other words, a function that takes some context and a graph describing the override order, and returning a modified graph.

**NOTE**: Graphs are described as lists of the private class `_OverrideNode` -- while this graph re-ordering functionality is public, it is both experimental and intended for advanced users only. The `_OverrideNode` class is to be used very carefully, and may change in the future.

This functionality can used by either setting the environment variable `TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN` to an importable python function with the above signature, or by adding the following to your top-level script, post `import torch`:

```
torch._native.reorder_graphs_from_user_function(
    my_ordering_fn,
    reregister_overrides=True,
)
```

Both methods are equivalent in functionality, but the environment-variable version is a little more efficient in that torch doesn't have to register **all** ops, before disabling/re-registering again based on the user-passed function.

**NOTE**: The passed ordering function can be destructive in nature - one can disable an op completely by returning `[]` for a given graph, indicating that no overrides exist / are allowed. **There is currently no supported way to retrieve the original graphs - they are considered gone for the lifetime of the process**.

An example user-ordering function is demonstrated below:

```
def example_ordering_fn(op_symbol, dispatch_key, nodes):
    out_nodes = []

    # disable overrides for these symbols completely
    if op_symbol in ["_scaled_mm_v2", "add"]:
         return []

    # Only keep triton overrides otherwise
    for node in nodes:
        if node.dsl_name != 'triton':
            continue
        out_nodes.append(node)

    return out_nodes
```

# Testing Native Ops

Adding operators means that they should be tested. Given that we're dealing with overrides, which by definition change behavior, we need to be a little careful.

The preferred testing method is to co-opt the existing `OpInfo` and `op_db` class/list and benefit from all the infrastructure built around that functionality.

Each op should have a corresponding entry added in `torch/testing/_internal/common_methods_invocations.py`, with a input method appropriate for the overrides(s) in terms of shapes and dtypes. Instead of adding directly into `op_db`, add to the appropriate entry in `dsl_ops_by_dsl`, a dictionary, where the key will be the name of the DSL used - this must be present in `torch.backends.python_native.available_dsl`. These entries are then later added to `op_db` as appropriate for use in:

* `test_ops.py`
* `test_unary_ufuncs.py`
* `test_ops_gradients.py`
* `test_torchinductor_opinfo.py`
* `test_export_opinfo.py`

## Testing only Native ops

Optionally, one can set the `OPINFO_RESTRICT_TO_DSL` environment variable to enable only the DSL specified for quick testing - this is then used via:

```
OPINFO_RESTRICT_TO_DSL=triton pytest -sv test/test_binary_ufuncs.py
```

This will test all triton-based binary operators.
