## Motivation

`torch.func.vmap` currently only supports `torch.Tensor` as inputs and outputs.
TensorDict define a tensor-like container type (ie, pytrees of tensors with metadata) have to resort to monkey-patching internal vmap functions (`_process_batched_inputs`, `_create_batched_inputs`, `_unwrap_batched`) to make their types work with vmap.

I'm proposing a lightweight, protocol-based extension mechanism that would let
any custom type participate in vmap without patching PyTorch internals.

## Design

### Protocol

A non-Tensor type becomes vmap-compatible by implementing four methods:

```python
class MyContainer:
    def _add_batch_dim(self, in_dim: int, vmap_level: int) -> "MyContainer":
        """Wrap internal tensors as batched tensors for the given vmap level."""
        ...

    def _maybe_remove_batch_dim(
        self,
        func_name: str,
        vmap_level: int,
        batch_size: int,
        out_dim: int,
    ) -> "MyContainer":
        """Unwrap batched tensors and insert the batch dimension at out_dim."""
        ...

    def dim(self) -> int:
        """Number of dimensions (used for in_dim validation)."""
        ...

    def size(self, dim: int) -> int:
        """Size along a dimension (used to validate consistent batch sizes)."""
        ...
```

Additionally, for `restore_vmap` / `autograd.Function` vmap support, a type can
implement:

```python
    def _unwrap_batched(self, level: int) -> tuple["MyContainer", int | None]:
        """Unwrap batched tensors and return (unwrapped, batch_dim_or_None)."""
        ...
```

### Detection

I have a helper function `_is_vmappable` that checks whether an object supports
the protocol. It uses a class-level cache so the check is O(1) for
previously-seen types:

```python
_vmappable_cls_cache: dict[type, bool] = {}

def _is_vmappable(obj: Any) -> bool:
    if isinstance(obj, torch.Tensor):
        return False
    cls = type(obj)
    if cls in _vmappable_cls_cache:
        return True
    if hasattr(cls, "_add_batch_dim"):
        _vmappable_cls_cache[cls] = True
        return True
    return False
```

The `isinstance(obj, torch.Tensor)` fast path ensures tensor subclasses
continue to go through the existing vmap path unchanged. Only positive results
are cached -- this is important because `_is_vmappable` gets called as
`is_leaf` in `tree_flatten` (so it sees tuples, ints, etc.), and caching
negative results would grow the dict and cause dynamo guard breaks.

### Explicit Registration API

For `torch.compile` users, I added a `register_vmappable_cls` function that
pre-populates the cache so that no dynamic `hasattr` check is needed during
tracing:

```python
def register_vmappable_cls(cls: type) -> None:
    _vmappable_cls_cache[cls] = True
```

Pre-registering at import time means `torch.compile` sees a stable dict lookup
and avoids guard breaks when a vmappable class is encountered for the first
time during tracing.

Auto-detection via `hasattr` remains as a fallback for eager-mode users who
don't call `register_vmappable_cls`.

## Tradeoffs

I imagine cache and pre-registration may trigger some discussion so here's a
breakdown of the pros and cons.

### Class cache + `hasattr` fallback vs. strict registration

| Aspect | Cache + fallback | Registration only |
|--------|-----------------|-------------------|
| Ease of use | Just implement the methods | Must also call `register_vmappable_cls` |
| Subclass support | Automatic (MRO walk) | Must register each subclass individually |
| `torch.compile` | First encounter triggers `hasattr` (guard break) | Clean, no guard breaks |
| Discoverability | Implicit duck typing | Explicit entry point |

The current design blends both: auto-detect for convenience, explicit
registration for compile friendliness.

### Protocol (duck typing) vs. ABC / `isinstance`

I considered an ABC (`VmapCompatible`) with `register()` but it adds ceremony
without clear benefit. The class cache achieves O(1) amortized detection just
like ABC caching, and the duck typing approach is more Pythonic and doesn't
require users to inherit or register.

In my experience [pure-python Protocols](https://peps.python.org/pep-0544/)
introduce overhead during instance checks and such and not much additional
clarity.

## What Changes in vmap

All changes are confined to `torch/_functorch/vmap.py`:

1. **`_process_batched_inputs`**: uses `tree_flatten(args, is_leaf=_is_vmappable)`
   so that vmappable types are treated as pytree leaves, and relaxes the
   Tensor-only validation to also accept vmappable types.

2. **`_create_batched_inputs`**: dispatches to `arg._add_batch_dim(in_dim, vmap_level)`
   for vmappable args.

3. **`_maybe_remove_batch_dim`**: dispatches to
   `output._maybe_remove_batch_dim(func_name, vmap_level, batch_size, out_dim)`
   for vmappable outputs.

4. **`_unwrap_batched`**: uses `tree_flatten(..., is_leaf=_is_vmappable)` and
   extends the single-output check to include vmappable types.

5. **`wrap_batched` / `unwrap_batched`**: uses `is_leaf=_is_vmappable` and
   dispatches to `arg._unwrap_batched(level)` for vmappable types.

6. **`_flatten_chunks_output`**: uses `is_leaf=_is_vmappable` so chunked vmap
   also works with custom types.

## API Stability of Protocol Methods

One thing I want to flag: the protocol currently asks implementors to define
private methods (`_add_batch_dim`, `_maybe_remove_batch_dim`,
`_unwrap_batched`). By Python convention, names starting with `_` are internal
and may change without notice in any release. That creates a real tension -- we're
asking third-party code to implement methods whose signatures PyTorch reserves
the right to change at any time, which would silently break downstream classes.

I see a few ways to address this:

1. **Make the methods public** (`add_batch_dim`, `maybe_remove_batch_dim`,
   `unwrap_batched`). This commits PyTorch to a stable signature, but it exposes
   low-level vmap internals (vmap levels, batch sizes) as public API, which
   feels undesirable.

2. **Add a thin public wrapper** that calls the private implementation. For
   example, vmap would call `obj.vmap_add_batch_dim(in_dim, vmap_level)` (public,
   stable signature), and the default implementation would forward to the
   internal `_add_batch_dim`. If the internal signature ever changes, only the
   wrapper needs updating. Third-party classes implement the public method and
   are shielded from internal churn.

3. **Keep the private names but document the signature as a contract.** This is
   the simplest path but offers no formal stability guarantee. Downstream
   libraries would need to pin to PyTorch versions or track changes manually.

My gut says option 2 is the right balance: a small public surface
(`vmap_add_batch_dim`, `vmap_remove_batch_dim`, `vmap_unwrap_batched`) with a
documented contract, while the internal plumbing remains free to evolve. But I'd
love to hear what you all think.

## Open Questions

1. **Experimental status**: should we mark this as experimental (e.g., under
   `torch.func.experimental` or behind a warning) to give ourselves room to
   iterate on the API before committing to backward compatibility?

2. **Chunked vmap**: `_get_chunked_inputs` calls `tensor_split` on flat args.
   Custom types would need to support `tensor_split` for chunked vmap to work.
   Should there be a protocol method for splitting?

3. **Public API surface**: should `register_vmappable_cls` be exposed in
   `torch.func` or remain in `torch._functorch.vmap`? Given the complexity of
   coding the entry points I suspect this should probably remain private for the
   time being.
