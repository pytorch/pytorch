# torch.compile support for custom_op

> STATUS: the package is scoped to `make_fx` (real + fake) only. The changes
> below were required to make `torch.compile` (dynamo -> aot_eager -> inductor)
> work and were **removed** from the code. This section records them so they can
> be re-applied. Everything below the "Representation" section is the earlier
> design narrative/log.

## Reconstruction: what torch.compile required (removed)

Verified working end-to-end for pure and mutating ops (aot_eager + inductor) with
these changes; each is here with its rationale.

1. **`functional.py` -- FunctionalTensorMode handler.** Reached from the
   overload's `_dispatch` when FunctionalTensorMode is active.
   ```python
   def functionalize_custom_op(op, mode, args, kwargs):
       from torch._higher_order_ops.auto_functionalize import (
           can_auto_functionalize, do_auto_functionalize)
       # Pop functional and pass the mode explicitly, as the C++ Functionalize
       # key does (redispatch_to_next is a nullcontext assuming it was popped).
       with _pop_mode_temporarily(torch._C._TorchDispatchModeKey.FUNCTIONAL):
           if can_auto_functionalize(op):        # mutating -> auto_functionalized HOP
               return do_auto_functionalize(mode, op, args, kwargs)
           api = PythonFunctionalizeAPI(mode)    # pure -> unwrap/redispatch/rewrap
           out = op(*api.unwrap_tensors(args), **api.unwrap_tensors(kwargs))
           return api.wrap_tensors(out)
   ```

2. **Overload `_dispatch` routes to functionalize** when FunctionalTensorMode is
   active (else defers to the CustomOp):
   ```python
   fmode = torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
   if fmode is not None:
       return functionalize_custom_op(self, fmode, args, kwargs)
   ```

3. **`flexible_layout` tag** on the overload: `super().__init__(..., schema,
   [torch.Tag.flexible_layout])`. The custom-op default `needs_exact_strides`
   routes Inductor through `constrain_to_fake_tensors`, which does
   `fake_kwargs[k]` and mismatches our node (kwargs-style call vs positional
   `eager_input_vals`; `normalize_function` echoes arrangement). Our impls are
   aten compositions (stride-agnostic), so opting out is correct.

4. **`_dispatch_has_kernel_for_dispatch_key` downgrade** in
   `install_global_hooks`: `can_auto_functionalize` queries the dispatcher by our
   unregistered op name and raises "operator does not exist"; wrap to return
   `False` on that `RuntimeError` (registered ops unaffected).

5. **Canonicalize the calling convention** in `trace_custom_op`:
   `bound = custom_op.sig.bind(*args, **kwargs); bound.apply_defaults();
   args, kwargs = bound.args, bound.kwargs`. Overload identity includes the
   `in_spec`, and dynamo calls positional while `auto_functionalized` calls
   kwargs -- without this they mint divergent overloads and Inductor references
   one not in the cache. RETAINED: identity stability matters for make_fx too
   (same logical call via different conventions must reuse one overload).

6. **Bare-FakeTensorMode branch** in `CustomOp.__call__` (dynamo fake-prop of a
   captured node, no proxy mode):
   ```python
   if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
       return self.fn(*args, **kwargs)
   ```

7. **Dispatcher-query neutralizations** on the overload (compile-stack code and
   `FallbackKernel` query these; unregistered -> "operator does not exist"):
   `_can_decompose`->False, `decompose`->NotImplemented,
   `has_kernel_for_dispatch_key`/`has_kernel_for_any_dispatch_key`->False,
   `redispatch`/`_handle`/`_get_dispatch`->raise. NOTE: do NOT hide
   `_overloadname`/`overload` -- inductor's `FallbackKernel` reads `_overloadname`.

8. **Overload identity round-trips as a real OpOverload** (so it survives as an
   `auto_functionalized` argument instead of collapsing to the packet):
   - `self.__name__ = f"{name}.{overload_name}"` (full overload name).
   - packet `__getattr__` resolves generated names back to the overload:
     `return self._custom_op.cache.get(key)`.
   - Because that makes `gm.code` show the mangled name, a `Graph.python_code`
     hook nicifies only the call-*target* position:
     `re.sub(r"(torch\.ops\.\w+\.\w+)\.pt_\w+\(", r"\1(", src)` (argument
     positions keep the real name so they still resolve to an OpOverload).
   - Dataclass args in `gm.code`: register the dataclass type via
     `torch.fx.graph._register_custom_builtin(cls.__name__, ...)` so its repr
     (`Box(t=...)`) resolves in generated-code globals. RETAINED: structured
     display renders dataclass pytree args in make_fx `gm.code` too.

The other items (1-4, 6, 7, and the `__name__`/`__getattr__`/regex identity
round-trip in item 8) are removed; make_fx only needs the packet-name spelling.

Dynamo frontend for pytree/dataclass *inputs* was NOT solved (dynamo can't
`as_proxy` a dataclass op-arg; needs a dedicated handler that flattens at
capture). Inductor mutating-op replay needed a clean disk cache (names are
deterministic, so stale pre-canonicalization caches referenced old overloads).

## Representation: flat identity, structured display (current)

Op nodes carry FLAT leaf args matching the leaf-only schema (BC: dynamo /
functionalize / inductor / serialize all treat them uniformly). Pytree structure
is metadata (`in_spec` on the overload) reconstructed ONLY at print sites:
- `str(gm.graph)`: `Node._pretty_print_target` (target) + a `Graph.__str__` hook
  that restructures args from `in_spec`.
- `gm.code`: a `Graph.python_code` hook that restructures args + nicifies the
  call-target spelling.
Overload identity is canonicalized via `sig.bind`, so `op(x, y)`, `op(x=x, y=y)`
map to the SAME overload (different layers call with different conventions --
dynamo positional, auto_functionalized kwargs -- so this prevents divergent
overloads). Same principle as the name: normal identity, pretty display.


## Goal

Make dispatch-less `custom_op` work under `torch.compile`. These ops never enter
the C++ dispatcher, so the dispatcher's automatic mode handling (Fake /
Functional / Proxy) does not happen for free. We reimplement it as a small
Python dispatcher in `CustomOp.__call__` that walks the active `TorchDispatchMode`
stack in the dispatcher's priority order (Functional > Proxy > Fake), handles the
top mode, and redispatches below it. Each mode integration lives in its own file.

## Files

- `custom_op.py` -- the router in `__call__` (detect top mode, delegate).
- `tracing.py`   -- ProxyTorchDispatchMode (already present; emits the FX node).
- `fake.py`      -- FakeTensorMode (run the fn so aten ops fake-propagate).
- `functional.py`-- FunctionalTensorMode (unwrap / redispatch / rewrap + mutations).

## Phases

### Phase 1 -- Fake + Proxy (done first; fully testable without compile)
- Router in `__call__`: Functional > Proxy > Fake > eager.
- `fake.py`: run the fn under the active fake mode -> fake outputs, no node.
- Drop the `tracing_mode == "real"` restriction; `trace_custom_op` already works
  for fake/symbolic because `disable_proxy_modes_tracing` leaves fake active.
- Tests: `make_fx(tracing_mode="fake"/"symbolic")`, bare `FakeTensorMode`.

### Phase 2 -- Functional  [DONE: pure and mutating ops compile via aot_eager]

Key fix (identity vs display): the overload now carries its **normal** qualified
name (`ns.name.overload`) so it round-trips as a real `OpOverload` -- crucial
when it appears as an *argument* to `auto_functionalized` (a packet-collapsed
name fails `isinstance(op, OpOverload)`). The nice packet-only spelling is
applied purely at print sites:
- `str(gm.graph)`: the `Node._pretty_print_target` hook.
- node names: an explicit clean `name=` at `create_proxy`.
- `gm.code`: a hook on `Graph.python_code` that nicifies only the call-*target*
  position (`ns.name.<overload>(` -> `ns.name(`); op-as-argument positions keep
  the real name, so `auto_functionalized` still sees an `OpOverload`.

Functionalization pops FunctionalTensorMode around `do_auto_functionalize`
(mirroring the C++ path) and relies on the `_dispatch_has_kernel_for_dispatch_key`
"does not exist" -> False patch. Both pure (pass-through) and mutating
(auto_functionalized) ops now compile through `aot_eager`.

--- (historical) earlier state: pure ops work; mutation is WIP ---
Design (matches torch.library ops), driven off the *concrete overload's* schema:
no mutation/alias -> pass through; mutates -> `auto_functionalized`; aliases ->
error. The blocker was that `can_auto_functionalize` / `do_auto_functionalize`
query the dispatcher by our unregistered name and raise "operator does not
exist". We unblock that with a narrow global patch in `install_global_hooks`:
`_dispatch_has_kernel_for_dispatch_key` now downgrades that specific
`RuntimeError` to `False` (registered ops unaffected).

Results:
- **Pure (non-mutating) ops functionalize and compile through `aot_eager`
  end-to-end.** `functional.py` unwraps, redispatches below FunctionalTensorMode
  (proxy builds the opaque node), and rewraps.
- **Mutating ops are BLOCKED by the overload<->packet identity collapse.**
  Fixed two layers: (a) pop FunctionalTensorMode around `do_auto_functionalize`
  (it assumes the mode is already popped, like the C++ path) -- this cleared the
  `output_tensor must not be a FunctionalTensor` error; (b) confirmed the op is
  the concrete overload at the call site. The remaining wall: `do_auto_
  functionalize` traces `auto_functionalized(op, ...)` with our overload as a
  *node argument*. Because the overload deliberately renders/resolves as the
  packet name (clean codegen), on execution that argument round-trips to the
  `DispatchlessOpOverloadPacket`, and `auto_functionalized`/`can_auto_
  functionalize` reject it (`is OpOverload: False` -> "Cannot auto-functionalize
  op"). So mutation-under-compile needs the overload to have a distinct,
  resolvable graph identity -- exactly what the clean-codegen trick removes and
  what a real dispatcher registration restores.
- FunctionalTensorMode is handled on the concrete overload's `__call__` (it needs
  that overload's schema), reached via the graph node during AOT. The packet's
  own router does not handle functional (direct packet-under-bare-functional is
  not a supported path).

### Phase 3 -- End-to-end torch.compile
Layer-by-layer spike results:

1. **Dynamo -- WORKS.** Dynamo recognizes a `DispatchlessOpOverload` and emits a
   `call_function` node, then fake-propagates it. This required the overload's
   `__call__` to route through the CustomOp mode dispatcher (so fake-prop runs
   the fn -> fake outputs) instead of hard-raising. `torch.compile(backend=
   "eager", fullgraph=True)` now works when a concrete overload is called.
   - Open gap for the packet spelling: `is_compiling()` can route the packet
     call to the overload, but producing the *concrete overload* needs the
     schema, which today comes from running the fn. That can't happen during
     dynamo tracing (no real inputs) without either eager warmup that
     pre-creates the overload, or an up-front schema. Precomputed overloads work.
2. **AOTAutograd (aot_eager):** pure ops **work** (Phase 2). Mutating ops WIP
   (re-entrancy in auto_functionalize).
3. **Inductor: pure AND mutating ops work end-to-end.**
   - Pure ops go through `FallbackKernel`. This reads `op._overloadname`, so the
     over-aggressive `__getattribute__` guard was removed -- identity attrs are
     accessible like a real overload; only print sites pretty-print.
   - Mutating ops go through `auto_functionalized`. They first hit
     `lowering.py:constrain_to_fake_tensors` doing `fake_kwargs['x']` ->
     `KeyError`, because our op defaults to the `needs_exact_strides` layout tag
     and Inductor's normalize step doesn't reconcile the node's kwargs-style call
     (`auto_functionalized_dense` does `_mutable_op(**new_kwargs)`) with the
     positional `eager_input_vals` (`normalize_function` echoes arrangement for
     our op instead of canonicalizing). Fix: tag the overload
     `torch.Tag.flexible_layout` -- correct for our aten-composition impls
     (stride-agnostic) and it skips that path entirely.
   (Local env note: this machine's g++ rejects the auto-detected `-march=armv9`;
   the inductor test sets `torch._inductor.config.cpp.simdlen = 0`, which just
   disables SIMD and is harmless elsewhere.)

Net: the full stack -- eager, make_fx (real/fake/symbolic), dynamo, aot_eager,
and inductor -- works for pure and mutating dispatch-less custom ops. Remaining
caveats: the narrow `_dispatch_has_kernel_for_dispatch_key` "does not exist" ->
False global patch, and the "produce the overload under compile from the packet
spelling" ergonomics (dynamo works when a concrete overload is called).

Net: dynamo and pure-op AOTAutograd work purely in Python (plus the narrow
has_kernel patch). Remaining: mutation re-entrancy, inductor execution, and the
"produce the overload under compile" problem for the packet spelling. The
registered-op route (`dispatch=False`) would make all of these behave like a
normal custom op; the pure-Python + monkey-patch route is viable but is
accumulating patches at each compile layer.

### Phase 3 -- End-to-end torch.compile (spike, not assumed working)
Two remaining obstacles, investigated separately:
1. Dynamo frontend: the packet call must become a graph node, not get traced
   into or cause a graph break.
2. Inductor execution: a dispatch-less op has no kernel to call and no
   decomposition, so codegen has nothing to emit. Likely needs a fallback that
   calls back into the Python packet, or a `dispatch=False` C++ registration
   used only for compilation. Spike `FallbackKernel` with an unregistered op
   before investing further.
