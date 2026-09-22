# custom_op Follow-ups

- [x] Replace the global `OpOverloadPacket` monkey patches with a `DispatchlessOpOverloadPacket` subclass.
  - The subclass is constructed directly in Python (it skips `super().__init__()`) and never registers with the dispatcher.
  - `__call__` routes packet calls like `torch.ops.ns.foo(pytree_arg)` back to the `CustomOp`'s Python function.
  - `__getattr__` resolves generated overload names like `pt_...` by decoding them (materializing the overload on demand).
  - The packet is assigned onto `torch.ops.<ns>.<name>` so generated `gm.code` keeps using the clean packet spelling.

- [x] Keep real `OpOverload` targets in the FX graph.
  - Pattern matching and schema/alias/mutation semantics use concrete generated `DispatchlessOpOverload` objects.
  - Graph targets are not replaced with a pure Python packet object.

- [x] Subclass instead of monkey-patching `OpOverload`.
  - `op._overloadname` / `op.overload` stay accessible (inductor's `FallbackKernel` reads `_overloadname`).
  - `name()` returns the public `ns::foo` spelling; `str(op)` / `repr(op)` stay stable for target counting and debugging.
  - `_pretty_print()` supplies the `[*]` graph spelling via an FX display hook.

- [x] Replace `mutates_args` with a path-based specifier API.
  - Identifies exactly which tensor leaves are mutated inside structured arguments (see `ArgPath`).
  - Supports paths through positional/keyword arguments, lists, tuples, dicts, and dataclasses.
  - Example shape: `mutates_args=("state['buf']", "buffers[0]", "box.tensor")`.
  - Resolved paths drive the alias/mutation annotations in the flattened leaf schema.
  - Keep an escape hatch for `mutates_args="unknown"` when precise mutation paths are not available.

- [x] Keep the graph-printing hook separate from executable codegen.
  - `str(gm.graph)` shows the pytree marker, e.g. `target=torch.ops.ns.foo[*]`.
  - `gm.code` remains executable and prints `torch.ops.ns.foo(...)` without `[*]`.
  - The `torch.fx.Node._pretty_print_target` hook is kept graph-only.

- [x] Add focused tests for the packet subclass behavior.
  - `torch.ops.ns.foo([[x, y], z])` runs through the `DispatchlessOpOverloadPacket`.
  - `torch.ops.ns.foo.pt_...` resolves (materializes) the overload; unknown/malformed names raise `AttributeError`.
  - `debug_overload_name(op)` is the explicit debug escape hatch (a test helper).
  - `str(gm.graph)` includes `[*]` and nested pytree args.
  - `gm.code` remains executable and omits `[*]`.
