"""Runtime driver for torch.compiler.precompile artifacts, authored as real code.

Nothing here is imported or run by torch at runtime, and the generated artifacts never
import this module. Instead torch._precompile emits these function bodies VERBATIM (via
inspect.getsource, on the to_python_code / emit path only) into the self-contained
python_code string, after the calling-convention metadata and the compiled/captured
graph (``_import_custom_op_modules`` goes before them, as section 0). Authoring the
driver as real code -- instead of a hand-written string literal -- lets pyrefly / ruff
/ IDEs type-check and navigate the load-bearing driver logic that would otherwise be
invisible inside a string (and drops the wall of ``# noqa: F821``).

Keeping python_code self-contained and version-frozen (its behavior is hashed via
code_hash, invariant 7) still holds: the artifact carries the driver TEXT, it does not
import it, so there is no torch-version skew. The emit path runs getsource in-process
where torch source is present; load() never touches this module.

The names the emitted bodies read from the artifact's own namespace -- the metadata
constants, the ``_torch`` / ``_pytree`` import aliases, and the graph's ``call`` -- are
declared under TYPE_CHECKING below so the bodies type-check here; at emit time they
resolve against the metadata + graph sections that precede the driver in python_code.

INVARIANT: ``_extract_param_buffers`` reproduces
``torch._precompile._intern_param_buffers``'s params-then-buffers, intern-by-identity
ordering VERBATIM; the two must stay in sync (see Note [precompile programming model]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch as _torch
import torch.utils._pytree as _pytree


if TYPE_CHECKING:
    # Calling-convention metadata the emitted driver reads from the artifact namespace,
    # where torch._precompile._build_metadata_section emits each as a literal assignment
    # ahead of the driver. Bound here with placeholder values (not bare annotations) so
    # static tools treat them as real names in the bodies below; this block is not emitted.
    MODULE_POSITIONS: list[int] = []
    # Multi-graph artifact state. _FRAMES is one record per Dynamo frame (its
    # code, its variants' guard state and transformed bytecode, the globals it
    # reads), _BACKENDS the compiled subgraphs, _ENTRY_BINDING the entry's
    # default arguments; the two versions lock the serialized code objects and the
    # pickled guard state to the interpreter and torch build that produced them.
    _FRAMES: str = ""
    _BACKENDS: str = ""
    _ENTRY_BINDING: str = ""
    # An installed artifact carries the pickled Dynamo package and its backend
    # artifacts instead of _FRAMES/_BACKENDS/_ENTRY_BINDING.
    _PACKAGE: str = ""
    # Each Inductor graph's output code, as editable text, keyed f"{backend_id}.fw" or
    # ".bw", and the sha256 of the text as captured. Artifacts predating these keep the
    # code inside _BACKENDS.
    GRAPH_SOURCES: dict[str, str] = {}
    _GRAPH_SOURCE_SHA256: dict[str, str] = {}
    # False as emitted; load(allow_serve_time_compiles=True) flips it to True.
    _ALLOW_SERVE_TIME_COMPILES: bool = False
    BACKEND: str = ""
    FN_NAME: str = ""
    ENTRY_MODULE: str = ""
    _DYNAMO_PYTHON_VERSION: tuple[int, int] = (0, 0)
    TORCH_VERSION: str = ""
    SERVED_MODULES: tuple[str, ...] = ()
    # Section 0 of an artifact that calls custom ops.
    CUSTOM_OP_MODULES: tuple[str, ...] = ()
    CUSTOM_OPS: tuple[tuple[str, str], ...] = ()
    CUSTOM_OP_KERNELS: dict[str, tuple[str, ...]] = {}
    CUSTOM_OP_OVERRIDE_MODULES: dict[str, dict[str, str]] = {}
    NUM_POSITIONAL_ARGS: int = 0
    PARAM_NAMES: list[str] = []
    BUFFER_NAMES: list[str] = []
    PARAM_SHAPES: list[tuple[int, ...]] = []
    BUFFER_SHAPES: list[tuple[int, ...]] = []
    PARAM_DTYPES: list[str] = []
    BUFFER_DTYPES: list[str] = []
    PARAM_DEVICES: list[str] = []
    BUFFER_DEVICES: list[str] = []
    GRAD_PARAM_INDICES: list[int] = []
    IN_SPEC: str | None = None
    OUT_SPEC: str = ""
    USER_INPUT_SHAPES: list[tuple[int | None, ...] | None] = []
    USER_INPUT_DTYPES: list[str | None] = []
    USER_INPUT_DEVICES: list[str | None] = []
    USER_INPUT_BOUNDS: list[dict[int, tuple[int | None, int | None]] | None] = []

    # The compiled/captured graph's entry point, emitted before the driver.
    def call(flat_inputs: list[object]) -> list[object]: ...


def _import_custom_op_modules(served=(), import_served=None):
    """Import the modules registering the custom ops the graphs call, then check
    that every op in CUSTOM_OPS and the CUSTOM_OP_KERNELS it had at capture are
    registered, and that every module in CUSTOM_OP_OVERRIDE_MODULES imported and
    still registers the active kernel it did at capture. Emitted as section 0 and
    called at exec time. The modules in ``served`` are imported with
    ``import_served``, which fails the load if they do not import."""
    import contextlib
    import importlib
    import importlib.util
    import os
    import re
    import sys
    import warnings

    import torch

    import_errors = {}

    # find_spec imports a dotted name's parents, so this follows the ordered imports.
    # None when the lookup raised, which import_served then reports.
    def found(module):
        try:
            return module != "__main__" and (
                module in sys.modules or importlib.util.find_spec(module) is not None
            )
        except ModuleNotFoundError:
            return False
        except Exception:
            return None

    # The entry may register a served module find_spec cannot locate (a plugin
    # loader), so a module not found yet is looked up again after the entry imports.
    # The module's own import then reports a failure, not the entry's.
    def import_or_entry(module, import_served):
        if found(module) is False and ENTRY_MODULE in served:
            with contextlib.suppress(Exception):
                import_served(ENTRY_MODULE)
        import_served(module)

    # A module imported from inside another finished first, so the order alone
    # would import an override nested in the module it overrides before that one.
    owners = {m for o in CUSTOM_OP_OVERRIDE_MODULES.values() for m in o.values()}
    for module in sorted(CUSTOM_OP_MODULES, key=lambda m: m in owners):
        if import_served is not None and module in served:
            import_or_entry(module, import_served)
            continue
        try:
            importlib.import_module(module)
        # Not just ImportError: e.g. importing torch.distributed.device_mesh before
        # torch.distributed.tensor raises ValueError.
        except Exception as e:
            import_errors[module] = e
    # A served module that does not import fails the load here, not at the op
    # check, which would report its ops missing. A __main__ entry is refused
    # before section 0 runs.
    for module in served:
        if not found(module) and import_served is not None:
            import_or_entry(module, import_served)
    missing = []
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
    for name, overload in CUSTOM_OPS:
        qualname = f"{name}.{overload}" if overload else name
        try:
            torch._C._dispatch_find_schema_or_throw(name, overload)
        except RuntimeError:
            missing.append(qualname)
            continue
        for key in CUSTOM_OP_KERNELS.get(qualname, ()):
            if not has_kernel(qualname, key):
                missing.append(f"{qualname} ({key} kernel)")
    # Another module's kernel under the same key would pass the check above, so
    # the active kernel must be registered from the overriding module's file.
    overridden = []
    real = os.path.realpath
    override_errors = set()
    for qualname, owners in CUSTOM_OP_OVERRIDE_MODULES.items():
        dump = torch._C._dispatch_dump(qualname)
        for key, module in owners.items():
            pattern = rf"^{key}(?:\[alias\])?: registered at (.+):\d+ ::"
            active = re.search(pattern, dump, re.MULTILINE)
            path = getattr(sys.modules.get(module), "__file__", None)
            if module in import_errors:
                override_errors.add(module)
            elif not (active and path and real(active[1]) == real(path)):
                overridden.append(f"{qualname} ({key} kernel of {module})")
    if missing or override_errors or overridden:
        # Lazily, as _fail does; _fail is emitted after section 0 and cannot chain.
        from torch._precompile import PrecompileError as _PrecompileError

        errors = [import_errors[m] for m in sorted(override_errors)]
        raise _PrecompileError(
            f"this artifact calls custom ops or kernels not registered here: "
            f"{missing or 'none'}. Modules whose kernels overrode another kernel "
            f"for the same key at capture that failed to import: "
            f"{sorted(override_errors) or 'none'}. Overriding kernels it ran at "
            f"capture that another kernel replaced: {overridden or 'none'}. "
            "Import the modules registering them, and register no other kernel "
            "over theirs, before loading it. Modules that failed to import: "
            f"{sorted(import_errors) or 'none'}."
        ) from next(iter(errors or import_errors.values()), None)
    for module, e in import_errors.items():
        warnings.warn(
            f"importing custom op module {module!r} raised {e!r}; none of its "
            "kernels overrode another for the same key, and the ops this artifact "
            "calls and their kernel keys are registered anyway, so it loads"
        )


def _extract_param_buffers(mods):
    """Lift the runtime modules' params then buffers, interning by identity, in the
    same order as capture, so the list lines up with the compiled/captured graph. Returns
    (pb, names) where names mirrors PARAM_NAMES + BUFFER_NAMES. This ordering AND the
    naming must match torch._precompile._intern_param_buffers verbatim (its INVARIANT)."""
    multi = len(mods) > 1
    seen = set()
    pb = []
    names = []

    def intern(mi, n, t):
        if id(t) not in seen:
            seen.add(id(t))
            pb.append(t)
            names.append(f"m{mi}.{n}" if multi else n)

    for mi, m in enumerate(mods):
        for n, p in m.named_parameters(remove_duplicate=False):
            intern(mi, n, p)
    for mi, m in enumerate(mods):
        for n, b in m.named_buffers(remove_duplicate=False):
            intern(mi, n, b)
    return pb, names


def _fail(msg):
    # Imported lazily (only when a guard fails) so a normal run does not couple the
    # standalone artifact to torch._precompile's import surface.
    from torch._precompile import PrecompileError as _PrecompileError

    raise _PrecompileError(msg)


def _check_structure(pb, names):
    # Verify the runtime model's extracted param/buffer NAMES match the baked
    # PARAM_NAMES + BUFFER_NAMES (count AND order/identity), so a reordered or
    # structurally-drifted same-count model is caught precisely (invariant 2) rather
    # than scattering grads onto the wrong slot. Then check each tensor's SHAPE, DTYPE and
    # DEVICE against the baked example: the graph is specialized to the example shapes and
    # can bake a device literal, so a same-named but differently shaped/typed/placed
    # runtime tensor would silently miscompute or fail deep in a kernel.
    expected = list(PARAM_NAMES) + list(BUFFER_NAMES)
    if names != expected:
        _fail(
            f"precompile: the runtime model's param/buffer names {names!r} do not match "
            f"the traced model's {expected!r}; the runtime model must be structurally "
            f"identical to the traced model (invariant 2)."
        )
    expected_shapes = list(PARAM_SHAPES) + list(BUFFER_SHAPES)
    expected_dtypes = list(PARAM_DTYPES) + list(BUFFER_DTYPES)
    expected_devices = list(PARAM_DEVICES) + list(BUFFER_DEVICES)
    for _nm, _t, _shp, _dt, _dev in zip(
        names, pb, expected_shapes, expected_dtypes, expected_devices
    ):
        if tuple(_t.shape) != tuple(_shp):
            _fail(
                f"precompile: the runtime param/buffer {_nm!r} has shape "
                f"{tuple(_t.shape)} but the traced model's was {tuple(_shp)}; the runtime "
                f"model must be structurally identical to the traced model (invariant 2)."
            )
        if str(_t.dtype) != _dt:
            _fail(
                f"precompile: the runtime param/buffer {_nm!r} has dtype {_t.dtype} but "
                f"the traced model's was {_dt}; the runtime model must be structurally "
                f"identical to the traced model (invariant 2)."
            )
        if str(_t.device) != _dev:
            _fail(
                f"precompile: the runtime param/buffer {_nm!r} is on device {_t.device} "
                f"but the traced model's was {_dev}; the runtime model must be "
                f"structurally identical to the traced model (invariant 2)."
            )


def _eager_forward(*args):
    """Run the captured ATen graph eagerly. Pass the same args the traced fn took --
    the module(s) in the same positions plus the runtime inputs. The module(s) must
    be structurally identical to the ones precompile traced (same param/buffer order
    and tying); only the weight values may differ.

    The eager backend runs the graph as captured: inputs (including tensor
    subclasses) are passed through unchanged (no dense flatten/unflatten), and the
    graph's flat outputs are reassembled into fn's output structure. If fn ran a
    backward, the trailing grad outputs (one per GRAD_PARAM_INDICES entry) are
    parameter grads, scattered (accumulated) onto the params that received one like
    eager .backward() -- frozen / non-contributing params keep .grad = None."""
    if len(args) != NUM_POSITIONAL_ARGS:
        _fail(
            f"precompile: expected {NUM_POSITIONAL_ARGS} positional args (the same as "
            f"the traced fn), got {len(args)} (invariant 2)."
        )
    mods = []
    for _i in MODULE_POSITIONS:
        if not isinstance(args[_i], _torch.nn.Module):
            _fail(
                f"precompile: argument at position {_i} must be the nn.Module the traced "
                f"fn took (invariant 2), got {type(args[_i]).__name__}."
            )
        mods.append(args[_i])
    user_inputs = [a for i, a in enumerate(args) if i not in set(MODULE_POSITIONS)]
    user_flat, _runtime_in_spec = _pytree.tree_flatten(tuple(user_inputs))
    if IN_SPEC is not None and _runtime_in_spec != _pytree.treespec_loads(IN_SPEC):
        _fail(
            "precompile: runtime inputs have a different structure than the traced "
            "example inputs (invariant 3); they must match in nesting and count."
        )
    # Reject a SHAPE / DTYPE / DEVICE mismatch (invariants 3 and 6) up front. Mirrors the
    # inductor driver checks (keep the two drivers in sync). The eager backend has no
    # assert_size_stride, so only these are checked (layout-flexible). The eager backend
    # rejects mark_unbacked up front, so every dim here is static and USER_INPUT_BOUNDS is
    # always all-None; there is no bounds branch (it would be dead code). USER_INPUT_BOUNDS
    # is still emitted in the metadata for the inductor driver, so it is not consumed here.
    if len(user_flat) != len(USER_INPUT_SHAPES):
        _fail(
            "precompile: runtime inputs flattened to a different number of leaves than "
            "the traced example (invariant 3); they must match the traced structure."
        )
    for _t, _shp, _dt, _dev in zip(
        user_flat, USER_INPUT_SHAPES, USER_INPUT_DTYPES, USER_INPUT_DEVICES
    ):
        if _shp is None or not isinstance(_t, _torch.Tensor):
            continue
        _act = tuple(_t.shape)
        if len(_act) != len(_shp) or any(a != e for a, e in zip(_act, _shp)):
            _fail(
                f"precompile: a runtime input has shape {_act} but the artifact was "
                f"traced with shape {tuple(_shp)}; the graph is specialized to the static "
                f"dims (invariant 3). Retrace for this shape, or use backend='eager'."
            )
        if _dt is not None and str(_t.dtype) != _dt:
            _fail(
                f"precompile: a runtime input has dtype {_t.dtype} but the artifact was "
                f"traced with dtype {_dt}; the graph is specialized to the example dtype "
                f"(invariant 6). Cast the input to the traced dtype, or retrace."
            )
        if _dev is not None and str(_t.device) != _dev:
            _fail(
                f"precompile: a runtime input is on device {_t.device} but the artifact "
                f"was traced on device {_dev}; the graph is specialized to the example "
                f"device (invariant 6). Move the input to the traced device, or retrace."
            )
    pb, _names = _extract_param_buffers(mods)
    _check_structure(pb, _names)
    # The casts the capture ran under are baked into the graph, which still
    # re-dispatches here, so ambient autocast must not cast a second time. Like
    # the no_grad, this pins autocast off regardless of capture or serve state.
    with _torch._C._DisableAutocast(), _torch.no_grad():
        out = list(call([*pb, *user_flat]))
    if GRAD_PARAM_INDICES:
        n = len(GRAD_PARAM_INDICES)
        grads = out[len(out) - n :]
        out = out[: len(out) - n]
        for idx, g in zip(GRAD_PARAM_INDICES, grads):
            p = pb[idx]
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)
    return _pytree.tree_unflatten(out, _pytree.treespec_loads(OUT_SPEC))


def _inductor_forward(*args):
    """Run the compiled computation. Pass the same args the traced fn took -- the
    module(s) in the same positions plus the runtime inputs. The module(s) must be
    structurally identical to the ones precompile traced (same param/buffer order
    and tying); only the weight values may differ.

    Module params/buffers are extracted (no weights are baked into the artifact) and,
    together with the runtime inputs, passed to the composed ``call`` -- which is the
    AOTAutograd+Inductor graph with its own prelude/epilogue, so it handles tensor-
    subclass wrap/unwrap and input mutation (e.g. BatchNorm running stats) internally
    and disables grad itself. If fn ran a backward, the trailing grad outputs (one per
    GRAD_PARAM_INDICES entry) are parameter grads: they are scattered (accumulated)
    onto the params that received one, mirroring eager .backward() (frozen /
    non-contributing params keep .grad = None), and the artifact returns fn's own
    result. Nothing here reads an external cache: the kernels JIT-compile from the
    inlined source on first call. A runtime input whose shape, dtype, or device differs
    from the traced example is rejected up front (invariants 3 and 6), and a differing
    stride / memory format is rejected via the inlined assert_size_stride (invariant 6);
    use backend="eager" for layout-flexible execution."""
    if len(args) != NUM_POSITIONAL_ARGS:
        _fail(
            f"precompile: expected {NUM_POSITIONAL_ARGS} positional args (the same as "
            f"the traced fn), got {len(args)} (invariant 2)."
        )
    mods = []
    for _i in MODULE_POSITIONS:
        if not isinstance(args[_i], _torch.nn.Module):
            _fail(
                f"precompile: argument at position {_i} must be the nn.Module the traced "
                f"fn took (invariant 2), got {type(args[_i]).__name__}."
            )
        mods.append(args[_i])
    user_inputs = [a for i, a in enumerate(args) if i not in set(MODULE_POSITIONS)]
    user_flat, _runtime_in_spec = _pytree.tree_flatten(tuple(user_inputs))
    if IN_SPEC is not None and _runtime_in_spec != _pytree.treespec_loads(IN_SPEC):
        _fail(
            "precompile: runtime inputs have a different structure than the traced "
            "example inputs (invariant 3); they must match in nesting and count."
        )
    # Reject a SHAPE / DTYPE / DEVICE / BOUNDS mismatch (invariants 3 and 6) up front.
    # Mirrors the eager driver checks (keep the two drivers in sync). Stride/memory-format
    # is enforced by the inlined assert_size_stride (pinned at capture).
    if len(user_flat) != len(USER_INPUT_SHAPES):
        _fail(
            "precompile: runtime inputs flattened to a different number of leaves than "
            "the traced example (invariant 3); they must match the traced structure."
        )
    for _t, _shp, _dt, _dev, _bnd in zip(
        user_flat,
        USER_INPUT_SHAPES,
        USER_INPUT_DTYPES,
        USER_INPUT_DEVICES,
        USER_INPUT_BOUNDS,
    ):
        if _shp is None or not isinstance(_t, _torch.Tensor):
            continue
        # A dim recorded as None was captured dynamic (unbacked); any size is valid.
        _act = tuple(_t.shape)
        if len(_act) != len(_shp) or any(
            e is not None and a != e for a, e in zip(_act, _shp)
        ):
            _fail(
                f"precompile: a runtime input has shape {_act} but the artifact was "
                f"traced with shape {tuple(_shp)} (None = a dynamic dim, any size); the "
                f"graph is specialized to the static dims (invariant 3). Retrace, mark "
                f"the dim dynamic via mark_unbacked, or use backend='eager'."
            )
        if _dt is not None and str(_t.dtype) != _dt:
            _fail(
                f"precompile: a runtime input has dtype {_t.dtype} but the artifact was "
                f"traced with dtype {_dt}; the graph is specialized to the example dtype "
                f"(invariant 6). Cast the input to the traced dtype, or retrace."
            )
        if _dev is not None and str(_t.device) != _dev:
            _fail(
                f"precompile: a runtime input is on device {str(_t.device)!r} but the "
                f"artifact was traced on device {_dev!r}; the graph is specialized to the "
                f"example device (invariant 6). Move the input to the traced device, or "
                f"retrace."
            )
        if _bnd is not None:
            for _d, (_lo, _hi) in _bnd.items():
                _sz = _t.shape[_d]
                if _lo is not None and _sz < _lo:
                    _fail(
                        f"precompile: runtime input dim {_d} has size {_sz} but "
                        f"mark_unbacked declared min={_lo} (invariant 3)."
                    )
                if _hi is not None and _sz > _hi:
                    _fail(
                        f"precompile: runtime input dim {_d} has size {_sz} but "
                        f"mark_unbacked declared max={_hi} (invariant 3)."
                    )
    pb, _names = _extract_param_buffers(mods)
    _check_structure(pb, _names)
    try:
        # A fallback op re-dispatches through its autocast-registered overload
        # (e.g. aten.addbmm.default on CPU), so ambient autocast must not cast
        # a second time over the casts the capture baked into the kernels.
        with _torch._C._DisableAutocast():
            out = list(call([*pb, *user_flat]))
    except AssertionError as _e:
        # Only relabel inductor's own assert_size_stride failure (a stride/memory-format
        # mismatch, or a size mismatch on an unbacked dim the static check above cannot
        # pre-validate; invariants 3 and 6). assert_size_stride raises one of two messages
        # -- "expected size A==B, stride C==D at dim=N" or "wrong number of dimensions" --
        # so match those. Any OTHER AssertionError (a user torch._assert, an internal
        # inductor invariant) is re-raised unchanged so its real message is not mislabeled.
        _m = str(_e)
        if not (
            ("expected size" in _m and "stride" in _m)
            or "wrong number of dimensions" in _m
        ):
            raise
        # When the artifact has dynamic (None) user-input dims, an "expected size"
        # assert_size_stride failure on a dynamic dim most likely means two inputs that
        # share a mark_unbacked shape_id (bound to ONE symbol, hence equal by
        # construction) were called with mismatched sizes. Call that out so the message
        # is not misleadingly only about memory format.
        _has_dynamic = any(
            _s is not None and any(_d is None for _d in _s) for _s in USER_INPUT_SHAPES
        )
        _shape_id_note = ""
        if _has_dynamic and "expected size" in _m:
            _shape_id_note = (
                " If two inputs share a mark_unbacked shape_id, their marked dims are "
                "bound to one symbol and so MUST have equal sizes at runtime; this can "
                "also be a shape_id equality violation."
            )
        _fail(
            f"precompile: a runtime tensor's shape or memory format differs from the "
            f"traced example; the inductor backend specializes on input shape and memory "
            f"format (invariants 3 and 6). The mismatch can be a user INPUT or a model "
            f"PARAMETER/BUFFER whose layout (memory format) differs from the example "
            f"weight, since the inductor backend also bakes each param/buffer's layout. "
            f"Pass the model/inputs in the example's shape and layout (.contiguous() to "
            f"match a contiguous example, or match the example weight's layout), or use "
            f"backend='eager'.{_shape_id_note} Underlying: {_e}"
        )
        # Unreachable: _fail always raises. The bare re-raise keeps `out` provably bound
        # for static tools (which do not model _fail as NoReturn) and re-raises the
        # original assert if _fail were ever changed to return.
        raise
    if GRAD_PARAM_INDICES:
        n = len(GRAD_PARAM_INDICES)
        grads = out[len(out) - n :]
        out = out[: len(out) - n]
        for idx, g in zip(GRAD_PARAM_INDICES, grads):
            p = pb[idx]
            if p.grad is None:
                p.grad = g
            else:
                p.grad.add_(g)
    return _pytree.tree_unflatten(out, _pytree.treespec_loads(OUT_SPEC))


def _build_multigraph_forward():
    """Reconstruct a multi-graph artifact and return the runnable ``forward``.

    A capture with graph breaks or several specializations is not one graph, so
    unlike the single-graph drivers this one has to DISPATCH. Dynamo produced,
    per frame, one transformed bytecode plus one guard tree per variant; the
    artifact carries those verbatim (_FRAMES) and this rebuilds them into a
    dispatcher per frame.

    Deliberately standalone: nothing is installed onto the running program's
    code objects and no frame evaluator is involved. A frame's dispatcher
    evaluates each variant's guards against the arguments it was handed and
    calls the first that matches, so an artifact serves only what it captured.
    The rebuilt bytecode runs, and the guards check, the LIVE globals of the
    module each frame was compiled in, as CompilePackage.install has it; the
    only names bound there are ones Dynamo minted while tracing: that module's
    import aliases and builtins dict key, the artifact's compiled subgraphs under
    their backend ids (every one, into each module opened) and the resume
    functions, bound the way install binds them. A continuation is reached the
    way Dynamo emits it -- the entry bytecode does LOAD_GLOBAL on the resume
    name -- so binding the resume dispatcher under that name in the module is
    all the wiring the graph-break path needs. A resume name held there by
    anything but this artifact (another standalone artifact, a live compile, a
    user binding) refuses the load before anything is seeded; a load refused
    for another reason partway through leaves the names it had already seeded.

    Because there is no compiler behind a source artifact, an uncovered call
    RAISES rather than falling back. That is the point: the artifact serves the
    calls it captured, and anything else is a coverage gap the caller has to
    hear about. The exception is a frame Dynamo gave up on during the capture
    (eager_fallback): the capture ran its other calls eagerly, so a miss of it
    runs eagerly here too, with a warning once per frame.
    """
    import base64
    import hashlib
    import importlib
    import inspect
    import pickle
    import sys as _sys
    import types

    import torch
    from torch._dynamo.aot_compile import bind_locals
    from torch._dynamo.output_graph import get_builtins_dict
    from torch._dynamo.package import (
        load_guard_manager,
        load_guards_state,
        SerializedCode,
    )
    from torch._dynamo.utils import CleanupHook

    # The documented contract (Note [precompile programming model]) is that the
    # version/build locks surface as a clean PrecompileError, so an
    # ``except torch.compiler.precompile.PrecompileError`` handler catches them.
    from torch._precompile import PrecompileError as _PrecompileError

    if tuple(_DYNAMO_PYTHON_VERSION) != _sys.version_info[:2]:
        # SerializedCode.to_code_object feeds the recorded co_* fields to
        # types.CodeType in the running interpreter's get_code_keys() order, so a
        # foreign artifact raises a bare TypeError or runs foreign bytecode.
        raise _PrecompileError(
            f"precompile: this artifact was produced on Python "
            f"{_DYNAMO_PYTHON_VERSION[0]}.{_DYNAMO_PYTHON_VERSION[1]} and cannot "
            f"load on {_sys.version_info[0]}.{_sys.version_info[1]}: it inlines "
            f"serialized code objects whose bytecode and field layout are "
            f"Python-version-locked. Regenerate the artifact under the serving "
            f"Python."
        )
    if TORCH_VERSION != torch.__version__:
        # _FRAMES carries pickled Dynamo guard state, which no other torch build
        # promises to unpickle; the import check below cannot see a build whose
        # modules all still import.
        raise _PrecompileError(
            f"precompile: this artifact was produced by torch {TORCH_VERSION} and "
            f"cannot load on torch {torch.__version__}: it inlines pickled Dynamo "
            f"guard state, which is torch-build-locked. Regenerate the artifact "
            f"under the serving torch."
        )
    # After the version checks: a torch without it fails them cleanly first.
    from torch._dynamo.eval_frame import _warn_eager_fallback_miss

    def _import(module_name):
        if module_name == "__main__":
            # import_module("__main__") succeeds in every process and hands back
            # whatever script is running, whose globals are not the capturing
            # script's; the guards would check the wrong objects silently.
            raise _PrecompileError(
                "precompile: this artifact was captured from a function defined "
                "in the capturing script's __main__ module, which no other "
                "process can import. Regenerate the artifact from a function "
                "defined in an importable module."
            )
        try:
            return importlib.import_module(module_name)
        # Not just ImportError, as in section 0.
        except Exception as _e:
            hint = ""
            if isinstance(_e, (ImportError, AttributeError)):
                # The torch-build / environment lock, surfaced per the documented
                # contract rather than as a raw ModuleNotFoundError.
                hint = (
                    " It was produced against a different torch build or "
                    "environment; regenerate the artifact in this one."
                )
            raise _PrecompileError(
                f"precompile: this artifact references module {module_name!r}, "
                f"which is not importable here ({_e}).{hint}"
            ) from _e

    if "_import_custom_op_modules" in globals():
        # Section 0 also lists them in case they register its ops; their failure
        # fails the load, so it is not warned about as a custom op module's.
        if "__main__" in SERVED_MODULES:
            _import("__main__")  # Refused before section 0 imports anything.
        _import_custom_op_modules(SERVED_MODULES, _import)
    frames = pickle.loads(base64.b64decode(_FRAMES))
    backends = pickle.loads(base64.b64decode(_BACKENDS))
    entry_binding = pickle.loads(base64.b64decode(_ENTRY_BINDING))
    # The graph code runs from GRAPH_SOURCES, edits included. An edited graph gets
    # its own PyCodeCache key and compiles the kernels its text defines, instead of
    # binding the ones precompiled from the captured text.
    _sha256 = globals().get("_GRAPH_SOURCE_SHA256", {})
    graph_sources = globals().get("GRAPH_SOURCES", {})
    for _key, _source in graph_sources.items():
        _backend_id, _, _part = _key.rpartition(".")
        _graph = getattr(backends[_backend_id].content, f"compiled_{_part}").result
        _digest = hashlib.sha256(_source.encode()).hexdigest()
        if _digest != _sha256.get(_key):
            _graph.cache_key = f"e{_digest}"
            _graph.kernel_sources = {}
            # Maps the captured text's lines to FX nodes, not the edited text's.
            _graph.cache_linemap = None
        _graph.source_code = _source
    compiled = {
        _backend_id: torch._dynamo.disable(_artifact.after_deserialization())
        for _backend_id, _artifact in backends.items()
    }
    # Resume names come from a per-process counter (backend ids carry a uuid), so
    # another artifact or a live compile can hold one this artifact mints; the
    # tag hashes both channels so two artifacts of one capture differ, and the
    # graph code, so an edited copy of an artifact is another version of it.
    graph_code = "".join(f"{k}\0{v}\0" for k, v in sorted(graph_sources.items()))
    tag = hashlib.sha256((_FRAMES + _BACKENDS + graph_code).encode()).hexdigest()[:16]
    # Each compiled graph is marked with a digest of its own code, so the same
    # graph from another artifact of the capture binds again, and an edited one
    # cannot repoint the graph another runnable already serves.
    for _backend_id, _fn in compiled.items():
        _code = "".join(
            graph_sources.get(f"{_backend_id}.{_p}", "") for _p in ("fw", "bw")
        )
        _fn.__precompile_graph__ = hashlib.sha256(_code.encode()).hexdigest()[:16]

    def _seed(scope, name, value):
        # A pre-reset compile's CleanupHook may still own the name; it must not
        # delete this binding once that code object is collected.
        CleanupHook.disown(scope, name)
        scope[name] = value

    # A frame's globals are the live dict of the module it was compiled in, for
    # the rebuilt bytecode and the guards alike: a global rebound after load
    # fails the guard that certified it instead of passing against a stale
    # copy, and a global the frame writes lands where the module reads it.
    # Only names Dynamo minted while tracing are bound into it (every backend id
    # of the artifact, plus that module's import aliases and builtins key),
    # never the artifact's own, so nothing here shadows a user global. A module
    # is opened only for a frame the driver serves; a dead record never imports
    # its module.
    scopes = {}

    def _scope(frame):
        module_name = frame["python_module"]
        scope = scopes.get(module_name)
        if scope is None:
            scope = scopes[module_name] = vars(_import(module_name))
            # Frames find a compiled graph by name in this module, so binding
            # different code under the name would repoint every runnable that
            # already serves it. All are checked before any is bound.
            for _backend_id, _fn in compiled.items():
                held = getattr(scope.get(_backend_id), "__precompile_graph__", None)
                if held not in (None, _fn.__precompile_graph__):
                    raise _PrecompileError(
                        f"precompile: {_backend_id!r} in module {module_name!r} is "
                        f"bound to other code for the same compiled graph (an edited "
                        f"copy of this capture, say); a process serves one version of "
                        f"a captured graph. Load this artifact in a fresh process."
                    )
            for _backend_id, _fn in compiled.items():
                _seed(scope, _backend_id, _fn)
        for _alias, _module_name in frame["import_sources"].items():
            _seed(scope, _alias, _import(_module_name))
        return scope

    def _make_dispatcher(frame):
        target = SerializedCode.to_code_object(frame["code"])
        is_entry = frame["is_entry"]
        if is_entry and target.co_freevars:
            # Capture refuses a closure entry, so this is a malformed artifact
            # rather than something to rebuild over empty cells.
            raise _PrecompileError(
                f"precompile: artifact entry {target.co_name!r} closes over "
                f"{list(target.co_freevars)!r}, which a self-contained artifact "
                f"cannot rebuild. Regenerate it from a module-level function."
            )
        if frame["trivial"] and frame["resume_names"]:
            # Dynamo compiled nothing of this continuation, so it ran as plain
            # Python during capture; rebuild it as one.
            # Its module is opened for the globals the bytecode reads.
            scope = _scope(frame)

            def _plain(closure):
                return types.FunctionType(target, scope, target.co_name, None, closure)

            if target.co_freevars:
                return _plain
            return _plain(None)
        if not frame["variants"]:
            # Nothing to dispatch, for one of two reasons the coverage-gap
            # error below would misdiagnose: adding examples fixes neither.
            # A continuation must still bind its resume names, so its refusal
            # is deferred to a call that reaches it. An unnamed trivial record
            # lands here too and binds nothing, because no frame names it.
            if frame["bypassed"]:
                cause = (
                    "was BYPASSED during capture (its guards could not be "
                    "serialized, or a backend artifact was missing when the "
                    "package was saved)"
                )
                advice = " Dynamo logged why; fix that and recapture."
            else:
                cause = "produced no guarded code (Dynamo ran it eager as trivial)"
                advice = ""
            msg = (
                f"precompile: {target.co_name!r} {cause}, so the artifact has no "
                f"variant of it to serve and no example can cover it.{advice}"
            )
            if is_entry:
                raise _PrecompileError(msg)

            # Bound under a resume name, so the frame ahead calls it with its
            # live locals; the diagnosis needs none of them.
            def refuse(*_args, **_kwargs):
                raise _PrecompileError(msg)

            return refuse
        scope = _scope(frame)
        # A code object carries no defaults, so the entry gets them back from
        # the artifact; a defaulted parameter the call omits is otherwise absent
        # from f_locals and every guard on it misses.
        defaults = entry_binding.get("defaults") if is_entry else None
        kwdefaults = entry_binding.get("kwdefaults") if is_entry else None
        # Dynamo gave up on the frame during capture: a miss runs eagerly.
        eager_fallback = frame["eager_fallback"]

        def _function(code, closure):
            f = types.FunctionType(code, scope, target.co_name, defaults, closure)
            if kwdefaults:
                f.__kwdefaults__ = dict(kwdefaults)
            return f

        variants = []
        for guarded in frame["variants"]:
            guards_state = load_guards_state(guarded["guards_state"])
            # Dynamo guards builtins through a dict it minted into the TRACING
            # process's globals under this key; a process that only loads never
            # traced, so the key is re-minted here (as CompilePackage.install
            # does) or every guard rooted at it misses. A key already bound to
            # anything but the module's builtins dict is refused, as install
            # refuses it: every builtin guard would read through that object.
            key = guards_state.output_graph.name_of_builtins_dict_key_in_fglobals
            if key:
                builtins_dict = get_builtins_dict(scope)
                if scope.get(key, builtins_dict) is not builtins_dict:
                    raise _PrecompileError(
                        f"precompile: {key!r} is bound in module "
                        f"{frame['python_module']!r} to something other than the "
                        f"module's builtins dict, which the artifact's builtin "
                        f"guards read through. Unbind it and load again."
                    )
                _seed(scope, key, builtins_dict)
            manager = load_guard_manager(guards_state, target, scope)
            body = SerializedCode.to_code_object(guarded["dynamo_code"])
            variants.append((manager, body))

        # Calls are bound to f_locals the way Python binds them (keyword-only,
        # *args and **kwargs included); the transformed bytecode keeps the
        # frame's parameter layout, so the original code object describes it.
        cells = tuple(types.CellType() for _ in target.co_freevars)
        signature = inspect.signature(_function(target, cells or None))

        def _dispatch_with(bound, closure, args, kwargs):
            # Guards are written against the frame's locals: the free variables
            # (a cell unbound at the graph break is absent, as it is from a real
            # frame's f_locals), then the bound arguments.
            f_locals = {}
            for name, cell in zip(target.co_freevars, closure or ()):
                try:
                    f_locals[name] = cell.cell_contents
                except ValueError:
                    pass
            f_locals.update(bind_locals(signature, *args, **kwargs))
            # inspect.signature spells a comprehension's iterator ".N" as
            # "implicitN"; the guards are rooted at the bytecode name.
            for name in target.co_varnames[: target.co_argcount]:
                if name.startswith("."):
                    f_locals[name] = f_locals.pop("implicit" + name[1:])
            # A tag-safe root's fast path (GuardManager::check_nopybind) can
            # refuse without running its tree and disarms itself; check twice.
            for _ in range(2):
                for manager, variant in bound:
                    if manager.check(f_locals):
                        return variant(*args, **kwargs)
            if eager_fallback:
                _warn_eager_fallback_miss(target)
                return _function(target, closure)(*args, **kwargs)
            raise _PrecompileError(
                f"precompile: no captured variant of {target.co_name!r} matches this "
                f"call ({len(variants)} captured). Either a guard on the call's "
                f"arguments or on a module global the graph baked in no longer "
                f"holds (restore that environment), or this call shape was never "
                f"captured: the artifact serves only what capture exercised, so add "
                f"an example covering it and recapture."
            )

        def _bind(closure):
            return [(manager, _function(body, closure)) for manager, body in variants]

        if target.co_freevars:
            # Dynamo binds a continuation that closes over locals as a FACTORY
            # taking the closure tuple, and the frame ahead of it passes one.
            def factory(closure):
                bound = _bind(closure)

                def dispatch(*args, **kwargs):
                    return _dispatch_with(bound, closure, args, kwargs)

                return dispatch

            return factory

        bound = _bind(None)

        def dispatch(*args, **kwargs):
            return _dispatch_with(bound, None, args, kwargs)

        return dispatch

    entry = None
    if not any(_frame["is_entry"] for _frame in frames):
        raise _PrecompileError("precompile: artifact has no entry frame")
    # Checked over every frame before anything is seeded, so a refusal leaves
    # the module as it was: a resume name is rebound only over this artifact's
    # own dispatcher (a rebuild), never over another artifact's, a live
    # compile's or a user's binding, whose LOAD_GLOBAL it would repoint.
    opened = set()
    for _frame in frames:
        module = _frame["python_module"]
        # torch._precompile._opens_scope: the frames _make_dispatcher opens a scope for.
        if _frame["variants"] or (_frame["trivial"] and _frame["resume_names"]):
            opened.add(module)
        for _name in _frame["resume_names"] if module in opened else ():
            existing = vars(_import(module)).get(_name)
            other = getattr(existing, "__precompile_artifact__", None)
            if existing is not None and other != tag:
                what = f"precompile: {_name!r} in module {module!r}"
                raise _PrecompileError(
                    f"{what} is bound by artifact {other} and this artifact ({tag}) mints "
                    f"the same resume name: only one standalone artifact per captured "
                    f"module can serve continuations in a process."
                    if other
                    else f"{what} is a resume name this artifact ({tag}) mints and this "
                    f"process already holds it (a live compile of the same function, or a "
                    f"user binding); load the artifact in a fresh process."
                )
    for _frame in frames:
        dispatcher = _make_dispatcher(_frame)
        if _frame["is_entry"]:
            entry = dispatcher
        else:
            dispatcher.__precompile_artifact__ = tag
        # A continuation is reached by LOAD_GLOBAL from the frame ahead of it,
        # in the module both were compiled in: Dynamo records a continuation
        # under its parent's module, except under config.nested_graph_breaks
        # (default False), where an inlined frame's continuation is recorded
        # under the inlined function's module while the root frame's bytecode
        # does the LOAD_GLOBAL. A dead record whose module no served frame
        # opened has no frame left to name it, so it binds nothing.
        scope = scopes.get(_frame["python_module"])
        if scope is not None:
            for _name in _frame["resume_names"]:
                _seed(scope, _name, dispatcher)
    return entry


def _build_installed_forward():
    """Install a multi-graph artifact onto the live code objects; return ``forward``.

    A graph break inside a function the entry calls (a child module's forward,
    say) compiles that function as its own frame, entered by an ordinary call
    the standalone dispatcher never sees. Serving those frames takes Dynamo's
    own frame evaluator: this installs the captured package's guarded entries
    onto the live code objects of the captured modules, the way a warm
    torch.compile cache load does, and calls the entry through it. The entry's
    callback is a _RefuseRecompileCallback bound at load, which
    _callback_from_stance keeps even under stances that would compile, so, as
    for a standalone artifact, a call no captured variant covers raises instead
    of compiling, or runs eagerly for a frame the capture ran eagerly (its
    code is in the callback's eager_codes). load(allow_serve_time_compiles=True)
    flips the artifact's _ALLOW_SERVE_TIME_COMPILES = False to True to keep the
    ordinary compiling callback instead.
    """
    import base64
    import importlib
    import inspect
    import operator
    import pickle
    import sys as _sys

    import torch
    from torch._C._dynamo.eval_frame import _debug_get_cache_entry_list
    from torch._dynamo.eval_frame import (
        _RecompileRefusedError,
        _RefuseRecompileCallback,
        OptimizeContext,
    )
    from torch._dynamo.package import _lookup_code, CompilePackage
    from torch._precompile import PrecompileError as _PrecompileError

    if tuple(_DYNAMO_PYTHON_VERSION) != _sys.version_info[:2]:
        raise _PrecompileError(
            f"precompile: this artifact was produced on Python "
            f"{_DYNAMO_PYTHON_VERSION[0]}.{_DYNAMO_PYTHON_VERSION[1]} and cannot "
            f"load on {_sys.version_info[0]}.{_sys.version_info[1]}: it carries "
            f"serialized code objects. Regenerate the artifact under the serving "
            f"Python."
        )
    if TORCH_VERSION != torch.__version__:
        raise _PrecompileError(
            f"precompile: this artifact was produced by torch {TORCH_VERSION} and "
            f"cannot load on torch {torch.__version__}: it carries pickled Dynamo "
            f"state. Regenerate the artifact under the serving torch."
        )
    if ENTRY_MODULE == "__main__":
        raise _PrecompileError(
            "precompile: this artifact was captured from a function defined in the "
            "capturing script's __main__ module, which no other process can import. "
            "Regenerate the artifact from a function defined in an importable module."
        )
    if "_import_custom_op_modules" in globals():

        def _import_served(module_name):
            try:
                importlib.import_module(module_name)
            # Not just ImportError, as in section 0.
            except Exception as _e:
                raise _PrecompileError(
                    f"precompile: this installed artifact serves frames of module "
                    f"{module_name!r}, which is not importable here ({_e})."
                ) from _e

        # Section 0 also lists them in case they register its ops; their failure
        # fails the load, so it is not warned about as a custom op module's.
        _import_custom_op_modules(SERVED_MODULES, _import_served)
    package_state = pickle.loads(base64.b64decode(_PACKAGE))
    dynamo = package_state["dynamo"]
    module_name = dynamo.codes[0].python_module
    try:
        fn = operator.attrgetter(FN_NAME)(importlib.import_module(module_name))
    # Not just ImportError, as in section 0.
    except Exception as _e:
        raise _PrecompileError(
            f"precompile: this installed artifact serves {module_name}.{FN_NAME}, "
            f"which is not importable here ({_e})."
        ) from _e
    # A decorator (torch.compiler.export_python's, say) can bind a functools.wraps
    # wrapper under the captured name: install onto the captured function it wraps.
    root = dynamo.codes[0].python_code
    captured = (root.co_name, root.co_firstlineno)

    def _is_captured(f):
        code = getattr(f, "__code__", None)
        return code is not None and (code.co_name, code.co_firstlineno) == captured

    fn = inspect.unwrap(fn, stop=_is_captured)
    try:
        # CompilePackage refuses a module whose captured source has changed.
        package = CompilePackage(fn, dynamo)
        if torch._dynamo.config.compiled_autograd:
            # OptimizeContext would compile backward graphs through a fresh,
            # non-refusing callback.
            raise _PrecompileError(
                "precompile: an installed artifact cannot load with "
                "torch._dynamo.config.compiled_autograd enabled: backward graphs "
                "would compile outside the artifact. Disable it before load()."
            )
        # The codes install() puts entries on; the entry's is innermost_fn's.
        entries = package._codes.items()
        targets = [_lookup_code(e) if e.code_source else c for c, e in entries]
        live = [c.co_name for c in targets if _debug_get_cache_entry_list(c)]
        if live:
            raise _PrecompileError(
                f"precompile: frames this artifact installs onto ({', '.join(live)}) "
                f"already have live Dynamo cache entries in this process (the capture "
                f"that produced this artifact, or a torch.compile of them), which "
                f"would still serve any call the installed entries miss: load this "
                f"artifact in a fresh process."
            )
        # A serve-time compile is not part of the artifact: record it nowhere.
        # caching_precompile would make optimize(package=None) create its own
        # package, which context(fn) fills from the on-disk Dynamo cache.
        with torch._dynamo.config.patch(caching_precompile=False):
            context = torch._dynamo.optimize(
                BACKEND, package=None if _ALLOW_SERVE_TIME_COMPILES else package
            )
            if not isinstance(context, OptimizeContext):
                raise _PrecompileError(
                    "precompile: an installed artifact serves through Dynamo, which "
                    "is disabled in this process (TORCHDYNAMO_DISABLE=1 or the "
                    "enable_dynamo killswitch)."
                )
            # The refusal is this callable's own callback, not the process-global
            # stance, so other threads' compiles are unaffected. It must be bound
            # before context(fn), which captures the callback when it wraps.
            if not _ALLOW_SERVE_TIME_COMPILES:
                flagged = [e.eager_fallback and not e.bypassed for _, e in entries]
                eager = [t for t, f in zip(targets, flagged) if f]
                context.callback = _RefuseRecompileCallback(context.callback, eager)
            compiled = context(fn)
        package.install(package_state["backends"])
    except _PrecompileError:
        raise
    except Exception as _e:
        raise _PrecompileError(
            f"precompile: this installed artifact could not be installed onto "
            f"{module_name}.{FN_NAME} ({type(_e).__name__}: {_e}). If that source "
            f"changed since capture, regenerate the artifact against it."
        ) from _e

    def forward(*args, **kwargs):
        try:
            return compiled(*args, **kwargs)
        except _RecompileRefusedError as _e:
            raise _PrecompileError(
                f"precompile: no captured variant matches this call. Either a "
                f"guard on the call's arguments or on a module global the graph "
                f"baked in no longer holds (restore that environment), or this "
                f"call shape was never captured: the artifact serves only what "
                f"capture exercised, so add an example covering it and "
                f"recapture. Dynamo reported: {_e}"
            ) from None

    return forward
