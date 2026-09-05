"""Runtime driver for torch.compiler.precompile artifacts, authored as real code.

Nothing here is imported or run by torch at runtime, and the generated artifacts never
import this module. Instead torch._precompile emits these function bodies VERBATIM (via
inspect.getsource, on the to_python_code / emit path only) into the self-contained
python_code string, after the calling-convention metadata and the compiled/captured
graph. Authoring the driver as real code -- instead of a hand-written string literal --
lets pyrefly / ruff / IDEs type-check and navigate the load-bearing driver logic that
would otherwise be invisible inside a string (and drops the wall of ``# noqa: F821``).

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
    # Multi-graph artifact state, emitted by
    # torch._precompile._build_multigraph_python_source. _FRAMES is one entry
    # per Dynamo frame (its code, its variants' guard state and transformed
    # bytecode, the globals it reads); _BACKENDS is the compiled subgraphs.
    # The backend the capture used; the drivers key their autocast handling and
    # their re-serve off it.
    BACKEND: str = ""
    _FRAMES: str = ""
    _BACKENDS: str = ""
    # An INSTALLED multi-graph artifact carries the whole captured package in
    # one blob instead of the per-frame records above, because it hands the
    # frames to the frame evaluator rather than dispatching them itself.
    _PACKAGE: str = ""
    _ENTRY_BINDING: str = ""
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
    # Device types the captured graph dispatches on. The drivers neutralize
    # ambient autocast on these; see _autocast_off.
    GRAPH_DEVICES: tuple[str, ...] = ()

    # The compiled/captured graph's entry point, emitted before the driver.
    def call(flat_inputs: list[object]) -> list[object]: ...


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


def _autocast_off(devices):
    """Neutralize ambient autocast on the devices the captured graph uses.

    Whatever the capture ran under is already baked into the artifact -- ATen
    casts for make_fx, generated kernels for inductor -- but the graph still
    re-dispatches (an inductor artifact calls extern_kernels, which hit the
    autocast key), so a serving process with autocast on would cast a second
    time. ``devices`` is GRAPH_DEVICES, recorded from the captured graph rather
    than from the runtime tensors: a graph can reach a device none of its
    inputs live on, and one built from factory ops has no input device at all.
    """
    import contextlib as _contextlib

    import torch as _t

    stack = _contextlib.ExitStack()
    for _dev in devices:
        if _t.amp.is_autocast_available(_dev):
            stack.enter_context(_t.amp.autocast(_dev, enabled=False))
    return stack


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
    with _autocast_off(GRAPH_DEVICES), _torch.no_grad():
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
        # The generated code re-dispatches through extern_kernels for anything
        # inductor did not fuse, so ambient autocast reaches it even though the
        # casts the capture ran under are already baked into the kernels.
        with _autocast_off(GRAPH_DEVICES):
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
    calls the first that matches, so an artifact serves only what it captured
    and mutates nothing. A continuation is reached the way Dynamo emits it --
    the entry bytecode does LOAD_GLOBAL on the resume name -- so binding the
    resume dispatcher under that name in this module's namespace is all the
    wiring the graph-break path needs.

    Because there is no compiler behind a source artifact, an uncovered call
    RAISES rather than falling back. That is the point: the artifact serves the
    calls it captured, and anything else is a coverage gap the caller has to
    hear about.
    """
    import base64
    import importlib
    import pickle
    import sys as _sys
    import types

    import torch
    from torch._dynamo.package import (
        load_guard_manager,
        load_guards_state,
        SerializedCode,
    )

    # The documented contract (Note [precompile programming model]) is that the
    # version/build locks surface as a clean PrecompileError, so an
    # ``except torch.compiler.precompile.PrecompileError`` handler catches them.
    from torch._precompile import PrecompileError as _PrecompileError

    produced_on = globals().get("_DYNAMO_PYTHON_VERSION")
    if produced_on is not None and tuple(produced_on) != _sys.version_info[:2]:
        # marshal only REJECTS a foreign blob across the 3.10 -> 3.11 layout
        # change; between 3.11 and 3.14 it loads and then segfaults when the
        # code object runs, so the version has to be checked explicitly.
        raise _PrecompileError(
            f"precompile: this artifact was produced on Python "
            f"{produced_on[0]}.{produced_on[1]} and cannot load on "
            f"{_sys.version_info[0]}.{_sys.version_info[1]}: it inlines marshalled "
            f"bytecode, which is Python-version-locked. Regenerate the artifact "
            f"under the serving Python."
        )

    frames = pickle.loads(base64.b64decode(_FRAMES))
    backends = pickle.loads(base64.b64decode(_BACKENDS)) if _BACKENDS else {}

    # One namespace for the whole artifact. Every name the transformed bytecode
    # can reach lives here: the compiled subgraphs, Dynamo's synthetic import
    # aliases, the plain globals it read, and the resume dispatchers bound
    # below. It is this module's own dict, never the user's.
    ns = globals()
    # Seed from the module each frame was compiled in: its bytecode reads that
    # module's globals by name, and its guards were written against them. This
    # is a READ -- the artifact binds its own names here, never in the user's
    # module, so loading mutates nothing and there is nothing to unload. The
    # values are therefore as of load time; a global rebound afterwards is not
    # seen, which for a frozen artifact is the intended reading.
    for _frame in frames:
        _module = _sys.modules.get(_frame["python_module"])
        if _module is None:
            try:
                _module = importlib.import_module(_frame["python_module"])
            except ImportError:
                _module = None
        if _module is not None:
            for _k, _v in vars(_module).items():
                ns.setdefault(_k, _v)
    # The artifact's own names win over anything seeded above.
    for _backend_id, _artifact in backends.items():
        ns[_backend_id] = torch._dynamo.disable(_artifact.after_deserialization())
    # Subgraphs emitted as readable source above already ran as part of this
    # module, so their compiled ``call`` is in _SUBGRAPHS. Inductor's entry point
    # is boxed (it takes one list), while the transformed bytecode calls the
    # subgraph with individual arguments, so adapt between the two.
    for _backend_id, _boxed in globals().get("_SUBGRAPHS", {}).items():

        def _adapt(_inner):
            def _compiled_subgraph(*args):
                return _inner(list(args))

            return _compiled_subgraph

        ns[_backend_id] = torch._dynamo.disable(_adapt(_boxed))
    for _frame in frames:
        for _alias, _module_name in _frame["import_sources"].items():
            try:
                ns[_alias] = importlib.import_module(_module_name)
            except ImportError as _e:
                # The torch-build / environment lock, surfaced per the
                # documented contract rather than as a raw ModuleNotFoundError.
                raise _PrecompileError(
                    f"precompile: this artifact references module "
                    f"'{_module_name}', which is not importable here ({_e}). It "
                    f"was produced against a different torch build or "
                    f"environment; regenerate the artifact in this one."
                ) from _e

    entry_binding = (
        pickle.loads(base64.b64decode(_ENTRY_BINDING)) if _ENTRY_BINDING else {}
    )

    def _make_dispatcher(frame):
        target = SerializedCode.to_code_object(frame["code"])
        arg_names = target.co_varnames[: target.co_argcount]
        is_entry = frame["is_entry"]
        # A code object carries neither defaults nor closure values, so the
        # entry gets them back from the artifact. Without the defaults an
        # omitted parameter is missing from f_locals and every guard misses;
        # without the closure a closure entry cannot be built at all.
        entry_defaults = entry_binding.get("defaults") if is_entry else None
        entry_kwdefaults = entry_binding.get("kwdefaults") if is_entry else None
        entry_cells = entry_binding.get("closure") if is_entry else None
        variants = []
        for guarded in frame["variants"]:
            guards_state = load_guards_state(guarded["guards_state"])
            manager = load_guard_manager(guards_state, target, ns)
            body = SerializedCode.to_code_object(guarded["dynamo_code"])
            variants.append((manager, body))

        def _bind(closure):
            bound = []
            for manager, body in variants:
                f = types.FunctionType(
                    body, ns, target.co_name, entry_defaults, closure
                )
                if entry_kwdefaults:
                    f.__kwdefaults__ = dict(entry_kwdefaults)
                bound.append((manager, f))
            return bound

        def _dispatch_with(bound, args, kwargs):
            # Guards are written against the frame's locals, so rebuild that
            # mapping from the call. Positional-only is enough: Dynamo compiles
            # the frame Python actually entered, and its parameters are bound by
            # then.
            f_locals = dict(zip(arg_names, args))
            # Fill omitted parameters from the entry's defaults BEFORE checking:
            # a guard written against a defaulted argument has nothing to bind to
            # otherwise, and every variant misses on a call that omitted it.
            if entry_defaults:
                # __defaults__ aligns with the LAST len(defaults) parameters,
                # not with the first parameter this call omitted.
                for name, value in zip(
                    arg_names[-len(entry_defaults) :], entry_defaults
                ):
                    f_locals.setdefault(name, value)
            if entry_kwdefaults:
                for name, value in entry_kwdefaults.items():
                    f_locals.setdefault(name, value)
            f_locals.update(kwargs)
            for manager, variant in bound:
                if manager.check(f_locals):
                    return variant(*args, **kwargs)
            raise _PrecompileError(
                f"precompile: no captured variant of {target.co_name!r} matches this "
                f"call. The artifact serves only what capture exercised; add an "
                f"example covering it and recapture. Captured "
                f"{len(variants)} variant(s)."
            )

        if is_entry and target.co_freevars:
            # The entry's cells come from the artifact, not from a caller: only
            # a continuation is handed a closure per call.
            bound = _bind(tuple(types.CellType(v) for v in (entry_cells or ())))

            def entry_dispatch(*args, **kwargs):
                return _dispatch_with(bound, args, kwargs)

            return entry_dispatch, target

        if target.co_freevars:
            # Dynamo binds a continuation that closes over locals as a FACTORY
            # taking the closure tuple, and the frame ahead of it passes one.
            # Mirror that: the closure is only known per call.
            def factory(closure):
                bound = _bind(closure)

                def dispatch(*args, **kwargs):
                    return _dispatch_with(bound, args, kwargs)

                return dispatch

            return factory, target

        bound = _bind(None)

        def dispatch(*args, **kwargs):
            return _dispatch_with(bound, args, kwargs)

        return dispatch, target

    entry = None
    for _frame in frames:
        dispatcher, _target = _make_dispatcher(_frame)
        if _frame["resume_names"]:
            # A continuation is reached by LOAD_GLOBAL from the frame ahead of
            # it, under the name capture minted. Bind it here rather than in the
            # user's module: nothing outside this artifact should resolve it.
            for _name in _frame["resume_names"]:
                ns[_name] = dispatcher
        elif _frame["is_entry"]:
            entry = dispatcher
    if entry is None:
        raise _PrecompileError("precompile: artifact has no entry frame")
    return entry


def _build_installed_forward():
    """Serve a multi-graph capture by INSTALLING it onto the live code objects.

    A self-contained driver dispatches only the entry frame and the
    continuations the entry's own bytecode names. A model that graph-breaks
    inside nested module forwards puts almost every compiled frame behind an
    ordinary method call, which no name in the entry reaches, so those frames
    would run eager. This driver instead rebuilds the captured package and hands
    it to the frame evaluator, which reaches every frame by code object.

    Nothing is installed here: building the handle only resolves and imports.
    Entering it, or calling it, installs; unload() takes it back out.
    """
    import base64
    import importlib
    import pickle
    import sys
    import types

    from torch._dynamo.package import SerializedCode
    from torch._precompile import _InstalledArtifact, PrecompileError

    cache_entry = pickle.loads(base64.b64decode(_PACKAGE))

    # install resolves every frame through sys.modules[...] directly, so each
    # module a captured frame came from has to be imported before it runs.
    for _code_entry in cache_entry.dynamo.codes:
        try:
            importlib.import_module(_code_entry.python_module)
        except ImportError as e:
            raise PrecompileError(
                f"precompile: this artifact holds a frame captured from module "
                f"{_code_entry.python_module!r}, which is not importable here "
                f"({e}). Load it where that module is, or pass fn= to load()."
            ) from e

    def _entry_function():
        # The entry records no qualname to resolve -- it is the callable handed
        # to precompile, not something reached from a module -- so rebuild a
        # function around its code object. A code object carries no defaults,
        # so they come back from the artifact; without them a defaulted
        # parameter is simply absent at the served call and every guard misses.
        # Capture refuses a closure entry. load(fn=...) lets a caller supply the
        # real function.
        code_entry = cache_entry.dynamo.codes[0]
        code = SerializedCode.to_code_object(code_entry.python_code)
        binding = (
            pickle.loads(base64.b64decode(_ENTRY_BINDING)) if _ENTRY_BINDING else {}
        )
        f = types.FunctionType(
            code,
            sys.modules[code_entry.python_module].__dict__,
            code.co_name,
            binding.get("defaults"),
        )
        kwdefaults = binding.get("kwdefaults")
        if kwdefaults:
            f.__kwdefaults__ = dict(kwdefaults)
        return f

    def _serve(fn, prepared=None):
        from torch._dynamo.precompile_context import PrecompileContext
        from torch._dynamo.precompile_package import serve_cache_entry

        # The backends have to be in the context before install deserializes
        # them, the same thing DynamoStore.load_cache_entry does for the path
        # form of this artifact.
        for _backend in cache_entry.backends.values():
            PrecompileContext.record_artifact(_backend)
        return serve_cache_entry(fn, cache_entry, backend=BACKEND, prepared=prepared)

    def _check_entry(fn):
        from torch._dynamo.precompile_package import (
            _check_artifact_matches,
            _entry_fn_of,
        )

        _check_artifact_matches(cache_entry.dynamo, _entry_fn_of(fn), "this artifact")

    return _InstalledArtifact(
        _serve,
        _entry_function,
        check_fn=_check_entry,
    )
