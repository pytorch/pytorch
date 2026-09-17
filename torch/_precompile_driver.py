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
    with _torch.no_grad():
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
