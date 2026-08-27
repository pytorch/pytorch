"""Runtime driver for torch.compiler.precompile artifacts, authored as real code.

Nothing here is imported or run by torch at runtime, and the generated artifacts never
import this module. Instead torch._precompile emits these function bodies VERBATIM (via
inspect.getsource, on the to_python_code / emit path only) into the python_code string,
after the calling-convention metadata and the compiled/captured graph. Authoring the
driver as real code -- instead of a hand-written string literal -- lets pyrefly / ruff /
IDEs type-check and navigate the load-bearing driver logic that would otherwise be
invisible inside a string (and drops the wall of ``# noqa: F821``).

Keeping the driver version-frozen (its behavior is hashed via code_hash, invariant 7)
still holds: the artifact carries the driver TEXT, it does not import it, so there is no
torch-version skew. The emit path runs getsource in-process where torch source is
present; load() never touches this module.

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
    from collections.abc import Callable

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
    _DYNAMO_BACKEND_IDS: tuple[str, ...] = ()
    _DYNAMO_BACKENDS: dict[str, Callable[[list[object]], object]] = {}
    _DYNAMO_PYTHON_VERSION: tuple[int, int] = (0, 0)
    _DYNAMO_STATE: str = ""
    TRAINING: bool = False

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


def _build_dynamo_forward():
    """Rebuild Dynamo's guards and transformed bytecode into a standalone dispatcher.

    The compiled graph sources stay ordinary Python in the artifact. Only the minimized
    Dynamo dispatch guards, transformed entry/resume code objects, and embedded disabled
    functions are opaque because none has a source form. There is no compiler behind
    this dispatcher: a miss against every retained guard set raises instead of compiling
    another specialization.
    """
    import base64
    import importlib
    import inspect
    import pickle
    import sys
    import types
    from typing import Any, cast

    import torch
    import torch.utils._pytree as _pytree
    from torch._dynamo.package import (
        load_guard_manager,
        load_guards_state,
        SerializedCode,
    )

    if tuple(_DYNAMO_PYTHON_VERSION) != sys.version_info[:2]:
        from torch._precompile import PrecompileError

        raise PrecompileError(
            "precompile artifact was produced on Python "
            f"{_DYNAMO_PYTHON_VERSION[0]}.{_DYNAMO_PYTHON_VERSION[1]}, but is "
            f"being loaded on Python {sys.version_info[0]}.{sys.version_info[1]}."
        )

    state = pickle.loads(base64.b64decode(_DYNAMO_STATE))
    namespace = dict(globals())

    def check_input_contract(args, kwargs):
        contract = state.input_contract
        if contract is None:
            return

        def module_signature(module):
            tensors = [
                ("parameter", name, tensor)
                for name, tensor in module.named_parameters(remove_duplicate=False)
            ]
            tensors.extend(
                ("buffer", name, tensor)
                for name, tensor in module.named_buffers(remove_duplicate=False)
            )
            aliases = {}
            metadata = []
            for kind, name, tensor in tensors:
                alias = aliases.setdefault(id(tensor), len(aliases))
                metadata.append(
                    (
                        kind,
                        name,
                        alias,
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                        str(tensor.dtype),
                        str(tensor.device),
                        tensor.requires_grad,
                    )
                )
            return (
                type(module).__module__,
                type(module).__qualname__,
                module.training,
                tuple(metadata),
            )

        leaves, spec = _pytree.tree_flatten((args, kwargs))
        serialized_spec = _pytree.treespec_dumps(spec)
        variant = next(
            (
                candidate
                for candidate in contract.variants
                if candidate.spec == serialized_spec
            ),
            None,
        )
        if variant is None:
            from torch._precompile import PrecompileError

            raise PrecompileError(
                "precompile: runtime inputs have a different structure from the "
                "captured Dynamo examples."
            )
        for index, (value, expected) in enumerate(
            zip(leaves, variant.leaves, strict=True)
        ):
            if expected is None:
                continue
            if expected["kind"] == "module":
                if not isinstance(value, torch.nn.Module):
                    from torch._precompile import PrecompileError

                    raise PrecompileError(
                        f"precompile: runtime input leaf {index} is not an nn.Module."
                    )
                if module_signature(value) not in expected["variants"]:
                    from torch._precompile import PrecompileError

                    raise PrecompileError(
                        f"precompile: runtime module input leaf {index} has a "
                        "different type, training mode, parameter/buffer structure, "
                        "shape, layout, dtype, device, or requires_grad contract."
                    )
                continue
            if not isinstance(value, torch.Tensor):
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    f"precompile: runtime input leaf {index} is not a tensor."
                )
            actual_type = (type(value).__module__, type(value).__qualname__)
            checks = (
                ("type", actual_type),
                ("dtype", str(value.dtype)),
                ("device", str(value.device)),
                ("requires_grad", value.requires_grad),
            )
            for name, actual in checks:
                wanted = expected[name]
                if wanted is not None and actual != wanted:
                    from torch._precompile import PrecompileError

                    raise PrecompileError(
                        f"precompile: runtime input leaf {index} has {name} "
                        f"{actual!r}, expected {wanted!r}."
                    )
            shape = expected["shape"]
            stride = expected["stride"]
            if shape is not None and len(value.shape) != len(shape):
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    f"precompile: runtime input leaf {index} has rank "
                    f"{value.dim()}, expected {len(shape)}."
                )
            if shape is not None:
                for dim, wanted in enumerate(shape):
                    if wanted is not None and value.shape[dim] != wanted:
                        from torch._precompile import PrecompileError

                        raise PrecompileError(
                            f"precompile: runtime input leaf {index} has shape "
                            f"{tuple(value.shape)}, expected dim {dim} to be {wanted}."
                        )
            if stride is not None:
                for dim, wanted in enumerate(stride):
                    if wanted is not None and value.stride(dim) != wanted:
                        from torch._precompile import PrecompileError

                        raise PrecompileError(
                            f"precompile: runtime input leaf {index} has stride "
                            f"{tuple(value.stride())}, expected dim {dim} to be "
                            f"{wanted}."
                        )

    for code_state in state.codes:
        for alias, (module_name, path) in code_state.global_bindings.items():
            value = importlib.import_module(module_name)
            for attr in path:
                value = getattr(value, attr)
            namespace[alias] = value
        namespace.update(code_state.value_globals)
        for alias, module_name in code_state.import_sources.items():
            namespace[alias] = importlib.import_module(module_name)

    def make_backend(call):
        def run(*args):
            return call(list(args))

        return torch._dynamo.disable(run)

    backend_calls = {}
    for backend_id in _DYNAMO_BACKEND_IDS:
        backend_call = make_backend(_DYNAMO_BACKENDS[backend_id])
        backend_calls[backend_id] = backend_call
        namespace[backend_id] = backend_call

    for global_name, function_state in state.disabled_functions.items():
        function_globals = dict(namespace)
        for name, module_name in function_state.module_globals.items():
            function_globals[name] = importlib.import_module(module_name)
        function_globals.update(function_state.value_globals)
        function = types.FunctionType(
            SerializedCode.to_code_object(function_state.code),
            function_globals,
            function_state.name,
            function_state.defaults,
        )
        if function_state.kwdefaults:
            function.__kwdefaults__ = dict(function_state.kwdefaults)
        function_globals[global_name] = function
        namespace[global_name] = function

    if state.serving_mode == "installed":
        import threading

        from torch._dynamo.package import CompilePackage

        if state.package is None:
            raise AssertionError("installed Dynamo artifact has no package")

        class SourceBackendArtifact:
            def __init__(self, function):
                self.function = function

            def after_deserialization(self):
                return self.function

        backends = {
            backend_id: SourceBackendArtifact(function)
            for backend_id, function in backend_calls.items()
        }

        def entry_function():
            code_state = state.codes[0]
            try:
                module = importlib.import_module(code_state.python_module)
            except ImportError as e:
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    "precompile: this installed artifact needs its defining module "
                    f"{code_state.python_module!r} to be importable."
                ) from e
            code = SerializedCode.to_code_object(code_state.code)
            function = types.FunctionType(
                code,
                module.__dict__,
                code.co_name,
                code_state.defaults,
            )
            if code_state.kwdefaults:
                function.__kwdefaults__ = dict(code_state.kwdefaults)
            return function

        class InstalledArtifact:
            def __init__(self):
                self.fn = None
                self.compiled = None
                self.package = None
                self.region = -1
                self.codes = ()
                self.state = threading.Condition()
                self.active_calls = 0
                self.unloading = False

            def _rebind(self, fn):
                with self.state:
                    if self.compiled is not None:
                        from torch._precompile import PrecompileError

                        raise PrecompileError(
                            "precompile: this artifact is already installed; pass "
                            "fn= to load() before the first call."
                        )
                    self.fn = fn

            def _ensure(self):
                with self.state:
                    if self.unloading:
                        raise RuntimeError("precompile artifact is being unloaded")
                    if self.compiled is not None:
                        return
                    fn = entry_function() if self.fn is None else self.fn
                    entry = fn.forward if isinstance(fn, torch.nn.Module) else fn

                    def fail_backend(gm, inputs):
                        from torch._precompile import PrecompileError

                        raise PrecompileError(
                            "precompile: no captured Dynamo variant matches this "
                            "call. Add an example covering it and precompile again."
                        )

                    package = CompilePackage(
                        entry,
                        state.package,
                        ignore_inlined_sources=True,
                    )
                    context = torch._dynamo.optimize(
                        fail_backend,
                        package=package,
                        dynamic=None,
                        isolate_recompiles=True,
                    )
                    compiled = context(fn)
                    region = context._isolate_recompiles_id  # type: ignore[attr-defined]
                    try:
                        package.install(
                            cast("dict[Any, Any]", backends),
                            isolate_recompiles_id=region,
                        )
                    except BaseException:
                        package.uninstall()
                        raise
                    self.fn = fn
                    self.compiled = compiled
                    self.package = package
                    self.region = region
                    self.codes = package.region_codes()

            def __call__(self, *args, **kwargs):
                self._ensure()
                check_input_contract(args, kwargs)
                with self.state:
                    if self.compiled is None or self.unloading:
                        raise RuntimeError("precompile artifact has been unloaded")
                    package = self.package
                    if package is None:
                        raise AssertionError("installed artifact was not prepared")
                    if package.installed_entries_dropped():
                        from torch._precompile import PrecompileError

                        raise PrecompileError(
                            "precompile: torch._dynamo.reset() cleared this loaded "
                            "artifact; load it again before calling it."
                        )
                    compiled = self.compiled
                    self.active_calls += 1
                try:
                    with torch.set_grad_enabled(TRAINING):
                        return compiled(*args, **kwargs)
                except torch._dynamo.exc.BackendCompilerFailed as e:
                    from torch._precompile import PrecompileError

                    if isinstance(e.inner_exception, PrecompileError):
                        raise e.inner_exception from e
                    raise
                finally:
                    with self.state:
                        self.active_calls -= 1
                        if not self.active_calls:
                            self.state.notify_all()

            def __enter__(self):
                self._ensure()
                return self

            def __exit__(self, *exc):
                self.unload()

            def unload(self):
                with self.state:
                    while self.unloading:
                        self.state.wait()
                    if self.compiled is None:
                        return
                    self.unloading = True
                    while self.active_calls:
                        self.state.wait()
                    package = self.package
                    codes = self.codes
                    region = self.region
                    self.compiled = None
                    self.package = None
                    self.codes = ()
                    self.region = -1
                try:
                    try:
                        if package is not None:
                            package.uninstall()
                    finally:
                        from torch._C._dynamo.eval_frame import (
                            _clear_cache_entries_for_region,
                        )

                        if region >= 0:
                            for code in codes:
                                _clear_cache_entries_for_region(code, region)
                finally:
                    with self.state:
                        self.unloading = False
                        self.state.notify_all()

        return InstalledArtifact()

    def prepare_code(code_state):
        target = SerializedCode.to_code_object(code_state.code)
        defaults = code_state.defaults
        kwdefaults = code_state.kwdefaults
        guarded_variants = []
        for guarded in code_state.variants:
            guards_state = load_guards_state(guarded.guards_state)
            manager = load_guard_manager(guards_state, target, namespace)
            code = SerializedCode.to_code_object(guarded.dynamo_code)
            guarded_variants.append((manager, code))

        def bind(closure=None):
            target_function = types.FunctionType(
                target, namespace, target.co_name, defaults, closure
            )
            if kwdefaults:
                target_function.__kwdefaults__ = dict(kwdefaults)
            if not guarded_variants:
                return target_function

            variants = []
            for manager, code in guarded_variants:
                function = types.FunctionType(
                    code, namespace, target.co_name, defaults, closure
                )
                if kwdefaults:
                    function.__kwdefaults__ = dict(kwdefaults)
                variants.append((manager, function))
            signature = inspect.signature(target_function)

            def dispatch(*args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                local_scope = dict(bound.arguments)
                for manager, function in variants:
                    if manager.check(local_scope):
                        return function(*args, **kwargs)
                from torch._precompile import PrecompileError

                raise PrecompileError(
                    f"precompile: no captured Dynamo variant of {target.co_name!r} "
                    f"matches this call. Add an example covering it and precompile "
                    f"again; the artifact contains {len(variants)} guarded variant(s)."
                )

            return dispatch

        return target, bind

    main_state, *resume_states = state.codes
    for code_state in resume_states:
        target, bind = prepare_code(code_state)
        function = bind if target.co_freevars else bind()
        for name in code_state.function_names:
            namespace[name] = function

    target, bind = prepare_code(main_state)
    if target.co_freevars:
        raise AssertionError("main Dynamo frame must not have free variables")
    entry = bind()

    def forward(*args, **kwargs):
        check_input_contract(args, kwargs)
        with torch.set_grad_enabled(TRAINING):
            return entry(*args, **kwargs)

    return forward
