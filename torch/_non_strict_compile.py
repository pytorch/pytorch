"""Non-strict, make_fx-based ahead-of-time compilation.

``torch.non_strict_compile`` traces a function or ``nn.Module`` with
``make_fx`` in a principled way: module parameters and buffers are lifted to
explicit graph inputs (via functional reparametrization) so that no live
tensor is ever baked into the graph as a constant. Any tensor that is neither
a graph input (parameter/buffer/user input) nor an intermediate produced by
the traced computation is rejected, since silently hard-coding such a tensor
into the graph would be a correctness footgun.

The captured graph is lowered through ``torch._inductor.standalone_compile``,
which runs the full AOTAutograd + Inductor pipeline (runtime wrappers plus
generated output code). The resulting artifact can be JIT-invoked, exported to
a human-readable Python file showing the calling convention and Inductor
output code, and exported to a binary cache holding the compiled kernels and
parameter/buffer values for later reload.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from typing import Any

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn.utils import stateless


__all__ = ["non_strict_compile", "NonStrictCompiled", "NonStrictCompileError"]


class NonStrictCompileError(RuntimeError):
    """Raised when non-strict tracing would bake a tensor into the graph."""


def _check_no_constant_tensors(gm: torch.fx.GraphModule) -> None:
    """Reject graphs that hard-code a tensor as a ``get_attr`` constant.

    Every legitimate tensor in a non-strict capture is a placeholder (a lifted
    parameter/buffer or user input) or the result of a ``call_function`` node.
    A ``get_attr`` pointing at a tensor therefore means some tensor was closed
    over and would be baked into the graph, which we forbid.
    """
    offending = []
    for node in gm.graph.nodes:
        if node.op != "get_attr":
            continue
        attr = gm
        for part in node.target.split("."):
            attr = getattr(attr, part, None)
        if isinstance(attr, torch.Tensor):
            offending.append((node.target, tuple(attr.shape), str(attr.dtype)))
    if offending:
        raise NonStrictCompileError(
            "non_strict_compile traced a tensor that is neither a graph input "
            "(module parameter/buffer or user input) nor an intermediate. Such "
            "tensors would be hard-coded into the graph. Offending constants "
            f"(target, shape, dtype): {offending}. Pass these tensors as inputs, "
            "or register them as parameters/buffers on the module."
        )


def _capture(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    decompositions: dict | None = None,
    step: Callable[..., Any] | None = None,
) -> _Capture:
    """Trace ``fn`` to an ATen graph with params/buffers lifted to inputs.

    With ``step`` set, the traced computation is an arbitrary user callable
    ``step(model, *inputs)`` run with the module's params/buffers lifted to graph
    inputs (via reparametrization). It may do anything -- forward, loss, and
    ``torch.autograd.grad`` for the backward -- expressed as one graph; whatever
    it returns becomes the graph outputs. Because the backward (if any) is just
    graph ops rather than an autograd.Function, the result lowers like an
    inference graph and is fully self-contained. The step must return its results
    (use ``torch.autograd.grad``, not ``.backward()``, which sets ``.grad`` as a
    side effect that tracing does not capture).
    """
    is_module = isinstance(fn, torch.nn.Module)
    if is_module:
        named_params = dict(fn.named_parameters(remove_duplicate=False))
        named_buffers = dict(fn.named_buffers(remove_duplicate=False))
    else:
        named_params, named_buffers = {}, {}

    param_names = list(named_params.keys())
    buffer_names = list(named_buffers.keys())
    pb_names = param_names + buffer_names
    pb_flat = list(named_params.values()) + list(named_buffers.values())
    num_pb = len(pb_flat)

    user_flat, in_spec = pytree.tree_flatten((args, kwargs))
    flat_args = [*pb_flat, *user_flat]

    out_spec_holder: dict[str, pytree.TreeSpec] = {}

    def flat_fn(flat: list[Any]) -> list[Any]:
        pb = flat[:num_pb]
        user = flat[num_pb:]
        user_args, user_kwargs = pytree.tree_unflatten(user, in_spec)
        if step is not None:
            assert isinstance(fn, torch.nn.Module)
            reparam = dict(zip(pb_names, pb))
            with stateless._reparametrize_module(fn, reparam, tie_weights=True):
                out = step(fn, *user_args, **user_kwargs)
        elif is_module:
            assert isinstance(fn, torch.nn.Module)
            reparam = dict(zip(pb_names, pb))
            with stateless._reparametrize_module(fn, reparam, tie_weights=True):
                out = fn(*user_args, **user_kwargs)
        else:
            out = fn(*user_args, **user_kwargs)
        out_flat, out_spec = pytree.tree_flatten(out)
        out_spec_holder["spec"] = out_spec
        return out_flat

    # Plain forward traces under no_grad; a custom step traces with grad enabled
    # so any backward it does (torch.autograd.grad) is built as graph ops.
    grad_ctx = torch.enable_grad() if step is not None else torch.no_grad()
    with grad_ctx:
        gm = make_fx(flat_fn, decomposition_table=decompositions)(flat_args)
    _check_no_constant_tensors(gm)

    return _Capture(
        gm=gm,
        flat_args=flat_args,
        param_names=param_names,
        buffer_names=buffer_names,
        param_buffer_flat=pb_flat,
        num_params_buffers=num_pb,
        in_spec=in_spec,
        out_spec=out_spec_holder["spec"],
    )


class _Capture:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        flat_args: list[Any],
        param_names: list[str],
        buffer_names: list[str],
        param_buffer_flat: list[Any],
        num_params_buffers: int,
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
    ) -> None:
        self.gm = gm
        self.flat_args = flat_args
        self.param_names = param_names
        self.buffer_names = buffer_names
        self.param_buffer_flat = param_buffer_flat
        self.num_params_buffers = num_params_buffers
        self.in_spec = in_spec
        self.out_spec = out_spec


# AOTAutograd codegens its runtime wrappers as Python source via a single helper
# (subclass_codegen._compile_and_exec_source). We intercept that helper during
# compilation to capture every generated wrapper, then re-emit and re-compose
# them here so the exported file runs the *same* prologue/epilogue, subclass
# unwrap/rewrap, dedup, synthetic-base, RNG and effect-token handling that the
# JIT path uses. Helpers are built first (referenced by the orchestration via
# globals); the chain is composed inner -> outer.
_WRAPPER_HELPER_NAMES = ["mutation_epilogue", "output_alias_wrapper"]
_WRAPPER_CHAIN_NAMES = [
    "effect_tokens_wrapper",
    "subclass_wrapper",
    "functionalized_rng_wrapper",
    "runtime_wrapper_orchestration",
    "synthetic_base_wrapper",
    "dedup_wrapper",
]


_GENERATED_HEADER = """\
# Generated by torch.non_strict_compile -- do not edit.
#
# This is a SELF-CONTAINED, EXECUTABLE end-to-end artifact. You can run it
# directly:
#
#     python this_file.py                       # reports it is ready
#
# or import / exec it and call forward() with your own inputs:
#
#     ns = {}
#     exec(open("this_file.py").read(), ns)
#     out = ns["forward"](my_input)             # returns the model output
#
# It contains, in order:
#   1. The Inductor-generated output code (the actual compiled kernels and the
#      ``call(args)`` entry point). The kernels are (re)compiled from the source
#      embedded below on first use, so no external kernel cache is required.
#   2. Calling-convention metadata: the parameter/buffer names lifted to graph
#      inputs, and the pytree specs for user inputs and outputs.
#   3. The AOTAutograd runtime wrappers, captured verbatim from the codegen that
#      runs during compilation (prologue/epilogue, input-mutation and output-
#      alias handling, tensor-subclass unwrap/rewrap such as DTensor, dedup,
#      synthetic-base, functionalized RNG, effect tokens). These are recomposed
#      inner -> outer to form the boxed runtime function, which ``forward`` drives.
#
# Parameter/buffer VALUES and the residual wrapper globals (subclass metadata,
# etc.) are not embedded here; they are loaded from the companion cache file
# written by ``export_cache`` (see WEIGHTS_PATH below).
"""


def _py_str_literal(s: str) -> str:
    """Emit ``s`` as a readable triple-quoted literal when safe, else repr."""
    if '"""' not in s and "\\" not in s:
        return 'r"""\n' + s + '\n"""'
    return repr(s)


def _build_metadata_section(
    compiled: NonStrictCompiled, weights_path: str
) -> list[str]:
    assert compiled._out_spec is not None
    out_spec_str = pytree.treespec_dumps(compiled._out_spec)
    parts = [
        "# " + "=" * 70,
        "# 2. Calling-convention metadata",
        "# " + "=" * 70,
        "import os as _os",
        "import torch as _torch",
        "import torch.utils._pytree as _pytree",
        "import importlib as _importlib",
        "import contextlib as _contextlib",
        "",
        f"CUSTOM_STEP = {compiled._has_step!r}",
        f"PARAM_NAMES = {compiled._param_names!r}",
        f"BUFFER_NAMES = {compiled._buffer_names!r}",
        f"NUM_PARAMS_BUFFERS = {compiled._num_params_buffers}",
        f"OUT_SPEC = {out_spec_str!r}",
        "",
        "# Companion cache file written by export_cache(): parameter/buffer values,",
        "# residual wrapper globals, and the inductor/AOTAutograd cache artifact.",
        "# Override with NSC_WEIGHTS_PATH or by setting this module global before",
        "# the first forward() call.",
        f'WEIGHTS_PATH = _os.environ.get("NSC_WEIGHTS_PATH", {weights_path!r})',
        "",
    ]
    return parts


def _build_wrapper_section(compiled: NonStrictCompiled) -> list[str]:
    parts = [
        "# " + "=" * 70,
        "# 3. AOTAutograd runtime wrappers (captured from codegen)",
        "# " + "=" * 70,
    ]
    for rec in compiled._wrapper_records:
        parts.append(f"# ---- generated wrapper: {rec['name']} ----")
        parts.append(f"_SRC_{rec['name']} = {_py_str_literal(rec['source'])}")
        parts.append("")
    record_entries = []
    for rec in compiled._wrapper_records:
        record_entries.append(
            "    {"
            f'"name": {rec["name"]!r}, '
            f'"fn_name": {rec["fn_name"]!r}, '
            f'"plan": {rec["plan"]!r}, '
            f'"source": _SRC_{rec["name"]},'
            "}"
        )
    parts.append("_RECORDS = [\n" + ",\n".join(record_entries) + "\n]")
    parts.append("")
    return parts


def _build_python_source(
    compiled: NonStrictCompiled,
    inductor_chunks: list[str],
    weights_path: str,
) -> str:
    parts = [_GENERATED_HEADER, ""]
    parts.append("# " + "=" * 70)
    parts.append("# 1. Inductor output code (generated kernels + ``call`` entry point)")
    parts.append("# " + "=" * 70)
    # Inference and a custom step (e.g. forward+loss+backward fused into one
    # graph) both inline the single runnable Inductor ``call``.
    parts.append(inductor_chunks[0])
    parts.append("")
    parts.extend(_build_metadata_section(compiled, weights_path))
    parts.extend(_build_wrapper_section(compiled))
    parts.append("# " + "=" * 70)
    parts.append("# 4. Composition + runtime calling-convention wrapper")
    parts.append("# " + "=" * 70)
    parts.append(_DRIVER_SOURCE)
    return "\n".join(parts)


_DRIVER_SOURCE = (
    """\
_HELPER_NAMES = """
    + repr(_WRAPPER_HELPER_NAMES)
    + """
_CHAIN_NAMES = """
    + repr(_WRAPPER_CHAIN_NAMES)
    + '''

_blob = None
_params_buffers = None
_runtime_fn = None


def _load_blob():
    global _blob
    if _blob is None:
        _blob = _torch.load(WEIGHTS_PATH, weights_only=False)
    return _blob


def _load_params_buffers():
    return _load_blob()["param_buffer_flat"]


def _build_runtime():
    """Recompose the captured AOTAutograd wrappers around the Inductor ``call``.

    Each wrapper's globals are reconstructed from its plan: ``INNER`` binds the
    running composed callable, ``CALL`` the Inductor entry point, ``REC:<name>``
    a previously built wrapper (mutation/alias helpers), ``MOD:<name>`` a module,
    and everything else (subclass metadata, helper fns) is loaded as a residual
    global from the cache. The orchestration wrapper receives its inner callable
    as a parameter, with a no-op profiling hook and a null first-call context.
    """
    residuals = _load_blob().get("wrapper_residuals", {})
    by_name = {r["name"]: r for r in _RECORDS}
    built = {}

    def build_one(rec, inner):
        g = dict(residuals.get(rec["name"], {}))
        for key, tag in rec["plan"]:
            if tag == "INNER":
                g[key] = inner
            elif tag == "CALL":
                g[key] = call  # noqa: F821  (inlined Inductor entry point)
            elif tag.startswith("REC:"):
                g[key] = built[tag[4:]]
            elif tag.startswith("MOD:"):
                g[key] = _importlib.import_module(tag[4:])
        loc = {}
        exec(compile(rec["source"], "<" + rec["name"] + ">", "exec"), g, loc)
        return loc[rec["fn_name"]]

    for name in _HELPER_NAMES:
        if name in by_name:
            built[name] = build_one(by_name[name], None)

    inner = call  # noqa: F821
    for name in _CHAIN_NAMES:
        if name not in by_name:
            continue
        rec = by_name[name]
        if name == "runtime_wrapper_orchestration":
            fn = build_one(rec, None)

            def _driver(args, _fn=fn, _inner=inner):
                return _fn(_inner, _contextlib.nullcontext, lambda: None, args)

            inner = _driver
        else:
            inner = build_one(rec, inner)
        built[name] = inner
    return inner


def forward(*args, **kwargs):
    """Run the compiled model on user inputs.

    The lifted parameters/buffers are prepended to the flattened user inputs to
    form the boxed flat list ``[*params, *buffers, *user_inputs]`` that the
    composed runtime function expects; it is run under no_grad (this is an
    inference artifact) and its flat outputs are unflattened to the original
    output structure.
    """
    global _runtime_fn, _params_buffers
    if _runtime_fn is None:
        _runtime_fn = _build_runtime()
    if _params_buffers is None:
        _params_buffers = _load_params_buffers()
    user_flat, _ = _pytree.tree_flatten((args, kwargs))
    flat = [*_params_buffers, *user_flat]
    with _torch.no_grad():
        out = _runtime_fn(list(flat))
    return _pytree.tree_unflatten(list(out), _pytree.treespec_loads(OUT_SPEC))


if __name__ == "__main__":
    _pb = _load_params_buffers()
    print("Loaded", len(_pb), "parameter/buffer tensors from", WEIGHTS_PATH)
    print("Composed", len(_RECORDS), "AOTAutograd runtime wrapper(s).")
    print("forward() is ready; call it with the model's user inputs.")
'''
)


def _innermost_compiled_fn(artifact: Any) -> Any:
    """Walk ``__wrapped__`` to the innermost callable (the Inductor boxed call)."""
    fn = getattr(artifact, "_compiled_fn", None) or getattr(
        artifact, "inner_fn", artifact
    )
    seen: set[int] = set()
    while True:
        nxt = getattr(fn, "__wrapped__", None)
        if nxt is None or id(nxt) in seen:
            return fn
        seen.add(id(nxt))
        fn = nxt


def _classify_globals(
    globals_dict: dict[str, Any],
    call_id: int,
    id_to_name: dict[int, str],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Split a wrapper's globals into a serializable plan and residual values.

    The plan records, per global name, how to rebind it at load time: ``INNER``
    (the running composed callable), ``CALL`` (the Inductor entry point),
    ``REC:<name>`` (another captured wrapper), or ``MOD:<name>`` (a module).
    Everything else (subclass metadata, helper functions, dim lists) is a
    residual value carried in the cache and reconstructed by unpickling.
    """
    import types as _types

    plan: list[tuple[str, str]] = []
    residual: dict[str, Any] = {}
    for key, value in globals_dict.items():
        if key == "__builtins__":
            continue
        if key in ("compiled_fn", "_compiled_fn_"):
            plan.append((key, "INNER"))
        elif id(value) == call_id:
            plan.append((key, "CALL"))
        elif id(value) in id_to_name:
            plan.append((key, "REC:" + id_to_name[id(value)]))
        elif isinstance(value, _types.ModuleType):
            plan.append((key, "MOD:" + value.__name__))
        else:
            residual[key] = value
    return plan, residual


def _capture_compiled_wrappers(
    gm: torch.fx.GraphModule, flat_args: list[Any]
) -> tuple[Any, list[dict[str, Any]]]:
    """Compile ``gm`` while capturing every codegen'd AOTAutograd wrapper.

    All runtime wrappers funnel through ``subclass_codegen._compile_and_exec_source``;
    we wrap it to record each ``(artifact_name, source, fn_name, globals, fn)``,
    then classify the globals against the final composition so the wrappers can
    be re-emitted and re-composed in the exported python. The graph is lowered
    like inference (any backward is already in the graph as ops).
    """
    from torch._functorch._aot_autograd import subclass_codegen
    from torch._inductor import standalone_compile

    raw: list[dict[str, Any]] = []
    orig = subclass_codegen._compile_and_exec_source

    def hook(source, globals_dict, fn_name, artifact_name, wrapped_fn=None):
        fn = orig(source, globals_dict, fn_name, artifact_name, wrapped_fn)
        raw.append(
            {
                "name": artifact_name,
                "fn_name": fn_name,
                "source": source,
                "globals": globals_dict,
                "fn": fn,
            }
        )
        return fn

    subclass_codegen._compile_and_exec_source = hook
    try:
        with torch.no_grad():
            artifact = standalone_compile(
                gm, flat_args, dynamic_shapes="from_example_inputs"
            )
    finally:
        subclass_codegen._compile_and_exec_source = orig

    call_id = id(_innermost_compiled_fn(artifact))
    id_to_name = {id(r["fn"]): r["name"] for r in raw}
    records = []
    for r in raw:
        plan, residual = _classify_globals(r["globals"], call_id, id_to_name)
        records.append(
            {
                "name": r["name"],
                "fn_name": r["fn_name"],
                "source": r["source"],
                "plan": plan,
                "residual": residual,
            }
        )
    return artifact, records


class NonStrictCompiled:
    """A non-strict compiled callable produced by ``torch.non_strict_compile``."""

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        step: Callable[..., Any] | None = None,
        decompositions: dict | None = None,
    ) -> None:
        if step is not None and not isinstance(fn, torch.nn.Module):
            raise NonStrictCompileError("step requires fn to be an nn.Module.")
        self._fn = fn
        # A custom step computation (e.g. forward+loss+backward) traced into one
        # graph; whatever it returns is the output. Lowers like inference.
        self._step = step
        self._has_step = step is not None
        self._decompositions = decompositions
        self._artifact: Any = None
        self._param_names: list[str] = []
        self._buffer_names: list[str] = []
        self._param_buffer_flat: list[Any] = []
        self._num_params_buffers: int = 0
        self._in_spec: pytree.TreeSpec | None = None
        self._out_spec: pytree.TreeSpec | None = None
        self._gm: torch.fx.GraphModule | None = None
        # Inductor output code chunk(s) and the captured AOTAutograd wrapper
        # records, populated by example() and re-emitted by export_python().
        self._inductor_chunks: list[str] = []
        self._wrapper_records: list[dict[str, Any]] = []
        # Absolute path of the most recent export_cache(), baked into the
        # exported python so it knows where to load parameter values from.
        self._exported_cache_path: str | None = None
        # Set only on the load() path, where we wrap a reconstructed callable.
        self._loaded_forward: Callable[..., Any] | None = None

    @property
    def compiled(self) -> bool:
        return self._artifact is not None or self._loaded_forward is not None

    def _compile(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        capture = _capture(self._fn, args, kwargs, self._decompositions, self._step)
        artifact, records = _capture_compiled_wrappers(capture.gm, capture.flat_args)

        self._artifact = artifact
        self._wrapper_records = records
        self._inductor_chunks = self._extract_inductor_code()
        self._param_names = capture.param_names
        self._buffer_names = capture.buffer_names
        self._param_buffer_flat = capture.param_buffer_flat
        self._num_params_buffers = capture.num_params_buffers
        self._in_spec = capture.in_spec
        self._out_spec = capture.out_spec
        self._gm = capture.gm

    def example(self, *args: Any, **kwargs: Any) -> Any:
        """JIT-compile against the given example inputs and run once."""
        self._compile(args, kwargs)
        return self(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._loaded_forward is not None:
            return self._loaded_forward(*args, **kwargs)
        if self._artifact is None:
            if self._fn is None:
                raise NonStrictCompileError("this callable has no function to compile.")
            # Lazily JIT-compile on first call, like torch.compile.
            self._compile(args, kwargs)
        assert self._artifact is not None
        user_flat, _ = pytree.tree_flatten((args, kwargs))
        full_flat = [*self._param_buffer_flat, *user_flat]
        # The compiled graph is functional (any backward is already in it as
        # ops), so it always runs under no_grad.
        with torch.no_grad():
            out_flat = self._artifact(*full_flat)
        assert self._out_spec is not None
        return pytree.tree_unflatten(list(out_flat), self._out_spec)

    def _extract_inductor_code(self) -> list[str]:
        """Unpack the artifact and return the Inductor output module(s).

        Of the ``.py`` files Inductor emits we want the one defining the module-
        level ``call`` entry point (``call = runner.call``); the others are
        compile-time autotuning helpers not needed at runtime. The trailing
        ``__main__`` benchmark block is stripped so it is a clean module exposing
        ``call``.
        """
        if self._artifact is None:
            raise NonStrictCompileError("nothing to extract; not compiled.")
        chunks: list[str] = []
        with tempfile.TemporaryDirectory() as unpack_dir:
            self._artifact.save(path=unpack_dir, format="unpacked")
            for root, _dirs, files in os.walk(unpack_dir):
                for name in sorted(files):
                    if not name.endswith(".py"):
                        continue
                    with open(os.path.join(root, name)) as f:
                        text = f.read()
                    if "def call(" in text and "call = runner.call" in text:
                        marker = '\nif __name__ == "__main__":'
                        idx = text.find(marker)
                        if idx != -1:
                            text = text[:idx].rstrip() + "\n"
                        chunks.append(text)
        if not chunks:
            raise NonStrictCompileError(
                "could not locate the runnable Inductor output code for this artifact."
            )
        return chunks

    def export_python(self, path: str, *, weights_path: str | None = None) -> None:
        """Write the self-contained, executable end-to-end Python artifact.

        The generated module embeds the Inductor output code and a ``forward()``
        that loads parameter/buffer values from ``weights_path``. If not given,
        it points at the most recent ``export_cache(...)`` path, falling back to
        a ``<stem>_cache.bin`` file next to ``path``.
        """
        if self._artifact is None:
            raise NonStrictCompileError(
                "nothing to export; call `.example(...)` first."
            )
        if weights_path is None:
            if self._exported_cache_path is not None:
                weights_path = self._exported_cache_path
            else:
                stem, _ = os.path.splitext(os.path.abspath(path))
                weights_path = stem + "_cache.bin"
        weights_path = os.path.abspath(weights_path)
        source = _build_python_source(self, self._inductor_chunks, weights_path)
        with open(path, "w") as f:
            f.write(source)

    def parameters(self) -> list[Any]:
        """Return the lifted parameter tensors (e.g. to build an optimizer)."""
        return self._param_buffer_flat[: len(self._param_names)]

    def export_cache(self, path: str) -> None:
        """Write the binary cache for the exported python.

        Stores parameter/buffer values, the residual wrapper globals (subclass
        metadata, dim lists, etc.) the captured wrappers close over, and the real
        Inductor/AOTAutograd compiled-artifact bytes. On load that artifact primes
        the inductor cache and is reconstructed via a FxGraphCache hit, so reload
        does not re-trace/re-lower the graph and (on GPU) restores bundled Triton
        kernels -- this is the kernel-caching the cache provides.
        """
        if self._artifact is None:
            raise NonStrictCompileError(
                "nothing to export; call `.example(...)` first."
            )
        assert self._in_spec is not None and self._out_spec is not None
        self._exported_cache_path = os.path.abspath(path)

        wrapper_residuals = {
            rec["name"]: rec["residual"] for rec in self._wrapper_records
        }
        blob: dict[str, Any] = {
            "has_step": self._has_step,
            "param_buffer_flat": self._param_buffer_flat,
            "param_names": self._param_names,
            "buffer_names": self._buffer_names,
            "num_params_buffers": self._num_params_buffers,
            "in_spec": pytree.treespec_dumps(self._in_spec),
            "out_spec": pytree.treespec_dumps(self._out_spec),
            "wrapper_residuals": wrapper_residuals,
            # None if the artifact is not serializable (uncacheable graph); load()
            # then falls back to executing the self-contained python.
            "artifact_bytes": self._try_artifact_binary_bytes(),
        }
        torch.save(blob, path)

    def _try_artifact_binary_bytes(self) -> bytes | None:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            tmp = tf.name
        try:
            self._artifact.save(path=tmp, format="binary")
            with open(tmp, "rb") as f:
                return f.read()
        except Exception:
            return None
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @staticmethod
    def load(python_path: str, cache_path: str) -> NonStrictCompiled:
        """Reconstruct a compiled callable from exported python + cache files.

        When the cache holds the serialized artifact (the common case), it is
        reconstructed via ``CompiledArtifact.load`` -- which primes the inductor
        cache (FxGraphCache hit, no re-lowering; restores bundled kernels) and
        rebuilds the full AOTAutograd runtime, so this path handles tensor
        subclasses (DTensor) and custom steps. If the artifact was not
        serializable, it falls back to executing the self-contained
        ``python_path`` (which recompiles kernels from the inlined source).
        """
        if not os.path.exists(python_path):
            raise NonStrictCompileError(f"python artifact not found: {python_path}")
        # Unpickling the cache references classes in AOTAutograd's runtime; import
        # dynamo first so that import completes in a non-circular order (otherwise
        # a cold load can hit a runtime_wrappers <-> _dynamo circular import).
        import torch._dynamo

        blob = torch.load(cache_path, weights_only=False)
        has_step = blob.get("has_step", False)
        params = blob["param_buffer_flat"]
        out_spec = pytree.treespec_loads(blob["out_spec"])

        artifact_bytes = blob.get("artifact_bytes")
        if artifact_bytes is not None:
            forward = _make_cached_forward(artifact_bytes, params, out_spec)
        else:
            forward = _make_inlined_forward(python_path, cache_path, params)

        obj = NonStrictCompiled.__new__(NonStrictCompiled)
        obj._fn = None  # type: ignore[assignment]
        obj._step = None
        obj._has_step = has_step
        obj._decompositions = None
        obj._artifact = None
        obj._inductor_chunks = []
        obj._wrapper_records = []
        obj._param_names = blob["param_names"]
        obj._buffer_names = blob["buffer_names"]
        obj._param_buffer_flat = params
        obj._num_params_buffers = blob["num_params_buffers"]
        obj._in_spec = pytree.treespec_loads(blob["in_spec"])
        obj._out_spec = out_spec
        obj._gm = None
        obj._exported_cache_path = os.path.abspath(cache_path)
        obj._loaded_forward = forward
        return obj


def _make_cached_forward(
    artifact_bytes: bytes,
    params: list[Any],
    out_spec: pytree.TreeSpec,
) -> Callable[..., Any]:
    """Reconstruct the compiled artifact from the cache and drive it.

    ``CompiledArtifact.load`` primes the inductor cache from ``artifact_bytes``
    (so the graph is not re-traced/re-lowered and bundled kernels are restored)
    and rebuilds the full AOTAutograd runtime (subclass aware). The graph is
    functional, so it runs under no_grad.
    """
    from torch._inductor import CompiledArtifact

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
        tmp = tf.name
        tf.write(artifact_bytes)
    try:
        artifact = CompiledArtifact.load(path=tmp, format="binary")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    def forward(*args: Any, **kwargs: Any) -> Any:
        user_flat, _ = pytree.tree_flatten((args, kwargs))
        full_flat = [*params, *user_flat]
        with torch.no_grad():
            out_flat = artifact(*full_flat)
        return pytree.tree_unflatten(list(out_flat), out_spec)

    return forward


def _make_inlined_forward(
    python_path: str, cache_path: str, params: list[Any]
) -> Callable[..., Any]:
    """Fallback: execute the self-contained python (recompiles kernels)."""
    module_ns: dict[str, Any] = {"__name__": "_nsc_compiled_artifact"}
    with open(python_path) as f:
        code = f.read()
    exec(compile(code, python_path, "exec"), module_ns)
    module_ns["WEIGHTS_PATH"] = os.path.abspath(cache_path)
    # Share one parameter list so backward-populated grads land on the tensors
    # returned by parameters() (which an optimizer steps).
    module_ns["_params_buffers"] = params
    return module_ns["forward"]


def non_strict_compile(
    fn: Callable[..., Any],
    *,
    step: Callable[..., Any] | None = None,
    decompositions: dict | None = None,
) -> NonStrictCompiled:
    """Non-strict, make_fx-based ahead-of-time compilation.

    Returns a :class:`NonStrictCompiled` wrapping ``fn`` (a function or an
    ``nn.Module``). Call ``.example(*args, **kwargs)`` to JIT-compile against
    example inputs, then ``.export_python(path)`` / ``.export_cache(path)`` to
    persist, and ``torch.non_strict_compile.load(py, cache)`` to reload.

    ``step`` is an arbitrary computation ``step(model, *inputs)`` (``fn`` must be
    an ``nn.Module``) traced into one graph with the module's params/buffers
    lifted to inputs. It may do anything -- forward, loss, and
    ``torch.autograd.grad`` for the backward -- expressed as one graph; whatever
    it returns becomes the output (use ``autograd.grad``, not ``.backward()``,
    whose ``.grad`` side effect tracing does not capture). Training is therefore
    just a step that computes a loss and its gradients; a typical step returns
    ``(loss, grads)``. Because the backward is plain graph ops, this lowers like
    inference and exports to a fully self-contained python artifact. Use
    ``parameters()`` to build an optimizer from the lifted params.
    """
    torch._C._log_api_usage_once("torch.non_strict_compile")
    return NonStrictCompiled(fn, step=step, decompositions=decompositions)


# Allow ``torch.non_strict_compile.load("foo.py", "cache.bin")``.
non_strict_compile.load = NonStrictCompiled.load  # type: ignore[attr-defined]
