from __future__ import annotations

import dataclasses
import functools
import inspect
import os
import pickle
import tempfile
import zipfile
from typing import Any, TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch.fx.experimental.proxy_tensor import make_fx as torch_make_fx


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def _static_names(names: str | Iterable[str] | None) -> tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        return (names,)
    return tuple(names)


def _backend_fn(backend: str | Callable[..., Any]) -> Callable[..., Any]:
    if callable(backend):
        return backend
    from torch._dynamo.backends.registry import lookup_backend

    return lookup_backend(backend)


@functools.lru_cache(maxsize=1)
def _tensor_metadata_fn() -> Callable[[torch.Tensor], Any]:
    try:
        from torch._inductor.codecache import extract_tensor_metadata_for_cache_key

        return extract_tensor_metadata_for_cache_key
    except Exception:
        from torch._dynamo.guards import extract_tensor_metadata

        return extract_tensor_metadata


def _freeze(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return (
            type(x),
            tuple(_freeze(getattr(x, f.name)) for f in dataclasses.fields(x)),
        )
    if isinstance(x, (tuple, list, torch.Size)):
        return tuple(_freeze(v) for v in x)
    return x


def _tensor_key(x: torch.Tensor) -> Any:
    return type(x), _freeze(_tensor_metadata_fn()(x))


def _check_static(name: str, value: Any) -> Any:
    if any(isinstance(leaf, torch.Tensor) for leaf in pytree.tree_leaves(value)):
        raise TypeError(
            f"static argument {name!r} must not contain Tensors; leave Tensor "
            "arguments dynamic so tensor metadata is used for caching"
        )
    try:
        hash(value)
    except TypeError as e:
        raise TypeError(f"static argument {name!r} must be hashable") from e
    return value


def _check_tensor_leaves(values: list[Any], what: str) -> None:
    for i, value in enumerate(values):
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"{what} must be Tensors or pytrees of Tensors; found leaf {i} "
                f"with type {type(value).__name__}"
            )


def _call(
    fn: Callable[..., Any], sig: inspect.Signature, values: dict[str, Any]
) -> Any:
    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name not in values:
            continue
        value = values[name]
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            args.append(value)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            args.extend(value)
        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            kwargs[name] = value
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            kwargs.update(value)
    return fn(*args, **kwargs)


def _signature_for_names(
    sig: inspect.Signature,
    names: tuple[str, ...],
) -> inspect.Signature:
    return sig.replace(parameters=[sig.parameters[name] for name in names])


def _flat_outputs(out: Any) -> tuple[tuple[Any, ...], pytree.TreeSpec]:
    flat_out, out_spec = pytree.tree_flatten(out)
    _check_tensor_leaves(flat_out, "outputs")
    return tuple(flat_out), out_spec


def _bind_dynamic(
    sig: inspect.Signature,
    static_order: tuple[str, ...],
    dynamic_order: tuple[str, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], tuple[Any, ...], list[Any], pytree.TreeSpec]:
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(bound.arguments)
    for name in static_order:
        _check_static(name, values[name])
    dynamic_values = tuple(values[name] for name in dynamic_order)
    flat_args, in_spec = pytree.tree_flatten(dynamic_values)
    _check_tensor_leaves(flat_args, "dynamic arguments")
    return values, tuple(values[name] for name in static_order), flat_args, in_spec


def _bind_artifact_call(
    sig: inspect.Signature,
    static_order: tuple[str, ...],
    dynamic_order: tuple[str, ...],
    static_values: tuple[Any, ...],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[list[Any], pytree.TreeSpec]:
    bound = _signature_for_names(sig, dynamic_order).bind(*args, **kwargs)
    bound.apply_defaults()
    values = dict(zip(static_order, static_values, strict=True))
    values.update(bound.arguments)
    flat_args, in_spec = pytree.tree_flatten(
        tuple(values[name] for name in dynamic_order)
    )
    _check_tensor_leaves(flat_args, "dynamic arguments")
    return flat_args, in_spec


def _trace(
    fn: Callable[..., Any],
    sig: inspect.Signature,
    values: dict[str, Any],
    dynamic_order: tuple[str, ...],
    flat_args: list[Any],
    in_spec: pytree.TreeSpec,
) -> tuple[torch.fx.GraphModule, pytree.TreeSpec]:
    out_spec_holder: dict[str, pytree.TreeSpec] = {}

    def flat_fn(*new_flat_args: Any) -> tuple[Any, ...]:
        new_values = dict(values)
        new_values.update(
            zip(
                dynamic_order,
                pytree.tree_unflatten(new_flat_args, in_spec),
                strict=True,
            )
        )
        flat_out, out_spec = _flat_outputs(_call(fn, sig, new_values))
        out_spec_holder["spec"] = out_spec
        return flat_out

    gm = torch_make_fx(flat_fn)(*flat_args)
    return gm, out_spec_holder["spec"]


def make_fx(
    fn: Callable[..., Any],
    *,
    static_argnames: str | Iterable[str] | None = None,
) -> Callable[..., torch.fx.GraphModule]:
    """Trace a function with ``make_fx`` using static named arguments.

    This is a prototype convenience wrapper around
    ``torch.fx.experimental.proxy_tensor.make_fx``. It always traces with
    ``tracing_mode="fake"``.

    Non-static arguments must be ``torch.Tensor`` values or pytrees of tensors.
    Arguments named in ``static_argnames`` are captured as compile-time
    constants, must be hashable, and must not contain tensors.

    The returned ``GraphModule`` uses the flattened dynamic tensor leaves as its
    calling convention, matching the graph shape consumed by ``jit`` and
    ``compile_artifact`` in this prototype.

    Example:
        >>> from nonstrict import make_fx
        >>> def f(xs, *, scale):
        ...     return {"y": xs["x"] * scale}
        >>> gm = make_fx(f, static_argnames="scale")({"x": torch.ones(2)}, scale=2)
        >>> gm(torch.ones(2))
        (tensor([2., 2.]),)
    """
    sig = inspect.signature(fn)
    static_names = frozenset(_static_names(static_argnames))
    unknown = static_names.difference(sig.parameters)
    if unknown:
        raise ValueError(f"unknown static_argnames: {sorted(unknown)}")

    static_order = tuple(name for name in sig.parameters if name in static_names)
    dynamic_order = tuple(name for name in sig.parameters if name not in static_names)

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> torch.fx.GraphModule:
        values, _, flat_args, in_spec = _bind_dynamic(
            sig, static_order, dynamic_order, args, kwargs
        )
        out_spec_holder: dict[str, pytree.TreeSpec] = {}

        def flat_fn(*new_flat_args: Any) -> tuple[Any, ...]:
            new_values = dict(values)
            new_values.update(
                zip(
                    dynamic_order,
                    pytree.tree_unflatten(new_flat_args, in_spec),
                    strict=True,
                )
            )
            flat_out, out_spec = _flat_outputs(_call(fn, sig, new_values))
            out_spec_holder["spec"] = out_spec
            return flat_out

        gm = torch_make_fx(flat_fn, tracing_mode="fake")(*flat_args)
        gm._nonstrict_jit_in_spec = in_spec
        gm._nonstrict_jit_out_spec = out_spec_holder["spec"]
        return gm

    return wrapped


def _call_flat_artifact(
    artifact: Callable[..., Any],
    out_spec: pytree.TreeSpec,
    flat_args: list[Any],
) -> Any:
    flat_out = artifact(*flat_args)
    if out_spec.num_leaves == 1 and not isinstance(flat_out, (tuple, list)):
        flat_out = (flat_out,)
    return pytree.tree_unflatten(flat_out, out_spec)


@dataclasses.dataclass(frozen=True)
class Target:
    device: str
    torch_version: str
    cuda_version: str | None

    @staticmethod
    def current() -> Target:
        return Target(
            device="cuda" if torch.cuda.is_available() else "cpu",
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
        )


class CompiledArtifact:
    def __init__(
        self,
        artifact: Any,
        *,
        target: Target,
        sig: inspect.Signature,
        static_order: tuple[str, ...],
        dynamic_order: tuple[str, ...],
        static_values: tuple[Any, ...],
        in_spec: pytree.TreeSpec,
        out_spec: pytree.TreeSpec,
    ) -> None:
        self.artifact = artifact
        self.target = target
        self.sig = sig
        self.static_order = static_order
        self.dynamic_order = dynamic_order
        self.static_values = static_values
        self.in_spec = in_spec
        self.out_spec = out_spec

    def run(self, *args: Any, **kwargs: Any) -> Any:
        flat_args, in_spec = _bind_artifact_call(
            self.sig,
            self.static_order,
            self.dynamic_order,
            self.static_values,
            args,
            kwargs,
        )
        if in_spec != self.in_spec:
            raise ValueError("compiled artifact called with different input pytree")
        return _call_flat_artifact(self.artifact, self.out_spec, flat_args)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def save(self, path: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "inductor_artifact.bin")
            self.artifact.save(path=artifact_path, format="binary")
            metadata = {
                "target": self.target,
                "sig": self.sig,
                "static_order": self.static_order,
                "dynamic_order": self.dynamic_order,
                "static_values": self.static_values,
                "in_spec": pytree.treespec_dumps(self.in_spec),
                "out_spec": pytree.treespec_dumps(self.out_spec),
            }
            with zipfile.ZipFile(path, "w") as zf:
                zf.write(artifact_path, "inductor_artifact.bin")
                zf.writestr("metadata.pkl", pickle.dumps(metadata))

    @staticmethod
    def load(path: str) -> CompiledArtifact:
        from torch._inductor import CompiledArtifact as InductorCompiledArtifact

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = os.path.join(tmpdir, "inductor_artifact.bin")
            with zipfile.ZipFile(path, "r") as zf:
                zf.extract("inductor_artifact.bin", tmpdir)
                metadata = pickle.loads(zf.read("metadata.pkl"))
            artifact = InductorCompiledArtifact.load(
                path=artifact_path, format="binary"
            )
        metadata["in_spec"] = pytree.treespec_loads(metadata["in_spec"])
        metadata["out_spec"] = pytree.treespec_loads(metadata["out_spec"])
        return CompiledArtifact(artifact, **metadata)


class _NonstrictCompiler:
    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        static_argnames: str | Iterable[str] | None,
        backend: str | Callable[..., Any],
    ) -> None:
        self.fn = fn
        self.sig = inspect.signature(fn)
        self.backend = backend

        static_names = frozenset(_static_names(static_argnames))
        unknown = static_names.difference(self.sig.parameters)
        if unknown:
            raise ValueError(f"unknown static_argnames: {sorted(unknown)}")

        self.static_order = tuple(
            name for name in self.sig.parameters if name in static_names
        )
        self.dynamic_order = tuple(
            name for name in self.sig.parameters if name not in static_names
        )
        self.compile_backend = _backend_fn(backend)
        self.cache: dict[Any, tuple[Callable[..., Any], pytree.TreeSpec]] = {}
        self.artifact_cache: dict[Any, CompiledArtifact] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        values, static_values, flat_args, in_spec = _bind_dynamic(
            self.sig, self.static_order, self.dynamic_order, args, kwargs
        )

        key = in_spec, static_values, tuple(_tensor_key(x) for x in flat_args)
        entry = self.cache.get(key)
        if entry is None:
            gm, out_spec = _trace(
                self.fn,
                self.sig,
                values,
                self.dynamic_order,
                flat_args,
                in_spec,
            )
            entry = self.compile_backend(gm, list(flat_args)), out_spec
            self.cache[key] = entry

        compiled, out_spec = entry
        return _call_flat_artifact(compiled, out_spec, flat_args)

    def cache_size(self) -> int:
        return len(self.cache)

    def cache_clear(self) -> None:
        self.cache.clear()
        self.artifact_cache.clear()

    def compile_artifact(
        self,
        *args: Any,
        artifact_target: Target | None = None,
        **kwargs: Any,
    ) -> CompiledArtifact:
        if self.backend != "inductor":
            raise NotImplementedError(
                "compile_artifact() currently only supports backend='inductor'"
            )
        target = Target.current() if artifact_target is None else artifact_target
        if target != Target.current():
            raise NotImplementedError(
                "compile_artifact() only supports the current target in this prototype"
            )

        values, static_values, flat_args, in_spec = _bind_dynamic(
            self.sig, self.static_order, self.dynamic_order, args, kwargs
        )
        key = target, in_spec, static_values, tuple(_tensor_key(x) for x in flat_args)
        entry = self.artifact_cache.get(key)
        if entry is None:
            gm, out_spec = _trace(
                self.fn,
                self.sig,
                values,
                self.dynamic_order,
                flat_args,
                in_spec,
            )
            import torch._inductor

            artifact = torch._inductor.standalone_compile(
                gm,
                list(flat_args),
                dynamic_shapes="from_graph",
            )
            entry = CompiledArtifact(
                artifact,
                target=target,
                sig=self.sig,
                static_order=self.static_order,
                dynamic_order=self.dynamic_order,
                static_values=static_values,
                in_spec=in_spec,
                out_spec=out_spec,
            )
            self.artifact_cache[key] = entry
        return entry


def jit(
    fn: Callable[..., Any] | None = None,
    *,
    static_argnames: str | Iterable[str] | None = None,
    backend: str | Callable[..., Any] = "inductor",
) -> Callable[..., Any]:
    """JIT compile a function from pytree Tensor inputs to pytree Tensor outputs.

    This is a small prototype of a JAX-like ``jit`` API. Non-static arguments
    must be ``torch.Tensor`` values or pytrees of tensors. Unlike ``jax.jit``,
    this prototype does not abstract Python numeric scalars as dynamic scalar
    array values; pass them as tensors or mark them static. Static arguments are
    named with ``static_argnames`` and are captured as compile-time constants;
    they must be hashable and must not contain tensors.

    Args:
        fn: Function to compile. May be omitted for decorator use.
        static_argnames: A string or iterable of argument names to treat as
            compile-time constants.
        backend: Dynamo backend name or backend callable. Defaults to
            ``"inductor"``.

    Example:
        >>> from nonstrict import jit
        >>> def f(xs, *, scale):
        ...     return {"y": xs["x"] * scale}
        >>> jf = jit(f, static_argnames="scale")
        >>> jf({"x": torch.ones(2)}, scale=2)
        {"y": tensor([2., 2.])}
    """
    if fn is None:
        return functools.partial(
            jit,
            static_argnames=static_argnames,
            backend=backend,
        )

    compiler = _NonstrictCompiler(
        fn,
        static_argnames=static_argnames,
        backend=backend,
    )

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return compiler(*args, **kwargs)

    wrapped.cache_size = compiler.cache_size  # type: ignore[attr-defined]
    wrapped.cache_clear = compiler.cache_clear  # type: ignore[attr-defined]
    return wrapped


def compile_artifact(
    fn: Callable[..., Any] | None = None,
    *,
    static_argnames: str | Iterable[str] | None = None,
    backend: str | Callable[..., Any] = "inductor",
    target: Target | None = None,
) -> Callable[..., CompiledArtifact]:
    """Return a callable that materializes one compiled artifact specialization.

    This is the artifact-facing counterpart to ``jit``. It shares the same
    function, static argument, and backend options, but does not require first
    constructing a jitted callable.

    Example:
        >>> artifact = compile_artifact(f, static_argnames="scale")(
        ...     torch.ones(2), scale=2
        ... )
        >>> artifact.run(torch.ones(2))
    """
    if fn is None:
        return functools.partial(
            compile_artifact,
            static_argnames=static_argnames,
            backend=backend,
            target=target,
        )

    compiler = _NonstrictCompiler(
        fn,
        static_argnames=static_argnames,
        backend=backend,
    )

    @functools.wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> CompiledArtifact:
        return compiler.compile_artifact(
            *args,
            artifact_target=target,
            **kwargs,
        )

    def cache_size() -> int:
        return len(compiler.artifact_cache)

    wrapped.cache_size = cache_size  # type: ignore[attr-defined]
    wrapped.cache_clear = compiler.cache_clear  # type: ignore[attr-defined]
    return wrapped


__all__ = ["jit", "make_fx", "compile_artifact", "CompiledArtifact", "Target"]
