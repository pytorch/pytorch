# mypy: allow-untyped-defs
"""
Prototype: Partial / Regional AOTInductor for inference.

This is an **experimental** API exposed as ``torch._inductor.regional_aoti``.
It lets users compile selected hot submodules of a model with AOTInductor while
keeping the rest of the model eager. Incompatible regions gracefully fall back to
eager execution instead of failing the whole model.

The implementation is a thin orchestration layer over existing PyTorch APIs:

- ``torch.func.functional_call`` to make params/buffers explicit runtime inputs,
- ``torch.export.export`` to capture each region independently,
- ``torch._inductor.aoti_compile_and_package`` / ``aoti_load_package`` to compile
  and load AOTI artifacts.

It intentionally does NOT depend on any Meta-internal serving system.

Public (experimental) entry points: ``mark_region`` + ``compile_regions``.
The returned model carries a ``RegionCompileResult`` and its regions are
``CompiledAOTIRegion`` instances; export/compile failures raise
``AOTIRegionExportError`` / ``AOTIRegionCompileError``.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import os
import tempfile
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch.utils._ordered_set import OrderedSet


log = logging.getLogger(__name__)

# Stable attribute used to tag a class / forward method / instance as an AOTI region.
_REGION_CONFIG_ATTR = "_torch_aoti_region_config"

# Attribute stashed on a model returned by ``compile_regions`` for later inspection.
_RESULT_ATTR = "_regional_aoti_result"

# Public (experimental) API. Supporting types (``region``, ``discover_regions``,
# ``CompiledAOTIRegion``, the ``RegionInfo`` / ``RegionCompileResult`` /
# ``AOTIRegionConfig`` dataclasses, and the ``AOTIRegion*Error`` classes) remain
# importable but are intentionally not advertised yet.
__all__ = [
    "mark_region",
    "compile_regions",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class AOTIRegionError(RuntimeError):
    """Base class for all regional AOTI errors."""


class AOTIRegionExportError(AOTIRegionError):
    """Raised when ``torch.export.export`` fails for a region."""


class AOTIRegionCompileError(AOTIRegionError):
    """Raised when AOTInductor compilation fails for a region."""


class AOTIRegionParityError(AOTIRegionError):
    """Raised when a compiled region fails the numerical parity check."""


class AOTIRegionLoadError(AOTIRegionError):
    """Raised when a compiled region artifact cannot be loaded."""


class AOTIRegionRuntimeError(AOTIRegionError):
    """Raised when a compiled region fails at runtime (e.g. schema mismatch)."""


# ---------------------------------------------------------------------------
# Config / info dataclasses
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class AOTIRegionConfig:
    """Per-region export / fallback configuration.

    Args:
        dynamic_shapes: Mapping from the region ``forward`` parameter name to an
            export dynamic-shape spec (e.g. ``{"x": {0: torch.export.Dim("batch")}}``).
            Params/buffers are always kept static.
        fallback: Optional per-region override of the global compile-time fallback
            policy (``"eager"`` or ``"error"``).
    """

    dynamic_shapes: dict[str, Any] | None = None
    fallback: str | None = None


@dataclasses.dataclass
class RegionInfo:
    """State tracked for a single discovered region."""

    name: str
    module: nn.Module = dataclasses.field(repr=False)
    config: AOTIRegionConfig = dataclasses.field(repr=False)
    # One of: discovered, captured, not_reached, exported, compiled, fallback.
    status: str = "discovered"
    fallback_reason: str | None = None
    package_path: str | None = None
    captured_args: tuple[Any, ...] | None = dataclasses.field(default=None, repr=False)
    captured_kwargs: dict[str, Any] | None = dataclasses.field(default=None, repr=False)


@dataclasses.dataclass
class RegionCompileResult:
    """Result of :func:`compile_regions`."""

    regions: list[RegionInfo]

    def compiled(self) -> list[RegionInfo]:
        return [r for r in self.regions if r.status == "compiled"]

    def fallback(self) -> list[RegionInfo]:
        return [r for r in self.regions if r.status == "fallback"]

    def not_reached(self) -> list[RegionInfo]:
        return [r for r in self.regions if r.status == "not_reached"]

    def report(self) -> str:
        lines = ["Regional AOTI report:"]
        for r in self.regions:
            name = r.name or "<root>"
            line = f"  [{r.status:>10}] {name}"
            if r.fallback_reason:
                line += f"  (reason: {r.fallback_reason})"
            lines.append(line)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


# ---------------------------------------------------------------------------
# Marking / discovery
# ---------------------------------------------------------------------------
_FALLBACK_CHOICES = ("eager", "error")


def _validate_fallback(fallback, argname: str = "fallback") -> None:
    """Reject an unrecognized fallback policy. ``None`` means "unset" and is allowed."""
    if fallback is not None and fallback not in _FALLBACK_CHOICES:
        raise ValueError(f"{argname} must be 'eager' or 'error', got {fallback!r}")


def region(_obj=None, *, dynamic_shapes=None, fallback=None):
    """Mark an ``nn.Module`` class or its ``forward`` method as an AOTI region.

    Can be used with or without arguments::

        class Attention(nn.Module):
            @region(dynamic_shapes={"x": {0: torch.export.Dim("batch")}})
            def forward(self, x, mask=None): ...


        @region()
        class Block(nn.Module): ...
    """
    config = AOTIRegionConfig(
        dynamic_shapes=dynamic_shapes,
        fallback=fallback,
    )

    def deco(obj):
        setattr(obj, _REGION_CONFIG_ATTR, config)
        return obj

    if _obj is not None:
        # Used as a bare decorator: @region
        return deco(_obj)
    return deco


def mark_region(
    module: nn.Module,
    config: AOTIRegionConfig | None = None,
    *,
    dynamic_shapes=None,
    fallback=None,
) -> nn.Module:
    """Programmatically mark a module *instance* as an AOTI region."""
    if not isinstance(module, nn.Module):
        raise TypeError(f"mark_region expects an nn.Module, got {type(module)!r}")
    if config is None:
        config = AOTIRegionConfig(
            dynamic_shapes=dynamic_shapes,
            fallback=fallback,
        )
    # Instance attribute (goes into __dict__, takes priority over class markers).
    module.__dict__[_REGION_CONFIG_ATTR] = config
    return module


def _get_region_config(module: nn.Module) -> AOTIRegionConfig | None:
    """Return the region config for a module, or None if it is not a region."""
    # 1. Instance-level (programmatic) marking wins.
    cfg = module.__dict__.get(_REGION_CONFIG_ATTR)
    if isinstance(cfg, AOTIRegionConfig):
        return cfg
    cls = type(module)
    # 2. Class-level decorator. Walk the MRO so a subclass of a decorated class
    #    is still recognized, consistent with the forward-method lookup below.
    for klass in cls.__mro__:
        cfg = klass.__dict__.get(_REGION_CONFIG_ATTR)
        if isinstance(cfg, AOTIRegionConfig):
            return cfg
    # 3. forward-method decorator.
    fwd = getattr(cls, "forward", None)
    cfg = getattr(fwd, _REGION_CONFIG_ATTR, None)
    if isinstance(cfg, AOTIRegionConfig):
        return cfg
    return None


def _is_nested_under(name: str, ancestors: list[str]) -> bool:
    """True if ``name`` is a strict descendant of any name in ``ancestors``."""
    for other in ancestors:
        if other == name:
            continue
        if other == "":
            # Root is an ancestor of every non-root module.
            return True
        if name.startswith(other + "."):
            return True
    return False


def discover_regions(model: nn.Module) -> list[RegionInfo]:
    """Walk ``model.named_modules()`` and return only top-level marked regions.

    Nested marked regions are skipped in favor of their marked ancestor so each
    tensor is compiled at most once.
    """
    marked: list[tuple[str, nn.Module, AOTIRegionConfig]] = []
    seen_ids: OrderedSet[int] = OrderedSet()
    for name, mod in model.named_modules():
        cfg = _get_region_config(mod)
        # A single instance mounted under multiple attribute paths must be
        # compiled once (under its first name); otherwise it would export
        # repeatedly and only one path would end up pointing at the wrapper.
        if cfg is not None and id(mod) not in seen_ids:
            seen_ids.add(id(mod))
            marked.append((name, mod, cfg))

    all_names = [n for n, _, _ in marked]
    regions: list[RegionInfo] = []
    for name, mod, cfg in marked:
        if _is_nested_under(name, all_names):
            continue
        regions.append(RegionInfo(name=name, module=mod, config=cfg))
    return regions


# ---------------------------------------------------------------------------
# Input capture
# ---------------------------------------------------------------------------
def _detach_inputs(obj, clone: bool):
    def f(x):
        if isinstance(x, torch.Tensor):
            y = x.detach()
            return y.clone() if clone else y
        return x

    return pytree.tree_map(f, obj)


def capture_region_inputs(
    model: nn.Module,
    regions: list[RegionInfo],
    example_inputs,
    example_kwargs=None,
    *,
    clone_inputs: bool = False,
) -> list[RegionInfo]:
    """Run one eager forward, capturing (args, kwargs) into each reached region.

    The outer model behavior is unchanged: this only observes inputs via
    forward pre-hooks and runs under ``torch.no_grad()``.
    """
    example_inputs = tuple(example_inputs) if example_inputs is not None else ()
    example_kwargs = dict(example_kwargs) if example_kwargs else {}

    captures: dict[int, tuple] = {}
    handles = []

    def make_hook():
        def hook(module, args, kwargs):
            captures[id(module)] = (
                _detach_inputs(args, clone_inputs),
                _detach_inputs(kwargs, clone_inputs),
            )

        return hook

    for r in regions:
        handles.append(
            r.module.register_forward_pre_hook(make_hook(), with_kwargs=True)
        )

    try:
        with torch.no_grad():
            model(*example_inputs, **example_kwargs)
    finally:
        for h in handles:
            h.remove()

    for r in regions:
        if id(r.module) in captures:
            r.captured_args, r.captured_kwargs = captures[id(r.module)]
            r.status = "captured"
        elif r.status == "discovered":
            r.status = "not_reached"
    return regions


# ---------------------------------------------------------------------------
# Functionalized export
# ---------------------------------------------------------------------------
class _FunctionalizedRegion(nn.Module):
    """Wrap a region so params/buffers are explicit runtime inputs.

    The original module is stored inside a plain Python list so it is NOT
    registered as a submodule; otherwise its parameters would be re-lifted as
    state instead of being passed in explicitly.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._region = [module]

    def forward(self, params, buffers, args, kwargs):
        return torch.func.functional_call(
            self._region[0], {**params, **buffers}, tuple(args), dict(kwargs)
        )


def _positional_param_names(fn) -> list[str]:
    names = []
    for p in inspect.signature(fn).parameters.values():
        if p.name == "self":
            continue
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            names.append(p.name)
    return names


def _spec_for(value, user_spec):
    """Translate a user dynamic-shape spec for a single input value.

    Non-tensor values and unspecified tensors are static (``None``). A user
    spec (dict/list of Dims) is passed through unchanged.
    """
    if not isinstance(value, torch.Tensor):
        return None
    if user_spec is None:
        return None
    return user_spec


def _build_dynamic_shapes(orig_module, params, buffers, args, kwargs, config):
    """Build the ``dynamic_shapes`` tuple for the functionalized signature.

    Signature is ``forward(params, buffers, args, kwargs)`` so the returned tuple
    is always length 4: ``(params_static, buffers_static, args_specs, kwargs_specs)``.
    Params/buffers are marked fully static.
    """
    user = (config.dynamic_shapes or {}) if config is not None else {}

    params_spec = dict.fromkeys(params, None)
    buffers_spec = dict.fromkeys(buffers, None)

    pos_names = _positional_param_names(orig_module.forward)
    args_specs = []
    for i, val in enumerate(args):
        nm = pos_names[i] if i < len(pos_names) else None
        args_specs.append(_spec_for(val, user.get(nm) if nm is not None else None))

    kwargs_specs = {k: _spec_for(v, user.get(k)) for k, v in kwargs.items()}

    return (params_spec, buffers_spec, tuple(args_specs), kwargs_specs)


def _export_region(rinfo: RegionInfo):
    module = rinfo.module
    params = {n: p.detach() for n, p in module.named_parameters()}
    buffers = {n: b.detach() for n, b in module.named_buffers()}

    args = tuple(rinfo.captured_args or ())
    kwargs = dict(rinfo.captured_kwargs or {})

    func_mod = _FunctionalizedRegion(module)
    export_inputs = (params, buffers, args, kwargs)
    dynamic_shapes = _build_dynamic_shapes(
        module, params, buffers, args, kwargs, rinfo.config
    )

    try:
        ep = torch.export.export(
            func_mod,
            export_inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )
    except Exception as e:
        raise AOTIRegionExportError(
            f"Failed to export region {rinfo.name or '<root>'!r}: {e}"
        ) from e
    rinfo.status = "exported"
    return ep


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------
def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("/", "_") if name else "__root__"


def _compile_region(rinfo: RegionInfo, ep, cache_dir: str | None) -> str:
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        pkg_path = os.path.join(cache_dir, _sanitize(rinfo.name) + ".pt2")
    else:
        fd, pkg_path = tempfile.mkstemp(suffix=".pt2")
        os.close(fd)

    try:
        path = torch._inductor.aoti_compile_and_package(ep, package_path=pkg_path)
    except Exception as e:
        if cache_dir is None:
            # Remove the orphaned temp artifact so failures don't leak .pt2 files.
            try:
                os.unlink(pkg_path)
            except OSError:
                pass
        raise AOTIRegionCompileError(
            f"Failed to AOTI-compile region {rinfo.name or '<root>'!r}: {e}"
        ) from e
    rinfo.package_path = path
    return path


# ---------------------------------------------------------------------------
# Runtime wrapper
# ---------------------------------------------------------------------------
class CompiledAOTIRegion(nn.Module):
    """Runtime wrapper that routes a region through its compiled AOTI artifact.

    The original module is kept as a child (named ``region``) so it continues to
    own its params/buffers. State-dict hooks make the extra ``region.`` prefix
    transparent, so ``state_dict()`` / ``load_state_dict()`` keep working with
    the original key names.
    """

    def __init__(
        self,
        original_module: nn.Module,
        package_path: str,
        *,
        runtime_fallback: str = "error",
        region_name: str = "",
    ) -> None:
        super().__init__()
        self.region = original_module
        self._package_path = package_path
        self._runtime_fallback = runtime_fallback
        self._region_name = region_name
        self._compiled = None  # lazily loaded AOTICompiledModel

        # Make the "region." infix transparent in state_dict / load_state_dict.
        self._register_state_dict_hook(_strip_region_prefix_hook)
        self._register_load_state_dict_pre_hook(
            _add_region_prefix_hook, with_module=True
        )

    def _load(self):
        if self._compiled is None:
            try:
                self._compiled = torch._inductor.aoti_load_package(self._package_path)
            except Exception as e:
                raise AOTIRegionLoadError(
                    f"Failed to load compiled region from {self._package_path!r}: {e}"
                ) from e
        return self._compiled

    def forward(self, *args, **kwargs):
        # Gather *current* params/buffers so load_state_dict updates take effect.
        params = dict(self.region.named_parameters())
        buffers = dict(self.region.named_buffers())
        try:
            compiled = self._load()
            return compiled(params, buffers, tuple(args), dict(kwargs))
        except AOTIRegionLoadError:
            raise
        except Exception as e:
            if self._runtime_fallback == "eager":
                log.warning(
                    "Regional AOTI: runtime fallback to eager for region %r: %s",
                    self._region_name or "<region>",
                    e,
                )
                return self.region(*args, **kwargs)
            raise AOTIRegionRuntimeError(
                f"Compiled region failed at runtime: {e}"
            ) from e


def _strip_region_prefix_hook(module, state_dict, prefix, local_metadata):
    region_prefix = prefix + "region."
    for key in list(state_dict.keys()):
        if key.startswith(region_prefix):
            new_key = prefix + key[len(region_prefix) :]
            state_dict[new_key] = state_dict.pop(key)
    return state_dict


def _add_region_prefix_hook(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # Only re-prefix keys that correspond to this region's own state; leave
    # unrelated keys (e.g. extras passed under strict=False) untouched. This
    # matters for the root wrapper where prefix is empty.
    region_prefix = prefix + "region."
    own_keys = OrderedSet(module.region.state_dict().keys())
    for key in list(state_dict.keys()):
        if not key.startswith(prefix):
            continue
        suffix = key[len(prefix) :]
        if suffix in own_keys:
            state_dict[region_prefix + suffix] = state_dict.pop(key)


def _replace_submodule(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if child.isdigit() and isinstance(parent, (nn.ModuleList, nn.Sequential)):
        # Ordered containers address children by position; assign through
        # __setitem__ so the container's internal _modules is updated.
        parent[int(child)] = new_module
    else:
        setattr(parent, child, new_module)


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------
def _check_parity(rinfo: RegionInfo, wrapper: CompiledAOTIRegion, rtol, atol) -> None:
    args = tuple(rinfo.captured_args or ())
    kwargs = dict(rinfo.captured_kwargs or {})
    # Force the compiled path while measuring parity. If the wrapper is left on
    # runtime_fallback="eager", a broken artifact would silently return eager
    # output and the check would compare eager-vs-eager and trivially pass.
    saved_fallback = wrapper._runtime_fallback
    wrapper._runtime_fallback = "error"
    try:
        with torch.no_grad():
            eager_out = rinfo.module(*args, **kwargs)
            comp_out = wrapper(*args, **kwargs)
    finally:
        wrapper._runtime_fallback = saved_fallback
    try:
        torch.testing.assert_close(comp_out, eager_out, rtol=rtol, atol=atol)
    except AssertionError as e:
        raise AOTIRegionParityError(
            f"Region {rinfo.name or '<root>'!r} failed parity check: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
def compile_regions(
    model: nn.Module,
    example_inputs,
    example_kwargs=None,
    *,
    cache_dir: str | None = None,
    fallback: str = "eager",
    runtime_fallback: str = "error",
    check_parity: bool = False,
    parity_rtol: float = 1e-3,
    parity_atol: float = 1e-3,
) -> nn.Module:
    """Compile marked regions of ``model`` with AOTInductor.

    Returns a model that routes successfully-compiled regions through their AOTI
    artifacts while keeping every other part (and any failed region under
    ``fallback="eager"``) eager. The returned object carries a
    :class:`RegionCompileResult` (see ``report()``).
    """
    if fallback not in ("eager", "error"):
        raise ValueError(f"fallback must be 'eager' or 'error', got {fallback!r}")
    if runtime_fallback not in ("eager", "error"):
        raise ValueError(
            f"runtime_fallback must be 'eager' or 'error', got {runtime_fallback!r}"
        )

    regions = discover_regions(model)
    if not regions:
        log.warning(
            "Regional AOTI: no marked regions found on %s", type(model).__name__
        )

    capture_region_inputs(model, regions, example_inputs, example_kwargs)

    root_wrapper: nn.Module | None = None

    for r in regions:
        eff_fallback = r.config.fallback or fallback

        if r.status == "not_reached":
            reason = "region was not reached during example-input capture"
            if eff_fallback == "error":
                # Not an export failure: export never ran for an unreached region.
                raise AOTIRegionError(f"Region {r.name or '<root>'!r}: {reason}")
            r.status = "not_reached"
            r.fallback_reason = reason
            log.warning("Regional AOTI: %s (%s)", reason, r.name or "<root>")
            continue

        path: str | None = None
        try:
            ep = _export_region(r)
            path = _compile_region(r, ep, cache_dir)
            wrapper = CompiledAOTIRegion(
                r.module,
                path,
                runtime_fallback=runtime_fallback,
                region_name=r.name,
            )
            if check_parity:
                _check_parity(r, wrapper, parity_rtol, parity_atol)
            if r.name == "":
                root_wrapper = wrapper
            else:
                _replace_submodule(model, r.name, wrapper)
            r.status = "compiled"
        except (
            AOTIRegionExportError,
            AOTIRegionCompileError,
            AOTIRegionParityError,
            AOTIRegionLoadError,
            AOTIRegionRuntimeError,
        ) as e:
            if eff_fallback == "error":
                raise
            # A downstream step (e.g. parity) can fail after _compile_region
            # already wrote the artifact. For temp artifacts (no cache_dir)
            # that would otherwise leak, unlink the orphaned .pt2 file.
            if path is not None and cache_dir is None:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            r.status = "fallback"
            r.fallback_reason = str(e)
            log.warning(
                "Regional AOTI: region %r fell back to eager: %s",
                r.name or "<root>",
                e,
            )

    result = RegionCompileResult(regions=regions)
    out_model = root_wrapper if root_wrapper is not None else model
    setattr(out_model, _RESULT_ATTR, result)
    return out_model
