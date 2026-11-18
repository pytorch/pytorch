"""torch.vm: a function run in a VM, with two lists of passes.

.. warning::
    This is an experiment and a temporary API. Everything about it may change.

``torch.vm`` is the engine of :mod:`torch.rewrite` with the passes where one
looks for them. A function is turned into one control-flow graph and run in a
VM; what is done to it is two lists::

    @torch.vm
    def forward(model, x): ...


    @torch.vm(
        cfg_passes=torch.vm.DefaultCFGPasses,
        block_passes=torch.vm.DefaultBlockPasses,
    )
    def forward(model, x): ...

``cfg_passes`` rewrite the control-flow graph of the function, once, before
it runs: calls are inlined, loops unrolled. ``block_passes`` rewrite one
basic block at a time, after its first runs were watched: tensor operations
become direct calls of their kernels, or go to a graph compiler. The bare
decorator is the two default lists.

A preset is the keyword arguments of a configuration, to be unpacked::

    @torch.vm(**torch.vm.FAST)  # the default passes, and Inductor for the tensor work
    def forward(model, x): ...

A pass that has a setting is made with it::

    # loops of up to 16 iterations, where the default is 8
    cfg_passes = (torch.vm.inlining_pass(), torch.vm.loop_unrolling_pass(16))

:func:`configure` makes the keyword arguments for the default passes with
some of them set differently::

    passes = torch.vm.configure(loop_unrolling_pass=16)


    @torch.vm(**passes)
    def forward(model, x): ...
"""

import sys
import types
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from torch.rewrite import (
    fuse_checked,
    Graphs,
    is_available,
    load_passes,
    memory,
    Mismatch,
    regions,
    rewrite as _rewrite,
    swap,
    Unsupported,
)
from torch.rewrite._passes import Builtin


__all__ = [
    "DefaultBlockPasses",
    "DefaultCFGPasses",
    "FAST",
    "Mismatch",
    "Unsupported",
    "configure",
    "dce_pass",
    "fold_pass",
    "fuse_checked",
    "graphs_pass",
    "inlining_pass",
    "is_available",
    "load_passes",
    "loop_unrolling_pass",
    "memory",
    "method_calls_pass",
    "module_shortcuts_pass",
    "narrow_dims_pass",
    "regions",
    "resolve_dispatch_pass",
    "swap",
    "tensor_accessors_pass",
    "vm",
]


# --------------------------------------------------------------------- passes


def inlining_pass() -> Builtin:
    """Calls of Python functions defined in the same module are replaced by
    their graph, under a guard on the callee; the call itself stays as the
    other side of that guard. Over the control-flow graph."""
    return Builtin("inline", True, "inlining_pass")


def loop_unrolling_pass(max_trip_count: int = 8) -> Builtin:
    """``for i in range(k)`` with a constant ``k`` of at most
    ``max_trip_count`` is replaced by ``k`` copies of its body, under a guard
    on ``range``. A loop whose copies would come to more than 600
    instructions is left as it is. Over the control-flow graph, after
    inlining."""
    if isinstance(max_trip_count, bool) or not isinstance(max_trip_count, int):
        kind = type(max_trip_count).__name__
        raise TypeError(f"loop_unrolling_pass takes a number of iterations, not {kind}")
    if max_trip_count < 1:
        raise ValueError(
            f"loop_unrolling_pass({max_trip_count}): at least one iteration; "
            "to unroll nothing, leave the pass out of cfg_passes"
        )
    settings = (("max_trip_count", max_trip_count),)
    return Builtin("unroll", True, "loop_unrolling_pass", settings)


def module_shortcuts_pass() -> Builtin:
    """The code of ``nn.Module`` in front of every call of a module and every
    read of a parameter, buffer or submodule is skipped whenever a few checks
    show what it would do: a call of a module without hooks goes straight to
    its ``forward``. Over the control-flow graph."""
    return Builtin("shortcuts", True, "module_shortcuts_pass")


class _GraphsPass(Graphs):
    __slots__ = ()

    def __repr__(self) -> str:
        name = self.backend if isinstance(self.backend, str) else "a callable"
        return f"torch.vm.graphs_pass({name!r})"


def graphs_pass(
    backend: str | Callable[..., Any] = "inductor",
    *,
    min_ops: int = 4,
    max_variants: int = 32,
    freeze: bool | None = None,
) -> Graphs:
    """The tensor work of a block, with what it calls, goes to a graph
    compiler: :func:`torch.rewrite.graphs`, which has the arguments. It sees
    the instructions as the function wrote them, so it is the first of
    ``block_passes``."""
    return _GraphsPass(backend, min_ops, max_variants, freeze)


def resolve_dispatch_pass() -> Builtin:
    """Operations that were seen to be one dispatcher call become direct
    calls of the kernel behind the dispatcher. Over a block."""
    return Builtin("resolve_dispatch", False, "resolve_dispatch_pass")


def tensor_accessors_pass() -> Builtin:
    """``x.shape[i]``, ``x.size(i)``, ``x.dim()`` and ``x.ndim`` of a tensor
    become plain integers of the graph. Over a block."""
    return Builtin("tensor_accessors", False, "tensor_accessors_pass")


def method_calls_pass() -> Builtin:
    """``x.f(...)`` calls that stay generic are made without a bound method
    object. Over a block."""
    return Builtin("method_calls", False, "method_calls_pass")


def narrow_dims_pass() -> Builtin:
    """A comparison of a dimension with a constant is folded to the outcome
    last seen, under a guard that pins the dimension to the sizes with that
    outcome. Over a block, in front of :func:`fold_pass`; not a default."""
    return Builtin("narrow_dims", False, "narrow_dims_pass")


def fold_pass() -> Builtin:
    """Truthiness tests and branches on constants are folded. Over a block."""
    return Builtin("fold", False, "fold_pass")


def dce_pass() -> Builtin:
    """Instructions whose result is not needed are dropped. Over a block."""
    return Builtin("dce", False, "dce_pass")


DefaultCFGPasses: tuple[Builtin, ...] = (
    inlining_pass(),
    loop_unrolling_pass(),
    module_shortcuts_pass(),
)
"""What ``@torch.vm`` does to the control-flow graph of a function:
:func:`inlining_pass`, :func:`loop_unrolling_pass` with its default of 8
iterations, :func:`module_shortcuts_pass`."""

DefaultBlockPasses: tuple[Builtin, ...] = (
    resolve_dispatch_pass(),
    tensor_accessors_pass(),
    method_calls_pass(),
    fold_pass(),
    dce_pass(),
)
"""What ``@torch.vm`` does to each basic block, in this order and again until
nothing changes: :func:`resolve_dispatch_pass`, :func:`tensor_accessors_pass`,
:func:`method_calls_pass`, :func:`fold_pass`, :func:`dce_pass`."""

FAST: Mapping[str, Any] = types.MappingProxyType(
    {
        "cfg_passes": DefaultCFGPasses,
        "block_passes": (graphs_pass("inductor"), *DefaultBlockPasses),
    }
)
"""The default passes, with the tensor work handed to AOTAutograd and
Inductor first (:func:`graphs_pass`): ``torch.vm(**torch.vm.FAST)``. It
compiles for seconds where the default does not compile at all."""


# ------------------------------------------------------------------ the entry


def _made_by(one: Any) -> str | None:
    """The name of the function of this module that makes such a pass."""
    if isinstance(one, Builtin):
        return one.made_by
    return "graphs_pass" if isinstance(one, Graphs) else None


def _checked(cfg_passes: Iterable[Any], block_passes: Iterable[Any]) -> list[Any]:
    """The two lists as the one the engine takes, each pass in its list."""
    cfg, block = tuple(cfg_passes), tuple(block_passes)
    for one in cfg:
        if not (isinstance(one, Builtin) and one.cfg):
            known = isinstance(one, (Builtin, Graphs))
            where = "; it goes in block_passes" if known else ""
            raise TypeError(
                f"cfg_passes has {one!r}, which is not a pass over the control-flow graph{where}"
            )
    for one in block:
        if isinstance(one, Builtin) and one.cfg:
            raise TypeError(
                f"block_passes has {one!r}, which is a pass over the control-flow "
                "graph; it goes in cfg_passes"
            )
        if not (isinstance(one, (Builtin, Graphs)) or callable(one)):
            kind = type(one).__name__
            raise TypeError(
                f"block_passes has a {kind}: a pass is what a function of torch.vm "
                "makes (torch.vm.fold_pass()), or a callable taking a region"
            )
    graphs = [i for i, one in enumerate(block) if isinstance(one, Graphs)]
    if graphs and any(isinstance(one, Builtin) for one in block[: graphs[0]]):
        raise ValueError(
            f"block_passes has {block[graphs[0]]!r} after a pass that rewrites single "
            "instructions: it sees the instructions as the function wrote them, so it goes first"
        )
    names = [_made_by(one) for one in (*cfg, *block) if _made_by(one) is not None]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise ValueError(f"{twice[0]} is given more than once")
    if "inlining_pass" in names and "loop_unrolling_pass" in names:
        if names.index("loop_unrolling_pass") < names.index("inlining_pass"):
            raise ValueError(
                "cfg_passes has loop_unrolling_pass in front of inlining_pass: "
                "the engine inlines first, so that the loops of an inlined "
                "function are unrolled too"
            )
    return [*cfg, *block]


def vm(
    fn: Any = None,
    *,
    cfg_passes: Iterable[Any] | None = None,
    block_passes: Iterable[Any] | None = None,
    fallback: bool = True,
    skip_autograd: bool = True,
    unboxed: bool = True,
    callees: bool = True,
    demote: bool | None = None,
    max_versions: int = 4,
    max_respecializations: int = 6,
    keep_memory: bool | None = None,
) -> Any:
    """Runs ``fn`` as a control-flow graph in the VM, with passes applied.

    ``torch.vm`` itself is this function: ``@torch.vm``, ``@torch.vm(...)``
    and ``torch.vm(fn, ...)`` all work.

    Args:
        fn: a Python function, a bound method, or an ``nn.Module`` (whose
            ``forward`` is run). Left out when used as a decorator with
            arguments.
        cfg_passes: the passes over the control-flow graph of the function,
            applied once before it runs. Defaults to
            :data:`DefaultCFGPasses`; an empty list leaves the graph as the
            bytecode gives it.
        block_passes: the passes over a basic block, applied in this order,
            and again until none changes the block, once the block has run
            and was watched. Defaults to :data:`DefaultBlockPasses`. An entry
            is what a function of this module makes (``fold_pass()``,
            ``graphs_pass("inductor")``), or a pass of one's own: a callable
            ``pass_(region) -> bool`` (see :func:`swap`), or one written in
            Rust (:func:`load_passes`). With both lists empty the function
            runs in the generic tier of the VM, which does what the
            interpreter does.
        fallback, skip_autograd, unboxed, callees, demote, max_versions,
        max_respecializations, keep_memory: as for
            :func:`torch.rewrite.rewrite`.

    Returns:
        A callable with the signature of ``fn``, with ``dump()``, ``stats()``
        and ``sites()``.
    """
    options: dict[str, Any] = {
        "fallback": fallback,
        "skip_autograd": skip_autograd,
        "unboxed": unboxed,
        "callees": callees,
        "demote": demote,
        "max_versions": max_versions,
        "max_respecializations": max_respecializations,
        "keep_memory": keep_memory,
    }
    passes = _checked(
        DefaultCFGPasses if cfg_passes is None else cfg_passes,
        DefaultBlockPasses if block_passes is None else block_passes,
    )
    return _rewrite(fn, passes=passes, **options)


# The passes that configure names, in the place each has in its list.
_CFG = ("inlining_pass", "loop_unrolling_pass", "module_shortcuts_pass")
_BLOCK = (
    "graphs_pass",
    "resolve_dispatch_pass",
    "tensor_accessors_pass",
    "method_calls_pass",
    "narrow_dims_pass",
    "fold_pass",
    "dce_pass",
)
_OPTIONS = (
    "fallback",
    "skip_autograd",
    "unboxed",
    "callees",
    "demote",
    "max_versions",
    "max_respecializations",
    "keep_memory",
)


def configure(
    preset: Mapping[str, Any] | None = None, /, **settings: Any
) -> dict[str, Any]:
    """The keyword arguments of ``torch.vm`` for the default passes, with
    some of them set differently::

        passes = torch.vm.configure(loop_unrolling_pass=16)


        @torch.vm(**passes)
        def forward(model, x): ...

    Args:
        preset: what to start from instead of the defaults: a preset such as
            :data:`FAST`, or what an earlier call returned.
        settings: per pass, by the name of the function that makes it, what
            to make it with: one argument (``loop_unrolling_pass=16``,
            ``graphs_pass="aot_eager"``), a tuple of arguments, a dict of
            keyword arguments, ``True`` for the pass as it comes, and
            ``False`` or ``None`` to leave it out. A pass that is in a list
            keeps its place there; one that is not goes where it belongs
            (``graphs_pass`` first, ``narrow_dims_pass`` in front of
            ``fold_pass``). Any other name is an option of ``torch.vm``
            (``demote=False``) and is handed through.

    Returns:
        A dict with ``cfg_passes``, ``block_passes`` and the options given.
    """
    found = dict(preset) if preset is not None else {}
    cfg = list(found.get("cfg_passes", DefaultCFGPasses))
    block = list(found.get("block_passes", DefaultBlockPasses))
    module = sys.modules[__name__]
    for name, value in settings.items():
        if name in _OPTIONS:
            found[name] = value
            continue
        if name not in _CFG and name not in _BLOCK:
            known = ", ".join((*_CFG, *_BLOCK, *_OPTIONS))
            raise TypeError(f"configure() does not know {name!r}; it knows {known}")
        passes, order = (cfg, _CFG) if name in _CFG else (block, _BLOCK)
        at = next((i for i, one in enumerate(passes) if _made_by(one) == name), None)
        if value is None or value is False:
            if at is not None:
                del passes[at]
            continue
        make = getattr(module, name)
        if value is True:
            made = make()
        elif isinstance(value, tuple):
            made = make(*value)
        elif isinstance(value, dict):
            made = make(**value)
        else:
            made = make(value)
        if at is not None:
            passes[at] = made
            continue
        # In front of the first pass that comes after it in the usual order.
        later = order[order.index(name) + 1 :]
        at = next(
            (i for i, one in enumerate(passes) if _made_by(one) in later), len(passes)
        )
        passes.insert(at, made)
    found["cfg_passes"], found["block_passes"] = tuple(cfg), tuple(block)
    return found


class _CallableModule(types.ModuleType):
    # torch.vm is both the entry point and the namespace: torch.vm(fn) and
    # torch.vm.FAST.
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return vm(*args, **kwargs)


sys.modules[__name__].__class__ = _CallableModule
