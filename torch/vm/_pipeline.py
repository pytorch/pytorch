# mypy: allow-untyped-defs
"""Runs a function's block passes on each region the VM traces, and infers the
region's versions from the guards they add.

Versions: v0 is the origin (the block as the VM runs it without a region).
v1 is the traced graph. A pass that adds no guard that varies between calls
edits the top version; one that does stacks a new version on top, whose
fallback is what the passes before it made. Each call runs the highest
version whose guards hold.

Block passes before ``Autograd`` (or without it) see the block's forward
graph. With ``Autograd``, the top version is differentiated with AOT
Autograd: passes up to ``Partition`` see the joint graph, and those after it
see the forward and backward graphs (or the inference graph, for a block
that needs no gradients). Lower versions run their graphs as they are, and
autograd differentiates the ops they run.
"""

import dataclasses
import logging
import os
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.fx
from torch.vm import _guards
from torch.vm._graph import (
    below_autocast,
    BoundLowering,
    copy_graph_module,
    direct,
    specialize_floats,
)
from torch.vm._types import exact, TensorType, Type


log = logging.getLogger(__name__)

VERIFY = bool(os.environ.get("TORCH_VM_VERIFY"))


@dataclasses.dataclass(frozen=True)
class VersionInfo:
    """One version of one region: v1 is the traced graph, higher numbers are
    more specialized and tried first (v0, the origin, runs when the region's
    own guard misses)."""

    region: str
    number: int
    passes: tuple[str, ...]
    guards: tuple[tuple[str, _guards.Check], ...]
    hits: int
    misses_by_guard: dict[str, int]


@dataclasses.dataclass
class Version:
    label: str
    passes: list[str]
    gm: torch.fx.GraphModule
    # Checked on every call: (input index, check).
    checks: list[tuple[int, _guards.Check]]
    run: Callable | None = None
    hits: int = 0
    # By the check that failed first.
    misses: dict[str, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Report:
    """What the passes did to one region, for ``vm.explain``."""

    name: str
    input_types: list[Type]
    versions: list[Version] = dataclasses.field(default_factory=list)
    notes: list[str] = dataclasses.field(default_factory=list)
    const_read: bool = False


def _name(p: Any) -> str:
    return getattr(p, "__name__", None) or repr(p)


def _copy(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # Graph deepcopy copies node meta shallowly, so example values are shared.
    out = copy_graph_module(gm)
    out.meta.update(gm.meta)
    out.meta.pop("vm_propagated", None)
    out.meta["vm_replacements"] = list(gm.meta.get("vm_replacements", ()))
    return out


def _tolerance(gm: torch.fx.GraphModule) -> dict[str, float]:
    reps = gm.meta.get("vm_replacements", ())
    if not reps:
        return {}
    return {"atol": max(r[1] for r in reps), "rtol": max(r[2] for r in reps)}


def _target(ph: list[str], i: int) -> str:
    return ph[i] if 0 <= i < len(ph) else "<state>" if i == -1 else "?"


class Pipeline:
    """A function's block passes, split at ``Autograd`` and ``Partition``."""

    def __init__(
        self,
        block_passes: Sequence[Any],
        autograd: Any,
        partition: Any,
        positions: dict[int, int],
        verify: bool = False,
    ) -> None:
        # positions: index in block_passes -> how many of Autograd/Partition precede it.
        self.forward = [p for i, p in enumerate(block_passes) if positions[i] == 0]
        self.joint = [p for i, p in enumerate(block_passes) if positions[i] == 1]
        self.post = [p for i, p in enumerate(block_passes) if positions[i] == 2]
        self.autograd = autograd
        self.partition = partition
        # With no block pass a region runs its graph as it was traced.
        self.as_traced = not block_passes and autograd is None
        self.reports: dict[str, Report] = {}
        self.verify = verify or VERIFY
        self.step_begins = [
            p.step_begin for p in block_passes if hasattr(p, "step_begin")
        ]

    def __call__(
        self,
        name: str,
        gm: torch.fx.GraphModule,
        inputs: list[Any],
        input_types: list[Type],
    ) -> tuple[Callable, bool]:
        """The region's function of its inputs, and whether a pass read a
        weight's value (so it's only for these weights)."""
        if name in self.reports:
            name = f"{name} #{sum(1 for k in self.reports if k.startswith(name))}"
        report = Report(name, input_types)
        self.reports[name] = report
        for t in input_types:
            if isinstance(t, TensorType) and t.is_const:
                t._on_const = lambda: setattr(report, "const_read", True)
        # The traced run's tensors, for the passes: every graph a pass is
        # given, and what it makes of it, refers to this one list.
        examples = gm.meta["vm_example_inputs"] = list(inputs)
        gm.meta["vm_stage"] = "forward"
        gm.meta["vm_const_inputs"] = [
            p.name
            for p, t in zip(gm.graph.find_nodes(op="placeholder"), input_types)
            if isinstance(t, TensorType) and t.is_const
        ]
        versions = [Version("v1", [], gm, [])]
        for p in self.forward:
            out = self._apply(p, versions[-1].gm, input_types, inputs, report)
            if out is None:
                continue
            new, lifted = out
            if lifted.dynamic:
                versions.append(
                    Version(
                        f"v{len(versions) + 1}",
                        [_name(p)],
                        new,
                        versions[-1].checks + lifted.dynamic,
                    )
                )
            else:
                versions[-1].gm = new
                versions[-1].passes.append(_name(p))
        # The graphs live as long as the function. Kept with them, the tensors
        # of the one call that was traced would too: the batch, every gradient
        # an optimizer's region was handed.
        del examples[:]
        for v in versions:
            v.gm.meta.pop("vm_example_inputs", None)
        report.versions = versions
        top = versions[-1]
        if self.autograd is not None:
            if any(isinstance(x, float) for x in inputs):
                # AOT Autograd bakes floats in: its version holds for these values.
                special = specialize_floats(top.gm, inputs)
                lifted = _guards.lift(special, inputs)
                versions.append(
                    Version(
                        f"v{len(versions) + 1}",
                        ["Autograd (floats specialized)"],
                        special,
                        top.checks + lifted.dynamic,
                    )
                )
                top = versions[-1]
            top.run = self._autograd(top.gm, inputs, input_types, report)
        for v in versions:
            if v.run is None:
                v.run = direct(v.gm) or v.gm.forward
        if self.verify:
            self._verify(versions, inputs, report)
        return _dispatch(versions), report.const_read

    def _apply(self, p, gm, input_types, inputs, report, stage="forward"):
        """Runs pass p on a copy of gm: (its graph, its guards), or None if it
        raised or added a guard that never holds here (its edits dropped)."""
        work = _copy(gm)
        try:
            new = p(work, input_types)
            if not isinstance(new, torch.fx.GraphModule):
                raise TypeError(
                    f"a block pass returns a GraphModule, not {type(new).__name__} (vm.lower_to wraps a callable)"
                )
            lifted = _guards.lift(new, gm.meta["vm_example_inputs"])
        except Exception as e:
            report.notes.append(
                f"{_name(p)} ({stage}): raised {type(e).__name__}: {e}; dropped"
            )
            warnings.warn(
                f"torch.vm: block pass {_name(p)} raised on {report.name} ({stage}), so it's skipped there: {type(e).__name__}: {e}",
                stacklevel=2,
            )
            return None
        if lifted.failed:
            shown = ", ".join(f"{c!r} on {n}" for n, c in lifted.failed)
            report.notes.append(
                f"{_name(p)} ({stage}): {shown} doesn't hold for this region's inputs; dropped"
            )
            return None
        if lifted.held:
            shown = ", ".join(f"{c!r} on {n}" for n, c in lifted.held)
            report.notes.append(
                f"{_name(p)} ({stage}): {shown} holds for every input this region runs on; merged"
            )
        changes = [
            r[0]
            for r in new.meta.get("vm_replacements", ())[
                len(gm.meta.get("vm_replacements", ())) :
            ]
            if r[3] == "changes"
        ]
        uneven = [c for _, c in lifted.dynamic if not c.rank_uniform]
        if changes and uneven:
            shown = ", ".join(map(repr, uneven))
            report.notes.append(
                f"{_name(p)} ({stage}): it changes collectives, so its guards must agree on every rank, and {shown} may not; dropped"
            )
            return None
        if lifted.dynamic and stage != "forward":
            report.notes.append(
                f"{_name(p)} ({stage}): guards that vary between calls are only supported before Autograd; dropped"
            )
            return None
        return new, lifted

    def _stage(self, passes, stage, report, input_types):
        """A compiler for AOT Autograd running ``passes`` on one graph."""

        def compile(g, ins):
            examples = list(ins)
            g.meta.update(vm_example_inputs=examples, vm_stage=stage)
            types = [
                input_types[i]
                if stage != "backward" and i < len(input_types)
                else exact(x)
                for i, x in enumerate(ins)
            ]
            for p in passes:
                out = self._apply(p, g, types, ins, report, stage)
                if out is not None:
                    g = out[0]
                    report.notes.append(f"{_name(p)} ran on the {stage} graph")
            # For the passes only, as the region's own (see __call__).
            del examples[:]
            g.meta.pop("vm_example_inputs", None)
            return direct(g) or g

        return compile

    def _autograd(self, gm, inputs, input_types, report):
        from torch._functorch.aot_autograd import aot_module_simplified
        from torch._inductor.decomposition import select_decomp_table
        from torch.vm.passes import Inductor

        def partition(joint, joint_inputs, **kwargs):
            joint = self._stage(self.joint, "joint", report, input_types)(
                joint, joint_inputs
            )
            if not isinstance(joint, torch.fx.GraphModule):
                raise TypeError(
                    "a pass between Autograd and Partition lowered the joint graph"
                )
            return self.partition.split(joint, joint_inputs, **kwargs)

        lowered_by_inductor = any(isinstance(p, Inductor) for p in self.post)
        return aot_module_simplified(
            below_autocast(gm),
            inputs,
            fw_compiler=self._stage(self.post, "forward_graph", report, input_types),
            bw_compiler=self._stage(self.post, "backward", report, input_types),
            inference_compiler=self._stage(self.post, "inference", report, input_types),
            partition_fn=partition,
            decompositions=select_decomp_table() if lowered_by_inductor else {},
            keep_inference_input_mutations=True,
        )

    def _verify(self, versions, inputs, report):
        """Compares each version with the one below on the traced run's inputs
        and drops those that differ (within their replacements' tolerances).
        They run as a region runs them: with autocast off (see
        torch._rust_inductor.trace_region)."""
        from torch._rust_inductor import autocast_on

        with torch.no_grad(), torch._C._ExcludeDispatchKeyGuard(autocast_on()):
            want = versions[0].run(*inputs)
            for v in versions[1:]:
                got = v.run(*inputs)
                try:
                    tolerance = _tolerance(v.gm)
                    torch.testing.assert_close(
                        list(got),
                        list(want),
                        rtol=tolerance.get("rtol"),
                        atol=tolerance.get("atol"),
                    )
                except AssertionError as e:
                    report.notes.append(
                        f"verify: {v.label} ({', '.join(v.passes)}) differs from the version below: {e}; dropped"
                    )
                    v.checks = [(-1, _Never())]
                    continue
                want = got


class _Never(_guards.Check):
    dynamic = True

    def holds(self, t, value_of):
        return False


def _dispatch(versions: list[Version]) -> Callable:
    def runner(v: Version) -> Callable:
        if v.run is None:
            raise AssertionError(f"version {v.label} has nothing to run")
        return v.run

    base, run_base = versions[0], runner(versions[0])
    guarded = [(v, runner(v)) for v in reversed(versions[1:])]

    def run(*inputs):
        for v, run_version in guarded:
            failed = _guards.failing(v.checks, inputs)
            if failed is None:
                v.hits += 1
                return run_version(*inputs)
            key = repr(failed[1])
            v.misses[key] = v.misses.get(key, 0) + 1
        base.hits += 1
        return run_base(*inputs)

    def bind(consts):
        """``run`` for one region: without its constants, which the versions
        lowered with ``lower_to(bind=)`` take here and the others are handed
        on every call."""

        def bound(call):
            if isinstance(call, BoundLowering):
                return call.bind(consts)
            return lambda *inputs: call(*inputs, *consts)

        calls = [(v, bound(call)) for v, call in guarded]
        call_base = bound(run_base)

        def run(*inputs):
            for v, call in calls:
                failed = _guards.failing(v.checks, (*inputs, *consts))
                if failed is None:
                    v.hits += 1
                    return call(*inputs)
                key = repr(failed[1])
                v.misses[key] = v.misses.get(key, 0) + 1
            base.hits += 1
            return call_base(*inputs)

        return run

    # For torch._rust_inductor.trace_region, which binds per region.
    calls = [run_base, *(call for _, call in guarded)]
    if any(isinstance(call, BoundLowering) for call in calls):
        run.bind = bind  # type: ignore[attr-defined]
    return run


def _selected(reports: dict[str, Report], block: int | None) -> list[Report]:
    return [r for name, r in reports.items() if block is None or f":bb{block} " in name]


def versions(reports: dict[str, Report], block: int | None = None) -> list[VersionInfo]:
    out = []
    for r in _selected(reports, block):
        for k, v in enumerate(r.versions):
            ph = [p.name for p in v.gm.graph.find_nodes(op="placeholder")]
            guards = tuple((_target(ph, i), c) for i, c in v.checks)
            out.append(
                VersionInfo(
                    r.name, k + 1, tuple(v.passes), guards, v.hits, dict(v.misses)
                )
            )
    return out


def explain(
    reports: dict[str, Report], block: int | None = None, view: str = "cfg"
) -> str:
    lines = []
    for r in _selected(reports, block):
        lines.append(
            f"{r.name}: v0 = origin, plus {len(r.versions)} version(s); tried highest first"
        )
        lines.append(f"  input_types: [{', '.join(map(repr, r.input_types))}]")
        for v in reversed(r.versions):
            misses = f", misses {v.misses}" if v.misses else ""
            lines.append(
                f"  version {v.label}  [{', '.join(v.passes) or 'traced graph'}]  hits {v.hits}{misses}"
            )
            ph = [p.name for p in v.gm.graph.find_nodes(op="placeholder")]
            for i, c in v.checks:
                lines.append(f"    guard {c!r} on {_target(ph, i)}")
            if view == "fx":
                lines.extend(f"    {line}" for line in v.gm.code.strip().splitlines())
        lines.extend(f"  note: {n}" for n in r.notes)
    return "\n".join(lines)
