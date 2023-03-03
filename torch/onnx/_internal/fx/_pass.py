from __future__ import annotations

import abc
import difflib

from typing import Any, Callable, Dict, Generator, Sequence, Tuple

import torch
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics


class FxReadableGraphDiffer(difflib.Differ):
    """Differ specialized for fx readable graph.

    The only difference between this and official `difflib.Differ` is the `autojunk`
    argument for `difflib.SequenceMatcher` class. In this class, `autojunk` is set
    to `False`. This is to prevent `difflib.SequenceMatcher` recognizing stacktrace
    messages in fx readable graph as junk, as these messages tend to be long (>200)
    and repeat multiple times, which falls under the junk filter criteria.

    `Reference: Automatic junk heuristic <https://docs.python.org/3/library/difflib.html>`_
    """

    def compare(self, a: Sequence[str], b: Sequence[str]) -> Generator[str, None, None]:
        cruncher = difflib.SequenceMatcher(None, a, b, autojunk=False)
        for tag, alo, ahi, blo, bhi in cruncher.get_opcodes():
            if tag == "replace":
                g = self._fancy_replace(a, alo, ahi, b, blo, bhi)  # type: ignore[attr-defined]
            elif tag == "delete":
                g = self._dump("-", a, alo, ahi)  # type: ignore[attr-defined]
            elif tag == "insert":
                g = self._dump("+", b, blo, bhi)  # type: ignore[attr-defined]
            elif tag == "equal":
                g = self._dump(" ", a, alo, ahi)  # type: ignore[attr-defined]
            else:
                raise ValueError("unknown tag %r" % (tag,))

            yield from g


class Pass(abc.ABC):
    ...


@_beartype.beartype
def _transform_diagnose_call_message_formatter(
    run: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> str:
    # TODO(bowbao): Update signature to varargs and varkwargs to avoid manual unpacking.
    assert (
        len(args) >= 2
        and isinstance(args[0], Transform)
        and isinstance(args[1], torch.fx.GraphModule)
    )
    transform = args[0]

    return f"Running {transform.__class__.__name__} pass."


class Transform(Pass):
    @abc.abstractmethod
    def _run(
        self, graph_module: torch.fx.GraphModule, *args, **kwargs
    ) -> torch.fx.GraphModule:
        ...

    @diagnostics.diagnose_call(
        rule=diagnostics.rules.fx_pass,
        exception_report_level=diagnostics.levels.ERROR,
        diagnostic_message_formatter=_transform_diagnose_call_message_formatter,
    )
    def run(
        self, graph_module: torch.fx.GraphModule, *args, **kwargs
    ) -> torch.fx.GraphModule:
        diagnostic = diagnostics.export_context().inflight_diagnostic(
            rule=diagnostics.rules.fx_pass
        )
        old_readable_graph = graph_module.print_readable(print_output=False)
        graph_module = self._run(graph_module, *args, **kwargs)
        new_readable_graph = graph_module.print_readable(print_output=False)
        graph_diff = "".join(
            FxReadableGraphDiffer().compare(
                old_readable_graph.splitlines(keepends=True),
                new_readable_graph.splitlines(keepends=True),
            )
        )
        diagnostic.with_additional_message(f"### Graph diff:\n```\n{graph_diff}\n```")

        return graph_module
