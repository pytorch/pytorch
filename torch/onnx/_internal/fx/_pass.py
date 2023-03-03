from __future__ import annotations

import abc

import contextlib
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

        # Patch `difflib.SequenceMatcher` to set `autojunk` to `False`.
        # More details are in the class docstring.
        @contextlib.contextmanager
        def patch_sequence_matcher_init():
            original_init = difflib.SequenceMatcher.__init__

            def patched_init(self, isjunk=None, a="", b="", autojunk=True):
                original_init(self, isjunk, a, b, autojunk=False)

            difflib.SequenceMatcher.__init__ = patched_init  # type: ignore[assignment]
            try:
                yield
            finally:
                difflib.SequenceMatcher.__init__ = original_init  # type: ignore[assignment]

        with patch_sequence_matcher_init():
            yield from super().compare(a, b)


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
