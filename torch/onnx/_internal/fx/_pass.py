from __future__ import annotations

import abc

import contextlib
import difflib

import io

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


@_beartype.beartype
def _transform_diagnose_call_message_formatter(
    run: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> str:
    # TODO(bowbao): Update signature to varargs and varkwargs to avoid manual unpacking.
    assert len(args) >= 1 and isinstance(args[0], Transform)
    transform = args[0]

    return f"Running {transform.__class__.__name__} pass."


def fx_graph_tabular(graph: torch.fx.Graph) -> str:
    """Return the Graph nodes in tabular format. Equivalent to stdout of `graph.print_tabular()`.

    Args:
        graph: The Graph to print.

    Returns:
        The Graph printed in a tabular format.
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        graph.print_tabular()
    return f.getvalue()


class Transform(abc.ABC):
    """Base class for FX graph transformations to be used by FX-ONNX exporter.

    This class provides builtin support for transformation recording using the diagnostics system.

    TODO(bowbao): Add more overrideable methods in call hierarchy
    Methods in the Transform class can be overridden to customize the behavior of the
    transform. The following methods can be overridden::

        _run()
            +-- run_node()
                +-- placeholder()
                +-- get_attr()
                +-- call_function()
                +-- call_method()
                +-- call_module()
                +-- output()

    The granularity of overriding is up to the user. And it affects the granularity of
    the diagnostics information. For example, if `_run()` is overridden, the
    diagnostics information will only contain graph level transformation. Instead,
    if `call_function()` is overridden, the diagnostics information will additionally
    contain the node level information of `call_function()`.

    Example: TODO(bowbao): Fill example once more overrideable methods are added.
    """

    """The module to be transformed."""
    module: torch.fx.GraphModule

    def __init__(self, module: torch.fx.GraphModule):
        """Initialize the transform.

        Args:
            module: The module to be transformed.
        """
        self.module = module

    @abc.abstractmethod
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        ...

    @diagnostics.diagnose_call(
        rule=diagnostics.rules.fx_pass,
        exception_report_level=diagnostics.levels.ERROR,
        diagnostic_message_formatter=_transform_diagnose_call_message_formatter,
    )
    def run(self, *args, **kwargs) -> torch.fx.GraphModule:
        """Run the transform on `self.module`.

        Note that this method may or may not mutate `self.module`, and the returned
        `GraphModule` could be either `self.module` or a new `GraphModule`.

        Args:
            *args: Positional arguments for `self.module` to run.
            **kwargs: Keyword arguments for `self.module` to run.
        """
        diagnostic = diagnostics.export_context().inflight_diagnostic(
            rule=diagnostics.rules.fx_pass
        )
        # Gather graph information before transform.
        old_readable_graph = self.module.print_readable(print_output=False)
        old_tabular = fx_graph_tabular(self.module.graph)

        module = self._run(*args, **kwargs)

        # Gather graph information after transform.
        new_readable_graph = module.print_readable(print_output=False)
        new_tabular = fx_graph_tabular(module.graph)

        graph_diff = "".join(
            FxReadableGraphDiffer().compare(
                old_readable_graph.splitlines(keepends=True),
                new_readable_graph.splitlines(keepends=True),
            )
        )
        diagnostic.with_additional_message(f"### Graph diff:\n```\n{graph_diff}\n```")

        tabular_diff = "".join(
            FxReadableGraphDiffer().compare(
                old_tabular.splitlines(keepends=True),
                new_tabular.splitlines(keepends=True),
            )
        )
        diagnostic.with_additional_message(
            f"### Tabular diff:\n```\n{tabular_diff}\n```"
        )

        return module
