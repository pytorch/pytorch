from __future__ import annotations

import abc

import contextlib
import difflib

import io
import logging
import sys

from typing import Any, Callable, Optional, Tuple

import torch
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics, onnxfunction_dispatcher


@contextlib.contextmanager
def _patch_difflib_sequence_matcher_init():
    """Context patching `difflib.SequenceMatcher` for fx readable graph.

    Under this context, the `autojunk` argument of `difflib.SequenceMatcher` will always
    be considered as `False`. This is to prevent `difflib.SequenceMatcher` recognizing
    stacktrace messages in fx readable graph as junk, as these messages tend to be long (>200)
    and repeat multiple times, which falls under the junk filter criteria.

    `difflib.SequenceMatcher` is used underneath by all sorts of diffing functions
    in `difflib`, including `difflib.unified_diff`, `difflib.ndiff`, `difflib.context_diff`.
    Unfortunately, there is no way to pass `autojunk` argument to these functions, and
    they all default to `True`. This context patching will affect all of them.

    `Reference: Automatic junk heuristic <https://docs.python.org/3/library/difflib.html>`_
    """
    original_init = difflib.SequenceMatcher.__init__

    def patched_init(self, isjunk=None, a="", b="", autojunk=True):
        original_init(self, isjunk, a, b, autojunk=False)

    difflib.SequenceMatcher.__init__ = patched_init  # type: ignore[assignment]
    try:
        yield
    finally:
        difflib.SequenceMatcher.__init__ = original_init  # type: ignore[assignment]


def _unified_diff(a: str, b: str) -> str:
    """Return a string containing the unified diff of two strings.

    This function calls a patched version of `difflib.unified_diff` with `autojunk` set
    to `False` for `difflib.SequenceMatcher` class. More details can be found in
    `_patch_difflib_sequence_matcher_init` function.

    Args:
        a: The first string.
        b: The second string.

    Returns:
        The unified diff of the two strings. If there is no diff, return "<no diff>".

    Example::

        >>> a = '''class GraphModule(torch.nn.Module):
        ...     def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view = input_ids.view(-1, 3);  input_ids = None
        ... '''
        >>> b = '''class <lambda>(torch.nn.Module):
        ...     def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
        ...         # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        ...         view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
        ... '''
        >>> print(_unified_diff(a, b))
        ---
        +++
        @@ -1,4 +1,4 @@
        -class GraphModule(torch.nn.Module):
        -    def forward(self, input_ids : torch.Tensor, attention_mask : torch.Tensor):
        +class <lambda>(torch.nn.Module):
        +    def forward(self, input_ids: i64[1, 3], attention_mask: i64[1, 3]):
                # File: /modeling.py:770, code: input_ids = input_ids.view(-1, input_shape[-1])
        -        view = input_ids.view(-1, 3);  input_ids = None
        +        view: i64[1, 3] = torch.ops.aten.view.default(input_ids, [-1, 3]);  input_ids = None
    """

    a_list = a.splitlines(keepends=True)
    b_list = b.splitlines(keepends=True)

    with _patch_difflib_sequence_matcher_init():
        # Set `n` to `sys.maxsize` to show entire graph when there is a diff.
        diff = "".join(difflib.unified_diff(a_list, b_list, n=sys.maxsize))

    if not diff:
        return "<no diff>"
    return diff


@_beartype.beartype
def _transform_diagnose_call_message_formatter(
    run: Callable,
    self: Transform,
    *args: Any,
    **kwargs: Any,
) -> str:
    return f"Running {self.__class__.__name__} pass. "


def maybe_fx_graph_tabular(graph: torch.fx.Graph) -> Optional[str]:
    """Return the Graph nodes in tabular format. Equivalent to stdout of `graph.print_tabular()`.
    If `tabulate` is not installed, return `None`.

    Args:
        graph: The Graph to print.

    Returns:
        The Graph printed in a tabular format. None if `tabulate` is not installed.
    """
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            graph.print_tabular()
        except ImportError:
            return None
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

    diagnostic_context: diagnostics.DiagnosticContext
    """The diagnostic context for recording diagnostics."""

    module: torch.fx.GraphModule
    """The module to be transformed."""

    fake_mode: Optional[fake_tensor.FakeTensorMode]
    """The existing fake mode detected from `self.module`."""

    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
    ):
        """Initialize the transform.

        Args:
            diagnostic_context: The diagnostic context for recording diagnostics.
            module: The module to be transformed.
        """
        self.diagnostic_context = diagnostic_context
        self.module = module
        self.fake_mode = self._detect_fake_mode()

    def _detect_fake_mode(self) -> Optional[fake_tensor.FakeTensorMode]:
        """Detect fake mode from the graph.

        Scan through all nodes in graph and their meta['val'] to detect fake mode.
        """
        fake_tensors = [node.meta.get("val") for node in self.module.graph.nodes]
        with maybe_disable_fake_tensor_mode():
            return torch._dynamo.utils.detect_fake_mode(fake_tensors)

    def _maybe_fakefy_args(
        self, fake_mode: Optional[fake_tensor.FakeTensorMode], *args: Any
    ) -> Tuple[Any, ...]:
        if fake_mode is None:
            return args
        # NB: This should hit the cache if tensors were fakefied before.
        # E.g., when the fx graph is produced by Dynamo.
        return tuple(
            fake_mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for t in args
        )

    @abc.abstractmethod
    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        ...

    @diagnostics.diagnose_call(
        diagnostics.rules.fx_pass,
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
        diagnostic = self.diagnostic_context.inflight_diagnostic(
            rule=diagnostics.rules.fx_pass
        )
        # Gather graph information before transform.
        graph_diff_log_level = logging.DEBUG
        if diagnostic.logger.isEnabledFor(graph_diff_log_level):
            # Cannot use LazyString because the graph may have been mutated at evaluation time.
            old_readable_graph = self.module.print_readable(print_output=False)
            old_tabular = maybe_fx_graph_tabular(self.module.graph)
        else:
            # Set to empty string to avoid unbound warning. This value should never be
            # used since the log level is not enabled.
            old_readable_graph = ""
            old_tabular = ""

        module = self._run(*args, **kwargs)

        # Gather graph information after transform.
        if diagnostic.logger.isEnabledFor(graph_diff_log_level):
            new_readable_graph = module.print_readable(print_output=False)
            new_tabular = maybe_fx_graph_tabular(module.graph)

            with diagnostic.log_section(graph_diff_log_level, "Graph diff:"):
                diagnostic.log(
                    graph_diff_log_level,
                    "```\n%s\n```",
                    diagnostics.LazyString(
                        _unified_diff, old_readable_graph, new_readable_graph
                    ),
                )

            with diagnostic.log_section(graph_diff_log_level, "Tabular diff:"):
                if old_tabular is None or new_tabular is None:
                    diagnostic.log(
                        graph_diff_log_level,
                        "Tabular diff is not available because `tabulate` is not installed.",
                    )
                else:
                    diagnostic.log(
                        graph_diff_log_level,
                        "```\n%s\n```",
                        diagnostics.LazyString(_unified_diff, old_tabular, new_tabular),
                    )

        return module


class AnalysisResult(abc.ABC):
    ...


class Analysis(abc.ABC):
    @_beartype.beartype
    def __init__(
        self,
        diagnostic_context: diagnostics.DiagnosticContext,
        module: torch.fx.GraphModule,
        onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher,
    ):
        self.diagnostic_context = diagnostic_context
        self.module = module
        self.onnxfunction_dispatcher = onnxfunction_dispatcher

    @abc.abstractmethod
    def analyze(self, diagnostic_level: diagnostics.infra.Level) -> AnalysisResult:
        ...
