"""A diagnostic engine based on SARIF."""

from __future__ import annotations

import contextlib

import dataclasses

import gzip

from typing import Callable, Generator, List, Mapping, Optional, Type, TypeVar

from typing_extensions import Literal

from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version


class DiagnosticError(RuntimeError):
    pass


# This is a workaround for mypy not supporting Self from typing_extensions.
_Diagnostic = TypeVar("_Diagnostic", bound="Diagnostic")


@dataclasses.dataclass
class Diagnostic:
    rule: infra.Rule
    level: infra.Level
    message: Optional[str] = None
    locations: List[infra.Location] = dataclasses.field(default_factory=list)
    stacks: List[infra.Stack] = dataclasses.field(default_factory=list)
    graphs: List[infra.Graph] = dataclasses.field(default_factory=list)
    thread_flow_locations: List[infra.ThreadFlowLocation] = dataclasses.field(
        default_factory=list
    )
    additional_message: Optional[str] = None
    tags: List[infra.Tag] = dataclasses.field(default_factory=list)

    def sarif(self) -> sarif.Result:
        """Returns the SARIF Result representation of this diagnostic."""
        message = self.message or self.rule.message_default_template
        if self.additional_message:
            message_markdown = (
                f"{message}\n\n## Additional Message:\n\n{self.additional_message}"
            )
        else:
            message_markdown = message

        kind: Literal["informational", "fail"] = (
            "informational" if self.level == infra.Level.NONE else "fail"
        )

        sarif_result = sarif.Result(
            message=sarif.Message(text=message, markdown=message_markdown),
            level=self.level.name.lower(),  # type: ignore[arg-type]
            rule_id=self.rule.id,
            kind=kind,
        )
        sarif_result.locations = [location.sarif() for location in self.locations]
        sarif_result.stacks = [stack.sarif() for stack in self.stacks]
        sarif_result.graphs = [graph.sarif() for graph in self.graphs]
        sarif_result.code_flows = [
            sarif.CodeFlow(
                thread_flows=[
                    sarif.ThreadFlow(
                        locations=[loc.sarif() for loc in self.thread_flow_locations]
                    )
                ]
            )
        ]
        sarif_result.properties = sarif.PropertyBag(
            tags=[tag.value for tag in self.tags]
        )
        return sarif_result

    def with_location(self: _Diagnostic, location: infra.Location) -> _Diagnostic:
        """Adds a location to the diagnostic."""
        self.locations.append(location)
        return self

    def with_thread_flow_location(
        self: _Diagnostic, location: infra.ThreadFlowLocation
    ) -> _Diagnostic:
        """Adds a thread flow location to the diagnostic."""
        self.thread_flow_locations.append(location)
        return self

    def with_stack(self: _Diagnostic, stack: infra.Stack) -> _Diagnostic:
        """Adds a stack to the diagnostic."""
        self.stacks.append(stack)
        return self

    def with_graph(self: _Diagnostic, graph: infra.Graph) -> _Diagnostic:
        """Adds a graph to the diagnostic."""
        self.graphs.append(graph)
        return self

    def with_additional_message(self: _Diagnostic, message: str) -> _Diagnostic:
        """Adds an additional message to the diagnostic."""
        if self.additional_message is None:
            self.additional_message = message
        else:
            self.additional_message = f"{self.additional_message}\n{message}"
        return self

    def record_python_call_stack(self, frames_to_skip: int) -> infra.Stack:
        """Records the current Python call stack."""
        frames_to_skip += 1  # Skip this function.
        stack = utils.python_call_stack(frames_to_skip=frames_to_skip)
        self.with_stack(stack)
        if len(stack.frames) > 0:
            self.with_location(stack.frames[0].location)
        return stack

    def record_python_call(
        self,
        fn: Callable,
        state: Mapping[str, str],
        message: Optional[str] = None,
        frames_to_skip: int = 0,
    ) -> infra.ThreadFlowLocation:
        """Records a python call as one thread flow step."""
        frames_to_skip += 1  # Skip this function.
        stack = utils.python_call_stack(frames_to_skip=frames_to_skip, frames_to_log=5)
        location = utils.function_location(fn)
        location.message = message
        # Add function location to the top of the stack.
        stack.frames.insert(0, infra.StackFrame(location=location))
        thread_flow_location = infra.ThreadFlowLocation(
            location=location,
            state=state,
            index=len(self.thread_flow_locations),
            stack=stack,
        )
        self.with_thread_flow_location(thread_flow_location)
        return thread_flow_location

    def pretty_print(
        self, verbose: bool = False, log_level: infra.Level = infra.Level.ERROR
    ):
        """Prints the diagnostics in a human-readable format.

        Args:
            verbose: If True, prints all information. E.g. stack frames, graphs, etc.
                Otherwise, only prints compact information. E.g., rule name and display message.
            log_level: The minimum level of diagnostics to print.
        """
        if self.level.value < log_level.value:
            return
        formatter.pretty_print_item_title(f"{self.level.name}: {self.rule.name}")
        print(self.message)
        print(self.additional_message)

        if not verbose:
            print("<Set verbose=True to see more details>\n")
            return

        formatter.pretty_print_title("Locations", fill_char="-")
        for location in self.locations:
            location.pretty_print()
        for stack in self.stacks:
            stack.pretty_print()
        formatter.pretty_print_title("Thread Flow Locations", fill_char="-")
        for thread_flow_location in self.thread_flow_locations:
            thread_flow_location.pretty_print(verbose=verbose)
        for graph in self.graphs:
            graph.pretty_print(verbose=verbose)

        print()

        # TODO: print help url to rule at the end.


@dataclasses.dataclass
class DiagnosticContext:
    name: str
    version: str
    options: infra.DiagnosticOptions = dataclasses.field(
        default_factory=infra.DiagnosticOptions
    )
    diagnostic_type: Type[Diagnostic] = dataclasses.field(default=Diagnostic)
    diagnostics: List[Diagnostic] = dataclasses.field(init=False, default_factory=list)
    # TODO(bowbao): Implement this.
    # _invocation: infra.Invocation = dataclasses.field(init=False)
    _inflight_diagnostics: List[Diagnostic] = dataclasses.field(
        init=False, default_factory=list
    )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

    def sarif(self) -> sarif.Run:
        """Returns the SARIF Run object."""
        unique_rules = {diagnostic.rule for diagnostic in self.diagnostics}
        return sarif.Run(
            tool=sarif.Tool(
                driver=sarif.ToolComponent(
                    name=self.name,
                    version=self.version,
                    rules=[rule.sarif() for rule in unique_rules],
                )
            ),
            results=[diagnostic.sarif() for diagnostic in self.diagnostics],
        )

    def add_diagnostic(self, diagnostic: Diagnostic) -> None:
        """Adds a diagnostic to the context.

        Use this method to add diagnostics that are not created by the context.
        Args:
            diagnostic: The diagnostic to add.
        """
        if not isinstance(diagnostic, Diagnostic):
            raise TypeError(
                f"Expected diagnostic of type {Diagnostic}, got {type(diagnostic)}"
            )
        self.diagnostics.append(diagnostic)

    @contextlib.contextmanager
    def add_inflight_diagnostic(
        self, diagnostic: Diagnostic
    ) -> Generator[Diagnostic, None, None]:
        """Adds a diagnostic to the context.

        Use this method to add diagnostics that are not created by the context.
        Args:
            diagnostic: The diagnostic to add.
        """
        self._inflight_diagnostics.append(diagnostic)
        try:
            yield diagnostic
        finally:
            self._inflight_diagnostics.pop()

    def diagnose(
        self,
        rule: infra.Rule,
        level: infra.Level,
        message: Optional[str] = None,
        **kwargs,
    ) -> Diagnostic:
        """Creates a diagnostic for the given arguments.

        Args:
            rule: The rule that triggered the diagnostic.
            level: The level of the diagnostic.
            message: The message of the diagnostic.
            **kwargs: Additional arguments to pass to the Diagnostic constructor.

        Returns:
            The created diagnostic.

        Raises:
            ValueError: If the rule is not supported by the tool.
        """
        diagnostic = self.diagnostic_type(rule, level, message, **kwargs)
        self.add_diagnostic(diagnostic)
        return diagnostic

    def push_inflight_diagnostic(self, diagnostic: Diagnostic) -> None:
        """Pushes a diagnostic to the inflight diagnostics stack.

        Args:
            diagnostic: The diagnostic to push.

        Raises:
            ValueError: If the rule is not supported by the tool.
        """
        self._inflight_diagnostics.append(diagnostic)

    def pop_inflight_diagnostic(self) -> Diagnostic:
        """Pops the last diagnostic from the inflight diagnostics stack.

        Returns:
            The popped diagnostic.
        """
        return self._inflight_diagnostics.pop()

    def inflight_diagnostic(self, rule: Optional[infra.Rule] = None) -> Diagnostic:
        if rule is None:
            # TODO(bowbao): Create builtin-rules and create diagnostic using that.
            if len(self._inflight_diagnostics) <= 0:
                raise DiagnosticError("No inflight diagnostics")

            return self._inflight_diagnostics[-1]
        else:
            # TODO(bowbao): Improve efficiency with Mapping[Rule, List[Diagnostic]]
            for diagnostic in reversed(self._inflight_diagnostics):
                if diagnostic.rule == rule:
                    return diagnostic
            raise DiagnosticError(f"No inflight diagnostic for rule {rule.name}")

    def pretty_print(
        self, verbose: Optional[bool] = None, log_level: Optional[infra.Level] = None
    ) -> None:
        """Prints the diagnostics in a human-readable format.

        Args:
            verbose: Whether to print the diagnostics in verbose mode. See Diagnostic.pretty_print.
                If not specified, uses the value of 'self.options.log_verbose'.
            log_level: The minimum level of diagnostics to print.
                If not specified, uses the value of 'self.options.log_level'.
        """
        if verbose is None:
            verbose = self.options.log_verbose
        if log_level is None:
            log_level = self.options.log_level

        formatter.pretty_print_title(
            f"Diagnostic Run {self.name} version {self.version}"
        )
        print(f"verbose: {verbose}, log level: {log_level}")
        diagnostic_stats = {level: 0 for level in infra.Level}
        for diagnostic in self.diagnostics:
            diagnostic_stats[diagnostic.level] += 1
        formatter.pretty_print_title(
            " ".join(f"{diagnostic_stats[level]} {level.name}" for level in infra.Level)
        )

        for diagnostic in self.diagnostics:
            diagnostic.pretty_print(verbose, log_level)

        unprinted_diagnostic_stats = [
            (level, count)
            for level, count in diagnostic_stats.items()
            if count > 0 and level.value < log_level.value
        ]
        if unprinted_diagnostic_stats:
            print(
                f"{' '.join(f'{count} {level.name}' for level, count in unprinted_diagnostic_stats)} "
                "were not printed due to the log level."
            )
        print()


class DiagnosticEngine:
    """A generic diagnostic engine based on SARIF.

    This class is the main interface for diagnostics. It manages the creation of diagnostic contexts.
    A DiagnosticContext provides the entry point for recording Diagnostics.
    See infra.DiagnosticContext for more details.

    Examples:
        Step 1: Create a set of rules.
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> rules = infra.RuleCollection.custom_collection_from_list(
        ...     "CustomRuleCollection",
        ...     [
        ...         infra.Rule(
        ...             id="r1",
        ...             name="rule-1",
        ...             message_default_template="Mising xxx",
        ...         ),
        ...     ],
        ... )

        Step 2: Create a diagnostic engine.
        >>> engine = DiagnosticEngine()

        Step 3: Start a new diagnostic context.
        >>> with engine.create_diagnostic_context("torch.onnx.export", version="1.0") as context:
        ...     ...

        Step 4: Add diagnostics in your code.
        ...     context.diagnose(rules.rule1, infra.Level.ERROR)

        Step 5: Afterwards, get the SARIF log.
        >>> sarif_log = engine.sarif_log()
    """

    contexts: List[DiagnosticContext]

    def __init__(self) -> None:
        self.contexts = []

    def sarif_log(self) -> sarif.SarifLog:
        return sarif.SarifLog(
            version=sarif_version.SARIF_VERSION,
            schema_uri=sarif_version.SARIF_SCHEMA_LINK,
            runs=[context.sarif() for context in self.contexts],
        )

    def __str__(self) -> str:
        # TODO: pretty print.
        return self.to_json()

    def __repr__(self) -> str:
        return self.to_json()

    def to_json(self) -> str:
        return formatter.sarif_to_json(self.sarif_log())

    def dump(self, file_path: str, compress: bool = False) -> None:
        """Dumps the SARIF log to a file."""
        if compress:
            with gzip.open(file_path, "wt") as f:
                f.write(self.to_json())
        else:
            with open(file_path, "w") as f:
                f.write(self.to_json())

    def clear(self) -> None:
        """Clears all diagnostic contexts."""
        self.contexts.clear()

    def create_diagnostic_context(
        self,
        name: str,
        version: str,
        options: Optional[infra.DiagnosticOptions] = None,
        diagnostic_type: Type[Diagnostic] = Diagnostic,
    ) -> DiagnosticContext:
        """Creates a new diagnostic context.

        Args:
            name: The subject name for the diagnostic context.
            version: The subject version for the diagnostic context.
            options: The options for the diagnostic context.

        Returns:
            A new diagnostic context.
        """
        if options is None:
            options = infra.DiagnosticOptions()
        context = DiagnosticContext(
            name, version, options, diagnostic_type=diagnostic_type
        )
        self.contexts.append(context)
        return context

    def pretty_print(
        self, verbose: bool = False, level: infra.Level = infra.Level.ERROR
    ) -> None:
        """Pretty prints all diagnostics in the diagnostic contexts.

        Args:
            verbose: Whether to print the diagnostics in verbose mode. See Diagnostic.pretty_print.
            level: The minimum level of diagnostics to print.
        """
        formatter.pretty_print_title(f"{len(self.contexts)} Diagnostic Run")
        for context in self.contexts:
            context.pretty_print(verbose, level)
