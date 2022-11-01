"""This file defines an additional layer of abstraction on top of the SARIF OM."""

from __future__ import annotations

import dataclasses
import enum
from typing import FrozenSet, List, Optional, Sequence, Tuple, Type, TypeVar

from torch.onnx._internal.diagnostics.infra import formatter, sarif


class Level(enum.Enum):
    """The level of a diagnostic.

    This class is used to represent the level of a diagnostic. The levels are defined
    by the SARIF specification, and are not modifiable. For alternative categories,
    please use infra.Tag instead.
    """

    NONE = enum.auto()
    NOTE = enum.auto()
    WARNING = enum.auto()
    ERROR = enum.auto()


levels = Level


class Tag(enum.Enum):
    """The tag of a diagnostic. This class can be inherited to define custom tags."""

    pass


class PatchedPropertyBag(sarif.PropertyBag):
    """Key/value pairs that provide additional information about the object.

    The definition of PropertyBag via SARIF spec is "A property bag is an object (§3.6)
    containing an unordered set of properties with arbitrary names." However it is not
    reflected in the json file, and therefore not captured by the python representation.
    This patch adds additional **kwargs to the `__init__` method to allow recording
    arbitrary key/value pairs.
    """

    def __init__(self, tags: Optional[List[str]] = None, **kwargs):
        super().__init__(tags=tags)
        self.__dict__.update(kwargs)


@dataclasses.dataclass(frozen=True)
class Rule:
    id: str
    name: str
    message_default_template: str
    short_description: Optional[str] = None
    full_description: Optional[str] = None
    full_description_markdown: Optional[str] = None
    help_uri: Optional[str] = None

    @classmethod
    def from_sarif(cls, **kwargs):
        """Returns a rule from the SARIF reporting descriptor."""
        short_description = kwargs.get("short_description", {}).get("text", None)
        full_description = kwargs.get("full_description", {}).get("text", None)
        full_description_markdown = kwargs.get("full_description", {}).get(
            "markdown", None
        )
        help_uri = kwargs.get("help_uri", None)

        rule = cls(
            id=kwargs["id"],
            name=kwargs["name"],
            message_default_template=kwargs["message_strings"]["default"]["text"],
            short_description=short_description,
            full_description=full_description,
            full_description_markdown=full_description_markdown,
            help_uri=help_uri,
        )
        return rule

    def sarif(self) -> sarif.ReportingDescriptor:
        """Returns a SARIF reporting descriptor of this Rule."""
        short_description = (
            sarif.MultiformatMessageString(text=self.short_description)
            if self.short_description is not None
            else None
        )
        full_description = (
            sarif.MultiformatMessageString(
                text=self.full_description, markdown=self.full_description_markdown
            )
            if self.full_description is not None
            else None
        )
        return sarif.ReportingDescriptor(
            id=self.id,
            name=self.name,
            short_description=short_description,
            full_description=full_description,
            help_uri=self.help_uri,
        )

    def format_message(self, *args, **kwargs) -> str:
        """Returns the formatted default message of this Rule.

        This method should be overridden (with code generation) by subclasses to reflect
        the exact arguments needed by the message template. This is a helper method to
        create the default message for a diagnostic.
        """
        return self.message_default_template.format(*args, **kwargs)

    def pretty_print(self):
        pass


@dataclasses.dataclass
class Location:
    uri: Optional[str] = None
    line: Optional[int] = None
    message: Optional[str] = None
    start_column: Optional[int] = None
    end_column: Optional[int] = None
    snippet: Optional[str] = None

    def sarif(self) -> sarif.Location:
        """Returns the SARIF representation of this location."""
        return sarif.Location(
            physical_location=sarif.PhysicalLocation(
                artifact_location=sarif.ArtifactLocation(uri=self.uri),
                region=sarif.Region(
                    start_line=self.line,
                    start_column=self.start_column,
                    end_column=self.end_column,
                    snippet=sarif.ArtifactContent(text=self.snippet),
                ),
            ),
            message=sarif.Message(text=self.message)
            if self.message is not None
            else None,
        )

    def pretty_print(self):
        """Prints the location in a human-readable format."""
        location_strs = ["frame:"]
        if self.snippet is not None:
            location_strs.append(self.snippet)
        if self.uri is not None:
            line_strs = [self.uri]
            line_strs.append(str(self.line)) if self.line is not None else "-1"
            line_strs.append(
                str(self.start_column)
            ) if self.start_column is not None else "-1"
            line_strs.append(
                str(self.end_column)
            ) if self.end_column is not None else "-1"
            location_strs.append(":".join(line_strs))
        if self.message is not None:
            location_strs.append(f"({self.message})")
        print(" ".join(location_strs))


@dataclasses.dataclass
class StackFrame:
    location: Location

    def sarif(self) -> sarif.StackFrame:
        """Returns the SARIF representation of this stack frame."""
        return sarif.StackFrame(location=self.location.sarif())

    def pretty_print(self):
        """Prints the stack frame in a human-readable format."""
        self.location.pretty_print()


@dataclasses.dataclass
class Stack:
    frames: List[StackFrame] = dataclasses.field(default_factory=list)
    message: Optional[str] = None

    def sarif(self) -> sarif.Stack:
        """Returns the SARIF representation of this stack."""
        return sarif.Stack(
            frames=[frame.sarif() for frame in self.frames],
            message=sarif.Message(text=self.message)
            if self.message is not None
            else None,
        )

    def pretty_print(self):
        """Prints the stack in a human-readable format."""
        formatter.pretty_print_title(f"Stack: {self.message}", fill_char="-")
        for frame in self.frames:
            frame.pretty_print()


# This is a workaround for mypy not supporting Self from typing_extensions.
_Diagnostic = TypeVar("_Diagnostic", bound="Diagnostic")


@dataclasses.dataclass
class Graph:
    """A graph of diagnostics.

    This class stores the string representation of a model graph.
    The `nodes` and `edges` fields are unused in the current implementation.
    """

    graph_str: str
    name: str
    description: Optional[str] = None

    def sarif(self) -> sarif.Graph:
        """Returns the SARIF representation of this graph."""
        return sarif.Graph(
            description=sarif.Message(text=self.graph_str),
            properties=PatchedPropertyBag(name=self.name, description=self.description),
        )

    def pretty_print(self):
        pass


@dataclasses.dataclass
class Diagnostic:
    rule: Rule
    level: Level
    message: Optional[str] = None
    locations: List[Location] = dataclasses.field(default_factory=list)
    stacks: List[Stack] = dataclasses.field(default_factory=list)
    graphs: List[Graph] = dataclasses.field(default_factory=list)
    additional_message: Optional[str] = None
    tags: List[Tag] = dataclasses.field(default_factory=list)

    def sarif(self) -> sarif.Result:
        """Returns the SARIF Result representation of this diagnostic."""
        message = self.message
        if message is None:
            message = self.rule.message_default_template
        if self.additional_message is not None:
            message = f"{message}\n{self.additional_message}"
        sarif_result = sarif.Result(
            message=sarif.Message(text=message),
            level=self.level.name.lower(),  # type: ignore[arg-type]
            rule_id=self.rule.id,
        )
        sarif_result.locations = [location.sarif() for location in self.locations]
        sarif_result.stacks = [stack.sarif() for stack in self.stacks]
        sarif_result.graphs = [graph.sarif() for graph in self.graphs]
        sarif_result.properties = sarif.PropertyBag(
            tags=[tag.value for tag in self.tags]
        )
        return sarif_result

    def with_location(self: _Diagnostic, location: Location) -> _Diagnostic:
        """Adds a location to the diagnostic."""
        self.locations.append(location)
        return self

    def with_stack(self: _Diagnostic, stack: Stack) -> _Diagnostic:
        """Adds a stack to the diagnostic."""
        self.stacks.append(stack)
        return self

    def with_graph(self: _Diagnostic, graph: Graph) -> _Diagnostic:
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

    def pretty_print(self, verbose: bool = False, log_level: Level = Level.ERROR):
        """Prints the diagnostics in a human-readable format.

        Args:
            verbose: If True, prints all information. E.g. stack frames, graphs, etc.
                Otherwise, only prints compact information. E.g., rule name and display message.
            level: The minimum level of diagnostics to print.
        """
        if self.level.value < log_level.value:
            return
        formatter.pretty_print_item_title(f"{self.level.name}: {self.rule.name}")
        print(self.message)

        if not verbose:
            print("<Set verbose=True to see more details>\n")
            return

        for location in self.locations:
            location.pretty_print()
        for stack in self.stacks:
            stack.pretty_print()
        for graph in self.graphs:
            graph.pretty_print()
        print()


@dataclasses.dataclass
class RuleCollection:
    _rule_id_name_set: FrozenSet[Tuple[str, str]] = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._rule_id_name_set = frozenset(
            {
                (field.default.id, field.default.name)
                for field in dataclasses.fields(self)
                if isinstance(field.default, Rule)
            }
        )

    def __contains__(self, rule: Rule) -> bool:
        """Checks if the rule is in the collection."""
        return (rule.id, rule.name) in self._rule_id_name_set

    @classmethod
    def custom_collection_from_list(
        cls, new_collection_class_name: str, rules: Sequence[Rule]
    ) -> RuleCollection:
        """Creates a custom class inherited from RuleCollection with the list of rules."""
        return dataclasses.make_dataclass(
            new_collection_class_name,
            [
                (
                    formatter.kebab_case_to_snake_case(rule.name),
                    type(rule),
                    dataclasses.field(default=rule),
                )
                for rule in rules
            ],
            bases=(cls,),
        )()


class Invocation:
    # TODO: Implement this.
    def __init__(self) -> None:
        raise NotImplementedError()


@dataclasses.dataclass
class DiagnosticOptions:
    """
    Options for diagnostic context.
    """


@dataclasses.dataclass
class DiagnosticContext:
    name: str
    version: str
    options: Optional[DiagnosticOptions] = None
    diagnostic_type: Type[Diagnostic] = dataclasses.field(default=Diagnostic)
    diagnostics: List[Diagnostic] = dataclasses.field(init=False, default_factory=list)
    _invocation: Invocation = dataclasses.field(init=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

    def sarif(self) -> sarif.Run:
        """Returns the SARIF Run object."""
        return sarif.Run(
            tool=sarif.Tool(
                driver=sarif.ToolComponent(
                    name=self.name,
                    version=self.version,
                    rules=[diagnostic.rule.sarif() for diagnostic in self.diagnostics],
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
        if not isinstance(diagnostic, self.diagnostic_type):
            raise TypeError(
                f"Expected diagnostic of type {self.diagnostic_type}, got {type(diagnostic)}"
            )
        self.diagnostics.append(diagnostic)

    def diagnose(
        self,
        rule: Rule,
        level: Level,
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

    def pretty_print(
        self, verbose: bool = False, log_level: Level = Level.ERROR
    ) -> None:
        """Prints the diagnostics in a human-readable format.

        Args:
            verbose: Whether to print the diagnostics in verbose mode. See Diagnostic.pretty_print.
            level: The minimum level of diagnostics to print.
        """
        formatter.pretty_print_title(
            f"Diagnostic Run {self.name} version {self.version}"
        )
        print(f"verbose: {verbose}, log level: {log_level}")
        diagnostic_stats = {level: 0 for level in Level}
        for diagnostic in self.diagnostics:
            diagnostic_stats[diagnostic.level] += 1
        formatter.pretty_print_title(
            " ".join(f"{diagnostic_stats[level]} {level.name}" for level in Level)
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
