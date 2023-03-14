"""This file defines an additional layer of abstraction on top of the SARIF OM."""

from __future__ import annotations

import dataclasses
import enum
import pprint
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple

from torch.onnx._internal.diagnostics.infra import formatter, sarif


class Level(enum.Enum):
    """The level of a diagnostic.

    This class is used to represent the level of a diagnostic. The levels are defined
    by the SARIF specification, and are not modifiable. For alternative categories,
    please use infra.Tag instead. When selecting a level, please consider the following
    guidelines:

    - NONE: Informational result that does not indicate the presence of a problem.
    - NOTE: An opportunity for improvement was found.
    - WARNING: A potential problem was found.
    - ERROR: A serious problem was found.
    """

    NONE = enum.auto()
    NOTE = enum.auto()
    WARNING = enum.auto()
    ERROR = enum.auto()


levels = Level


class Tag(enum.Enum):
    """The tag of a diagnostic. This class can be inherited to define custom tags."""


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
        short_description = kwargs.get("short_description", {}).get("text")
        full_description = kwargs.get("full_description", {}).get("text")
        full_description_markdown = kwargs.get("full_description", {}).get("markdown")
        help_uri = kwargs.get("help_uri")

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

    def format(self, level: Level, *args, **kwargs) -> Tuple[Rule, Level, str]:
        """Returns a tuple of (rule, level, message) for a diagnostic.

        This method is used to format the message of a diagnostic. The message is
        formatted using the default template of this rule, and the arguments passed in
        as `*args` and `**kwargs`. The level is used to override the default level of
        this rule.
        """
        return (self, level, self.format_message(*args, **kwargs))

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
    function: Optional[str] = None

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
        """Prints the location in a traceback style format."""
        unknown = "<unknown>"
        snippet = self.snippet or unknown
        uri = self.uri or unknown
        function = self.function or unknown
        lineno = self.line if self.line is not None else unknown
        message = f"  # {self.message}" if self.message is not None else ""
        print(f'  File "{uri}", line {lineno}, in {function}\n    {snippet}{message}')


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
    """Records a stack trace. The top of the stack is the first element in the list."""

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
        for frame in reversed(self.frames):
            frame.pretty_print()


@dataclasses.dataclass
class ThreadFlowLocation:
    """Records code location and the initial state."""

    location: Location
    state: Mapping[str, str]
    index: int
    stack: Optional[Stack] = None

    def sarif(self) -> sarif.ThreadFlowLocation:
        """Returns the SARIF representation of this thread flow location."""
        return sarif.ThreadFlowLocation(
            location=self.location.sarif(),
            state=self.state,
            stack=self.stack.sarif() if self.stack is not None else None,
        )

    def pretty_print(self, verbose: bool = False):
        """Prints the thread flow location in a human-readable format."""
        formatter.pretty_print_title(f"Step {self.index}", fill_char="-")
        self.location.pretty_print()
        if verbose:
            print(f"State: {pprint.pformat(self.state)}")
            if self.stack is not None:
                self.stack.pretty_print()


@dataclasses.dataclass
class Graph:
    """A graph of diagnostics.

    This class stores the string representation of a model graph.
    The `nodes` and `edges` fields are unused in the current implementation.
    """

    graph: str
    name: str
    description: Optional[str] = None

    def sarif(self) -> sarif.Graph:
        """Returns the SARIF representation of this graph."""
        return sarif.Graph(
            description=sarif.Message(text=self.graph),
            properties=PatchedPropertyBag(name=self.name, description=self.description),
        )

    def pretty_print(
        self,
        verbose: bool = False,
    ):
        """Prints the diagnostics in a human-readable format.

        Args:
            verbose: If True, prints all information. Otherwise, only prints compact
                information. E.g., graph name and description.
            log_level: The minimum level of diagnostics to print.
        """
        formatter.pretty_print_title(f"Graph: {self.name}", fill_char="-")
        print(self.description)
        if verbose:
            print(self.graph)


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
    # Tracks top level call arguments and diagnostic options.
    def __init__(self) -> None:
        raise NotImplementedError()


@dataclasses.dataclass
class DiagnosticOptions:
    """
    Options for diagnostic context.
    """

    log_verbose: bool = dataclasses.field(default=False)
    log_level: Level = dataclasses.field(default=Level.ERROR)
