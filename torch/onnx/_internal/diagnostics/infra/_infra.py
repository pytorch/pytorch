"""This file defines an additional layer of abstraction on top of the SARIF OM."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, FrozenSet, List, Optional, Sequence, Set, Tuple

from torch.onnx._internal.diagnostics.infra import formatter, sarif_om


class Level(enum.Enum):
    """The level of a diagnostic."""

    NONE = "none"
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


class Rule:
    _sarif_reporting_descriptor: sarif_om.ReportingDescriptor

    def __init__(
        self,
        id: str,
        name: str,
        message_default_template: str,
        short_description: Optional[str] = None,
        full_description: Optional[str] = None,
        help_uri: Optional[str] = None,
    ) -> None:
        self._sarif_reporting_descriptor = sarif_om.ReportingDescriptor(
            id=id, name=name
        )
        self.message_default_template = message_default_template
        self.short_description = short_description
        self.full_description = full_description
        self.help_uri = help_uri

    @classmethod
    def from_sarif(cls, **kwargs) -> Rule:
        """Returns a rule from the SARIF representation of this rule."""

        rule = cls(
            kwargs["id"],
            kwargs["name"],
            kwargs["message_strings"]["default"]["text"],
        )
        if "short_description" in kwargs:
            rule.short_description = kwargs["short_description"]["text"]
        if "full_description" in kwargs:
            rule.full_description = kwargs["full_description"]["text"]
        if "help_uri" in kwargs:
            rule.help_uri = kwargs["help_uri"]

        return rule

    def sarif(self) -> sarif_om.ReportingDescriptor:
        """Returns the SARIF representation of this rule."""
        return self._sarif_reporting_descriptor

    @property
    def id(self) -> str:
        """The unique identifier for the rule."""
        return self._sarif_reporting_descriptor.id

    @property
    def name(self) -> str:
        """A stable, opaque identifier for the rule."""
        return self._sarif_reporting_descriptor.name

    @property
    def message_default_template(self) -> str:
        """The default message template for the rule report."""
        return self._sarif_reporting_descriptor.message_strings["default"].text

    @message_default_template.setter
    def message_default_template(self, new_message_default_template: str) -> None:
        """Sets the default message template for the rule report."""
        self._sarif_reporting_descriptor.message_strings = {
            "default": sarif_om.Message(text=new_message_default_template)
        }

    @property
    def short_description(self) -> Optional[str]:
        """A brief description of the rule."""
        return self._sarif_reporting_descriptor.short_description.text

    @short_description.setter
    def short_description(self, new_short_description: Optional[str]) -> None:
        """Sets the short description of the rule."""
        if new_short_description is None:
            self._sarif_reporting_descriptor.short_description = None
            return
        self._sarif_reporting_descriptor.short_description = sarif_om.Message(
            text=new_short_description
        )

    @property
    def full_description(self) -> Optional[str]:
        """A comprehensive description of the rule in markdown."""
        return self._sarif_reporting_descriptor.full_description.markdown

    @full_description.setter
    def full_description(self, new_full_description: Optional[str]) -> None:
        """Sets the full description of the rule."""
        if new_full_description is None:
            self._sarif_reporting_descriptor.full_description = None
            return
        self._sarif_reporting_descriptor.full_description = sarif_om.Message(
            text="", markdown=new_full_description
        )

    @property
    def help_uri(self) -> Optional[str]:
        """A URI where the rule is documented."""
        return self._sarif_reporting_descriptor.help_uri

    @help_uri.setter
    def help_uri(self, new_help_uri: Optional[str]) -> None:
        """Sets the help URI of the rule."""
        self._sarif_reporting_descriptor.help_uri = new_help_uri


class Location:
    _sarif_location: sarif_om.Location

    def __init__(
        self,
        uri: str,
        message: str,
        line: Optional[int] = None,
        start_column: Optional[int] = None,
        end_column: Optional[int] = None,
    ):
        self._sarif_location = sarif_om.Location()
        self.uri = uri
        self.message = message
        self.line = line
        self.start_column = start_column
        self.end_column = end_column

    def sarif(self) -> sarif_om.Location:
        """Returns the SARIF Location object."""
        return self._sarif_location

    @property
    def uri(self) -> str:
        """The URI of the source code location."""
        return self._sarif_location.physical_location.artifact_location.uri

    @uri.setter
    def uri(self, new_uri: str) -> None:
        """Set the URI of the source code location."""
        self._sarif_location.physical_location.artifact_location.uri = new_uri

    @property
    def message(self) -> str:
        """The message associated with the location."""
        return self._sarif_location.message.text

    @message.setter
    def message(self, new_message: str) -> None:
        """Set the message associated with the location."""
        self._sarif_location.message.text = new_message

    @property
    def line(self) -> Optional[int]:
        """The line number of the source code location."""
        return self._sarif_location.physical_location.region.start_line

    @line.setter
    def line(self, new_line: Optional[int]) -> None:
        """Set the line number of the source code location."""
        self._sarif_location.physical_location.region.start_line = new_line

    @property
    def start_column(self) -> Optional[int]:
        """The start column number of the source code location."""
        return self._sarif_location.physical_location.region.start_column

    @start_column.setter
    def start_column(self, new_start_column: Optional[int]) -> None:
        """Set the start column number of the source code location."""
        self._sarif_location.physical_location.region.start_column = new_start_column

    @property
    def end_column(self) -> Optional[int]:
        """The end column number of the source code location."""
        return self._sarif_location.physical_location.region.end_column

    @end_column.setter
    def end_column(self, new_end_column: Optional[int]) -> None:
        """Set the end column number of the source code location."""
        self._sarif_location.physical_location.region.end_column = new_end_column


class Stack:
    _sarif_stack: sarif_om.Stack

    def __init__(
        self,
        message: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._sarif_stack = sarif_om.Stack(message=message, **kwargs)

    def sarif(self) -> sarif_om.Stack:
        """Returns the underlying SARIF stack object."""
        return self._sarif_stack

    def add_frame(
        self,
        uri: str,
        message: str,
        line: Optional[int] = None,
        start_column: Optional[int] = None,
        end_column: Optional[int] = None,
    ) -> None:
        """Adds a frame to the stack."""
        self._sarif_stack.frames.append(
            Location(
                uri,
                message,
                line,
                start_column,
                end_column,
            ).sarif()
        )


class Diagnostic:
    _sarif_result: sarif_om.Result
    _locations: List[Location]
    _stacks: List[Stack]
    _rule: Rule
    _additional_message: str

    def __init__(
        self,
        rule: Rule,
        level: Level,
        message_args: Optional[Tuple[Any, ...]],
        **kwargs,
    ) -> None:
        if message_args is None:
            message_args = tuple()
        message = rule.message_default_template.format(*message_args)
        self._sarif_result = sarif_om.Result(
            message=message, level=level.value, rule_id=rule.id
        )
        self._rule = rule
        self._locations = []
        self._stacks = []
        self._additional_message = ""

    def sarif(self) -> sarif_om.Result:
        """Returns the SARIF Result object."""
        self._sarif_result.locations = [
            location.sarif() for location in self._locations
        ]
        self._sarif_result.stacks = [stack.sarif() for stack in self._stacks]

        return self._sarif_result

    @property
    def message(self) -> str:
        """The message associated with the diagnostic."""
        return self._sarif_result.message.text

    @property
    def level(self) -> Level:
        """The level of the diagnostic."""
        return getattr(Level, self._sarif_result.level.upper())

    @property
    def rule(self) -> Rule:
        """The rule associated with the diagnostic."""
        return self._rule

    @property
    def locations(self) -> Sequence[Location]:
        """The locations of the diagnostic."""
        return self._locations

    @property
    def stacks(self) -> Sequence[Stack]:
        """The stacks of the diagnostic."""
        return self._stacks

    def with_location(self, location: Location) -> "Diagnostic":
        """Adds a location to the diagnostic."""
        self._locations.append(location)
        return self

    def with_stack(self, stack: Stack) -> "Diagnostic":
        """Adds a stack to the diagnostic."""
        self._stacks.append(stack)
        return self

    def with_additional_message(self, message: str) -> "Diagnostic":
        """Adds an additional message to the diagnostic."""
        self._sarif_result.message.text += f"\n{message}"
        return self


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
                    formatter._kebab_case_to_snake_case(rule.name),
                    type(rule),
                    dataclasses.field(default=rule),
                )
                for rule in rules
            ],
            bases=(cls,),
        )()


class DiagnosticTool:
    _sarif_tool: sarif_om.Tool
    _rules: RuleCollection
    _triggered_rules: Set[Rule]

    def __init__(
        self,
        name: str,
        version: str,
        rules: RuleCollection,
        diagnostic_type: type = Diagnostic,
    ) -> None:
        self._sarif_tool = sarif_om.Tool(
            driver=sarif_om.ToolComponent(name=name, version=version)
        )
        self._rules = rules
        self._triggered_rules = set()
        if not issubclass(diagnostic_type, Diagnostic):
            raise TypeError(
                "Expected diagnostic_type to be a subclass of Diagnostic, "
                f"but got {diagnostic_type}"
            )
        self._diagnostic_type = diagnostic_type

    def sarif(self) -> sarif_om.Tool:
        """Returns the SARIF Tool object."""
        self._sarif_tool.driver.rules = [rule.sarif() for rule in self._triggered_rules]
        return self._sarif_tool

    @property
    def name(self) -> str:
        """The name of the tool."""
        return self._sarif_tool.driver.name

    @property
    def version(self) -> str:
        """The version of the tool."""
        return self._sarif_tool.driver.version

    @property
    def rules(self) -> RuleCollection:
        """The rules supported by the tool."""
        return self._rules

    def create_diagnostic(
        self,
        rule: Rule,
        level: Level,
        message_args: Optional[Tuple[Any, ...]],
        **kwargs,
    ) -> Diagnostic:
        """Creates a diagnostic for the given arguments.

        Args:
            rule: The rule that triggered the diagnostic.
            level: The level of the diagnostic.
            message_args: The arguments to format the rule's message template.
            **kwargs: Additional arguments to pass to the Diagnostic constructor.

        Returns:
            The created diagnostic.

        Raises:
            ValueError: If the rule is not supported by the tool.
        """
        if rule not in self._rules:
            raise ValueError(
                f"Rule '{rule.id}:{rule.name}' is not supported by this tool '{self.name} {self.version}'."
                f" Supported rules are: {self._rules._rule_id_name_set}"
            )
        self._triggered_rules.add(rule)
        return self._diagnostic_type(rule, level, message_args, **kwargs)


class Invocation:
    _sarif_invocation: sarif_om.Invocation
    # TODO: Implement this.


@dataclasses.dataclass
class DiagnosticOptions:
    """
    Options for diagnostic context.
    """


class DiagnosticContext:
    _sarif_run: sarif_om.Run
    _diagnostics: List[Diagnostic]
    _tool: DiagnosticTool
    _options: DiagnosticOptions
    _invocation: Invocation
    _is_active: bool

    def __init__(
        self, tool: DiagnosticTool, options: Optional[DiagnosticOptions]
    ) -> None:
        self._tool = tool
        self._diagnostics = []
        self._is_active = True
        self._invocation = Invocation()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
        return True

    def sarif(self) -> sarif_om.Run:
        """Returns the SARIF Run object."""
        self._sarif_run = sarif_om.Run(tool=self._tool.sarif())
        self._sarif_run.results = [
            diagnostic.sarif() for diagnostic in self._diagnostics
        ]
        return self._sarif_run

    @property
    def diagnostics(self) -> Sequence[Diagnostic]:
        """The diagnostics collected in the context."""
        return self._diagnostics

    def diagnose(
        self,
        rule: Rule,
        level: Level,
        message_args: Optional[Tuple[Any, ...]] = None,
        **kwargs,
    ) -> Diagnostic:
        """Creates a diagnostic for the given arguments.

        Args:
            rule: The rule that triggered the diagnostic.
            level: The level of the diagnostic.
            message_args: The arguments to format the rule's message template.
            **kwargs: Additional arguments to pass to the Diagnostic constructor.

        Returns:
            The created diagnostic.

        Raises:
            RuntimeError: If the context is not active.
            ValueError: If the rule is not supported by the tool.
        """
        if not self._is_active:
            raise RuntimeError("The diagnostics context is not active.")

        diagnostic = self._tool.create_diagnostic(rule, level, message_args, **kwargs)
        self._diagnostics.append(diagnostic)
        return diagnostic

    def end(self) -> None:
        """Ends the context."""
        print("end is called on DiagnosticContext")
        self._is_active = False
        # TODO: Update info in invocation.
        # TODO: Emit report.
