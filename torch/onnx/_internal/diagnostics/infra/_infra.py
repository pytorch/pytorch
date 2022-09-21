"""This file defines an additional layer of abstraction on top of the SARIF OM."""

from __future__ import annotations

import enum
from typing import Any, List, Optional, Sequence, Set, Tuple

from torch.onnx._internal.diagnostics.infra import sarif_om


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
        self._sarif_reporting_descriptor = sarif_om.ReportingDescriptor(id=id)
        self.name = name
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

    @name.setter
    def name(self, new_name: str) -> None:
        """Sets the name of the rule."""
        self._sarif_reporting_descriptor.name = new_name

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


class DiagnosticTool:
    _sarif_tool: sarif_om.Tool
    _rules: Sequence[Rule]
    _triggered_rules: Set[Rule]

    def __init__(self, name: str, version: str, rules: Sequence[Rule]) -> None:
        self._sarif_tool = sarif_om.Tool(
            driver=sarif_om.ToolComponent(name=name, version=version)
        )
        self._rules = rules
        self._triggered_rules = set()

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
    def rules(self) -> Sequence[Rule]:
        """The rules supported by the tool."""
        return self._rules

    @rules.setter
    def rules(self, new_rules: Sequence[Rule]) -> None:
        """Sets the rules supported by the tool."""
        self._rules = new_rules

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
            )
        self._triggered_rules.add(rule)
        return Diagnostic(rule, level, message_args, **kwargs)


class Run:
    _sarif_run: sarif_om.Run
    _diagnostics: List[Diagnostic]
    _tool: DiagnosticTool

    def __init__(self, tool: DiagnosticTool) -> None:
        self._tool = tool
        self._diagnostics = []

    def sarif(self) -> sarif_om.Run:
        """Returns the SARIF Run object."""
        self._tool.rules = list(
            set([diagnostic.rule for diagnostic in self._diagnostics])
        )
        self._sarif_run = sarif_om.Run(tool=self._tool.sarif())
        self._sarif_run.results = [
            diagnostic.sarif() for diagnostic in self._diagnostics
        ]

        return self._sarif_run

    @property
    def diagnostics(self) -> List[Diagnostic]:
        """The diagnostics captured."""
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(
        self,
        new_diagnostics: List[Diagnostic],
    ):
        """Sets the diagnostics captured."""
        self._diagnostics = new_diagnostics

    def add_diagnostic(
        self,
        rule: Rule,
        level: Level,
        message_args: Optional[Tuple[Any, ...]] = None,
        **kwargs,
    ) -> Diagnostic:
        """Adds a diagnostic for the given arguments.

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
        diagnostic = self._tool.create_diagnostic(rule, level, message_args, **kwargs)
        self._diagnostics.append(diagnostic)
        return diagnostic
