import textwrap
from dataclasses import dataclass
from typing import List


@dataclass
class GraphBreakReason:
    # Dynamo-internal reason for graph break, for developers
    reason: str
    # English/Python description of the graph break, for users
    descr: str
    # Suggested workaround for users
    workaround: str


def format_graph_break_reasons(graph_break_reasons: List[GraphBreakReason]) -> str:
    reason_msg = ""
    descr_msg = ""
    workaround_msg = ""
    for reason in graph_break_reasons:
        reason_msg += f" - {reason.reason}\n"
        if reason.descr:
            descr_msg += f" - ({reason.reason}) {reason.descr}\n"
        if reason.workaround:
            workaround_msg += f" - ({reason.reason}) {reason.workaround}\n"
    return f"""\
Reasons:
{reason_msg}


Descriptions:
{descr_msg}


Workarounds:
{workaround_msg}
"""


TROUBLESHOOTING_DOC_URL = (
    "https://pytorch.org/docs/main/torch.compiler_troubleshooting.html"
)


@dataclass
class Supportable(GraphBreakReason):
    reason: str = "Supportable"
    descr: str = "Supporting this user code is possible, but not yet implemented."
    workaround: str = (
        "Avoid tracing this code - see the `torch.compile` troubleshooting doc "
        f"{TROUBLESHOOTING_DOC_URL} for workaround suggestions. "
        "If you would like us to add support, please submit an issue."
    )


@dataclass
class Unsupportable(GraphBreakReason):
    reason: str = "Unsupportable"
    descr: str = "It is not theoretically possible to support this user code."
    workaround: str = (
        "Avoid tracing this code - see the `torch.compile` troubleshooting doc "
        f"{TROUBLESHOOTING_DOC_URL} for workaround suggestions."
    )


@dataclass
class Weird(GraphBreakReason):
    reason: str = "Weird"
    descr: str = "We're not sure why the user code is causing the graph break."
    workaround: str = (
        "Avoid tracing this code - see the `torch.compile` troubleshooting doc "
        f"{TROUBLESHOOTING_DOC_URL} for workaround suggestions. "
        "If you would like us to add support, please submit an issue."
    )
