import logging
from dataclasses import fields
from textwrap import indent


logger = logging.getLogger(__name__)


def generate_dataclass_help(cls) -> str:
    """Auto-generate help text for dataclass default values."""
    lines = []
    for field in fields(cls):
        default = field.default
        if default is not None and default != "":
            lines.append(f"{field.name:<22} = {repr(default)}")
        else:
            lines.append(f"{field.name:<22} = ''")
    return indent("\n".join(lines), "    ")
