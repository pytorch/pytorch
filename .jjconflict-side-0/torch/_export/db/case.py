# mypy: allow-untyped-defs
import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from types import ModuleType

import torch

_TAGS: dict[str, dict[str, Any]] = {
    "torch": {
        "cond": {},
        "dynamic-shape": {},
        "escape-hatch": {},
        "map": {},
        "dynamic-value": {},
        "operator": {},
        "mutation": {},
    },
    "python": {
        "assert": {},
        "builtin": {},
        "closure": {},
        "context-manager": {},
        "control-flow": {},
        "data-structure": {},
        "standard-library": {},
        "object-model": {},
    },
}


class SupportLevel(Enum):
    """
    Indicates at what stage the feature
    used in the example is handled in export.
    """

    SUPPORTED = 1
    NOT_SUPPORTED_YET = 0


ArgsType = tuple[Any, ...]


def check_inputs_type(args, kwargs):
    if not isinstance(args, tuple):
        raise ValueError(
            f"Expecting args type to be a tuple, got: {type(args)}"
        )
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"Expecting kwargs type to be a dict, got: {type(kwargs)}"
        )
    for key in kwargs:
        if not isinstance(key, str):
            raise ValueError(
                f"Expecting kwargs keys to be a string, got: {type(key)}"
            )

def _validate_tag(tag: str):
    parts = tag.split(".")
    t = _TAGS
    for part in parts:
        if not set(part) <= set(string.ascii_lowercase + "-"):
            raise AssertionError(f"Tag contains invalid characters: {part}")
        if part in t:
            t = t[part]
        else:
            raise ValueError(f"Tag {tag} is not found in registered tags.")


@dataclass(frozen=True)
class ExportCase:
    example_args: ArgsType
    description: str  # A description of the use case.
    model: torch.nn.Module
    name: str
    example_kwargs: dict[str, Any] = field(default_factory=dict)
    extra_args: ArgsType | None = None  # For testing graph generalization.
    # Tags associated with the use case. (e.g dynamic-shape, escape-hatch)
    tags: set[str] = field(default_factory=set)
    support_level: SupportLevel = SupportLevel.SUPPORTED
    dynamic_shapes: dict[str, Any] | None = None

    def __post_init__(self):
        check_inputs_type(self.example_args, self.example_kwargs)
        if self.extra_args is not None:
            check_inputs_type(self.extra_args, {})

        for tag in self.tags:
            _validate_tag(tag)

        if not isinstance(self.description, str) or len(self.description) == 0:
            raise ValueError(f'Invalid description: "{self.description}"')


_EXAMPLE_CASES: dict[str, ExportCase] = {}
_MODULES: set[ModuleType] = set()
_EXAMPLE_CONFLICT_CASES: dict[str, list[ExportCase]] = {}
_EXAMPLE_REWRITE_CASES: dict[str, list[ExportCase]] = {}


def register_db_case(case: ExportCase) -> None:
    """
    Registers a user provided ExportCase into example bank.
    """
    if case.name in _EXAMPLE_CASES:
        if case.name not in _EXAMPLE_CONFLICT_CASES:
            _EXAMPLE_CONFLICT_CASES[case.name] = [_EXAMPLE_CASES[case.name]]
        _EXAMPLE_CONFLICT_CASES[case.name].append(case)
        return

    _EXAMPLE_CASES[case.name] = case


def to_snake_case(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def _make_export_case(m, name, configs):
    if not isinstance(m, torch.nn.Module):
        raise TypeError("Export case class should be a torch.nn.Module.")

    if "description" not in configs:
        # Fallback to docstring if description is missing.
        if m.__doc__ is None:
            raise AssertionError(
                f"Could not find description or docstring for export case: {m}"
            )
        configs = {**configs, "description": m.__doc__}
    # pyrefly: ignore [bad-argument-type]
    return ExportCase(**{**configs, "model": m, "name": name})


def export_case(**kwargs):
    """
    Decorator for registering a user provided case into example bank.
    """

    def wrapper(m):
        configs = kwargs
        module = inspect.getmodule(m)
        if module in _MODULES:
            raise RuntimeError("export_case should only be used once per example file.")

        if module is None:
            raise AssertionError("module must not be None")
        _MODULES.add(module)
        module_name = module.__name__.split(".")[-1]
        case = _make_export_case(m, module_name, configs)
        register_db_case(case)
        return case

    return wrapper


def export_rewrite_case(**kwargs):
    def wrapper(m):
        configs = kwargs

        parent = configs.pop("parent")
        if not isinstance(parent, ExportCase):
            raise AssertionError(f"expected ExportCase, got {type(parent)}")
        key = parent.name
        if key not in _EXAMPLE_REWRITE_CASES:
            _EXAMPLE_REWRITE_CASES[key] = []

        configs["example_args"] = parent.example_args
        case = _make_export_case(m, to_snake_case(m.__name__), configs)
        _EXAMPLE_REWRITE_CASES[key].append(case)
        return case

    return wrapper
