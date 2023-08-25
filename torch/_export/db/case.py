import inspect
import re
import string
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.export import Constraint

_TAGS: Dict[str, Dict[str, Any]] = {
    "torch": {
        "cond": {},
        "dynamic-shape": {},
        "escape-hatch": {},
        "map": {},
    },
    "python": {
        "assert": {},
        "builtin": {},
        "closure": {},
        "context-manager": {},
        "control-flow": {},
        "data-structure": {},
        "standard-library": {},
    },
}


class SupportLevel(Enum):
    """
    Indicates at what stage the feature
    used in the example is handled in export.
    """

    SUPPORTED = 1
    NOT_SUPPORTED_YET = 0


class ExportArgs:
    __slots__ = ("args", "kwargs")

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


InputsType = Union[Tuple[Any, ...], ExportArgs]


def check_inputs_type(x):
    if not isinstance(x, (ExportArgs, tuple)):
        raise ValueError(
            f"Expecting inputs type to be either a tuple, or ExportArgs, got: {type(x)}"
        )


def _validate_tag(tag: str):
    parts = tag.split(".")
    t = _TAGS
    for part in parts:
        assert set(part) <= set(
            string.ascii_lowercase + "-"
        ), f"Tag contains invalid characters: {part}"
        if part in t:
            t = t[part]
        else:
            raise ValueError(f"Tag {tag} is not found in registered tags.")


@dataclass(frozen=True)
class ExportCase:
    example_inputs: InputsType
    description: str  # A description of the use case.
    model: torch.nn.Module
    name: str
    extra_inputs: Optional[InputsType] = None  # For testing graph generalization.
    # Tags associated with the use case. (e.g dynamic-shape, escape-hatch)
    tags: Set[str] = field(default_factory=lambda: set())
    support_level: SupportLevel = SupportLevel.SUPPORTED
    constraints: List[Constraint] = field(default_factory=list)

    def __post_init__(self):
        check_inputs_type(self.example_inputs)
        if self.extra_inputs is not None:
            check_inputs_type(self.extra_inputs)

        for tag in self.tags:
            _validate_tag(tag)

        if not isinstance(self.description, str) or len(self.description) == 0:
            raise ValueError(f'Invalid description: "{self.description}"')


_EXAMPLE_CASES: Dict[str, ExportCase] = {}
_MODULES = set()
_EXAMPLE_CONFLICT_CASES = {}
_EXAMPLE_REWRITE_CASES: Dict[str, List[ExportCase]] = {}


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
    if inspect.isclass(m):
        if not issubclass(m, torch.nn.Module):
            raise TypeError("Export case class should be a torch.nn.Module.")
        m = m()

    if "description" not in configs:
        # Fallback to docstring if description is missing.
        assert (
            m.__doc__ is not None
        ), f"Could not find description or docstring for export case: {m}"
        configs = {**configs, "description": m.__doc__}
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

        _MODULES.add(module)
        normalized_name = to_snake_case(m.__name__)
        assert module is not None
        module_name = module.__name__.split(".")[-1]
        if module_name != normalized_name:
            raise RuntimeError(
                f'Module name "{module.__name__}" is inconsistent with exported program '
                + f'name "{m.__name__}". Please rename the module to "{normalized_name}".'
            )

        case = _make_export_case(m, module_name, configs)
        register_db_case(case)
        return case

    return wrapper


def export_rewrite_case(**kwargs):
    def wrapper(m):
        configs = kwargs

        parent = configs.pop("parent")
        assert isinstance(parent, ExportCase)
        key = parent.name
        if key not in _EXAMPLE_REWRITE_CASES:
            _EXAMPLE_REWRITE_CASES[key] = []

        configs["example_inputs"] = parent.example_inputs
        case = _make_export_case(m, to_snake_case(m.__name__), configs)
        _EXAMPLE_REWRITE_CASES[key].append(case)
        return case

    return wrapper


def normalize_inputs(x: InputsType) -> ExportArgs:
    if isinstance(x, tuple):
        return ExportArgs(*x)

    assert isinstance(x, ExportArgs)
    return x
