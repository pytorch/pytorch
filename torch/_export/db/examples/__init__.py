# mypy: allow-untyped-defs
import glob
import importlib
from os.path import basename, dirname, isfile, join

import torch
from torch._export.db.case import (
    _EXAMPLE_CASES,
    _EXAMPLE_CONFLICT_CASES,
    _EXAMPLE_REWRITE_CASES,
    SupportLevel,
)


modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]

# Import all module in the current directory.
from . import *  # noqa: F403


def all_examples():
    return _EXAMPLE_CASES


if len(_EXAMPLE_CONFLICT_CASES) > 0:

    def get_name(case):
        model = case.model
        if isinstance(model, torch.nn.Module):
            model = type(model)
        return model.__name__

    msg = "Error on conflict export case name.\n"
    for case_name, cases in _EXAMPLE_CONFLICT_CASES.items():
        msg += f"Case name {case_name} is associated with multiple cases:\n  "
        msg += f"[{','.join(map(get_name, cases))}]\n"

    raise RuntimeError(msg)


def filter_examples_by_support_level(support_level: SupportLevel):
    return {
        key: val
        for key, val in all_examples().items()
        if val.support_level == support_level
    }


def get_rewrite_cases(case):
    return _EXAMPLE_REWRITE_CASES.get(case.name, [])
