# mypy: allow-untyped-defs
import dataclasses
import glob
import inspect
from os.path import basename, dirname, isfile, join

import torch
from torch._export.db.case import (
    _EXAMPLE_CASES,
    _EXAMPLE_CONFLICT_CASES,
    _EXAMPLE_REWRITE_CASES,
    SupportLevel,
    export_case,
    ExportCase,
)


def _collect_examples():
    case_names = glob.glob(join(dirname(__file__), "*.py"))
    case_names = [
        basename(f)[:-3] for f in case_names if isfile(f) and not f.endswith("__init__.py")
    ]

    case_fields = {f.name for f in dataclasses.fields(ExportCase)}
    for case_name in case_names:
        case = __import__(case_name, globals(), locals(), [], 1)
        variables = [name for name in dir(case) if name in case_fields]
        export_case(**{v: getattr(case, v) for v in variables})(case.model)

_collect_examples()

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
