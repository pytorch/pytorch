"""pytest plugin that deselects the tests this machine can't run, going by their tci
constraints. Enable it with ``-p torch.testing._internal.tci.plugin``. With
``--tci-deselected=FILE`` it also writes those tests to FILE as JSON, each with the
constraint the machine doesn't meet."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from torch.testing._internal import tci
from torch.testing._internal.tci import machine


if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.config.argparsing import Parser


def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--tci-deselected",
        metavar="FILE",
        help="write the tests tci deselects to FILE as JSON, each with the "
        "constraint this machine doesn't meet",
    )


@pytest.hookimpl(trylast=True)  # after -k and -m, to judge only the tests they keep
def pytest_collection_modifyitems(config: Config, items: list[pytest.Item]) -> None:
    deselected = {}
    for item in items:
        cls, test = getattr(item, "cls", None), getattr(item, "obj", None)
        for constraint in tci.constraints(cls, test):
            # Here, so the machine is read only once a test has a constraint
            if not constraint.met_by(machine.capture()):
                deselected[item] = constraint.code
                break
    if deselected:
        config.hook.pytest_deselected(items=list(deselected))
        items[:] = [item for item in items if item not in deselected]
    if path := config.getoption("tci_deselected"):
        with open(path, "w") as file:
            json.dump({item.nodeid: unmet for item, unmet in deselected.items()}, file)
