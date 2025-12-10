# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import ast
import hashlib
import inspect
import math
import sys
from ast import Constant, Expr, NodeVisitor, UnaryOp, USub
from collections.abc import Iterator, MutableSet
from functools import lru_cache
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import TypeAlias

import hypothesis
from hypothesis.configuration import storage_directory
from hypothesis.internal.conjecture.choice import ChoiceTypeT
from hypothesis.internal.escalation import is_hypothesis_file

ConstantT: TypeAlias = int | float | bytes | str

# unfortunate collision with builtin. I don't want to name the init arg bytes_.
bytesT = bytes


class Constants:
    def __init__(
        self,
        *,
        integers: MutableSet[int] | None = None,
        floats: MutableSet[float] | None = None,
        bytes: MutableSet[bytes] | None = None,
        strings: MutableSet[str] | None = None,
    ):
        self.integers: MutableSet[int] = set() if integers is None else integers
        self.floats: MutableSet[float] = set() if floats is None else floats
        self.bytes: MutableSet[bytesT] = set() if bytes is None else bytes
        self.strings: MutableSet[str] = set() if strings is None else strings

    def set_for_type(
        self, constant_type: type[ConstantT] | ChoiceTypeT
    ) -> MutableSet[int] | MutableSet[float] | MutableSet[bytes] | MutableSet[str]:
        if constant_type is int or constant_type == "integer":
            return self.integers
        elif constant_type is float or constant_type == "float":
            return self.floats
        elif constant_type is bytes or constant_type == "bytes":
            return self.bytes
        elif constant_type is str or constant_type == "string":
            return self.strings
        raise ValueError(f"unknown constant_type {constant_type}")

    def add(self, constant: ConstantT) -> None:
        self.set_for_type(type(constant)).add(constant)  # type: ignore

    def __contains__(self, constant: ConstantT) -> bool:
        return constant in self.set_for_type(type(constant))

    def __or__(self, other: "Constants") -> "Constants":
        return Constants(
            integers=self.integers | other.integers,  # type: ignore
            floats=self.floats | other.floats,  # type: ignore
            bytes=self.bytes | other.bytes,  # type: ignore
            strings=self.strings | other.strings,  # type: ignore
        )

    def __iter__(self) -> Iterator[ConstantT]:
        return iter(chain(self.integers, self.floats, self.bytes, self.strings))

    def __len__(self) -> int:
        return (
            len(self.integers) + len(self.floats) + len(self.bytes) + len(self.strings)
        )

    def __repr__(self) -> str:
        return f"Constants({self.integers=}, {self.floats=}, {self.bytes=}, {self.strings=})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Constants):
            return False
        return (
            self.integers == other.integers
            and self.floats == other.floats
            and self.bytes == other.bytes
            and self.strings == other.strings
        )


class TooManyConstants(Exception):
    # a control flow exception which we raise in ConstantsVisitor when the
    # number of constants in a module gets too large.
    pass


class ConstantVisitor(NodeVisitor):
    CONSTANTS_LIMIT: int = 1024

    def __init__(self, *, limit: bool):
        super().__init__()
        self.constants = Constants()
        self.limit = limit

    def _add_constant(self, value: object) -> None:
        if self.limit and len(self.constants) >= self.CONSTANTS_LIMIT:
            raise TooManyConstants

        if isinstance(value, str) and (
            value.isspace()
            or value == ""
            # long strings are unlikely to be useful.
            or len(value) > 20
        ):
            return
        if isinstance(value, bytes) and (
            value == b""
            # long bytes seem plausibly more likely to be useful than long strings
            # (e.g. AES-256 has a 32 byte key), but we still want to cap at some
            # point to avoid performance issues.
            or len(value) > 50
        ):
            return
        if isinstance(value, bool):
            return
        if isinstance(value, float) and math.isinf(value):
            # we already upweight inf.
            return
        if isinstance(value, int) and -100 < value < 100:
            # we already upweight small integers.
            return

        if isinstance(value, (int, float, bytes, str)):
            self.constants.add(value)
            return

        # I don't kow what case could go here, but am also not confident there
        # isn't one.
        return  # pragma: no cover

    def visit_UnaryOp(self, node: UnaryOp) -> None:
        # `a = -1` is actually a combination of a USub and the constant 1.
        if (
            isinstance(node.op, USub)
            and isinstance(node.operand, Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)
        ):
            self._add_constant(-node.operand.value)
            # don't recurse on this node to avoid adding the positive variant
            return

        self.generic_visit(node)

    def visit_Expr(self, node: Expr) -> None:
        if isinstance(node.value, Constant) and isinstance(node.value.value, str):
            return

        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        # dont recurse on JoinedStr, i.e. f strings. Constants that appear *only*
        # in f strings are unlikely to be helpful.
        return

    def visit_Constant(self, node):
        self._add_constant(node.value)
        self.generic_visit(node)


def _constants_from_source(source: str | bytes, *, limit: bool) -> Constants:
    tree = ast.parse(source)
    visitor = ConstantVisitor(limit=limit)

    try:
        visitor.visit(tree)
    except TooManyConstants:
        # in the case of an incomplete collection, return nothing, to avoid
        # muddying caches etc.
        return Constants()

    return visitor.constants


def _constants_file_str(constants: Constants) -> str:
    return str(sorted(constants, key=lambda v: (str(type(v)), v)))


@lru_cache(4096)
def constants_from_module(module: ModuleType, *, limit: bool = True) -> Constants:
    try:
        module_file = inspect.getsourcefile(module)
        # use type: ignore because we know this might error
        source_bytes = Path(module_file).read_bytes()  # type: ignore
    except Exception:
        return Constants()

    if limit and len(source_bytes) > 512 * 1024:
        # Skip files over 512kb. For reference, the largest source file
        # in Hypothesis is strategies/_internal/core.py at 107kb at time
        # of writing.
        return Constants()

    source_hash = hashlib.sha1(source_bytes).hexdigest()[:16]
    # separate cache files for each limit param. see discussion in pull/4398
    cache_p = storage_directory("constants") / (
        source_hash + ("" if limit else "_nolimit")
    )
    try:
        return _constants_from_source(cache_p.read_bytes(), limit=limit)
    except Exception:
        # if the cached location doesn't exist, or it does exist but there was
        # a problem reading it, fall back to standard computation of the constants
        pass

    try:
        constants = _constants_from_source(source_bytes, limit=limit)
    except Exception:
        # A bunch of things can go wrong here.
        # * ast.parse may fail on the source code
        # * NodeVisitor may hit a RecursionError (see many related issues on
        #   e.g. libcst https://github.com/Instagram/LibCST/issues?q=recursion),
        #   or a MemoryError (`"[1, " * 200 + "]" * 200`)
        return Constants()

    try:
        cache_p.parent.mkdir(parents=True, exist_ok=True)
        cache_p.write_text(
            f"# file: {module_file}\n# hypothesis_version: {hypothesis.__version__}\n\n"
            # somewhat arbitrary sort order. The cache file doesn't *have* to be
            # stable... but it is aesthetically pleasing, and means we could rely
            # on it in the future!
            + _constants_file_str(constants),
            encoding="utf-8",
        )
    except Exception:  # pragma: no cover
        pass

    return constants


@lru_cache(4096)
def is_local_module_file(path: str) -> bool:
    from hypothesis.internal.scrutineer import ModuleLocation

    return (
        # Skip expensive path lookup for stdlib modules.
        # This will cause false negatives if a user names their module the
        # same as a stdlib module.
        path not in sys.stdlib_module_names
        # A path containing site-packages is extremely likely to be
        # ModuleLocation.SITE_PACKAGES. Skip the expensive path lookup here.
        and "/site-packages/" not in path
        and ModuleLocation.from_path(path) is ModuleLocation.LOCAL
        # normally, hypothesis is a third-party library and is not returned
        # by local_modules. However, if it is installed as an editable package
        # with pip install -e, then we will pick up on it. Just hardcode an
        # ignore here.
        and not is_hypothesis_file(path)
        # avoid collecting constants from test files
        and not (
            "test" in (p := Path(path)).parts
            or "tests" in p.parts
            or p.stem.startswith("test_")
            or p.stem.endswith("_test")
        )
    )
