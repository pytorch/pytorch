#!/usr/bin/env python3
"""Validate test case requirements.

This linter enforces test case requirements, including hardware classification
declaration, test instantiation patterns, test method signatures, and
supported decorators.

A JSON allowlist tracks test files that are not yet migrated to this linter's
requirements. Files in the allowlist are skipped silently. Files not in the
allowlist must satisfy the test case requirements defined below.

All test classes inheriting from `TestCase` must first declare a valid
`hw_classification` attribute. Supported values are `GENERIC`, `ACCELERATOR`,
`CPU`, `CUDA`, `MPS`, and `XPU`.

The requirements for each `hw_classification` are summarized below:

  GENERIC
    - Class must not be used with instantiate_device_type_tests.
    - Test methods must not accept device/devices parameter.
    - Class body must not check accelerator availability, e.g.
      torch.cuda.is_available().

  ACCELERATOR
    - Class must be used with instantiate_device_type_tests.
    - Every test method must accept device/devices parameter.
    - Test methods must not use device-specific @only* decorators (except @onlyAccelerator).
    - instantiate_device_type_tests must not use only_for (use except_for
      as a blacklist approach instead).

  CPU / CUDA / MPS / XPU (device-specific)
    - Only the hw_classification declaration is required; nothing else is
      checked for these classes.

The scan covers module-level statements and statements recursively nested within
``if``, ``try`` (including ``except`` handlers), ``with``, ``for``, and ``while``
bodies, so conditionally defined test classes and instantiation calls are linted
as well.

Usage:

    # Lint the given files (every test file if none are given)
    python tools/linter/adapters/test_linter.py [filenames ...]

    # Regenerate the allowlist from the linter's own results
    python tools/linter/adapters/test_linter.py --regenerate
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import fnmatch
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import NamedTuple, TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


LINTER_CODE = "TEST_LINTER"
HW_CLASSIFICATION_ATTR = "hw_classification"
INSTANTIATE_FN_NAME = "instantiate_device_type_tests"
REPO_ROOT = Path(__file__).resolve().parents[3]


# Files in this allowlist are temporarily excluded from test linter checks
ALLOWLIST_PATH = Path(__file__).resolve().parent / "test_linter_allowlist.json"
ALLOWLIST_REL_PATH = os.path.relpath(ALLOWLIST_PATH, REPO_ROOT)

INCLUDE_PATTERNS = ("test/**/test_*.py", "test/**/*_test.py")
EXCLUDE_PREFIXES = (
    "test/cpp_extensions/open_registration_extension/",
    "test/cpython/",
)


class _UnknownKwarg:
    pass


# Mirrors the member names of `torch.testing._internal.common_utils.HardwareClassification`.
# Values differ from upstream; only member names are used for matching.
# Defined locally to avoid importing test infrastructure into the linter.
class HardwareClassification(Enum):
    GENERIC = "GENERIC"
    ACCELERATOR = "ACCELERATOR"
    CPU = "CPU"
    CUDA = "CUDA"
    MPS = "MPS"
    XPU = "XPU"


# Lint message types
class LintSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    ADVICE = "advice"
    DISABLED = "disabled"


class LintMessage(NamedTuple):
    path: str
    line: int | None
    char: int | None
    code: str
    severity: LintSeverity
    name: str
    original: str | None
    replacement: str | None
    description: str | None


error_msg = partial(
    LintMessage,
    char=None,
    code=LINTER_CODE,
    severity=LintSeverity.ERROR,
    original=None,
    replacement=None,
)


def _load_allowlist() -> set[str]:
    if ALLOWLIST_PATH.exists():
        with open(ALLOWLIST_PATH) as f:
            return set(json.load(f))
    return set()


_allowlist: set[str] = _load_allowlist()

_KWARG_UNKNOWN = _UnknownKwarg()  # sentinel: kwarg present but not a literal


def _is_test_file(filename: str) -> bool:
    """True for test files this linter applies to."""
    rel_path = os.path.relpath(filename, REPO_ROOT).replace("\\", "/")
    if rel_path.startswith(EXCLUDE_PREFIXES):
        return False

    return any(
        fnmatch.fnmatchcase(rel_path, pattern)
        or fnmatch.fnmatchcase(rel_path, pattern.replace("**/", ""))
        for pattern in INCLUDE_PATTERNS
    )


def _discover_files() -> list[Path]:
    """Return the test files this linter applies to, sorted."""
    files: set[Path] = set()
    for pattern in INCLUDE_PATTERNS:
        for path in REPO_ROOT.glob(pattern):
            if _is_test_file(str(path)):
                files.add(path)
    return sorted(files)


def _is_test_class(node: ast.ClassDef) -> bool:
    # Identify test classes by the presence of test methods rather than
    # inheritance. Resolving TestCase inheritance through AST is incomplete
    # because subclasses can be defined indirectly (e.g. NNTestCase, JitTestCase)
    # and across different files. Therefore, intermediate base classes and
    # concrete test classes are treated uniformly: any class defining test
    # methods must declare hw_classification.
    for stmt in node.body:
        if isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and stmt.name.startswith("test"):
            return True
    return False


def _get_hw_classification(
    node: ast.ClassDef,
) -> HardwareClassification | None:
    """Parse the `hw_classification` attribute from *node*.

    The value is returned as a `HardwareClassification` enum member.

    Only accepts the exact forms:

        hw_classification = HardwareClassification.<MEMBER>
        hw_classification: HardwareClassification = HardwareClassification.<MEMBER>

    Returns `None` if the attribute is absent or does not match one of the
    supported forms.
    """
    for stmt in node.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not (
                isinstance(target, ast.Name) and target.id == HW_CLASSIFICATION_ATTR
            ):
                continue
            value = stmt.value
        elif isinstance(stmt, ast.AnnAssign):
            target = stmt.target
            if not (
                isinstance(target, ast.Name) and target.id == HW_CLASSIFICATION_ATTR
            ):
                continue
            if not (
                isinstance(stmt.annotation, ast.Name)
                and stmt.annotation.id == HardwareClassification.__name__
                and stmt.value is not None
            ):
                return None
            value = stmt.value
        else:
            continue

        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == HardwareClassification.__name__
        ):
            try:
                return HardwareClassification[value.attr]
            except KeyError:
                return None

        return None

    return None


class _ScannedStatementCollector(ast.NodeVisitor):
    """Collect statements from the module and nested ``if``/``try``/``with``/
    ``for``/``while`` bodies, so conditionally defined test classes and
    ``instantiate_device_type_tests`` calls are linted too.
    """

    def __init__(self) -> None:
        self.statements: list[ast.stmt] = []

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, ast.stmt):
            self.statements.append(node)

    def visit_Module(self, node: ast.Module) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> None:
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    def visit_Try(self, node: ast.Try) -> None:
        for stmt in node.body + node.orelse + node.finalbody:
            self.visit(stmt)
        for handler in node.handlers:
            for stmt in handler.body:
                self.visit(stmt)

    def visit_With(self, node: ast.With) -> None:
        for stmt in node.body:
            self.visit(stmt)

    def visit_For(self, node: ast.For) -> None:
        for stmt in node.body + node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> None:
        for stmt in node.body + node.orelse:
            self.visit(stmt)


def _scanned_statements(tree: ast.Module) -> list[ast.stmt]:
    collector = _ScannedStatementCollector()
    collector.visit(tree)
    return collector.statements


@dataclass(frozen=True)
class ClassEntry:
    """A test class definition and its optional instantiation call."""

    class_def: ast.ClassDef
    instantiation: ast.Call | None = None


def _is_instantiate_call(call: ast.Call) -> bool:
    """True for direct ``instantiate_device_type_tests(...)`` and attribute-form
    ``common_device_type.instantiate_device_type_tests(...)`` calls."""
    if isinstance(call.func, ast.Name):
        return call.func.id == INSTANTIATE_FN_NAME
    if isinstance(call.func, ast.Attribute):
        return call.func.attr == INSTANTIATE_FN_NAME
    return False


def _collect_test_classes(tree: ast.Module) -> dict[str, ClassEntry]:
    """Map each test class name to its definition and instantiation call, if any."""
    class_defs: dict[str, ast.ClassDef] = {}
    instantiations: dict[str, ast.Call] = {}
    for stmt in _scanned_statements(tree):
        if isinstance(stmt, ast.ClassDef) and _is_test_class(stmt):
            class_defs[stmt.name] = stmt
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            if (
                _is_instantiate_call(call)
                and call.args
                and isinstance(call.args[0], ast.Name)
            ):
                instantiations[call.args[0].id] = call
    return {
        name: ClassEntry(class_defs[name], instantiations.get(name))
        for name in class_defs
    }


def _get_string_list_kwarg(
    call: ast.Call, param_name: str
) -> list[str] | None | _UnknownKwarg:
    """Return statically known string list value of a keyword argument.

    Returns:
        - None: keyword argument is absent.
        - list[str]: keyword argument is a statically known string list.
        - _KWARG_UNKNOWN: keyword argument exists but cannot be statically resolved.
    """
    for kw_item in call.keywords:
        if kw_item.arg != param_name:
            continue

        node = kw_item.value

        if isinstance(node, ast.Constant) and node.value is None:
            return None

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return [node.value]

        if isinstance(node, (ast.List, ast.Tuple)):
            result = [
                elt.value
                for elt in node.elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]

            # list contains non-string elements
            if len(result) != len(node.elts):
                return _KWARG_UNKNOWN
            return result

        return _KWARG_UNKNOWN

    return None


def _accelerator_check(call: ast.Call) -> str | None:
    """Return the dotted name of an accelerator is_available() call, or None."""

    # Device modules an availability check may probe. "accelerator" is included on
    # purpose: torch.accelerator.is_available() still needs some accelerator.
    ACCELERATOR_MODULES = {
        "accelerator",
        "cuda",
        "hpu",
        "ipu",
        "mps",
        "mtia",
        "xla",
        "xpu",
    }

    func = call.func
    if not isinstance(func, ast.Attribute):
        return None

    name = ast.unparse(func)
    parts = name.split(".")
    if parts[0] != "torch" or not ACCELERATOR_MODULES.intersection(parts):
        return None
    return name


@dataclass(frozen=True)
class InstantiationContext:
    """Context for an ``instantiate_device_type_tests`` call."""

    call: ast.Call
    only_for: list[str] | None | _UnknownKwarg

    @classmethod
    def from_call(cls, call: ast.Call) -> InstantiationContext:
        return cls(call=call, only_for=_get_string_list_kwarg(call, "only_for"))


@dataclass(frozen=True)
class RuleContext:
    """Context passed to each rule function during test class linter checks."""

    filename: str
    class_node: ast.ClassDef
    classification: HardwareClassification
    test_methods: list[ast.FunctionDef | ast.AsyncFunctionDef] = field(
        default_factory=list
    )
    instantiation: InstantiationContext | None = None

    @classmethod
    def from_node(
        cls,
        filename: str,
        class_node: ast.ClassDef,
        instantiation: ast.Call | None,
        classification: HardwareClassification,
    ) -> RuleContext:
        test_methods = [
            stmt
            for stmt in class_node.body
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and stmt.name.startswith("test")
        ]

        inst = (
            InstantiationContext.from_call(instantiation)
            if instantiation is not None
            else None
        )
        return cls(
            filename=filename,
            class_node=class_node,
            classification=classification,
            test_methods=test_methods,
            instantiation=inst,
        )


class RuleId(Enum):
    """Short names of lint rules, used as message names and registry keys."""

    HW_CLASSIFICATION = "hw_classification"
    DEVICE_PARAM = "device_param"
    INSTANTIATION = "instantiation"
    ACCELERATOR_AVAILABILITY = "accelerator_availability"
    DECORATOR = "decorator"
    ONLY_FOR = "only_for"


class Rule:
    """Base class for lint rules.

    Each rule is a class declaring its id, a summary of what it checks and why
    it is needed, and a ``check`` classmethod implementing the rule. The check
    builds its own messages with ``error_msg``; check signatures are
    rule-specific.
    """

    id: RuleId
    summary: str

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if getattr(cls, "id", None) is None or not getattr(cls, "summary", None):
            raise TypeError(
                f"Rule {cls.__name__} must define 'id' and a non-empty 'summary'"
            )

    @classmethod
    def tag(cls) -> str:
        return f"[{cls.id.value}]"

    @classmethod
    def hint(cls) -> str:
        """Pointer to this rule's summary, appended to every message."""
        return (
            f"See the '{cls.id.value}' rule summary in "
            f"{os.path.relpath(__file__, REPO_ROOT)} for details."
        )

    @classmethod
    def check(cls, ctx: RuleContext) -> list[LintMessage]:
        raise NotImplementedError


rules: dict[HardwareClassification, list[type[Rule]]] = {}


def _register(*groups: HardwareClassification) -> Callable[[type[Rule]], type[Rule]]:
    """Decorator: register a Rule subclass into one or more classification groups.

    Example::

        @_register(HardwareClassification.ACCELERATOR)
        class DecoratorRule(Rule): ...
    """

    def decorator(rule_cls: type[Rule]) -> type[Rule]:
        for group in groups:
            rules.setdefault(group, []).append(rule_cls)
        return rule_cls

    return decorator


# ---------------------------------------------------------------------------
# Rules. Keep this list in sync with the module docstring.
# ---------------------------------------------------------------------------


# The gate rule: it must run before dispatch, when the classification is not
# known yet, so it is called explicitly instead of being registered.
class HwClassificationRule(Rule):
    id = RuleId.HW_CLASSIFICATION
    summary = (
        "Every test class must declare a valid hw_classification attribute "
        "(GENERIC, ACCELERATOR, CPU, CUDA, MPS, or XPU) so test "
        "infrastructure knows which hardware the tests target.\n"
        "GENERIC marks device-agnostic tests, ACCELERATOR marks tests that "
        "run on every accelerator via instantiate_device_type_tests, and "
        "CPU/CUDA/MPS/XPU mark device-specific classes. Declare the "
        "attribute in the class body, e.g.:\n"
        "    hw_classification = HardwareClassification.GENERIC"
    )

    @classmethod
    def parse(
        cls, filename: str, node: ast.ClassDef
    ) -> tuple[HardwareClassification | None, list[LintMessage]]:
        """Parse and validate the class's hw_classification declaration.

        Returns the classification and any errors (non-empty iff the
        declaration is missing or invalid).
        """
        classification = _get_hw_classification(node)
        if classification is None:
            return None, [
                error_msg(
                    name=cls.tag(),
                    path=filename,
                    line=node.lineno,
                    description=(
                        f"Test class '{node.name}': missing or invalid hw_classification."
                        f"\n{cls.hint()}"
                    ),
                )
            ]
        return classification, []


@_register(HardwareClassification.GENERIC, HardwareClassification.ACCELERATOR)
class InstantiationRule(Rule):
    id = RuleId.INSTANTIATION
    summary = (
        "GENERIC classes must not be used with instantiate_device_type_tests; "
        "ACCELERATOR classes must be.\n"
        "instantiate_device_type_tests turns a device-parameterized test "
        "class into per-accelerator test classes, which only makes sense for "
        "accelerator tests. For ACCELERATOR classes, add the call at module "
        "level, e.g.:\n"
        "    instantiate_device_type_tests(TestFoo, globals())"
    )

    @classmethod
    def check(cls, ctx: RuleContext) -> list[LintMessage]:
        is_instantiated = ctx.instantiation is not None
        required = ctx.classification is HardwareClassification.ACCELERATOR
        if is_instantiated == required:
            return []
        class_name = ctx.class_node.name
        if required:
            description = (
                f"Test class '{class_name}' ({ctx.classification.value}): "
                f"must be used with {INSTANTIATE_FN_NAME}."
            )
        else:
            description = (
                f"Test class '{class_name}' ({ctx.classification.value}): "
                f"must not be used with {INSTANTIATE_FN_NAME}."
            )
        return [
            error_msg(
                name=cls.tag(),
                path=ctx.filename,
                line=ctx.class_node.lineno,
                description=f"{description}\n{cls.hint()}",
            )
        ]


@_register(HardwareClassification.GENERIC, HardwareClassification.ACCELERATOR)
class DeviceParamRule(Rule):
    id = RuleId.DEVICE_PARAM
    summary = (
        "GENERIC test methods must not accept a device or devices parameter; "
        "ACCELERATOR test methods must accept one.\n"
        "GENERIC tests exercise device-agnostic logic and must not depend on "
        "an accelerator. ACCELERATOR tests run on every accelerator and "
        "receive the device under test through this parameter."
    )

    @classmethod
    def check(cls, ctx: RuleContext) -> list[LintMessage]:
        required = ctx.classification is HardwareClassification.ACCELERATOR
        messages: list[LintMessage] = []
        for stmt in ctx.test_methods:
            params = {
                a.arg
                for a in stmt.args.args + stmt.args.posonlyargs + stmt.args.kwonlyargs
            }
            has_device_param = "device" in params or "devices" in params
            if has_device_param == required:
                continue
            action = "must accept" if required else "must not accept"
            messages.append(
                error_msg(
                    name=cls.tag(),
                    path=ctx.filename,
                    line=stmt.lineno,
                    description=(
                        f"Test method '{ctx.class_node.name}.{stmt.name}' "
                        f"({ctx.classification.value}): {action} a 'device' or 'devices' parameter."
                        f"\n{cls.hint()}"
                    ),
                )
            )
        return messages


@_register(HardwareClassification.GENERIC)
class AcceleratorAvailabilityRule(Rule):
    id = RuleId.ACCELERATOR_AVAILABILITY
    summary = (
        "GENERIC classes must not check accelerator availability in the "
        "class body.\n"
        "Such checks make test behavior depend on the available accelerators, "
        "which is inconsistent with the GENERIC classification. Move the "
        "check to an ACCELERATOR or device-specific test class."
    )

    @classmethod
    def check(cls, ctx: RuleContext) -> list[LintMessage]:
        messages: list[LintMessage] = []
        for stmt in ctx.class_node.body:
            owner = ctx.class_node.name
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                owner = f"{owner}.{stmt.name}"
            for node in ast.walk(stmt):
                if not isinstance(node, ast.Call):
                    continue
                check = _accelerator_check(node)
                if check is None:
                    continue
                messages.append(
                    error_msg(
                        name=cls.tag(),
                        path=ctx.filename,
                        line=node.lineno,
                        description=(
                            f"Test class '{ctx.class_node.name}' ({ctx.classification.value}): "
                            f"must not check accelerator availability in '{owner}': "
                            f"'{check}()'."
                            f"\n{cls.hint()}"
                        ),
                    )
                )
        return messages


@_register(HardwareClassification.ACCELERATOR)
class DecoratorRule(Rule):
    id = RuleId.DECORATOR
    summary = (
        "ACCELERATOR test methods must not use device-specific only* "
        "decorators; @onlyAccelerator is permitted.\n"
        "Device-specific decorators restrict tests to particular devices, "
        "which conflicts with the device-generic intent of the ACCELERATOR "
        "classification."
    )

    @classmethod
    def check(cls, ctx: RuleContext) -> list[LintMessage]:
        # Device-specific only* decorators forbidden in ACCELERATOR classes.
        forbidden = {
            "onlyCPU",
            "onlyCUDA",
            "onlyMPS",
            "onlyXPU",
            "onlyHPU",
            "onlyPRIVATEUSE1",
            "onlyOn",
            "onlyCUDAAndPRIVATEUSE1",
            "onlyNativeDeviceTypes",
            "onlyNativeDeviceTypesAnd",
        }

        def get_decorator_name(dec: ast.expr) -> str | None:
            """Extract the decorator name from forms like @foo, @obj.foo, and @foo(...)."""
            if isinstance(dec, ast.Name):
                return dec.id
            if isinstance(dec, ast.Attribute):
                return dec.attr
            if isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    return dec.func.id
                if isinstance(dec.func, ast.Attribute):
                    return dec.func.attr
            return None

        messages: list[LintMessage] = []
        for stmt in ctx.test_methods:
            for dec in stmt.decorator_list:
                name = get_decorator_name(dec)
                if name in forbidden:
                    messages.append(
                        error_msg(
                            name=cls.tag(),
                            path=ctx.filename,
                            line=stmt.lineno,
                            description=(
                                f"Test method '{ctx.class_node.name}.{stmt.name}' "
                                f"({ctx.classification.value}): must not use '@{name}'; "
                                f"only '@onlyAccelerator' is allowed."
                                f"\n{cls.hint()}"
                            ),
                        )
                    )
        return messages


@_register(HardwareClassification.ACCELERATOR)
class OnlyForRule(Rule):
    id = RuleId.ONLY_FOR
    summary = (
        "ACCELERATOR classes must not use only_for in "
        "instantiate_device_type_tests.\n"
        "only_for restricts tests to specific devices, preventing coverage "
        "on other accelerators. Use except_for to exclude known-unsupported "
        "devices when necessary."
    )

    @classmethod
    def check(cls, ctx: RuleContext) -> list[LintMessage]:
        if ctx.instantiation is None or ctx.instantiation.only_for is None:
            return []
        return [
            error_msg(
                name=cls.tag(),
                path=ctx.filename,
                line=ctx.instantiation.call.lineno,
                description=(
                    f"Test class '{ctx.class_node.name}' ({ctx.classification.value}): "
                    f"must not use only_for in {INSTANTIATE_FN_NAME}; use except_for instead."
                    f"\n{cls.hint()}"
                ),
            )
        ]


def check_file(filename: str) -> list[LintMessage]:
    # Callers pre-filter with _is_test_file; only the allowlist gate remains.
    rel_path = os.path.relpath(filename, REPO_ROOT).replace("\\", "/")
    if rel_path in _allowlist:
        return []

    try:
        with open(filename, encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filename)
    except (OSError, SyntaxError) as e:
        logging.error("Failed to parse '%s': %s", filename, e)
        return []

    test_classes = _collect_test_classes(tree)

    messages: list[LintMessage] = []
    for entry in test_classes.values():
        node = entry.class_def
        classification, errors = HwClassificationRule.parse(filename, node)
        messages.extend(errors)
        if classification is None:
            continue

        # Dispatch to registered rules for this classification
        ctx = RuleContext.from_node(
            filename=filename,
            class_node=node,
            instantiation=entry.instantiation,
            classification=classification,
        )
        for rule in rules.get(classification, []):
            messages.extend(rule.check(ctx))

    return messages


def _regenerate_allowlist() -> None:
    """Regenerate the allowlist from the linter's results on all test files."""

    # check_file() returns nothing for files already in the allowlist, which
    # would drop every existing entry; clear it so the list is rebuilt from the
    # linter's real output rather than from its current contents.
    global _allowlist
    _allowlist = set()

    files = _discover_files()
    entries = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in files
        if check_file(str(path))
    ]

    old_content = (
        ALLOWLIST_PATH.read_text(encoding="utf-8") if ALLOWLIST_PATH.exists() else ""
    )
    old: list[str] = json.loads(old_content) if old_content else []

    added = sorted(set(entries) - set(old))
    removed = sorted(set(old) - set(entries))

    print(f"Checked {len(files)} test files; {len(entries)} require allowlisting.")
    print(f"Allowlist changes: {len(added)} added, {len(removed)} removed.")

    for prefix, paths in (("+", added), ("-", removed)):
        for path in paths:
            print(f"  {prefix} {path}")

    ALLOWLIST_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} entries to {ALLOWLIST_REL_PATH}")


def _default_num_workers() -> int | None:
    max_jobs = os.environ.get("MAX_JOBS")
    if max_jobs and max_jobs.isdigit() and int(max_jobs) > 0:
        return int(max_jobs)
    return os.cpu_count()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ensure test classes declare hw_classification.",
        fromfile_prefix_chars="@",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="verbose logging",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="regenerate the allowlist from actual linter results",
    )
    parser.add_argument(
        "filenames", nargs="*", help="paths to lint (all test files if omitted)"
    )
    args = parser.parse_args()

    if args.regenerate:
        if args.filenames:
            parser.error("filenames cannot be combined with --regenerate")
        _regenerate_allowlist()
        return

    logging.basicConfig(
        format="<%(threadName)s:%(levelname)s> %(message)s",
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
    )

    # No filenames means lint every test file this linter applies to.
    candidates = args.filenames or [str(p) for p in _discover_files()]
    filenames = [x for x in candidates if _is_test_file(x)]

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=_default_num_workers(),
    ) as executor:
        futures = {executor.submit(check_file, x): x for x in filenames}
        for future in concurrent.futures.as_completed(futures):
            try:
                for lint_message in future.result():
                    print(json.dumps(lint_message._asdict()), flush=True)
            except Exception:
                logging.critical('Failed at "%s".', futures[future])


if __name__ == "__main__":
    main()
