from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterable

from tools.testing.discover_tests import TESTS


def test_command(file: str, root: str | Path) -> list[str]:
    tree = ast.parse(Path(root, file).read_text(), filename=file)
    # Ramjet does not rerun tests; this plugin starts an unrelated Python thread.
    pytest_args = ["-p", "no:rerunfailures"]
    runners = {
        "torch.testing._internal.common_utils.run_tests",
        "torch._dynamo.test_case.run_tests",
        "torch._inductor.test_case.run_tests",
    }
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
                if alias.name == "*":
                    imported = Path(root, file).parent / (
                        node.module.replace(".", "/") + ".py"
                    )
                    if not imported.is_file():
                        continue
                    imported_tree = ast.parse(
                        imported.read_text(), filename=str(imported)
                    )
                    for binding in ast.walk(imported_tree):
                        if isinstance(binding, ast.ImportFrom) and binding.module:
                            for name in binding.names:
                                qualified = f"{binding.module}.{name.name}"
                                if qualified in runners:
                                    aliases[name.asname or name.name] = qualified
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.split(".")[0]
                aliases[alias.asname or name] = alias.name if alias.asname else name
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and len(node.test.ops) == 1
            and isinstance(node.test.ops[0], ast.Eq)
        ):
            continue
        names = {ast.unparse(node.test.left), ast.unparse(node.test.comparators[0])}
        if names != {"__name__", "'__main__'"}:
            continue
        for call in ast.walk(node):
            if isinstance(call, ast.Call):
                name, separator, suffix = ast.unparse(call.func).partition(".")
                resolved = f"{aliases.get(name, name)}{separator}{suffix}"
                if resolved in runners:
                    launcher = str(Path(__file__).with_name("ramjet_run.py"))
                    return [
                        sys.executable,
                        launcher,
                        file,
                        "--use-pytest",
                        *pytest_args,
                    ]
    return [sys.executable, "-m", "pytest", file, *pytest_args]


def collection_targets(
    targets: list[str],
    root: str | Path,
    *,
    test_modules: Iterable[str] | None = None,
    python_version: tuple[int, int] | None = None,
) -> list[list[str]]:
    root = Path(root)
    version = sys.version_info[:2] if python_version is None else python_version
    version_dir = f"v{version[0]}_{version[1]}"
    modules = TESTS if test_modules is None else test_modules
    catalog = {f"test/{module}.py" for module in modules}
    catalog = {file for file in catalog if (root / file).is_file()}
    compatible = {
        file
        for file in catalog
        if not file.startswith("test/cpython/")
        or file.startswith(f"test/cpython/{version_dir}/")
    }
    by_file: dict[str, set[str]] = {}
    trees: dict[str, ast.Module] = {}

    def tree(file: str) -> ast.Module:
        if file not in trees:
            trees[file] = ast.parse((root / file).read_text(), filename=file)
        return trees[file]

    def imported_targets(parent: str, module: str, selector: str) -> list[str]:
        helper = "test/" + module.replace(".", "/") + ".py"
        classes = {
            node.name
            for node in ast.walk(tree(helper))
            if isinstance(node, ast.ClassDef)
        }
        imports = {
            alias.name: alias.asname or alias.name
            for node in ast.walk(tree(parent))
            if isinstance(node, ast.ImportFrom) and node.module == module
            for alias in node.names
            if alias.name in classes
            or alias.name.startswith("Test")
            or alias.name.endswith("Test")
        }
        generated = {
            node.args[0].id
            for node in ast.walk(tree(parent))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "instantiate_device_type_tests"
            and node.args
            and isinstance(node.args[0], ast.Name)
        }
        selected, separator, suffix = selector.partition("::")
        names = {selected: imports[selected]} if selected in imports else imports
        if (selected and selected not in imports) or not names:
            raise ValueError(
                f"Cannot resolve {module}{'::' + selector if selector else ''} "
                f"through its parent runner {parent}; select a class from {parent}"
            )
        if generated.intersection(names.values()):
            raise ValueError(
                f"{parent} generates device classes for {module}; "
                f"select {parent} or one of its generated class names directly"
            )
        return [f"{parent}::{name}{separator}{suffix}" for name in names.values()]

    for target in targets:
        file, separator, selector = target.partition("::")
        path = root / file
        if path.is_dir():
            files = [name for name in compatible if Path(name).is_relative_to(file)]
            for name in files:
                by_file.setdefault(name, set()).add(name)
            # Helpers are imported by entrypoints outside the selected directory.
            module_prefix = ".".join(Path(file).parts[1:]) + "."
            for parent in sorted(compatible - set(files)):
                source = (root / parent).read_text()
                if not re.search(r"\bfrom\s+" + re.escape(module_prefix), source):
                    continue
                modules = {
                    node.module
                    for node in ast.walk(tree(parent))
                    if isinstance(node, ast.ImportFrom)
                    and node.module is not None
                    and node.module.startswith(module_prefix)
                    and node.module.rsplit(".", 1)[-1].startswith("test_")
                }
                for module in sorted(modules):
                    mapped = imported_targets(parent, module, "")
                    by_file.setdefault(parent, set()).update(mapped)
            continue
        if file.startswith("test/cpython/") and not file.startswith(
            f"test/cpython/{version_dir}/"
        ):
            required = Path(file).parts[2].removeprefix("v").replace("_", ".")
            raise ValueError(
                f"{file} requires Python {required}; Ramjet uses Python "
                f"{version[0]}.{version[1]}"
            )
        if not path.is_file() or path.suffix != ".py":
            raise ValueError(f"Invalid Python test target: {target}")
        source = path.read_text()
        parents = (
            {
                node.args[0].value
                for node in ast.walk(tree(file))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "raise_on_run_directly"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            }
            if "raise_on_run_directly" in source
            else set()
        )
        if parents:
            if len(parents) != 1 or not parents.issubset(compatible):
                raise ValueError(
                    f"Unsupported parent runner for {file}: {sorted(parents)}"
                )
            parent = next(iter(parents))
            module = ".".join(Path(file).with_suffix("").parts[1:])
            by_file.setdefault(parent, set()).update(
                imported_targets(parent, module, selector)
            )
        else:
            by_file.setdefault(file, set()).add(f"{file}{separator}{selector}")
    if not by_file:
        raise ValueError(
            "No test files found in the selected targets for Python "
            f"{version[0]}.{version[1]}; directories use PyTorch's CI test entrypoints"
        )
    return [sorted(by_file[file]) for file in sorted(by_file)]
