"""Prints, as JSON, the test files test/run_test.py runs and the test/ files each needs.

A test needs its own file, the conftest.py files pytest loads for it, and every
file under test/ those import. modulefinder cannot work this out because it
does not follow namespace packages, which is what most test/ directories are.
"""

import ast
import functools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


ROOT = Path(sys.argv[1])
TEST = ROOT / "test"


def resolve(module: list[str], search: list[Path]) -> list[Path]:
    """The files that importing `module` loads, searching each directory in turn."""
    *packages, name = module
    for directory in search:
        inits = []
        for package in packages:
            directory = directory / package
            inits.append(directory / "__init__.py")
        if (directory / f"{name}.py").is_file():
            inits.append(directory / f"{name}.py")
        elif (directory / name).is_dir():
            inits.append(directory / name / "__init__.py")
        else:
            continue
        return [f for f in inits if f.is_file()]
    return []


def imports(path: Path) -> list[Path]:
    """The files under test/ that `path` imports directly."""
    found = []
    for node in ast.walk(ast.parse(path.read_bytes())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found += resolve(alias.name.split("."), [path.parent, TEST])
        elif isinstance(node, ast.ImportFrom):
            # A script's own directory comes before test/ on sys.path.
            search = [path.parent, TEST]
            if node.level:
                search = [path.parents[node.level - 1]]
            module = node.module.split(".") if node.module else []
            if module:
                found += resolve(module, search)
            for alias in node.names:
                found += resolve(module + [alias.name], search)
    return found


def needed_files(tests: list[str], direct: dict[Path, list[Path]]) -> dict:
    @functools.cache
    def closure(path: Path) -> frozenset[Path]:
        seen, pending = {path}, [path]
        while pending:
            for dependency in direct[pending.pop()]:
                if dependency not in seen:
                    seen.add(dependency)
                    pending.append(dependency)
        return frozenset(seen)

    needed = {}
    for test in tests:
        script = TEST / f"{test}.py"
        parents = [p for p in script.parents if p.is_relative_to(TEST)]
        entry_points = [script] + [p / "conftest.py" for p in parents]
        files = set().union(*(closure(e) for e in entry_points if e.is_file()))
        needed[test] = sorted(str(f.relative_to(TEST)) for f in files)
    return needed


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from tools.testing.discover_tests import TESTS

    tests = [test for test in TESTS if (TEST / f"{test}.py").is_file()]
    sources = sorted(TEST.rglob("*.py"))
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() or 1)) as pool:
        direct = dict(zip(sources, pool.map(imports, sources, chunksize=16)))
    print(json.dumps(needed_files(tests, direct)))


if __name__ == "__main__":
    main()
