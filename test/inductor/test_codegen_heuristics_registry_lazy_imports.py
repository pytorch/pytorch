# Owner(s): ["module: inductor"]

import importlib
import subprocess
import sys

from torch._inductor.test_case import run_tests, TestCase


_TEST_SCRIPT = """\
import importlib
import sys

from torch._inductor.heuristics.registry import clear_registry, get_codegen_heuristic

if not getattr(importlib, "is_lazy_imports_enabled", lambda: False)():
    raise AssertionError("Cinder lazy imports are not enabled")

package_name = "torch._inductor.heuristics.triton_codegen"
expected_types = {
    "pointwise": (
        "torch._inductor.heuristics.triton_codegen.pointwise",
        "PointwiseHeuristic",
    ),
    "reduction": (
        "torch._inductor.heuristics.triton_codegen.reduction",
        "ReductionHeuristic",
    ),
}

def check_heuristics() -> None:
    for name, expected_type in expected_types.items():
        heuristic_type = type(get_codegen_heuristic(name, "cpu"))
        actual_type = (heuristic_type.__module__, heuristic_type.__name__)
        if actual_type != expected_type:
            raise AssertionError(f"{name}: expected {expected_type}, got {actual_type}")

print("PHASE: cold lookup", flush=True)
if package_name in sys.modules:
    raise AssertionError(f"{package_name} was loaded before the cold lookup")

check_heuristics()

parent = sys.modules[package_name]
module_names = {module_name for module_name, _ in expected_types.values()}

for module_name in module_names:
    if module_name not in sys.modules:
        raise AssertionError(f"{module_name} was not loaded by the cold lookup")

print("PHASE: cached-parent recovery", flush=True)
for module_name in module_names:
    sys.modules.pop(module_name)
    parent.__dict__.pop(module_name.rsplit(".", 1)[1], None)

clear_registry()

if package_name not in sys.modules:
    raise AssertionError(f"{package_name} is not cached")
for module_name in module_names:
    if module_name in sys.modules:
        raise AssertionError(f"{module_name} is still loaded")

check_heuristics()
for module_name in module_names:
    if module_name not in sys.modules:
        raise AssertionError(f"{module_name} was not loaded by the cached lookup")

print("PHASE: unknown-name fallback", flush=True)
for module_name in module_names:
    sys.modules.pop(module_name)
    parent.__dict__.pop(module_name.rsplit(".", 1)[1], None)

clear_registry()

if package_name not in sys.modules:
    raise AssertionError(f"{package_name} is not cached before the unknown lookup")
for module_name in module_names:
    if module_name in sys.modules:
        raise AssertionError(f"{module_name} is still loaded before the unknown lookup")

try:
    get_codegen_heuristic("unknown", "cpu")
except ValueError:
    pass
else:
    raise AssertionError("unknown heuristic did not raise ValueError")

for module_name in module_names:
    if module_name not in sys.modules:
        raise AssertionError(f"{module_name} was not loaded by the unknown lookup")
"""


class CodegenHeuristicsRegistryLazyImportsTest(TestCase):
    def test_codegen_heuristics_are_registered_on_demand(self) -> None:
        lazy_import_capability = getattr(importlib, "is_lazy_imports_enabled", None)
        if not callable(lazy_import_capability) or not lazy_import_capability():
            self.skipTest("Cinder lazy imports are unavailable or disabled")

        result = subprocess.run(
            [sys.executable, "-L", "-c", _TEST_SCRIPT],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"subprocess failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

        phases = [
            line for line in result.stdout.splitlines() if line.startswith("PHASE: ")
        ]
        self.assertEqual(
            phases,
            [
                "PHASE: cold lookup",
                "PHASE: cached-parent recovery",
                "PHASE: unknown-name fallback",
            ],
            "unexpected subprocess phases:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}",
        )


if __name__ == "__main__":
    run_tests()
