# Owner(s): ["module: cpp-extensions"]

"""Import-time invariants for :mod:`torch.utils.cpp_extension`.

The package is split so that importing it does not transitively pull in
``setuptools``. The setuptools-dependent adapters (``BuildExtension``,
``CppExtension``, ``CUDAExtension``, ``SyclExtension``) live in
``torch.utils.cpp_extension.setuptools`` and are resolved lazily on first
attribute access. These tests lock in that invariant and are meant to be
the early-warning signal if a future refactor accidentally re-introduces
an eager setuptools import.

Each test runs in a fresh subprocess so that ``sys.modules`` state from
other tests can't hide a regression.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import torch.utils.cpp_extension
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCppExtensionImports(TestCase):
    def _run(self, script: str) -> None:
        subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_plain_import_does_not_load_setuptools_submodule(self):
        self._run(
            """
            import sys
            import torch.utils.cpp_extension  # noqa: F401

            assert "torch.utils.cpp_extension.setuptools" not in sys.modules, (
                "importing torch.utils.cpp_extension eagerly loaded the "
                "setuptools adapter submodule; it must stay lazy"
            )
            """
        )

    def test_jit_and_discovery_access_does_not_load_setuptools_submodule(self):
        self._run(
            """
            import sys
            import torch.utils.cpp_extension as m

            # Public JIT entry points
            assert callable(m.load)
            assert callable(m.load_inline)
            assert callable(m.remove_extension_h_precompiler_headers)

            # Discovery helpers and constants
            assert callable(m.include_paths)
            assert callable(m.library_paths)
            assert callable(m.is_ninja_available)
            assert callable(m.verify_ninja_availability)
            m.CUDA_HOME  # just touch
            m._TORCH_PATH

            assert "torch.utils.cpp_extension.setuptools" not in sys.modules, (
                "accessing JIT/discovery names pulled in the setuptools "
                "adapter submodule"
            )
            """
        )

    def test_setuptools_adapter_access_triggers_lazy_load(self):
        self._run(
            """
            import sys
            import torch.utils.cpp_extension as m

            assert "torch.utils.cpp_extension.setuptools" not in sys.modules

            bx = m.BuildExtension
            assert "torch.utils.cpp_extension.setuptools" in sys.modules, (
                "accessing BuildExtension should have loaded the setuptools "
                "adapter submodule"
            )

            # Lazy resolution must be stable: repeated access returns the
            # same object and all four names resolve.
            assert m.BuildExtension is bx
            assert m.CppExtension is not None
            assert m.CUDAExtension is not None
            assert m.SyclExtension is not None
            """
        )

    def test_non_adapter_modules_do_not_import_setuptools(self):
        """Static check: only ``setuptools.py`` may reference setuptools.

        Guards against a future refactor silently re-introducing an import
        of setuptools (eager or lazy) in ``_discovery.py``, ``_jit.py``, or
        ``__init__.py``. The adapter submodule is the single allowed home.
        """
        pkg_dir = Path(torch.utils.cpp_extension.__file__).parent
        for name in ("__init__.py", "_discovery.py", "_jit.py"):
            src = (pkg_dir / name).read_text()
            # Strip comments and docstrings by parsing and re-dumping imports.
            import ast

            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name == "setuptools"
                            or alias.name.startswith("setuptools."),
                            f"{name} imports {alias.name}; setuptools must live "
                            "only in torch.utils.cpp_extension.setuptools",
                        )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    self.assertFalse(
                        mod == "setuptools" or mod.startswith("setuptools."),
                        f"{name} imports from {mod}; setuptools must live "
                        "only in torch.utils.cpp_extension.setuptools",
                    )


if __name__ == "__main__":
    run_tests()
