"""Compatibility shim for composable_kernel's ck_tile/01_fmha/generate.py.

That script registers its codegen ops via ``loader.load_module()``, which was
removed in Python 3.15 (deprecated since 3.4). We re-add an equivalent
``load_module`` to importlib's source file loader, then run generate.py
unchanged. This unblocks ROCm wheel builds on Python 3.15 without bumping the
CK submodule. See https://github.com/pytorch/pytorch/issues/184900.

Remove this shim and inline the direct ``python3 .../generate.py`` calls in
CMakeLists.txt once third_party/composable_kernel is bumped past the upstream
load_module fix.

Usage: python3 generate_compat.py <path-to-generate.py> [generate.py args...]
"""

import importlib.machinery
import importlib.util
import os
import runpy
import sys


def _install_load_module_shim() -> None:
    loader = importlib.machinery.SourceFileLoader
    if hasattr(loader, "load_module"):
        return

    def load_module(self, fullname):
        spec = importlib.util.spec_from_loader(fullname, self)
        module = importlib.util.module_from_spec(spec)
        self.exec_module(module)
        return module

    loader.load_module = load_module


def main() -> None:
    generate_py = os.path.abspath(sys.argv[1])
    _install_load_module_shim()
    # Run generate.py as if invoked directly: argv[0] is the script and its
    # directory leads sys.path so its 'codegen' package imports resolve.
    sys.argv = [generate_py, *sys.argv[2:]]
    sys.path.insert(0, os.path.dirname(generate_py))
    runpy.run_path(generate_py, run_name="__main__")


if __name__ == "__main__":
    main()
