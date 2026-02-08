#!/usr/bin/env python3
"""
Static Import Graph Smoke Test

This tool performs AST-based static analysis of Python imports in the PyTorch
source tree to detect import graph integrity issues WITHOUT requiring a C++
build or executing any code.

Purpose:
  - Validate internal torch.* import paths remain resolvable
  - Catch import refactoring errors early (before expensive CI build)
  - Establish baseline import graph for INV-050 (Import Path Stability)

Usage:
  python -m tools.refactor.import_smoke_static --targets torch,torch.nn,torch.optim

Exit Codes:
  0 = Success (all imports resolved)
  1 = Failure (unresolved imports found)
  2 = Invalid arguments or runtime error
"""

import argparse
import ast
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple


# =============================================================================
# ALLOWLIST PHILOSOPHY
# =============================================================================
# This tool intentionally allowlists certain modules that cannot be resolved
# statically from the source tree. These are NOT import failures; they are
# excluded by design because they fall into one of these categories:
#
#   1. C Extensions (torch._C.*) - Compiled at build time, not Python source
#   2. Build-Generated Modules (torch.version) - Created during installation
#   3. FB-Internal Modules (torch._inductor.fb.*) - Not in open source repo
#   4. Optional Third-Party Packages - Conditionally imported, may not be installed
#
# The allowlist is EXPLICIT rather than pattern-based to maintain auditability.
# Each entry should be justified and documented. When adding new entries,
# prefer specificity over wildcards.
#
# If this tool reports a false positive, the correct fix is usually to add
# the module to the appropriate allowlist with a comment explaining why.
# =============================================================================

COMPILED_MODULE_ALLOWLIST = {
    # C extensions (compiled at build time)
    "torch._C",
    "torch._C._nn",
    "torch._C._distributed_c10d",
    "torch._C._autograd",
    "torch._C._jit",
    "torch._C._onnx",
    
    # Build-time generated modules
    "torch.version",
    "torch.utils._config_typing",
    
    # Facebook-internal modules (not in open source)
    "torch._inductor.fb",
    "torch._inductor.runtime.caching.fb",
    
    # Optional metrics init (generated/optional)
    "torch.distributed.elastic.metrics.static_init",
}

# Optional third-party packages that PyTorch conditionally imports
# These are NOT part of PyTorch itself and may not be installed
OPTIONAL_THIRD_PARTY_PACKAGES = {
    "torch_xla",
    "torcharrow",
    "torchaudio",
    "torchdistx",
    "torchrec",
    "torchvision",
}


class ImportGraphAnalyzer:
    """AST-based static import analyzer for Python packages"""

    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo_root = repo_root
        self.verbose = verbose
        self.scanned_files = 0
        self.total_imports = 0
        self.unresolved: Set[Tuple[str, str, str]] = set()  # (module, import, file)

    def analyze_targets(self, targets: List[str]) -> bool:
        """
        Analyze import graph for given target modules.
        
        Args:
            targets: List of module names (e.g., ['torch', 'torch.nn'])
        
        Returns:
            True if all imports resolved, False otherwise
        """
        print(f"[*] Analyzing import graph for {len(targets)} target(s)...")
        print(f"    Repository root: {self.repo_root}")
        print()

        for target in targets:
            self._analyze_module(target)

        return len(self.unresolved) == 0

    def _analyze_module(self, module_name: str) -> None:
        """Analyze a single module (package or file)"""
        module_path = self._resolve_module_path(module_name)

        if module_path is None:
            print(f"[!] Module not found: {module_name}", file=sys.stderr)
            return

        if module_path.is_dir():
            # Package - scan all .py files recursively
            self._scan_package(module_name, module_path)
        else:
            # Single file module
            self._scan_file(module_name, module_path)

    def _resolve_module_path(self, module_name: str) -> Path | None:
        """Convert module name to filesystem path"""
        # Convert torch.nn -> torch/nn
        parts = module_name.split(".")
        
        # Try as package (directory with __init__.py)
        dir_path = self.repo_root.joinpath(*parts)
        if dir_path.is_dir() and (dir_path / "__init__.py").exists():
            return dir_path

        # Try as module file (torch/nn.py)
        if len(parts) > 1:
            file_path = self.repo_root.joinpath(*parts[:-1]) / f"{parts[-1]}.py"
            if file_path.exists():
                return file_path

        # Try as top-level module (torch.py - unlikely but possible)
        file_path = self.repo_root / f"{module_name}.py"
        if file_path.exists():
            return file_path

        return None

    def _scan_package(self, package_name: str, package_path: Path) -> None:
        """Recursively scan all .py files in a package"""
        if self.verbose:
            print(f"[+] Scanning package: {package_name}")

        for py_file in sorted(package_path.rglob("*.py")):
            # Skip test files and vendored code
            rel_path = py_file.relative_to(self.repo_root)
            if any(part.startswith("test") for part in rel_path.parts):
                continue
            if "third_party" in rel_path.parts:
                continue

            self._scan_file(package_name, py_file)

    def _scan_file(self, context_module: str, file_path: Path) -> None:
        """Scan a single Python file for imports"""
        try:
            source = file_path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(file_path))
            self.scanned_files += 1

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self._check_import(node, file_path)
                elif isinstance(node, ast.ImportFrom):
                    self._check_import_from(node, file_path)

        except SyntaxError as e:
            if self.verbose:
                print(f"[!] Syntax error in {file_path}: {e}", file=sys.stderr)
        except Exception as e:
            if self.verbose:
                print(f"[!] Error reading {file_path}: {e}", file=sys.stderr)

    def _check_import(self, node: ast.Import, source_file: Path) -> None:
        """Check 'import foo' statements"""
        for alias in node.names:
            self.total_imports += 1
            module_name = alias.name

            # Only check torch.* imports (internal)
            if not module_name.startswith("torch"):
                continue

            if not self._is_module_resolvable(module_name):
                self.unresolved.add((
                    module_name,
                    f"import {module_name}",
                    str(source_file.relative_to(self.repo_root))
                ))

    def _check_import_from(self, node: ast.ImportFrom, source_file: Path) -> None:
        """Check 'from foo import bar' statements"""
        if node.module is None:
            # Relative import without module (from . import foo)
            return

        self.total_imports += 1
        module_name = node.module

        # Handle relative imports (from . import X, from .. import Y)
        if node.level > 0:
            # Resolve relative to source file's package
            source_package = self._get_package_name(source_file)
            if source_package:
                # Go up 'level - 1' parents (level 1 means current package)
                # from .foo means: current_package.foo (go up 0 levels)
                # from ..foo means: parent_package.foo (go up 1 level)
                parts = source_package.split(".")
                levels_up = node.level - 1
                if levels_up < len(parts):
                    base = ".".join(parts[:len(parts) - levels_up]) if levels_up > 0 else source_package
                    module_name = f"{base}.{node.module}" if node.module else base
                else:
                    # Relative import goes above package root - skip
                    return

        # Only check torch.* imports (internal)
        if not module_name.startswith("torch"):
            return

        if not self._is_module_resolvable(module_name):
            self.unresolved.add((
                module_name,
                f"from {node.module} import ...",
                str(source_file.relative_to(self.repo_root))
            ))

    def _get_package_name(self, file_path: Path) -> str | None:
        """Convert file path to package name (e.g., torch/nn/modules/linear.py -> torch.nn.modules)"""
        try:
            rel_path = file_path.relative_to(self.repo_root)
            parts = list(rel_path.parts[:-1])  # Drop filename
            
            # Remove __init__.py special case
            if file_path.name == "__init__.py" and len(parts) > 0:
                return ".".join(parts)
            
            return ".".join(parts) if parts else None
        except ValueError:
            return None

    def _is_module_resolvable(self, module_name: str) -> bool:
        """
        Check if a module can be resolved statically.
        
        Returns True if:
          - Module exists on disk (package or file)
          - Module is in the compiled allowlist (torch._C.*, torch.version, etc.)
          - Module is an optional third-party package (torchvision, etc.)
        """
        # Check allowlist first (exact match)
        if module_name in COMPILED_MODULE_ALLOWLIST:
            return True

        # Check if it's a submodule of an allowlisted module
        for allowed in COMPILED_MODULE_ALLOWLIST:
            if module_name.startswith(allowed + "."):
                return True

        # Check if it's an optional third-party package
        for pkg in OPTIONAL_THIRD_PARTY_PACKAGES:
            if module_name == pkg or module_name.startswith(pkg + "."):
                return True

        # Check if module exists on disk
        return self._resolve_module_path(module_name) is not None

    def print_summary(self) -> None:
        """Print analysis summary"""
        print()
        print("=" * 70)
        print("Import Graph Analysis Summary")
        print("=" * 70)
        print(f"Files scanned:       {self.scanned_files}")
        print(f"Total imports found: {self.total_imports}")
        print(f"Unresolved imports:  {len(self.unresolved)}")
        print()

        if self.unresolved:
            print("[X] UNRESOLVED IMPORTS:")
            print()
            # Group by module for readability
            by_module: Dict[str, List[Tuple[str, str]]] = {}
            for module, import_stmt, file_path in sorted(self.unresolved):
                if module not in by_module:
                    by_module[module] = []
                by_module[module].append((import_stmt, file_path))

            for module in sorted(by_module.keys()):
                print(f"  {module}:")
                for import_stmt, file_path in sorted(by_module[module]):
                    print(f"    - {file_path}")
                print()
        else:
            print("[OK] All imports resolved successfully")

        print("=" * 70)


def main() -> int:
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Static import graph checker for PyTorch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--targets",
        required=True,
        help="Comma-separated list of modules to check (e.g., torch,torch.nn)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root path (default: auto-detect from script location)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Auto-detect repo root if not provided
    if args.repo_root is None:
        # Assume script is in tools/refactor/, so repo root is 2 levels up
        script_path = Path(__file__).resolve()
        repo_root = script_path.parent.parent.parent
    else:
        repo_root = args.repo_root.resolve()

    if not repo_root.exists():
        print(f"[X] Repository root not found: {repo_root}", file=sys.stderr)
        return 2

    targets = [t.strip() for t in args.targets.split(",")]

    print("PyTorch Static Import Smoke Test")
    print(f"Targets: {', '.join(targets)}")
    print()

    start_time = time.time()
    analyzer = ImportGraphAnalyzer(repo_root, verbose=args.verbose)
    
    try:
        success = analyzer.analyze_targets(targets)
        elapsed = time.time() - start_time
        
        analyzer.print_summary()
        print(f"Analysis completed in {elapsed:.2f}s")
        print()

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n[!] Analysis interrupted by user", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\n[X] Fatal error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())

