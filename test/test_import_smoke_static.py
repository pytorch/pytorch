#!/usr/bin/env python3
"""
Test suite for static import smoke checker

This test validates that the import_smoke_static tool correctly identifies
resolvable and unresolvable imports in the PyTorch codebase.

Test Strategy:
  - Test against KNOWN GOOD targets (torch, torch.nn, etc.)
  - Verify tool exits 0 on current codebase
  - Establish baseline for INV-050 regression detection

Note: This is a SMOKE TEST, not a comprehensive import test. It validates
      the tool works correctly, not that every import is perfect.
"""

import subprocess
import sys
import unittest
from pathlib import Path


class TestImportSmokeStatic(unittest.TestCase):
    """Test static import graph checker"""

    @classmethod
    def setUpClass(cls):
        """Find repository root"""
        # Test is in test/, repo root is parent
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.tool_path = cls.repo_root / "tools" / "refactor" / "import_smoke_static.py"

        if not cls.tool_path.exists():
            raise RuntimeError(f"Tool not found: {cls.tool_path}")

    def _run_tool(self, targets: str, expect_success: bool = True) -> subprocess.CompletedProcess:
        """
        Run the import smoke tool with given targets.
        
        Args:
            targets: Comma-separated module list
            expect_success: If True, assert exit code 0; if False, allow non-zero
        
        Returns:
            CompletedProcess result
        """
        cmd = [
            sys.executable,
            "-m",
            "tools.refactor.import_smoke_static",
            "--targets",
            targets,
            "--repo-root",
            str(self.repo_root),
        ]

        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if expect_success:
            if result.returncode != 0:
                print("\n=== STDOUT ===", file=sys.stderr)
                print(result.stdout, file=sys.stderr)
                print("\n=== STDERR ===", file=sys.stderr)
                print(result.stderr, file=sys.stderr)
                self.fail(f"Tool failed with exit code {result.returncode}")

        return result

    def test_single_target_torch(self):
        """Test analyzing single target: torch"""
        result = self._run_tool("torch")
        
        # Check output contains expected summary
        self.assertIn("Import Graph Analysis Summary", result.stdout)
        self.assertIn("Files scanned:", result.stdout)
        self.assertIn("Total imports found:", result.stdout)

    def test_multiple_targets_core(self):
        """Test analyzing multiple core targets"""
        targets = "torch,torch.nn,torch.optim,torch.utils,torch.autograd"
        result = self._run_tool(targets)

        # Should succeed with core modules
        self.assertEqual(result.returncode, 0)
        self.assertIn("All imports resolved successfully", result.stdout)

    def test_torch_nn_only(self):
        """Test analyzing torch.nn subpackage"""
        result = self._run_tool("torch.nn")
        
        # torch.nn is well-structured, should pass
        self.assertEqual(result.returncode, 0)

    def test_torch_optim_only(self):
        """Test analyzing torch.optim subpackage"""
        result = self._run_tool("torch.optim")
        
        # torch.optim is well-structured, should pass
        self.assertEqual(result.returncode, 0)

    def test_output_format_deterministic(self):
        """Test that output is deterministic (sorted)"""
        result1 = self._run_tool("torch.nn")
        result2 = self._run_tool("torch.nn")

        # Running twice should produce identical output
        self.assertEqual(result1.stdout, result2.stdout)

    def test_invalid_target_handling(self):
        """Test tool handles non-existent modules gracefully"""
        # Tool should not crash on invalid target, just report it
        result = self._run_tool("torch.nonexistent_module_xyz", expect_success=False)
        
        # Should exit cleanly (either 0 or 1, not crash)
        self.assertIn(result.returncode, [0, 1])

    def test_compiled_module_allowlist(self):
        """Test that torch._C is correctly allowlisted"""
        # torch itself imports torch._C - should not fail
        result = self._run_tool("torch")
        
        # Should succeed because torch._C is allowlisted
        self.assertEqual(result.returncode, 0)
        
        # Verify torch._C is NOT reported as unresolved
        self.assertNotIn("torch._C:", result.stdout)


class TestImportSmokeBaseline(unittest.TestCase):
    """
    Baseline regression test - locks down M01 success criteria.
    
    This test establishes the baseline import graph state at M01 completion.
    Future changes that break this test indicate INV-050 violations.
    """

    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent

    def test_m01_baseline_targets(self):
        """
        M01 BASELINE: Core 5 targets must pass static import check.
        
        Protected Invariant: INV-050 (Import Path Stability)
        
        Locked Targets (M01):
          - torch
          - torch.nn
          - torch.optim
          - torch.utils
          - torch.autograd
        
        If this test fails in future commits, it indicates an import graph
        regression that must be fixed before merge.
        """
        targets = "torch,torch.nn,torch.optim,torch.utils,torch.autograd"
        
        cmd = [
            sys.executable,
            "-m",
            "tools.refactor.import_smoke_static",
            "--targets",
            targets,
            "--repo-root",
            str(self.repo_root),
        ]

        result = subprocess.run(
            cmd,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # CRITICAL: This must pass on M01 baseline
        if result.returncode != 0:
            print("\n[X] M01 BASELINE REGRESSION DETECTED", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            print("\nStatic import check failed for core targets.", file=sys.stderr)
            print("This violates INV-050 (Import Path Stability).\n", file=sys.stderr)
            print("Tool output:", file=sys.stderr)
            print(result.stdout, file=sys.stderr)
            print("\nErrors:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            print("=" * 70, file=sys.stderr)

        self.assertEqual(
            result.returncode,
            0,
            "M01 baseline import check must pass (INV-050 protected)"
        )


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)

