# Owner(s): ["module: dataloader"]

"""
Tests for datapipes deprecation warning behavior.

This test verifies that the deprecation warning for datapipes symbols in torch.utils.data:
1. Does NOT show for `import torch`
2. Shows for `from torch.utils.data import IterDataPipe, MapDataPipe`
3. Does NOT show when datapipes is called from other torch code internally
"""

import subprocess
import sys
import textwrap

from torch.testing._internal.common_utils import run_tests, TestCase


class TestDatapipesDeprecationWarning(TestCase):
    """Test the datapipes deprecation warning behavior."""

    def _run_python_code(self, code: str) -> tuple[str, str]:
        result = subprocess.run(
            [sys.executable, "-W", "always", "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
        )
        return result.stdout, result.stderr

    def test_no_warning_on_torch_or_internal_calls(self):
        """Verify that torch top-level imports do not trigger the deprecation warning."""
        code = """
            import torch
            import torch.utils.data
            loader = torch.utils.data.DataLoader([1, 2, 3])
            print("done")
        """
        stdout, stderr = self._run_python_code(code)
        self.assertIn("done", stdout)
        self.assertNotIn("datapipes", stderr.lower())
        self.assertNotIn("deprecated", stderr.lower())

    def test_no_warning_on_dataloader_with_workers(self):
        """Verify that DataLoader with num_workers does not trigger the deprecation warning."""
        code = """
            import torch
            from torch.utils.data import DataLoader
            dl = DataLoader([1, 2, 3, 4, 10, 100, 10000], num_workers=1)
            # Iterate to trigger worker spawning
            for batch in dl:
                pass
            print("done")
        """
        stdout, stderr = self._run_python_code(code)
        self.assertIn("done", stdout)
        self.assertNotIn("datapipes", stderr.lower())
        self.assertNotIn("deprecated", stderr.lower())

    def test_warning_on_datapipes_imports(self):
        """Verify warning triggers when importing datapipes symbols from torch.utils.data."""
        import_statements = [
            "from torch.utils.data import IterDataPipe",
            "from torch.utils.data import DataChunk",
            "from torch.utils.data import MapDataPipe",
        ]
        for import_stmt in import_statements:
            with self.subTest(import_stmt=import_stmt):
                code = f"""
                    {import_stmt}
                    print("done")
                """
                stdout, stderr = self._run_python_code(code)
                self.assertIn("done", stdout)
                self.assertIn("datapipes", stderr.lower())
                self.assertIn("deprecated", stderr.lower())

    def test_warning_only_once(self):
        """Verify the warning only shows once even with multiple imports."""
        code = """
            from torch.utils.data import IterDataPipe
            from torch.utils.data import MapDataPipe
            print("done")
        """
        stdout, stderr = self._run_python_code(code)
        self.assertIn("done", stdout)
        # Count occurrences of the warning
        warning_count = stderr.lower().count("deprecated")
        self.assertEqual(
            warning_count, 1, f"Expected exactly 1 warning, got {warning_count}"
        )


if __name__ == "__main__":
    run_tests()
