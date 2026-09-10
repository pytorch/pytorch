import os
import tempfile
import unittest
from pathlib import Path

from cli.lib.common.utils import temp_environ, working_directory  # <-- replace import


class EnvIsolatedTestCase(unittest.TestCase):
    """Base class that snapshots os.environ and CWD for isolation."""

    def setUp(self):
        import os
        import tempfile

        self._env_backup = dict(os.environ)

        # Snapshot/repair CWD if it's gone
        try:
            self._cwd_backup = os.getcwd()
        except FileNotFoundError:
            # If CWD no longer exists, switch to a safe place and record that
            self._cwd_backup = tempfile.gettempdir()
            os.chdir(self._cwd_backup)

        # Create a temporary directory for the test to run in
        self._temp_dir = tempfile.mkdtemp()
        os.chdir(self._temp_dir)

    def tearDown(self):
        import os
        import shutil
        import tempfile

        # Restore cwd first (before cleaning up temp dir)
        try:
            os.chdir(self._cwd_backup)
        except OSError:
            os.chdir(tempfile.gettempdir())

        # Clean up temporary directory
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors

        # Restore env
        to_del = set(os.environ.keys()) - set(self._env_backup.keys())
        for k in to_del:
            os.environ.pop(k, None)
        for k, v in self._env_backup.items():
            os.environ[k] = v


class TestTempEnviron(EnvIsolatedTestCase):
    def test_sets_and_restores_new_var(self):
        var = "TEST_TMP_ENV_NEW"
        self.assertNotIn(var, os.environ)

        with temp_environ({var: "123"}):
            self.assertEqual(os.environ[var], "123")

        self.assertNotIn(var, os.environ)  # removed after exit

    def test_overwrites_and_restores_existing_var(self):
        var = "TEST_TMP_ENV_OVERWRITE"
        os.environ[var] = "orig"

        with temp_environ({var: "override"}):
            self.assertEqual(os.environ[var], "override")

        self.assertEqual(os.environ[var], "orig")  # restored

    def test_multiple_vars_and_missing_cleanup(self):
        v1, v2 = "TEST_ENV_V1", "TEST_ENV_V2"
        os.environ.pop(v1, None)
        os.environ[v2] = "keep"

        with temp_environ({v1: "a", v2: "b"}):
            self.assertEqual(os.environ[v1], "a")
            self.assertEqual(os.environ[v2], "b")

        self.assertNotIn(v1, os.environ)  # newly-added -> removed
        self.assertEqual(os.environ[v2], "keep")  # pre-existing -> restored

    def test_restores_even_on_exception(self):
        var = "TEST_TMP_ENV_EXCEPTION"
        self.assertNotIn(var, os.environ)

        with self.assertRaises(RuntimeError):
            with temp_environ({var: "x"}):
                self.assertEqual(os.environ[var], "x")
                raise RuntimeError("boom")

        self.assertNotIn(var, os.environ)  # removed after exception


class TestWorkingDirectory(EnvIsolatedTestCase):
    def test_changes_and_restores(self):
        start = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "wd"
            target.mkdir()

            with working_directory(str(target)):
                self.assertEqual(Path.cwd().resolve(), target.resolve())

        self.assertEqual(Path.cwd(), start)

    def test_noop_when_empty_path(self):
        start = Path.cwd()
        with working_directory(""):
            self.assertEqual(Path.cwd(), start)
        self.assertEqual(Path.cwd(), start)

    def test_restores_on_exception(self):
        start = Path.cwd()

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "wd_exc"
            target.mkdir()

            with self.assertRaises(ValueError):
                with working_directory(str(target)):
                    # Normalize both sides to handle /var -> /private/var
                    self.assertEqual(Path.cwd().resolve(), target.resolve())
                    raise ValueError("boom")

        self.assertEqual(Path.cwd().resolve(), start.resolve())

    def test_raises_for_missing_dir(self):
        start = Path.cwd()
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "does_not_exist"
            with self.assertRaises(FileNotFoundError):
                # os.chdir should raise before yielding
                with working_directory(str(missing)):
                    pass
        self.assertEqual(Path.cwd(), start)


if __name__ == "__main__":
    unittest.main(verbosity=2)
