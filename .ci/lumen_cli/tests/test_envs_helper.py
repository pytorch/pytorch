import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import cli.lib.common.envs_helper as m


class TestEnvHelpers(unittest.TestCase):
    def setUp(self):
        # Keep a copy of the original environment to restore later
        self._env_backup = dict(os.environ)

    def tearDown(self):
        # Restore environment to original state
        os.environ.clear()
        os.environ.update(self._env_backup)

    # -------- get_env --------
    def test_get_env_unset_returns_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(m.get_env("FOO", "default"), "default")

    def test_get_env_empty_returns_default(self):
        with patch.dict(os.environ, {"FOO": ""}, clear=True):
            self.assertEqual(m.get_env("FOO", "default"), "default")

    def test_get_env_set_returns_value(self):
        with patch.dict(os.environ, {"FOO": "bar"}, clear=True):
            self.assertEqual(m.get_env("FOO", "default"), "bar")

    def test_get_env_not_exist_returns_default(self):
        with patch.dict(os.environ, {"FOO": "bar"}, clear=True):
            self.assertEqual(m.get_env("TEST_NOT_EXIST", "default"), "default")

    def test_get_env_not_exist_without_default(self):
        with patch.dict(os.environ, {"FOO": "bar"}, clear=True):
            self.assertEqual(m.get_env("TEST_NOT_EXIST"), "")

    # -------- env_bool --------
    def test_env_bool_uses_default_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(m.env_bool("FLAG", default=True))
            self.assertFalse(m.env_bool("FLAG", default=False))

    def test_env_bool_uses_str2bool_when_set(self):
        # Patch str2bool used by env_bool so we don't depend on its exact behavior
        def fake_str2bool(s: str) -> bool:
            return s.lower() in {"1", "true", "yes", "on", "y"}

        with (
            patch.dict(os.environ, {"FLAG": "yEs"}, clear=True),
            patch.object(m, "str2bool", fake_str2bool),
        ):
            self.assertTrue(m.env_bool("FLAG", default=False))

    # -------- env_path_optional / env_path --------
    def test_env_path_optional_unset_returns_none_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(m.env_path_optional("P"))

    def test_env_path_optional_unset_returns_none_when_env_var_is_empty(self):
        with patch.dict(os.environ, {"P": ""}, clear=True):
            self.assertIsNone(m.env_path_optional("P"))

    def test_env_path_optional_unset_returns_default_str(self):
        # default as string; resolve=True by default -> absolute path
        default_str = "x/y"
        with patch.dict(os.environ, {}, clear=True):
            p = m.env_path_optional("P", default=default_str)
            self.assertIsInstance(p, Path)
            self.assertIsNotNone(p)
            if p:
                self.assertTrue(p.is_absolute())
                self.assertEqual(p.parts[-2:], ("x", "y"))

    def test_env_path_optional_unset_returns_default_path_no_resolve(self):
        d = Path("z")
        with patch.dict(os.environ, {}, clear=True):
            p = m.env_path_optional("P", default=d, resolve=False)
            self.assertEqual(p, d)

    def test_env_path_optional_respects_resolve_true(self):
        with patch.dict(os.environ, {"P": "a/b"}, clear=True):
            p = m.env_path_optional("P", resolve=True)
            self.assertIsInstance(p, Path)
            if p:
                self.assertTrue(p.is_absolute())

    def test_env_path_optional_respects_resolve_false(self):
        with patch.dict(os.environ, {"P": "rel/dir"}, clear=True):
            p = m.env_path_optional("P", resolve=False)
            self.assertEqual(p, Path("rel/dir"))
            if p:
                self.assertFalse(p.is_absolute())

    def test_env_path_raises_when_missing_and_default_none(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                m.env_path("P", None, resolve=True)

    def test_env_path_returns_path_when_present(self):
        tmp = Path("./b").resolve()
        with patch.dict(os.environ, {"P": str(tmp)}, clear=True):
            p = m.env_path("P", None, resolve=True)
            self.assertEqual(p, tmp)

    # -------- dataclass field helpers --------
    def test_dataclass_fields_read_env_at_instantiation(self):
        @dataclass
        class Cfg:
            flag: bool = m.env_bool_field("FLAG", default=False)
            out: Path = m.env_path_field("OUT", default="ab", resolve=True)
            name: str = m.env_str_field("NAME", default="anon")

        # First instantiation
        with patch.dict(
            os.environ, {"FLAG": "true", "OUT": "outdir", "NAME": "alice"}, clear=True
        ):
            cfg1 = Cfg()
            self.assertTrue(cfg1.flag)
            self.assertIsInstance(cfg1.out, Path)
            self.assertTrue(cfg1.out.is_absolute())
            self.assertEqual(cfg1.name, "alice")
            cfg1.name = "bob"  # change instance value
            self.assertEqual(cfg1.name, "bob")  # change is reflected

        # Change env; new instance should reflect new values
        with patch.dict(os.environ, {"FLAG": "false", "NAME": ""}, clear=True):
            cfg2 = Cfg()
            self.assertFalse(cfg2.flag)  # str2bool("false") -> False
            self.assertTrue("ab" in str(cfg2.out))
            self.assertIsInstance(cfg2.out, Path)
            self.assertTrue(cfg2.out.is_absolute())
            self.assertEqual(cfg2.name, "anon")  # empty -> fallback to default

    def test_dataclass_path_field_with_default_value(self):
        @dataclass
        class C2:
            out: Path = m.env_path_field("OUT", default="some/dir", resolve=False)

        with patch.dict(os.environ, {}, clear=True):
            c = C2()
            self.assertEqual(c.out, Path("some/dir"))


if __name__ == "__main__":
    unittest.main()
