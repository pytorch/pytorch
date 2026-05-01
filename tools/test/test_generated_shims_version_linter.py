import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT))

from tools.linter.adapters.generated_shims_version_linter import (
    check_file,
    parse_fallback_ops,
)


class TestParseFallbackOps(unittest.TestCase):
    def test_basic_two_dicts(self):
        src = """
inductor_fallback_ops = {
    "aten.foo.default": {},
    "aten.bar.default": {"since": "TORCH_VERSION_2_12_0"},
}
aten_shimified_ops = {
    "aten.foo.default": {"since": "TORCH_VERSION_2_10_0"},
}
"""
        parsed = parse_fallback_ops(src)
        # Returns a map keyed by the module-level variable name.
        self.assertEqual(set(parsed), {"inductor_fallback_ops", "aten_shimified_ops"})
        # Same op key in both dicts should not interfere
        self.assertEqual(
            parsed["inductor_fallback_ops"]["aten.foo.default"][1][0], None
        )
        self.assertEqual(
            parsed["aten_shimified_ops"]["aten.foo.default"][1][0],
            "TORCH_VERSION_2_10_0",
        )
        self.assertEqual(
            parsed["inductor_fallback_ops"]["aten.bar.default"][1][0],
            "TORCH_VERSION_2_12_0",
        )

    def test_different_op_versions(self):
        src = """
inductor_fallback_ops = {
    "aten.baz.default": {
        "v2": {"new_args": ["x"], "since": "TORCH_VERSION_2_12_0"},
        "v3": {"new_args": ["y"], "since": "TORCH_VERSION_2_13_0"},
    },
}
aten_shimified_ops = {}
"""
        parsed = parse_fallback_ops(src)
        per_op = parsed["inductor_fallback_ops"]["aten.baz.default"]
        # v1 ungated (no top-level since)
        self.assertEqual(per_op[1][0], None)
        # v2 + v3 carry their own since
        self.assertEqual(per_op[2][0], "TORCH_VERSION_2_12_0")
        self.assertEqual(per_op[3][0], "TORCH_VERSION_2_13_0")


class TestGeneratedShimsVersionLinter(unittest.TestCase):
    """Test the four rules of the generated shims version linter."""

    EMPTY_BASE = """
inductor_fallback_ops = {}
aten_shimified_ops = {}
"""

    def _check(self, current_src, base_src, current_version=(2, 13, 0)):
        """Helper: write current_src to a temp file, run check_file with mocks."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(current_src)
            f.flush()
            temp_file = f.name

        with patch(
            "tools.linter.adapters.generated_shims_version_linter._read_at_merge_base",
            return_value=base_src,
        ):
            with patch(
                "tools.linter.adapters.generated_shims_version_linter.get_current_version",
                return_value=current_version,
            ):
                return check_file(temp_file)

    def test_missing_version_on_new_op(self):
        current = """
inductor_fallback_ops = {
    "aten.bad.default": {},
}
aten_shimified_ops = {}
"""
        msgs = self._check(current, self.EMPTY_BASE)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "missing-version-on-new-op")

    def test_missing_version_on_new_variant(self):
        # Op pre-existed (ungated v1 in base); a new v2 is added without `since`.
        base = """
inductor_fallback_ops = {
    "aten.foo.default": {},
}
aten_shimified_ops = {}
"""
        current = """
inductor_fallback_ops = {
    "aten.foo.default": {"v2": {"new_args": ["x"]}},
}
aten_shimified_ops = {}
"""
        msgs = self._check(current, base)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "missing-version-on-new-op")
        self.assertIn("v2", msgs[0].description)

    def test_wrong_version_on_new_op(self):
        current = """
inductor_fallback_ops = {
    "aten.bad.default": {"since": "TORCH_VERSION_2_10_0"},
}
aten_shimified_ops = {}
"""
        msgs = self._check(current, self.EMPTY_BASE, current_version=(2, 13, 0))
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "wrong-version-on-new-op")
        self.assertIn("TORCH_VERSION_2_13_0", msgs[0].description)

    def test_malformed_version(self):
        current = """
inductor_fallback_ops = {
    "aten.bad.default": {"since": "TORCH_2_13"},
}
aten_shimified_ops = {}
"""
        msgs = self._check(current, self.EMPTY_BASE)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "malformed-version")

    def test_op_version_modified(self):
        base = """
inductor_fallback_ops = {
    "aten.shipped.default": {"since": "TORCH_VERSION_2_12_0"},
}
aten_shimified_ops = {}
"""
        current = """
inductor_fallback_ops = {
    "aten.shipped.default": {"since": "TORCH_VERSION_2_14_0"},
}
aten_shimified_ops = {}
"""
        msgs = self._check(current, base)
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "op-version-modified")
        self.assertIn("TORCH_VERSION_2_12_0", msgs[0].description)
        self.assertIn("TORCH_VERSION_2_14_0", msgs[0].description)

    def test_new_op_with_correct_since_passes(self):
        current = """
inductor_fallback_ops = {
    "aten.new_good.default": {"since": "TORCH_VERSION_2_13_0"},
}
aten_shimified_ops = {}
"""
        msgs = self._check(current, self.EMPTY_BASE, current_version=(2, 13, 0))
        self.assertEqual(msgs, [])

    def test_unchanged_file_passes(self):
        src = """
inductor_fallback_ops = {
    "aten.old.default": {},
    "aten.recent.default": {"since": "TORCH_VERSION_2_12_0"},
}
aten_shimified_ops = {
    "aten.old.default": {"since": "TORCH_VERSION_2_10_0"},
}
"""
        msgs = self._check(src, src)
        self.assertEqual(msgs, [])

    def test_first_time_file_creation(self):
        # File didn't exist at merge-base — every entry is new.
        current = """
inductor_fallback_ops = {
    "aten.a.default": {"since": "TORCH_VERSION_2_13_0"},
    "aten.b.default": {},
}
aten_shimified_ops = {}
"""
        msgs = self._check(current, base_src=None, current_version=(2, 13, 0))
        # aten.a is fine; aten.b has no since
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "missing-version-on-new-op")
        self.assertIn("aten.b.default", msgs[0].description)

    def test_collision_keys_checked_independently(self):
        # Same op key in both dicts: bumping the since in one dict should be
        # flagged even though the other dict's entry stays unchanged.
        base = """
inductor_fallback_ops = {
    "aten.collide.default": {},
}
aten_shimified_ops = {
    "aten.collide.default": {"since": "TORCH_VERSION_2_10_0"},
}
"""
        current = """
inductor_fallback_ops = {
    "aten.collide.default": {},
}
aten_shimified_ops = {
    "aten.collide.default": {"since": "TORCH_VERSION_2_11_0"},
}
"""
        msgs = self._check(current, base)
        # Only the aten_shimified entry's bumped 'since' should fire.
        # The inductor entry is unchanged (still ungated), so no error there.
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0].name, "op-version-modified")
        self.assertIn("TORCH_VERSION_2_10_0", msgs[0].description)
        self.assertIn("TORCH_VERSION_2_11_0", msgs[0].description)


if __name__ == "__main__":
    unittest.main()
