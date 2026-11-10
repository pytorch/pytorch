"""The torch.variant build token, resolved with scikit-build-core's own reader.

The rule under test: PYTORCH_VARIANT (or -C torch.variant=...) keys the build
tree and the editable install tree, env wins over -C, unset keeps the plain
build/ layout, and any value outside the declared vocabulary fails the build.
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path


try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

try:
    # pyrefly: ignore [missing-import]
    from scikit_build_core import __version__ as SKBC_VERSION

    # pyrefly: ignore [missing-import]
    from scikit_build_core.settings.skbuild_read_settings import SettingsReader
except ImportError:  # the class below is skipped, so the name stays unused
    SKBC_VERSION = "0"


REPO = Path(__file__).resolve().parents[2]
PYPROJECT = REPO / "pyproject.toml"
SETTING = "torch.variant"
ENV_NAME = "PYTORCH_VARIANT"
SCRUBBED = (ENV_NAME, "SKBUILD_BUILD_DIR", "SKBUILD_EDITABLE_REBUILD_DIR")


def _version_tuple(version: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", version)[:3])


def _skbc_at_least(version: str) -> bool:
    return _version_tuple(SKBC_VERSION) >= _version_tuple(version)


def _overrides() -> list[dict]:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)["tool"]["scikit-build"]["overrides"]


def declared_variants() -> list[str]:
    """The values with an override block of their own, in file order."""
    values = []
    for block in _overrides():
        cond = block.get("if", {}).get("config-setting", {}).get(SETTING)
        if cond is not None and not cond.startswith("^"):
            values.append(cond)
    return values


def rejection_regex() -> str:
    (regex,) = [
        block["if"]["config-setting"][SETTING]
        for block in _overrides()
        if block.get("fail")
        and block.get("if", {})
        .get("config-setting", {})
        .get(SETTING, "")
        .startswith("^")
    ]
    return regex


@unittest.skipUnless(
    _skbc_at_least("1.1.1"), f"scikit-build-core >= 1.1.1 needed, have {SKBC_VERSION}"
)
class TestVariantToken(unittest.TestCase):
    def resolve(
        self,
        config_settings: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ):
        base = {k: v for k, v in os.environ.items() if k not in SCRUBBED}
        base.update(env or {})
        return SettingsReader.from_file(PYPROJECT, config_settings, env=base).settings

    def test_unset_keeps_plain_build_dir(self) -> None:
        s = self.resolve()
        self.assertEqual(s.build_dir, "build")
        self.assertEqual(s.editable.rebuild_dir, "build/install/{wheel_tag}")
        self.assertFalse(s.fail)

    def test_environment_keys_both_trees(self) -> None:
        for v in declared_variants():
            with self.subTest(variant=v):
                s = self.resolve(env={ENV_NAME: v})
                self.assertEqual(s.build_dir, f"build/{v}")
                self.assertEqual(
                    s.editable.rebuild_dir, f"build/{v}/install/{{wheel_tag}}"
                )
                self.assertFalse(s.fail)

    def test_flag_keys_both_trees(self) -> None:
        for v in declared_variants():
            with self.subTest(variant=v):
                s = self.resolve({SETTING: v})
                self.assertEqual(s.build_dir, f"build/{v}")
                self.assertEqual(
                    s.editable.rebuild_dir, f"build/{v}/install/{{wheel_tag}}"
                )

    def test_environment_wins_over_flag(self) -> None:
        first, second = declared_variants()[:2]
        s = self.resolve({SETTING: first}, env={ENV_NAME: second})
        self.assertEqual(s.build_dir, f"build/{second}")

    def test_unknown_value_fails(self) -> None:
        for bad in ("cu999", "CU132", "cuda", "cu132 "):
            with self.subTest(value=bad):
                self.assertTrue(self.resolve(env={ENV_NAME: bad}).fail)

    def test_empty_value_means_unset(self) -> None:
        s = self.resolve(env={ENV_NAME: ""})
        self.assertEqual(s.build_dir, "build")
        self.assertFalse(s.fail)

    def test_build_dir_env_still_overrides(self) -> None:
        s = self.resolve(env={ENV_NAME: "cu132", "SKBUILD_BUILD_DIR": "elsewhere"})
        self.assertEqual(s.build_dir, "elsewhere")
        self.assertEqual(s.editable.rebuild_dir, "build/cu132/install/{wheel_tag}")


class TestVariantVocabulary(unittest.TestCase):
    """The vocabulary is spelled three times in pyproject.toml; keep them in step."""

    def test_rejection_regex_matches_exactly_the_undeclared(self) -> None:
        variants = declared_variants()
        self.assertGreater(len(variants), 0)
        regex = re.compile(rejection_regex())
        for v in variants:
            self.assertIsNone(regex.match(v), v)
        for bad in ("cu999", "CU132", "cpu ", "x"):
            self.assertIsNotNone(regex.match(bad), bad)
        self.assertIsNone(regex.match(""))

    def test_help_lists_every_variant(self) -> None:
        with open(PYPROJECT, "rb") as f:
            declared = tomllib.load(f)["tool"]["scikit-build"]["config-setting"][
                SETTING
            ]
        self.assertEqual(declared["env"], ENV_NAME)
        for v in declared_variants():
            self.assertIn(v, declared["help"])

    def test_variants_are_unique(self) -> None:
        variants = declared_variants()
        self.assertEqual(len(variants), len(set(variants)))


if __name__ == "__main__":
    unittest.main()
