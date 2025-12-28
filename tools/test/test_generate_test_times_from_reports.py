"""Tests for tools/stats/generate_test_times_from_reports.py.

Run with:
    pytest tools/stats/test_generate_test_times_from_reports.py -v

All tests are stdlib + pytest only; torch is not required.
"""

from __future__ import annotations

import json
import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tools.stats.generate_test_times_from_reports import (
    _build_payload,
    collect_times,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_xml(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content), encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit tests: _parse_xml / collect_times
# ---------------------------------------------------------------------------


class TestCollectTimes(unittest.TestCase):
    """Tests for XML scanning and time aggregation."""

    def test_basic_single_module(self) -> None:
        """A single XML file with two testcases produces correct module total."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_xml(
                root / "test_foo" / "TEST-test_foo.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="TestFoo" name="test_a" time="1.5"/>
                  <testcase classname="TestFoo" name="test_b" time="0.5"/>
                </testsuite>
                """,
            )
            mod_times, cls_times = collect_times(root)

        self.assertAlmostEqual(mod_times["test_foo"], 2.0)
        self.assertAlmostEqual(cls_times["TestFoo::test_foo"], 2.0)

    def test_multiple_modules_are_aggregated_independently(self) -> None:
        """Two modules in different subdirectories accumulate separately."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_xml(
                root / "test_alpha" / "TEST-test_alpha.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="Alpha" name="test_1" time="3.0"/>
                </testsuite>
                """,
            )
            _write_xml(
                root / "test_beta" / "TEST-test_beta.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="Beta" name="test_1" time="7.0"/>
                </testsuite>
                """,
            )
            mod_times, _ = collect_times(root)

        self.assertAlmostEqual(mod_times["test_alpha"], 3.0)
        self.assertAlmostEqual(mod_times["test_beta"], 7.0)
        self.assertEqual(len(mod_times), 2)

    def test_multiple_xml_files_in_same_module_directory_are_summed(self) -> None:
        """Multiple XML files under one directory are summed into one module entry."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, t in enumerate([1.0, 2.0, 3.0]):
                _write_xml(
                    root / "test_multi" / f"TEST-shard{i}.xml",
                    f"""\
                    <?xml version="1.0" ?>
                    <testsuite>
                      <testcase classname="Multi" name="test_{i}" time="{t}"/>
                    </testsuite>
                    """,
                )
            mod_times, _ = collect_times(root)

        self.assertAlmostEqual(mod_times["test_multi"], 6.0)

    def test_nested_subdirectory_structure(self) -> None:
        """XML files in a deeply nested path use the immediate parent as the module name."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_xml(
                root / "a" / "b" / "test_deep" / "TEST-test_deep.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="Deep" name="test_x" time="4.2"/>
                </testsuite>
                """,
            )
            mod_times, _ = collect_times(root)

        # The invoking module is always report.parent.name, i.e. the immediate parent dir.
        self.assertIn("test_deep", mod_times)
        self.assertAlmostEqual(mod_times["test_deep"], 4.2)

    def test_testcase_missing_time_attribute_is_skipped(self) -> None:
        """A testcase with no 'time' attribute does not raise and contributes 0."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_xml(
                root / "test_notimes" / "TEST.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="NoTime" name="test_a"/>
                  <testcase classname="NoTime" name="test_b" time="2.0"/>
                </testsuite>
                """,
            )
            mod_times, cls_times = collect_times(root)

        self.assertAlmostEqual(mod_times["test_notimes"], 2.0)
        self.assertAlmostEqual(cls_times["NoTime::test_notimes"], 2.0)

    def test_testcase_with_non_numeric_time_is_skipped(self) -> None:
        """A time attribute that cannot be parsed as float is silently ignored."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_xml(
                root / "test_badtime" / "TEST.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="C" name="test_a" time="N/A"/>
                  <testcase classname="C" name="test_b" time="1.0"/>
                </testsuite>
                """,
            )
            mod_times, _ = collect_times(root)

        self.assertAlmostEqual(mod_times["test_badtime"], 1.0)

    def test_malformed_xml_is_skipped_gracefully(self) -> None:
        """A corrupt XML file prints a warning and does not abort the entire run."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "test_corrupt" / "bad.xml"
            bad.parent.mkdir(parents=True)
            bad.write_text("<<<not xml>>>", encoding="utf-8")

            good = root / "test_good" / "good.xml"
            _write_xml(
                good,
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="Good" name="test_ok" time="1.0"/>
                </testsuite>
                """,
            )
            mod_times, _ = collect_times(root)

        # The corrupt file contributes nothing; the good file is processed normally.
        self.assertNotIn("test_corrupt", mod_times)
        self.assertIn("test_good", mod_times)

    def test_empty_reports_directory_returns_empty_dicts(self) -> None:
        """An empty reports directory produces empty dicts without raising."""
        with TemporaryDirectory() as tmp:
            mod_times, cls_times = collect_times(Path(tmp))

        self.assertEqual(mod_times, {})
        self.assertEqual(cls_times, {})

    def test_testcase_without_classname_is_excluded_from_class_times(self) -> None:
        """Testcases with an empty or absent classname do not appear in class_times."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_xml(
                root / "test_noclassname" / "TEST.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase name="test_a" time="5.0"/>
                  <testcase classname="" name="test_b" time="3.0"/>
                </testsuite>
                """,
            )
            mod_times, cls_times = collect_times(root)

        # Module total includes all timed cases.
        self.assertAlmostEqual(mod_times["test_noclassname"], 8.0)
        # No class entry because classname was absent / empty.
        self.assertEqual(cls_times, {})


# ---------------------------------------------------------------------------
# Unit tests: _build_payload
# ---------------------------------------------------------------------------


class TestBuildPayload(unittest.TestCase):
    """Tests for the three-level JSON payload construction."""

    def setUp(self) -> None:
        self.times = {"test_foo": 1.5, "test_bar": 2.5}
        self.job_name = "riscv64-linux / test"
        self.config = "default"

    def test_exact_job_and_config_key_is_present(self) -> None:
        payload = _build_payload(self.times, self.job_name, self.config)
        self.assertIn(self.job_name, payload)
        self.assertIn(self.config, payload[self.job_name])
        self.assertEqual(payload[self.job_name][self.config], self.times)

    def test_default_job_fallback_is_present(self) -> None:
        payload = _build_payload(self.times, self.job_name, self.config)
        self.assertIn("default", payload)
        self.assertIn(self.config, payload["default"])

    def test_default_default_fallback_is_present(self) -> None:
        payload = _build_payload(self.times, self.job_name, self.config)
        self.assertIn("default", payload["default"])
        self.assertEqual(payload["default"]["default"], self.times)

    def test_non_default_config_name(self) -> None:
        """A non-'default' config name is stored under both the job and 'default'."""
        payload = _build_payload(self.times, self.job_name, "slow")
        self.assertIn("slow", payload[self.job_name])
        self.assertIn("slow", payload["default"])


# ---------------------------------------------------------------------------
# Integration tests: main() end-to-end
# ---------------------------------------------------------------------------


class TestMainEndToEnd(unittest.TestCase):
    """End-to-end tests that exercise the CLI entry point."""

    def _run(self, reports_dir: Path, output_dir: Path, **kwargs: str) -> int:
        argv = [
            "--reports-dir", str(reports_dir),
            "--output-dir", str(output_dir),
            "--job-name", kwargs.get("job_name", "test-job"),
            "--test-config", kwargs.get("test_config", "default"),
        ]
        return main(argv)

    def test_output_files_are_created(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            output = root / "output"
            _write_xml(
                reports / "test_bar" / "TEST.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="Bar" name="test_1" time="2.0"/>
                </testsuite>
                """,
            )
            rc = self._run(reports, output)

        self.assertEqual(rc, 0)
        self.assertTrue((output / "test-times.json").exists())
        self.assertTrue((output / "test-class-times.json").exists())

    def test_output_json_structure_and_values(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            output = root / "output"
            _write_xml(
                reports / "test_baz" / "TEST.xml",
                """\
                <?xml version="1.0" ?>
                <testsuite>
                  <testcase classname="Baz" name="test_x" time="3.0"/>
                  <testcase classname="Baz" name="test_y" time="1.0"/>
                </testsuite>
                """,
            )
            self._run(reports, output, job_name="my-job", test_config="slow")

            with open(output / "test-times.json") as f:
                times = json.load(f)
            with open(output / "test-class-times.json") as f:
                class_times = json.load(f)

        # Module times
        self.assertAlmostEqual(times["my-job"]["slow"]["test_baz"], 4.0)
        self.assertAlmostEqual(times["default"]["default"]["test_baz"], 4.0)

        # Class times
        self.assertAlmostEqual(
            class_times["my-job"]["slow"]["Baz::test_baz"], 4.0
        )

    def test_nonexistent_reports_dir_returns_nonzero(self) -> None:
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "output"
            rc = main([
                "--reports-dir", "/this/does/not/exist",
                "--output-dir", str(output),
                "--job-name", "x",
            ])
        self.assertNotEqual(rc, 0)

    def test_output_dir_is_created_if_missing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            reports = root / "reports"
            reports.mkdir()
            output = root / "deeply" / "nested" / "output"
            self._run(reports, output)
        self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
