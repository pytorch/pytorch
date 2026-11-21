import tempfile
import unittest
from pathlib import Path

from summarize_xml_testreports import (
    _canonical_classname,
    _test_config_from_dir,
    get_test_status,
    parse_xml_reports_as_dict,
)


SUITE = (
    '<testsuites><testsuite name="pytest" errors="0" failures="0" '
    'skipped="3" tests="14" time="{time}">{testcase}</testsuite></testsuites>'
)


class TestXmlReportMerging(unittest.TestCase):
    def _write(self, root, shard, parent, filename, contents):
        path = root / shard / parent / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    def test_four_gpu_directory_merges_into_distributed(self):
        self.assertEqual(
            _test_config_from_dir("test-distributed_4gpu-1-2_1001"),
            "distributed",
        )

    def test_testsuites_remain_distinct_across_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for shard, runtime in (
                ("test-default-1-2_1001", 10),
                ("test-default-2-2_1002", 20),
            ):
                self._write(
                    root,
                    shard,
                    "golden",
                    "pytest.xml",
                    SUITE.format(time=runtime, testcase=""),
                )

            suites = parse_xml_reports_as_dict(-1, -1, "testsuite", str(root))

            self.assertEqual(len(suites), 2)
            self.assertEqual(
                sum(case["running_time_xml"] for case in suites.values()),
                30,
            )
            self.assertEqual(
                {case["shard"] for case in suites.values()},
                {"1/2", "2/2"},
            )

    def test_testcase_retry_prefers_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = "test-default-1-1_1001"
            failed = '<testcase classname="TestOps" name="test_retry" time="1"><failure /></testcase>'
            passed = '<testcase classname="TestOps" name="test_retry" time="1" />'
            self._write(
                root,
                shard,
                "test_ops",
                "failed.xml",
                SUITE.format(time=1, testcase=failed),
            )
            self._write(
                root,
                shard,
                "test_ops",
                "passed.xml",
                SUITE.format(time=1, testcase=passed),
            )

            cases = parse_xml_reports_as_dict(-1, -1, "testcase", str(root))

            self.assertEqual(len(cases), 1)
            self.assertEqual(get_test_status(next(iter(cases.values()))), "PASSED")

    def test_testcase_retry_deduplicates_module_qualified_classname(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = "test-distributed-1-1_1001"
            failed = (
                '<testcase classname="test.distributed.test_nccl.'
                'NCCLSymmetricMemoryTest" name="test_nccl_symmem_put" time="300">'
                "<failure /></testcase>"
            )
            passed = (
                '<testcase classname="NCCLSymmetricMemoryTest" '
                'name="test_nccl_symmem_put" time="10" />'
            )
            for filename, testcase in (("failed.xml", failed), ("passed.xml", passed)):
                self._write(
                    root,
                    shard,
                    "distributed.test_nccl",
                    filename,
                    SUITE.format(time=1, testcase=testcase),
                )

            cases = parse_xml_reports_as_dict(-1, -1, "testcase", str(root))

            self.assertEqual(len(cases), 1)
            key, case = next(iter(cases.items()))
            self.assertEqual(key[1], "NCCLSymmetricMemoryTest")
            self.assertEqual(get_test_status(case), "PASSED")

    def test_unrelated_dotted_classname_is_preserved(self):
        self.assertEqual(
            _canonical_classname("distributed.test_nccl", "package.NestedTest"),
            "package.NestedTest",
        )

    def test_checked_in_junit_fixtures_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = "test-default-1-1_1001"
            fixture = (
                '<testcase classname="TestJunitOutcomes" '
                'name="test_phantom" time="1"><failure /></testcase>'
            )
            real = '<testcase classname="TestReal" name="test_real" time="1" />'
            self._write(
                root,
                shard,
                "junit_xml_testdata/expected",
                "pytest.xml",
                SUITE.format(time=10, testcase=fixture),
            )
            self._write(
                root,
                shard,
                "test/test-reports",
                "real.xml",
                SUITE.format(time=20, testcase=real),
            )

            cases = parse_xml_reports_as_dict(-1, -1, "testcase", str(root))
            suites = parse_xml_reports_as_dict(-1, -1, "testsuite", str(root))

            self.assertEqual(len(cases), 1)
            self.assertEqual(next(iter(cases.values()))["name"], "test_real")
            self.assertEqual(len(suites), 1)
            self.assertEqual(next(iter(suites.values()))["running_time_xml"], 20)

    def test_duplicate_testsuite_in_same_shard_is_not_double_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = "test-default-1-1_1001"
            for copy in ("copy1", "copy2"):
                self._write(
                    root,
                    shard,
                    f"{copy}/golden",
                    "pytest.xml",
                    SUITE.format(time=10, testcase=""),
                )

            suites = parse_xml_reports_as_dict(-1, -1, "testsuite", str(root))

            self.assertEqual(len(suites), 1)
            self.assertEqual(next(iter(suites.values()))["running_time_xml"], 10)

    def test_conflicting_duplicate_testsuite_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard = "test-default-1-1_1001"
            for copy, runtime in (("copy1", 10), ("copy2", 20)):
                self._write(
                    root,
                    shard,
                    f"{copy}/golden",
                    "pytest.xml",
                    SUITE.format(time=runtime, testcase=""),
                )

            with self.assertRaisesRegex(ValueError, "Conflicting duplicate testsuite"):
                parse_xml_reports_as_dict(-1, -1, "testsuite", str(root))

    def test_checked_in_junit_fixtures_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            failed = (
                '<testcase classname="TestJunitOutcomes" '
                'name="test_assert_failure" time="1"><failure /></testcase>'
            )
            self._write(
                root,
                "test-default-1-1_1001",
                "junit_xml_testdata/expected",
                "pytest.xml",
                SUITE.format(time=1, testcase=failed),
            )

            self.assertEqual(
                parse_xml_reports_as_dict(-1, -1, "testcase", str(root)),
                {},
            )
            self.assertEqual(
                parse_xml_reports_as_dict(-1, -1, "testsuite", str(root)),
                {},
            )

    def test_empty_primary_set_is_allowed_for_cuda_only_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                parse_xml_reports_as_dict(-1, -1, "testcase", tmp),
                {},
            )
            self.assertEqual(
                parse_xml_reports_as_dict(-1, -1, "testsuite", tmp),
                {},
            )


if __name__ == "__main__":
    unittest.main()
