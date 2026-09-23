# Owner(s): ["module: ci"]

from types import SimpleNamespace

from _pytest.reports import TestReport
from conftest import _NodeReporterReruns, _pytest_testcase_identity
from torch.testing._internal.common_utils import run_tests, TestCase


class TestPytestTestcaseIdentity(TestCase):
    def test_identity(self):
        cases = [
            (
                "test/distributed/fsdp/test_fsdp_meta.py::"
                "TestFSDPWithMetaDevice::test_bad_arg_meta",
                "test/distributed/fsdp/test_fsdp_meta.py",
                ("TestFSDPWithMetaDevice", "distributed/fsdp/test_fsdp_meta.py"),
            ),
            (
                "test/test_example.py::Outer::Inner::test_nested[a::b]",
                "test/test_example.py",
                ("Outer.Inner", "test_example.py"),
            ),
            (
                "test/test_example.py::test_function",
                "test/test_example.py",
                ("test_example", "test_example.py"),
            ),
            (
                "test/test_example.py::TestExample::test_windows",
                "test\\distributed\\test_example.py",
                ("TestExample", "distributed/test_example.py"),
            ),
        ]
        for nodeid, filename, expected in cases:
            with self.subTest(nodeid=nodeid):
                report = SimpleNamespace(
                    nodeid=nodeid,
                    location=(filename, 0, ""),
                )
                self.assertEqual(_pytest_testcase_identity(report), expected)

    def test_reporter_normalizes_before_serialization(self):
        report = TestReport(
            nodeid="test/test_example.py::TestFailure::test_fails",
            location=("test/test_example.py", 2, "test_fails"),
            keywords={},
            outcome="failed",
            longrepr="assert False",
            when="call",
            sections=[],
            duration=1.0,
            user_properties=[],
        )
        reporter = _NodeReporterReruns(
            report.nodeid,
            SimpleNamespace(
                prefix="pytorch",
                family="xunit2",
                add_stats=lambda _: None,
            ),
        )

        reporter.record_testreport(report)
        testcase = reporter.to_xml()

        self.assertEqual(testcase.attrib["classname"], "pytorch.TestFailure")
        self.assertEqual(testcase.attrib["file"], "test_example.py")


if __name__ == "__main__":
    run_tests()
