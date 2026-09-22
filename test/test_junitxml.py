import unittest
from types import SimpleNamespace

from _pytest.reports import TestReport
from conftest import _NodeReporterReruns, _pytest_testcase_identity


class TestPytestTestcaseIdentity(unittest.TestCase):
    def test_class_test(self):
        report = SimpleNamespace(
            nodeid=(
                "test/distributed/fsdp/test_fsdp_meta.py::"
                "TestFSDPWithMetaDevice::test_bad_arg_meta"
            ),
            location=("test/distributed/fsdp/test_fsdp_meta.py", 0, ""),
        )

        self.assertEqual(
            _pytest_testcase_identity(report),
            (
                "TestFSDPWithMetaDevice",
                "distributed/fsdp/test_fsdp_meta.py",
            ),
        )

    def test_nested_class(self):
        report = SimpleNamespace(
            nodeid="test/test_example.py::Outer::Inner::test_nested[param]",
            location=("test/test_example.py", 0, ""),
        )

        self.assertEqual(
            _pytest_testcase_identity(report),
            ("Outer.Inner", "test_example.py"),
        )

    def test_windows_test_prefix(self):
        report = SimpleNamespace(
            nodeid="test/test_example.py::TestExample::test_windows",
            location=("test\\test_example.py", 0, ""),
        )

        self.assertEqual(
            _pytest_testcase_identity(report),
            ("TestExample", "test_example.py"),
        )

    def test_parameter_id_with_double_colon(self):
        report = SimpleNamespace(
            nodeid="test/test_example.py::TestExample::test_param[a::b]",
            location=("test/test_example.py", 0, ""),
        )

        self.assertEqual(
            _pytest_testcase_identity(report),
            ("TestExample", "test_example.py"),
        )

    def test_module_function(self):
        report = SimpleNamespace(
            nodeid="test/test_example.py::test_function",
            location=("test/test_example.py", 0, ""),
        )

        self.assertEqual(
            _pytest_testcase_identity(report),
            ("test_example", "test_example.py"),
        )

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
    unittest.main()
