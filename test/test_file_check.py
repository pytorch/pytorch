# Owner(s): ["module: unknown"]

from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestFileCheck(TestCase):
    def test_not_run(self):
        stdout, _ = self.run_process_no_exception(
            """\
from torch.testing import FileCheck
file_check = FileCheck().check("not run")
del file_check
""",
        )
        FileCheck().check("You have not run this instance of FileCheck!").check_next(
            "FileCheck checks:"
        ).check_next("\tCHECK: not run").run(stdout)

    @parametrize(
        "checks,text,error",
        [
            ("# CHECK: 22", "232", 'Expected to find "22"'),
            ("# CHECK: 3\n# CHECK: 3", "232", "CHECK: 3"),
            ("# CHECK-COUNT-EXACTLY-4: 2", "22222", "Expected to not find"),
            ("# CHECK-COUNT-6: 2", "22222", "CHECK-COUNT-6: 2"),
            ("# CHECK-SAME: 33", "22\n33", "Expected to not find"),
            ("# CHECK: 1\n# CHECK-NEXT: 2", "12", "Expected to find"),
            ("# CHECK: 1\n# CHECK-NEXT: 2", "1\n\n2", "Expected to not find"),
            (
                "# CHECK-DAG: 1\n# CHECK-DAG: 2\n# CHECK: 3",
                "1 3 2",
                'Expected to find "3"',
            ),
            ("# CHECK-NOT: 2\n# CHECK: 1", "21", 'Expected to not find "2"'),
            ("# CHECK: 2\n# CHECK-NOT: 1", "21", 'Expected to not find "1"'),
            (
                "# CHECK-COUNT-2: 2\n# CHECK-NOT: 1\n# CHECK-COUNT-2: 2",
                "22 1 22",
                'Expected to not find "1"',
            ),
            ("", "1", "Check for bad input"),
            ("# CHECK1", "", "Could not parse check"),
        ],
    )
    def test_rejects_invalid_matches(self, checks, text, error):
        with self.assertRaisesRegex(RuntimeError, error):
            FileCheck().run(checks, text)

    def test_negative_checks_are_bounded(self):
        FileCheck().check_not("2").check("1").run("12")
        FileCheck().check("2").check_not("2").run("12")
        FileCheck().check_not("3").check_dag("1").check_dag("2").check_not("3").run(
            "1 3 2"
        )

    def test_all_python_api(self):
        test_string = """
check check_same
check_next
check_count
check_dag
check_source_highlighted
~~~~~~~~~~~~~~~~~~~~~~~~
check_regex
"""
        FileCheck().check("check").check_not("check_not").check_same(
            "check_same"
        ).check_next("check_next").check_count("check_count", 1).check_dag(
            "check_dag"
        ).check_source_highlighted("check_source_highlighted").check_regex(
            r"check_.+"
        ).run(test_string)

        FileCheck().run(
            """
# CHECK: check
# CHECK-NOT: check_not
# CHECK-SAME: check_same
# CHECK-NEXT: check_next
# CHECK-DAG: check_dag
# CHECK-SOURCE-HIGHLIGHTED: check_source_highlighted
# CHECK-REGEX: check_.+
        """,
            test_string,
        )


instantiate_parametrized_tests(TestFileCheck)


if __name__ == "__main__":
    run_tests()
