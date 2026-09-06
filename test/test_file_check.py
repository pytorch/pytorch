# Owner(s): ["module: unknown"]

from torch.testing import FileCheck
from torch.testing._internal.common_utils import run_tests, TestCase


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

    def test_string_captures_and_substitutions(self):
        test_string = """
clone = torch.ops.aten.clone.default(t_1)
generated_all_reduce_7 = torch.ops._c10d_functional.all_reduce.default(clone)
wait = torch.ops._c10d_functional.wait_tensor.default(generated_all_reduce_7)
"""
        capture = r"[[ALL_REDUCE_NODE:[A-Za-z0-9_]+]]"
        FileCheck().check(
            capture + " = torch.ops._c10d_functional.all_reduce.default"
        ).check(
            "torch.ops._c10d_functional.wait_tensor.default([[ALL_REDUCE_NODE]])"
        ).run(test_string)

        FileCheck().run(
            r"""
# CHECK: [[ALL_REDUCE_NODE:[A-Za-z0-9_]+]] = torch.ops._c10d_functional.all_reduce.default
# CHECK: torch.ops._c10d_functional.wait_tensor.default([[ALL_REDUCE_NODE]])
""",
            test_string,
        )

    def test_string_capture_same_line_and_redefinition(self):
        FileCheck().check(r"op [[REGISTER:r[0-9]+]], [[REGISTER]]").check_next(
            r"op [[REGISTER:r[0-9]+]]"
        ).check_next("use [[REGISTER]]").run(
            "op r12, r12\nop r7\nuse r7"
        )

    def test_string_capture_in_check_not(self):
        FileCheck().check(r"[[REGISTER:r[0-9]+]]").check_not(
            "clobber [[REGISTER]]"
        ).check("done").run("r1\nclobber r2\ndone")

        with self.assertRaisesRegex(RuntimeError, "Expected to not find pattern"):
            FileCheck().check(r"[[REGISTER:r[0-9]+]]").check_not(
                "clobber [[REGISTER]]"
            ).check("done").run("r1\nclobber r1\ndone")

    def test_string_capture_errors(self):
        with self.assertRaisesRegex(RuntimeError, "Undefined FileCheck variable"):
            FileCheck().check("use [[REGISTER]]").run("use r1")

        with self.assertRaisesRegex(RuntimeError, "Expected to find pattern"):
            FileCheck().check(r"def [[REGISTER:r[0-9]+]]").check_next(
                "use [[REGISTER]]"
            ).run("def r1\nuse r2")


if __name__ == "__main__":
    run_tests()
