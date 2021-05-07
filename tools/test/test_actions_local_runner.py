import textwrap
import unittest
import sys
import contextlib
import io
import os
from typing import List, Dict, Any

from tools import actions_local_runner


if sys.version_info >= (3, 8):
    # actions_local_runner uses asyncio features not available in 3.6, and
    # IsolatedAsyncioTestCase was added in 3.8, so skip testing on
    # unsupported systems
    class TestRunner(unittest.IsolatedAsyncioTestCase):
        def run(self, *args: List[Any], **kwargs: List[Dict[str, Any]]) -> Any:
            return super().run(*args, **kwargs)

        def test_step_extraction(self) -> None:
            fake_job = {
                "steps": [
                    {
                        "name": "test1",
                        "run": "echo hi"
                    },
                    {
                        "name": "test2",
                        "run": "echo hi"
                    },
                    {
                        "name": "test3",
                        "run": "echo hi"
                    },
                ]
            }

            actual = actions_local_runner.grab_specific_steps(["test2"], fake_job)
            expected = [
                {
                    "name": "test2",
                    "run": "echo hi"
                },
            ]
            self.assertEqual(actual, expected)

        async def test_runner(self) -> None:
            fake_step = {
                "name": "say hello",
                "run": "echo hi"
            }
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                await actions_local_runner.run_steps([fake_step], "test", None, True)

            result = f.getvalue()
            self.assertIn("say hello", result)
            self.assertIn("hi", result)

    class TestQuicklint(unittest.IsolatedAsyncioTestCase):
        test_files = [
            os.path.join("caffe2", "some_cool_file.py"),
            os.path.join("torch", "some_cool_file.py"),
            os.path.join("aten", "some_cool_file.py"),
            os.path.join("torch", "some_stubs.pyi"),
        ]
        maxDiff = None

        def setUp(self, *args, **kwargs):
            for name in self.test_files:
                bad_code = textwrap.dedent("""
                    some_variable = '2'
                    some_variable = None
                    some_variable = 11.2
                """).rstrip("\n")

                with open(name, "w") as f:
                    f.write(bad_code)

        def tearDown(self, *args, **kwargs):
            for name in self.test_files:
                os.remove(name)

        def test_file_selection(self):
            files = actions_local_runner.find_changed_files()
            for name in self.test_files:
                self.assertIn(name, files)

        async def test_flake8(self):
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                await actions_local_runner.run_flake8(self.test_files, True)

            # Should exclude the caffe2/ file
            expected = textwrap.dedent("""
                x flake8
                torch/some_cool_file.py:4:21: W292 no newline at end of file
                aten/some_cool_file.py:4:21: W292 no newline at end of file
            """).lstrip("\n")
            self.assertEqual(expected, f.getvalue())

        async def test_mypy(self):
            self.maxDiff = None
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                await actions_local_runner.run_mypy(self.test_files, True)

            # Should exclude the aten/ file
            expected = textwrap.dedent("""
                x mypy (skipped typestub generation)
                caffe2/some_cool_file.py:3:17: error: Incompatible types in assignment (expression has type "None", variable has type "str")  [assignment]
                caffe2/some_cool_file.py:4:17: error: Incompatible types in assignment (expression has type "float", variable has type "str")  [assignment]
                torch/some_cool_file.py:3:17: error: Incompatible types in assignment (expression has type "None", variable has type "str")  [assignment]
                torch/some_cool_file.py:4:17: error: Incompatible types in assignment (expression has type "float", variable has type "str")  [assignment]
                torch/some_stubs.pyi:3:17: error: Incompatible types in assignment (expression has type "None", variable has type "str")  [assignment]
                torch/some_stubs.pyi:4:17: error: Incompatible types in assignment (expression has type "float", variable has type "str")  [assignment]
            """).lstrip("\n")  # noqa: B950
            self.assertEqual(expected, f.getvalue())


if __name__ == '__main__':
    unittest.main()
