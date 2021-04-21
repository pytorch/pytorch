import unittest
import sys
import contextlib
import io
from typing import List, Dict, Any

from tools import actions_local_runner


if __name__ == '__main__':
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
                    await actions_local_runner.run_steps([fake_step], "test", None)

                result = f.getvalue()
                self.assertIn("say hello", result)
                self.assertIn("hi", result)


    unittest.main()
