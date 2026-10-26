from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from tools.ramjet_discovery import collection_targets, test_command as command_for_test
from tools.testing.discover_tests import TESTS

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestRamjetDiscovery(TestCase):
    root: Path
    catalog: list[str]

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-discovery-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.catalog = [
            "test_parent",
            "nested/test_standalone",
            "cpython/v3_11/test_version",
            "cpython/v3_13/test_version",
            "test_virtual",
            "cpp/test_binary",
        ]
        sources = {
            "test/test_parent.py": (
                "from helpers.test_first import TestFirst as TestAlias, TestSecond\n"
                "from helpers.nested.test_third import TestThird\n"
            ),
            "test/helpers/test_first.py": (
                "raise_on_run_directly('test/test_parent.py')\n"
            ),
            "test/helpers/nested/test_third.py": (
                "raise_on_run_directly('test/test_parent.py')\n"
            ),
            "test/helpers/test_unused.py": "",
            "test/nested/test_standalone.py": "",
            "test/cpython/v3_11/test_version.py": "",
            "test/cpython/v3_13/test_version.py": "this uses newer syntax!\n",
            "test/external_test.py": "",
        }
        for name, source in sources.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def discover(self, targets, version=(3, 11)):
        return collection_targets(
            targets, self.root, test_modules=self.catalog, python_version=version
        )

    def test_directory_uses_canonical_entrypoints(self):
        self.assertEqual(
            self.discover(["test"]),
            [
                ["test/cpython/v3_11/test_version.py"],
                ["test/nested/test_standalone.py"],
                ["test/test_parent.py"],
            ],
        )

    def test_swa_helper_uses_optimizer_entrypoint(self):
        self.assertNotIn("optim/test_swa_utils", TESTS)
        self.assertIn("test_optim", TESTS)
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(
            collection_targets(
                ["test/optim/test_swa_utils.py::TestSWAUtilsGeneric"], root
            ),
            [["test/test_optim.py::TestSWAUtilsGeneric"]],
        )

    def test_version_filter_matches_remote_interpreter(self):
        self.assertEqual(
            self.discover(["test/cpython"], version=(3, 13)),
            [["test/cpython/v3_13/test_version.py"]],
        )

    def test_explicit_incompatible_version_explains_requirement(self):
        with self.assertRaisesRegex(
            ValueError, "requires Python 3.13.*uses Python 3.11"
        ):
            self.discover(["test/cpython/v3_13/test_version.py::TestVersion"])

    @parametrize("selector", ["", "::Suite", "::Suite::test_method"])
    def test_explicit_files_are_not_restricted_to_ci_catalog(self, selector):
        target = "test/external_test.py" + selector
        self.assertEqual(self.discover([target]), [[target]])

    @parametrize(
        "selector,expected",
        [
            ("", ["TestAlias", "TestSecond"]),
            ("::TestFirst", ["TestAlias"]),
            ("::TestFirst::test_method", ["TestAlias::test_method"]),
        ],
    )
    def test_helper_uses_parent_without_expanding_selection(self, selector, expected):
        self.assertEqual(
            self.discover(["test/helpers/test_first.py" + selector]),
            [[f"test/test_parent.py::{name}" for name in expected]],
        )

    def test_helper_subdirectory_preserves_scope(self):
        self.assertEqual(
            self.discover(["test/helpers/nested"]),
            [["test/test_parent.py::TestThird"]],
        )

    def test_helper_directory_combines_imported_classes(self):
        self.assertEqual(
            self.discover(["test/helpers"]),
            [
                [
                    "test/test_parent.py::TestAlias",
                    "test/test_parent.py::TestSecond",
                    "test/test_parent.py::TestThird",
                ]
            ],
        )

    def test_unknown_helper_selector_is_not_broadened(self):
        with self.assertRaisesRegex(ValueError, "Cannot resolve.*TestMissing"):
            self.discover(["test/helpers/test_first.py::TestMissing"])

    def test_unimported_helper_is_not_broadened(self):
        (self.root / "test/helpers/test_unused.py").write_text(
            "raise_on_run_directly('test/test_parent.py')\n"
        )
        with self.assertRaisesRegex(ValueError, "Cannot resolve.*test_unused"):
            self.discover(["test/helpers/test_unused.py"])

    def test_generated_parent_class_explains_supported_selection(self):
        with (self.root / "test/test_parent.py").open("a") as stream:
            stream.write("instantiate_device_type_tests(TestAlias, globals())\n")
        with self.assertRaisesRegex(ValueError, "generates device classes"):
            self.discover(["test/helpers/test_first.py"])
        self.assertEqual(
            self.discover(["test/helpers/test_first.py::TestSecond"]),
            [["test/test_parent.py::TestSecond"]],
        )

    def test_duplicate_selectors_are_merged(self):
        target = "test/nested/test_standalone.py::Suite"
        self.assertEqual(
            self.discover([target, target, "test/nested"]),
            [["test/nested/test_standalone.py", target]],
        )

    def test_empty_directory_reports_canonical_discovery(self):
        (self.root / "test/empty").mkdir()
        with self.assertRaisesRegex(ValueError, "No test files.*CI test entrypoints"):
            self.discover(["test/empty"])

    @parametrize(
        "imports,call",
        [
            ("from torch.testing._internal.common_utils import run_tests", "run_tests"),
            (
                "from torch.testing._internal.common_utils import run_tests as run",
                "run",
            ),
            (
                "from torch.testing._internal import common_utils as common",
                "common.run_tests",
            ),
            (
                "import torch.testing._internal.common_utils as common",
                "common.run_tests",
            ),
            ("from torch._dynamo.test_case import run_tests", "run_tests"),
            ("from torch._inductor.test_case import run_tests", "run_tests"),
            ("import torch._dynamo.test_case", "torch._dynamo.test_case.run_tests"),
        ],
    )
    def test_command_preserves_standard_main_runner(self, imports, call):
        file = "test/external_test.py"
        (self.root / file).write_text(
            f"{imports}\nif __name__ == '__main__':\n    setup()\n    {call}()\n"
        )
        self.assertEqual(
            command_for_test(file, self.root),
            [
                sys.executable,
                str(Path(__file__).parents[1] / "ramjet_run.py"),
                file,
                "--use-pytest",
                "-p",
                "no:rerunfailures",
            ],
        )

    def test_command_handles_conditional_main_runner(self):
        file = "test/external_test.py"
        (self.root / file).write_text(
            "if '__main__' == __name__:\n"
            "    from torch._inductor.test_case import run_tests\n"
            "    if enabled:\n"
            "        run_tests(needs='dependency')\n"
        )
        self.assertEqual(
            command_for_test(file, self.root),
            [
                sys.executable,
                str(Path(__file__).parents[1] / "ramjet_run.py"),
                file,
                "--use-pytest",
                "-p",
                "no:rerunfailures",
            ],
        )

    @parametrize("runner", ["run_tests", "run_tests as run"])
    def test_command_resolves_runner_from_local_star_import(self, runner):
        (self.root / "test/test_imported.py").write_text(
            f"from torch.testing._internal.common_utils import {runner}\n"
            "raise RuntimeError('Discovery must not import this module')\n"
        )
        file = "test/external_test.py"
        name = "run" if " as " in runner else "run_tests"
        (self.root / file).write_text(
            f"from test_imported import *\nif __name__ == '__main__':\n    {name}()\n"
        )
        self.assertEqual(
            command_for_test(file, self.root),
            [
                sys.executable,
                str(Path(__file__).parents[1] / "ramjet_run.py"),
                file,
                "--use-pytest",
                "-p",
                "no:rerunfailures",
            ],
        )

    @parametrize(
        "source",
        [
            "def test_function():\n    pass\n",
            "from torch.testing._internal.common_utils import run_tests\n",
            "from other_module import run_tests\n"
            "if __name__ == '__main__':\n    run_tests()\n",
        ],
    )
    def test_command_uses_pytest_for_modules_without_standard_main(self, source):
        file = "test/external_test.py"
        (self.root / file).write_text(source)
        self.assertEqual(
            command_for_test(file, self.root),
            [sys.executable, "-m", "pytest", file, "-p", "no:rerunfailures"],
        )


if __name__ == "__main__":
    run_tests()
