from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import mock

from tools.ramjet_fixtures import (
    build_fixtures,
    EDGE_PROFILER,
    FIXTURE_FILES,
    FIXTURE_SOURCES,
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestRamjetFixtures(TestCase):
    root: Path
    installed: Path

    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory(prefix="ramjet-fixture-tests-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.installed = self.root / "site-packages" / "torch"
        self.installed.joinpath("lib").mkdir(parents=True)
        for relative in FIXTURE_FILES:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture source")

    @parametrize("kineto", [True, False])
    def test_builds_and_installs_cpu_fixtures(self, kineto):
        def compile_fixture(**kwargs):
            self.assertFalse(kwargs["with_cuda"])
            self.assertFalse(kwargs["is_python_module"])
            self.assertEqual(kwargs["extra_include_paths"], [str(self.root)])
            self.assertEqual(os.environ["MAX_JOBS"], "32")
            self.assertTrue(Path(os.environ["CCACHE_DIR"]).is_dir())
            self.assertTrue(Path(os.environ["CCACHE_TEMPDIR"]).is_dir())
            directory = Path(kwargs["build_directory"])
            self.assertTrue(directory.is_dir())
            library = directory / f"{kwargs['name']}.so"
            library.write_text(kwargs["name"])
            return str(library)

        with (
            mock.patch("torch.__file__", str(self.installed / "__init__.py")),
            mock.patch("torch.profiler.kineto_available", return_value=kineto),
            mock.patch("tempfile.gettempdir", return_value=str(self.root)),
            mock.patch.dict(os.environ, {"CCACHE_DIR": "/missing/volume"}),
            mock.patch(
                "torch.utils.cpp_extension.load", side_effect=compile_fixture
            ) as load,
        ):
            build_fixtures(str(self.root))

        self.assertEqual(load.call_count, len(FIXTURE_SOURCES))
        for call in load.call_args_list:
            name = call.kwargs["name"]
            library = self.installed / "lib" / f"lib{name}.so"
            self.assertEqual(library.read_text(), name)
            sources = [str(self.root / path) for path in FIXTURE_SOURCES[name]]
            flags = []
            if name == "backend_with_compiler":
                if kineto:
                    sources.append(str(self.root / EDGE_PROFILER))
                flags = ["-DUSE_KINETO" if kineto else "-DNO_PROFILING"]
            self.assertEqual(call.kwargs["sources"], sources)
            self.assertEqual(call.kwargs["extra_cflags"], flags)

    def test_missing_source_fails_before_compilation(self):
        missing = FIXTURE_FILES[-1]
        self.root.joinpath(missing).unlink()
        with mock.patch("torch.utils.cpp_extension.load") as load:
            with self.assertRaisesRegex(FileNotFoundError, missing):
                build_fixtures(str(self.root))
        load.assert_not_called()

    def test_fixture_manifest_matches_checkout(self):
        checkout = Path(__file__).resolve().parents[2]
        self.assertEqual(len(FIXTURE_FILES), len(set(FIXTURE_FILES)))
        for relative in FIXTURE_FILES:
            self.assertTrue(checkout.joinpath(relative).is_file(), relative)


if __name__ == "__main__":
    run_tests()
