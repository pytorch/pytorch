#!/usr/bin/env python3

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from check_taxonomy import (
    analyze,
    compile_glob,
    parse_patterns,
    TaxonomyPattern,
    tracked_paths,
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


class TestCodeownersTaxonomy(TestCase):
    def parse_entry(self, entry: str):
        """Parse one taxonomy entry from a minimal CODEOWNERS fixture."""
        with TemporaryDirectory() as directory:
            codeowners = Path(directory) / "CODEOWNERS"
            codeowners.write_text(
                f"# Grouped review-surface definitions:\n# [surface]\n{entry}\n"
            )
            return parse_patterns(codeowners)

    @parametrize(
        "entry",
        [
            "/path/ @user",
            "/path/ @org/team",
            "/path/ owner@example.com",
            "/path/ @user # explanation",
            "/path/*.py @user",
        ],
    )
    def test_valid_active_owners(self, entry):
        patterns, groups = self.parse_entry(entry)
        self.assertEqual(groups, ["surface"])
        self.assertEqual(patterns[0].path, entry.split()[0][1:])

    @parametrize("entry", ["/path/ not-an-owner", "/path/ # explanation"])
    def test_invalid_active_owners(self, entry):
        with self.assertRaisesRegex(
            ValueError, "active pattern has no owners|invalid active owners"
        ):
            self.parse_entry(entry)

    def test_active_rule_outside_grouped_taxonomy(self):
        with TemporaryDirectory() as directory:
            codeowners = Path(directory) / "CODEOWNERS"
            codeowners.write_text(
                "/legacy/ @owner\n"
                "# Grouped review-surface definitions:\n"
                "# [surface]\n"
                "# /path/\n"
            )
            with self.assertRaisesRegex(ValueError, "active rule outside"):
                parse_patterns(codeowners)

    def test_tracked_paths_ignore_index_only_files(self):
        with TemporaryDirectory() as directory:
            repo = Path(directory)
            git = ["git", "-C", repo]
            subprocess.run([*git, "init"], check=True, capture_output=True)
            (repo / ".gitignore").write_text("generated.pyi\n")
            (repo / "tracked.py").touch()
            subprocess.run([*git, "add", ".gitignore", "tracked.py"], check=True)
            subprocess.run(
                [
                    *git,
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.com",
                    "commit",
                    "-m",
                    "initial",
                ],
                check=True,
                capture_output=True,
            )
            (repo / "generated.pyi").touch()
            subprocess.run([*git, "add", "--force", "generated.pyi"], check=True)
            self.assertEqual(tracked_paths(repo), [".gitignore", "tracked.py"])

    def test_active_future_pattern_is_not_stale(self):
        patterns = [
            TaxonomyPattern("current", "current.py", 1),
            TaxonomyPattern("future", "future/", 2, active=True),
        ]
        report = analyze(["current.py"], patterns)
        self.assertEqual(report.stale, [])

    def test_codeowners_glob(self):
        glob = compile_glob("aten/src/ATen/native/**/*LinearAlgebra*")
        self.assertIsNotNone(
            glob.fullmatch("aten/src/ATen/native/cuda/linalg/BatchLinearAlgebra.cpp")
        )
        self.assertIsNone(glob.fullmatch("aten/src/ATen/native/cuda/Blas.cpp"))
        with self.assertRaisesRegex(ValueError, "unsupported CODEOWNERS glob"):
            compile_glob("foo/**bar")

    def test_glob_precedes_literal_override(self):
        path = "torch/csrc/autograd/profiler_kineto.cpp"
        patterns = [
            TaxonomyPattern("autograd", "torch/csrc/autograd/", 1),
            TaxonomyPattern("profiler", "torch/csrc/autograd/profiler*", 2),
            TaxonomyPattern("profiler", path, 3),
        ]
        report = analyze([path], patterns)
        self.assertEqual(report.uncovered, [])
        self.assertEqual(
            {pattern.path for pattern in report.overridden[path]},
            {"torch/csrc/autograd/", "torch/csrc/autograd/profiler*", path},
        )
        self.assertEqual(report.source_order_mismatches, [])

    def test_source_order_must_preserve_specificity(self):
        path = "torch/_higher_order_ops/flex_attention.py"
        specific = TaxonomyPattern("inductor", path, 1)
        broad = TaxonomyPattern("export", "torch/_higher_order_ops/", 2)
        report = analyze([path], [specific, broad])
        self.assertEqual(
            report.source_order_mismatches,
            [f"{path}: export:torch/_higher_order_ops/ overrides inductor:{path}"],
        )

    def test_broad_rule_before_specific_rule(self):
        path = "torch/_higher_order_ops/flex_attention.py"
        broad = TaxonomyPattern("export", "torch/_higher_order_ops/", 1)
        specific = TaxonomyPattern("inductor", path, 2)
        self.assertEqual(analyze([path], [broad, specific]).source_order_mismatches, [])


instantiate_parametrized_tests(TestCodeownersTaxonomy)

if __name__ == "__main__":
    run_tests()
