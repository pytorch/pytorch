#!/usr/bin/env python3

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from check_taxonomy import analyze, parse_patterns, TaxonomyPattern, tracked_paths

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
                f"# Draft grouped review-surface definitions:\n# [surface]\n{entry}\n"
            )
            return parse_patterns(codeowners)

    @parametrize(
        "entry",
        [
            "/path/ @user",
            "/path/ @org/team",
            "/path/ owner@example.com",
            "/path/ @user # explanation",
        ],
    )
    def test_valid_active_owners(self, entry):
        patterns, groups = self.parse_entry(entry)
        self.assertEqual(groups, ["surface"])
        self.assertEqual(patterns[0].path, "path/")

    @parametrize("entry", ["/path/ not-an-owner", "/path/ # explanation"])
    def test_invalid_active_owners(self, entry):
        with self.assertRaisesRegex(
            ValueError, "active pattern has no owners|invalid active owners"
        ):
            self.parse_entry(entry)

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
