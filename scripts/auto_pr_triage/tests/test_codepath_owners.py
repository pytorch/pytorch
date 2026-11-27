from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codepath_owners import (
    build_llm_artifact,
    matches,
    parse_rules,
    resolve_for_llm,
    resolve_paths,
    resolve_rule,
)
from trusted_config import (
    CODEPATH_OWNERS_PATH,
    load_codepath_owners,
    MAX_CODEPATH_OWNERS_BYTES,
)


COMMIT_SHA = "a" * 40


class PatternTest(unittest.TestCase):
    def assert_matches(
        self, *, pattern: str, matching: list[str], rejected: list[str]
    ) -> None:
        for path in matching:
            self.assertTrue(matches(pattern=pattern, path=path), (pattern, path))
        for path in rejected:
            self.assertFalse(matches(pattern=pattern, path=path), (pattern, path))

    def test_literal_directory_and_wildcard_patterns(self) -> None:
        cases = [
            (
                "foo",
                ["foo", "foo/bar", "bar/foo", "bar/foo/baz"],
                ["foo.txt", "bar/foo.txt", "bar/baz"],
            ),
            (
                "/foo",
                ["foo", "foo/bar", "foo/bar/baz"],
                ["bar/foo", "bar/foo/baz"],
            ),
            (
                "foo/",
                ["foo/bar", "foo/bar/baz", "bar/foo/baz"],
                ["foo", "bar/foo", "bar/baz"],
            ),
            (
                "*",
                ["foo", "foo/bar", "bar/foo/baz"],
                [],
            ),
            (
                "f*",
                ["foo", "foo/bar", "bar/foo", "bar/foo/baz"],
                ["xfoo", "bar/baz"],
            ),
            ("/*", ["foo", "bar"], ["foo/bar", "foo/bar/baz"]),
            (
                "/f*",
                ["foo", "foo/bar", "foo/bar/baz"],
                ["bar/foo", "xfoo"],
            ),
            ("foo/*", ["foo/bar"], ["foo", "foo/bar/baz", "bar/foo/baz"]),
            (
                "foo/*.txt",
                ["foo/bar.txt"],
                ["foo/bar/baz.txt", "qux/foo/bar.txt"],
            ),
            (
                "**/foo/bar",
                ["foo/bar", "qux/foo/bar", "qux/foo/bar/baz"],
                ["foo/baz/bar"],
            ),
            (
                "foo/bar/**",
                ["foo/bar/baz", "foo/bar/baz/qux"],
                ["foo/bar", "qux/foo/bar/baz"],
            ),
            (
                "foo/**/bar",
                ["foo/bar", "foo/qux/bar", "foo/qux/quux/bar/baz"],
                ["qux/foo/bar"],
            ),
            (
                "foo**bar",
                ["foobar", "fooXXbar", "x/foobar/z"],
                ["foo/x/bar"],
            ),
        ]
        for pattern, matching, rejected in cases:
            with self.subTest(pattern=pattern):
                self.assert_matches(
                    pattern=pattern, matching=matching, rejected=rejected
                )

    def test_escaping_and_literal_brackets(self) -> None:
        self.assert_matches(pattern="f\\*o", matching=["f*o"], rejected=["foo"])
        self.assert_matches(pattern="f\\?o", matching=["f?o"], rejected=["foo"])
        self.assert_matches(
            pattern="/apps/[param]/file.ts",
            matching=["apps/[param]/file.ts"],
            rejected=["apps/param/file.ts", "apps/other/file.ts"],
        )
        self.assertTrue(matches(pattern="/foo", path="foo/bar\nbaz"))
        self.assertFalse(matches(pattern="foo", path="foo/bar\nbaz"))
        self.assertFalse(matches(pattern="/f*", path="foo/bar\nbaz"))
        self.assertFalse(matches(pattern="**", path="a\nb"))
        self.assertFalse(matches(pattern="foo/**", path="foo/a\nb"))

    def test_invalid_and_empty_patterns(self) -> None:
        with self.assertRaises(ValueError):
            matches(pattern="foo/***/bar", path="foo/x/bar")
        self.assertFalse(matches(pattern="/", path="foo"))


class ResolutionTest(unittest.TestCase):
    def test_parser_preserves_comments_and_skips_invalid_rules(self) -> None:
        diagnostics: list[dict[str, object]] = []
        rules = parse_rules(
            contents=r"""# broad owner
* @org/all # inline comment

# path with a space
foo\ bar @person
bad invalid.owner
private/ # deliberately unowned
""",
            blob_sha="2" * 40,
            diagnostics=diagnostics,
        )
        self.assertEqual([rule["line"] for rule in rules], [2, 5, 7])
        self.assertEqual(rules[0]["comment"], "inline comment")
        self.assertEqual(rules[1]["preceding_comments"], ["path with a space"])
        self.assertEqual(rules[2]["owners"], [])
        self.assertEqual(rules[2]["rule_id"], f"{'2' * 40}:L7")
        self.assertEqual([item["line"] for item in diagnostics], [6])
        with self.assertRaises(ValueError):
            parse_rules(contents="bad invalid.owner\n", strict=True)

    def test_last_match_wins_including_ownerless_override(self) -> None:
        rules = parse_rules(contents="* @all\n/docs/ @docs\n/docs/private/\n")
        self.assertEqual(
            resolve_rule(path="new/file.txt", rules=rules)["owners"], ["@all"]
        )
        self.assertEqual(
            resolve_rule(path="docs/new.txt", rules=rules)["owners"], ["@docs"]
        )
        self.assertEqual(
            resolve_rule(path="docs/private/new.txt", rules=rules)["owners"], []
        )

    def test_compact_artifact_is_an_exact_ordered_partition(self) -> None:
        rules = parse_rules(
            contents="* @all\n/docs/ @docs @org/writers\n/docs/private/\n"
        )
        resolutions = resolve_paths(
            paths=[
                "src/a.py",
                "docs/a.rst",
                "docs/b.rst",
                "docs/private/key.txt",
                "src/a.py",
            ],
            rules=rules,
        )
        artifact = build_llm_artifact(resolutions)
        self.assertEqual(artifact["owners"], ["@all", "@docs", "@org/writers"])
        self.assertEqual(
            artifact["matched_path_groups"],
            [
                {"owners": ["@all"], "paths": ["src/a.py"]},
                {
                    "owners": ["@docs", "@org/writers"],
                    "paths": ["docs/a.rst", "docs/b.rst"],
                },
            ],
        )
        self.assertEqual(artifact["paths_without_owners"], ["docs/private/key.txt"])

    def test_no_match_and_ownerless_override_are_both_unowned(self) -> None:
        rules = parse_rules(contents="/private/\n")
        artifact = build_llm_artifact(
            resolve_paths(paths=["private/a", "outside/a"], rules=rules)
        )
        self.assertEqual(artifact["paths_without_owners"], ["private/a", "outside/a"])


class PolicyLoadingTest(unittest.TestCase):
    def load(self, content: bytes) -> dict[str, object]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        path = root / CODEPATH_OWNERS_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return load_codepath_owners(
            repository_root=root, repo="pytorch/pytorch", ref=COMMIT_SHA
        )

    def test_loads_single_policy_file_with_content_provenance(self) -> None:
        content = b"* @root-owner\n"
        snapshot = self.load(content)
        header = f"blob {len(content)}\0".encode()
        self.assertEqual(
            snapshot["source"],
            {
                "repository": "pytorch/pytorch",
                "path": "CODEOWNERS",
                "ref": COMMIT_SHA,
                "blob_sha": hashlib.sha1(header + content).hexdigest(),
            },
        )
        self.assertEqual(snapshot["rules"][0]["owners"], ["@root-owner"])

    def test_checked_in_codeowners_loads(self) -> None:
        repository_root = Path(__file__).resolve().parents[3]
        snapshot = load_codepath_owners(
            repository_root=repository_root, repo="pytorch/ciforge", ref=COMMIT_SHA
        )
        self.assertTrue(snapshot["rules"])

    def test_normalizes_and_deduplicates_github_handle_casing(self) -> None:
        snapshot = self.load(
            b"/one/ @Chillee @chillee @PyTorch/Core @pytorch/core\n/two/ @CHILLEE @PYTORCH/CORE\n"
        )

        self.assertEqual(
            [rule["owners"] for rule in snapshot["rules"]],
            [
                ["@chillee", "@pytorch/core"],
                ["@chillee", "@pytorch/core"],
            ],
        )
        artifact = resolve_for_llm(paths=["one/a.py", "two/b.py"], snapshot=snapshot)
        self.assertEqual(artifact["owners"], ["@chillee", "@pytorch/core"])
        self.assertEqual(
            artifact["matched_path_groups"],
            [
                {
                    "owners": ["@chillee", "@pytorch/core"],
                    "paths": ["one/a.py", "two/b.py"],
                }
            ],
        )

    def test_builds_exact_dynamic_codepath_owners_contract(self) -> None:
        snapshot = self.load(b"/torch/ @pytorch/core\n")
        artifact = resolve_for_llm(paths=["torch/a.py", "README.md"], snapshot=snapshot)
        self.assertEqual(
            set(artifact),
            {"owners", "matched_path_groups", "paths_without_owners"},
        )
        self.assertEqual(artifact["owners"], ["@pytorch/core"])
        self.assertNotIn("rules", artifact)
        self.assertNotIn("parse_diagnostics", artifact)

    def test_skips_team_owner_ids(self) -> None:
        snapshot = self.load(b"* @root-owner\n/torch/ compiler\n")
        self.assertEqual(
            [rule["owners"] for rule in snapshot["rules"]], [["@root-owner"]]
        )
        self.assertEqual(snapshot["parse_diagnostics"][0]["line"], 2)

    def test_rejects_oversize_invalid_utf8_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / CODEPATH_OWNERS_PATH
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * (MAX_CODEPATH_OWNERS_BYTES + 1))
            with self.assertRaisesRegex(RuntimeError, "size is invalid"):
                load_codepath_owners(
                    repository_root=root, repo="pytorch/pytorch", ref=COMMIT_SHA
                )
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(RuntimeError, "not valid UTF-8"):
                load_codepath_owners(
                    repository_root=root, repo="pytorch/pytorch", ref=COMMIT_SHA
                )
            outside = root / "outside.txt"
            outside.write_text("* @owner\n")
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "config file is unavailable"):
                load_codepath_owners(
                    repository_root=root, repo="pytorch/pytorch", ref=COMMIT_SHA
                )

    def test_preserves_teams(self) -> None:
        snapshot = self.load(b"* @person @pytorch/compiler\n")
        self.assertEqual(
            snapshot["rules"][0]["owners"],
            ["@person", "@pytorch/compiler"],
        )

    def test_email_is_not_a_valid_owner_handle(self) -> None:
        with self.assertRaisesRegex(ValueError, "codepath owner"):
            parse_rules(contents="* person@example.com\n", strict=True)

    def test_invalid_github_usernames_are_not_owner_handles(self) -> None:
        for owner in ("@bad_user", "@-bad", "@bad-"):
            with (
                self.subTest(owner=owner),
                self.assertRaisesRegex(ValueError, "codepath owner"),
            ):
                parse_rules(contents=f"* {owner}\n", strict=True)

    def test_loader_skips_invalid_lines_like_github(self) -> None:
        snapshot = self.load(
            b"* @root-owner\n/docs/ person@example.com\n/a/ @a-owner\n"
        )
        self.assertEqual(
            [rule["owners"] for rule in snapshot["rules"]],
            [["@root-owner"], ["@a-owner"]],
        )
        self.assertEqual([d["line"] for d in snapshot["parse_diagnostics"]], [2])


class CommandLineTest(unittest.TestCase):
    def test_cli_returns_json_handles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codepath_owners.txt"
            path.write_text("* @all\n/docs/ @docs\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "codepath_owners.py"),
                    "--codepath-owners",
                    str(path),
                    "--owners-only",
                    "src/a.py",
                    "docs/a.rst",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        self.assertEqual(json.loads(result.stdout), ["@all", "@docs"])


if __name__ == "__main__":
    unittest.main()
