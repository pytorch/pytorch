from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from codepath_owners import (
    MAX_CODEPATH_OWNERS_BYTES,
    build_codepath_owners,
    build_llm_artifact,
    load_codepath_owners,
    matches,
    parse_rules,
    resolve_paths,
    resolve_rule,
)
from ownership import (
    owner_label,
)
from prepare_llm_input import fetch_pull_request_files


COMMIT_SHA = "a" * 40


class PagingGitHub:
    def __init__(self, pages: list[list[dict[str, object]]]) -> None:
        self.pages = pages
        self.calls: list[str] = []

    def json(self, endpoint: str) -> object:
        self.calls.append(endpoint)
        page = int(endpoint.rsplit("=", 1)[1])
        return self.pages[page - 1]


class PatternTest(unittest.TestCase):
    def assert_matches(
        self, pattern: str, matching: list[str], rejected: list[str]
    ) -> None:
        for path in matching:
            self.assertTrue(matches(pattern, path), (pattern, path))
        for path in rejected:
            self.assertFalse(matches(pattern, path), (pattern, path))

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
                self.assert_matches(pattern, matching, rejected)

    def test_escaping_and_literal_brackets(self) -> None:
        self.assert_matches("f\\*o", ["f*o"], ["foo"])
        self.assert_matches("f\\?o", ["f?o"], ["foo"])
        self.assert_matches(
            "/apps/[param]/file.ts",
            ["apps/[param]/file.ts"],
            ["apps/param/file.ts", "apps/other/file.ts"],
        )
        self.assertTrue(matches("/foo", "foo/bar\nbaz"))
        self.assertFalse(matches("foo", "foo/bar\nbaz"))
        self.assertFalse(matches("/f*", "foo/bar\nbaz"))
        self.assertFalse(matches("**", "a\nb"))
        self.assertFalse(matches("foo/**", "foo/a\nb"))

    def test_invalid_and_empty_patterns(self) -> None:
        with self.assertRaises(ValueError):
            matches("foo/***/bar", "foo/x/bar")
        self.assertFalse(matches("/", "foo"))


class ResolutionTest(unittest.TestCase):
    def test_parser_preserves_comments_and_skips_invalid_rules(self) -> None:
        diagnostics: list[dict[str, object]] = []
        rules = parse_rules(
            r"""# broad owner
* @org/all # inline comment

# path with a space
foo\ bar @person
bad invalid.owner
private/ # deliberately unowned
""",
            "2" * 40,
            diagnostics,
        )
        self.assertEqual([rule["line"] for rule in rules], [2, 5, 7])
        self.assertEqual(rules[0]["comment"], "inline comment")
        self.assertEqual(rules[1]["preceding_comments"], ["path with a space"])
        self.assertEqual(rules[2]["owners"], [])
        self.assertEqual(rules[2]["rule_id"], f"{'2' * 40}:L7")
        self.assertEqual([item["line"] for item in diagnostics], [6])
        with self.assertRaises(ValueError):
            parse_rules("bad invalid.owner\n", strict=True)

    def test_last_match_wins_including_ownerless_override(self) -> None:
        rules = parse_rules("* @all\n/docs/ @docs\n/docs/private/\n")
        self.assertEqual(resolve_rule("new/file.txt", rules)["owners"], ["@all"])
        self.assertEqual(
            resolve_rule("docs/new.txt", rules)["owners"], ["@docs"]
        )
        self.assertEqual(resolve_rule("docs/private/new.txt", rules)["owners"], [])

    def test_compact_artifact_is_an_exact_ordered_partition(self) -> None:
        rules = parse_rules(
            "* @all\n/docs/ @docs @org/writers\n/docs/private/\n"
        )
        resolutions = resolve_paths(
            [
                "src/a.py",
                "docs/a.rst",
                "docs/b.rst",
                "docs/private/key.txt",
                "src/a.py",
            ],
            rules,
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
        self.assertEqual(
            artifact["paths_without_owners"],
            [{"path": "docs/private/key.txt", "reason": "ownerless_override"}],
        )

    def test_no_match_is_distinct_from_ownerless_override(self) -> None:
        rules = parse_rules("/private/\n")
        artifact = build_llm_artifact(
            resolve_paths(["private/a", "outside/a"], rules)
        )
        self.assertEqual(
            artifact["paths_without_owners"],
            [
                {"path": "private/a", "reason": "ownerless_override"},
                {"path": "outside/a", "reason": "no_matching_rule"},
            ],
        )


class PolicyLoadingTest(unittest.TestCase):
    def load(self, content: bytes) -> dict[str, object]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / ".github/auto-pr-triage/codepath_owners.txt"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        return load_codepath_owners(path, "pytorch/pytorch", COMMIT_SHA)

    def test_loads_single_policy_file_with_content_provenance(self) -> None:
        content = b"* @root-owner\n"
        snapshot = self.load(content)
        header = f"blob {len(content)}\0".encode()
        self.assertEqual(
            snapshot["source"],
            {
                "repository": "pytorch/pytorch",
                "path": ".github/auto-pr-triage/codepath_owners.txt",
                "ref": COMMIT_SHA,
                "blob_sha": hashlib.sha1(header + content).hexdigest(),
            },
        )
        self.assertEqual(snapshot["rules"][0]["owners"], ["@root-owner"])

    def test_shadow_policy_matches_native_codeowners(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        native = (repository_root / "CODEOWNERS").read_bytes()
        shadow = (
            repository_root / ".github/auto-pr-triage/codepath_owners.txt"
        ).read_bytes()

        self.assertEqual(shadow, native)
        rules = parse_rules(shadow.decode("utf-8"), strict=True)
        self.assertTrue(rules)
        self.assertTrue(
            all(
                rule["owners"]
                and all(owner.startswith("@") for owner in rule["owners"])
                for rule in rules
            )
        )

    def test_builds_exact_dynamic_codepath_owners_contract(self) -> None:
        snapshot = self.load(b"/torch/ @pytorch/core\n")
        artifact = build_codepath_owners(
            ["torch/a.py", "README.md"], snapshot
        )
        self.assertEqual(
            set(artifact),
            {
                "source",
                "owners",
                "matched_path_groups",
                "paths_without_owners",
            },
        )
        self.assertEqual(artifact["owners"], ["@pytorch/core"])
        self.assertNotIn("rules", artifact)
        self.assertNotIn("parse_diagnostics", artifact)

    def test_accepts_internal_owner_ids(self) -> None:
        snapshot = self.load(b"/torch/ compiler\n")
        artifact = build_codepath_owners(["torch/a.py"], snapshot)

        self.assertEqual(artifact["owners"], ["compiler"])
        self.assertEqual(
            artifact["matched_path_groups"],
            [{"owners": ["compiler"], "paths": ["torch/a.py"]}],
        )

    def test_rejects_mutable_ref_wrong_path_oversize_and_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / ".github/auto-pr-triage/codepath_owners.txt"
            path.parent.mkdir(parents=True)
            path.write_text("* @owner\n")
            with self.assertRaisesRegex(ValueError, "immutable commit SHA"):
                load_codepath_owners(path, "pytorch/pytorch", "main")
            wrong_path = root / "other/codepath_owners.txt"
            wrong_path.parent.mkdir()
            wrong_path.write_text("* @owner\n")
            with self.assertRaisesRegex(ValueError, "path must name .*codepath_owners"):
                load_codepath_owners(wrong_path, "pytorch/pytorch", COMMIT_SHA)
            path.write_bytes(b"x" * MAX_CODEPATH_OWNERS_BYTES)
            with self.assertRaisesRegex(ValueError, "3 MB limit"):
                load_codepath_owners(path, "pytorch/pytorch", COMMIT_SHA)
            path.write_bytes(b"\xff")
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                load_codepath_owners(path, "pytorch/pytorch", COMMIT_SHA)

    def test_rejects_foreign_teams_but_preserves_target_org_teams(self) -> None:
        snapshot = self.load(b"* @person @pytorch/compiler\n")
        self.assertEqual(
            snapshot["rules"][0]["owners"],
            ["@person", "@pytorch/compiler"],
        )

        with self.assertRaisesRegex(RuntimeError, "outside the target"):
            self.load(b"* @other/compiler\n")

    def test_email_is_not_a_valid_owner_handle(self) -> None:
        with self.assertRaisesRegex(ValueError, "codepath owner"):
            parse_rules("* person@example.com\n", strict=True)

    def test_invalid_github_usernames_are_not_owner_handles(self) -> None:
        for owner in ("@bad_user", "@-bad", "@bad-"):
            with self.subTest(owner=owner), self.assertRaisesRegex(
                ValueError, "codepath owner"
            ):
                parse_rules(f"* {owner}\n", strict=True)

    def test_policy_loader_rejects_invalid_rules(self) -> None:
        with self.assertRaisesRegex(ValueError, "codepath owner"):
            self.load(b"* invalid.owner\n")


class CollectorIntegrationTest(unittest.TestCase):
    def test_fetches_all_pull_request_file_pages_once(self) -> None:
        first = [
            {
                "filename": f"src/{index}.py",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": "+pass",
            }
            for index in range(100)
        ]
        second = [
            {
                "filename": "README.md",
                "status": "modified",
                "additions": 1,
                "deletions": 0,
                "patch": "+text",
            }
        ]
        github = PagingGitHub([first, second])
        files = fetch_pull_request_files(github, "pytorch/ciforge", 123)
        self.assertEqual(len(files), 101)
        self.assertEqual(files[-1]["filename"], "README.md")
        self.assertEqual(
            github.calls,
            [
                "repos/pytorch/ciforge/pulls/123/files?per_page=100&page=1",
                "repos/pytorch/ciforge/pulls/123/files?per_page=100&page=2",
            ],
        )

    def test_file_pagination_bound_fails_closed(self) -> None:
        pages = [
            [
                {"filename": f"file-{page * 100 + index}"}
                for index in range(100)
            ]
            for page in range(30)
        ]
        with self.assertRaisesRegex(RuntimeError, "3,000 or more"):
            fetch_pull_request_files(
                PagingGitHub(pages),
                "pytorch/ciforge",
                123,
            )

    def test_owner_labels_are_derived_and_bounded(self) -> None:
        self.assertEqual(owner_label("compiler"), "owner: compiler")
        for owner_id in ("Compiler", "a" * 44):
            with self.subTest(owner_id=owner_id), self.assertRaises(ValueError):
                owner_label(owner_id)


class CommandLineTest(unittest.TestCase):
    def test_cli_returns_json_handles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "codepath_owners.txt"
            path.write_text("* @all\n/docs/ @docs\n")
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("codepath_owners.py")),
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
