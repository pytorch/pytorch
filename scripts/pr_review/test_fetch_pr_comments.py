#!/usr/bin/env python3
"""Tests for the maintainer-conversation fetcher run by `prepare`.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_pr_comments as fpc  # noqa: E402

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import run_this_suite, TestTheSuiteIsWhole  # noqa: E402,F401


REPO = "o/r"
PR = 7


def user(login, kind="User"):
    return {"login": login, "type": kind}


class FakeApi:
    pull = {"user": user("author")}

    def __init__(
        self,
        *,
        issue=(),
        reviews=(),
        inline=(),
        maintainers=(),
        unknown=(),
        reactions=None,
    ):
        self.pages = {
            f"repos/{REPO}/issues/{PR}/comments": list(issue),
            f"repos/{REPO}/pulls/{PR}/reviews": list(reviews),
            f"repos/{REPO}/pulls/{PR}/comments": list(inline),
            **{f"repos/{REPO}/{k}/reactions": v for k, v in (reactions or {}).items()},
        }
        self.maintainers = set(maintainers)
        self.unknown = set(unknown)
        self.deleted = set()
        self.calls = []

    def __call__(self, endpoint):
        self.calls.append(endpoint)
        if endpoint == f"repos/{REPO}/pulls/{PR}":
            return self.pull
        base, _, query = endpoint.partition("?")
        if base in self.deleted:
            raise fpc.NotFound(endpoint)
        if base in self.pages:
            page = int(query.split("&page=", 1)[1]) if "&page=" in query else 1
            items = self.pages[base]
            return items[(page - 1) * fpc.PER_PAGE : page * fpc.PER_PAGE]
        login = base.split("/collaborators/", 1)[1].split("/", 1)[0]
        if login in self.unknown:
            raise fpc.NotFound(endpoint)
        perms = {p: False for p in ("admin", "maintain", "push", "triage", "pull")}
        perms["triage"] = login in self.maintainers
        return {"user": {"permissions": perms}}


def comment(
    login, body="please fix", when="2026-01-01T00:00:00Z", kind="User", **extra
):
    return {
        "id": 1,
        "user": user(login, kind),
        "body": body,
        "created_at": when,
        **extra,
    }


class TestCollect(unittest.TestCase):
    def test_keeps_the_author_and_maintainers_only(self):
        api = FakeApi(
            issue=[
                comment("author", when="2026-01-03"),
                comment("maint", when="2026-01-02"),
                comment("drive-by"),
                comment("pytorchbot", kind="Bot"),
            ],
            maintainers={"maint"},
        )
        out = fpc.collect(api, REPO, PR)
        self.assertEqual(out["pr_author"], "author")
        self.assertEqual(
            [(i["author"], i["role"]) for i in out["items"]],
            [("maint", "maintainer"), ("author", "author")],
        )
        self.assertFalse(out["truncated"])

    def test_each_login_is_checked_once_and_bots_and_the_author_never(self):
        api = FakeApi(
            issue=[
                comment("maint"),
                comment("maint"),
                comment("author"),
                comment("b", kind="Bot"),
            ],
            inline=[comment("maint", path="a.py", line=3)],
            maintainers={"maint"},
        )
        fpc.collect(api, REPO, PR)
        checks = [c for c in api.calls if "/collaborators/" in c]
        self.assertEqual(checks, [f"repos/{REPO}/collaborators/maint/permission"])

    def test_machine_accounts_registered_as_users_are_dropped(self):
        api = FakeApi(
            issue=[comment("pytorchmergebot")], maintainers={"pytorchmergebot"}
        )
        self.assertEqual(fpc.collect(api, REPO, PR)["items"], [])
        self.assertFalse([c for c in api.calls if "/collaborators/" in c])

    def test_an_unknown_user_is_not_a_maintainer(self):
        api = FakeApi(issue=[comment("ghost")], unknown={"ghost"})
        self.assertEqual(fpc.collect(api, REPO, PR)["items"], [])

    def test_empty_reviews_are_dropped_unless_they_request_changes(self):
        reviews = [
            {
                "id": 1,
                "user": user("maint"),
                "body": "",
                "state": "APPROVED",
                "submitted_at": "1",
            },
            {
                "id": 2,
                "user": user("maint"),
                "body": "",
                "state": "CHANGES_REQUESTED",
                "submitted_at": "2",
            },
            {
                "id": 3,
                "user": user("maint"),
                "body": "nit",
                "state": "COMMENTED",
                "submitted_at": "3",
            },
        ]
        out = fpc.collect(FakeApi(reviews=reviews, maintainers={"maint"}), REPO, PR)
        self.assertEqual(
            [(i["id"], i["state"]) for i in out["items"]],
            [(2, "CHANGES_REQUESTED"), (3, "COMMENTED")],
        )
        self.assertEqual(out["items"][0]["created_at"], "2")

    def test_inline_comments_carry_their_anchor_and_are_capped(self):
        inline = [
            comment(
                "maint",
                body="x" * (fpc.MAX_BODY + 10),
                path="torch/a.py",
                line=12,
                original_line=10,
                in_reply_to_id=None,
                diff_hunk="@" * (fpc.MAX_HUNK + 10),
            )
        ]
        (item,) = fpc.collect(FakeApi(inline=inline, maintainers={"maint"}), REPO, PR)[
            "items"
        ]
        self.assertEqual(
            (item["kind"], item["path"], item["line"], item["original_line"]),
            ("review_comment", "torch/a.py", 12, 10),
        )
        self.assertTrue(item["body"].endswith(" [truncated]"))
        self.assertEqual(len(item["body"]), fpc.MAX_BODY + len(" [truncated]"))
        self.assertEqual(len(item["diff_hunk"]), fpc.MAX_HUNK + len(" [truncated]"))

    def test_every_page_is_read(self):
        issue = [comment("maint", when=f"{i:05d}") for i in range(fpc.PER_PAGE + 5)]
        out = fpc.collect(FakeApi(issue=issue, maintainers={"maint"}), REPO, PR)
        self.assertEqual(len(out["items"]), fpc.PER_PAGE + 5)

    def test_too_many_pages_is_an_error_not_a_silent_cut(self):
        issue = [comment("maint")] * (fpc.PER_PAGE * fpc.MAX_PAGES)
        with self.assertRaises(ValueError):
            fpc.collect(FakeApi(issue=issue, maintainers={"maint"}), REPO, PR)

    def test_the_oldest_items_are_dropped_first(self):
        issue = [
            comment("maint", body=str(i), when=f"{i:05d}")
            for i in range(fpc.MAX_ITEMS + 3)
        ]
        out = fpc.collect(FakeApi(issue=issue, maintainers={"maint"}), REPO, PR)
        self.assertTrue(out["truncated"])
        self.assertEqual(out["omitted_items"], 3)
        self.assertEqual(out["items"][0]["body"], "3")
        self.assertEqual(out["items"][-1]["body"], str(fpc.MAX_ITEMS + 2))

    def test_the_byte_cap_holds(self):
        issue = [
            comment("maint", body="y" * fpc.MAX_BODY, when=f"{i:05d}")
            for i in range(80)
        ]
        out = fpc.collect(FakeApi(issue=issue, maintainers={"maint"}), REPO, PR)
        self.assertTrue(out["truncated"])
        self.assertLessEqual(len(fpc.dumps(out)), fpc.MAX_TOTAL_BYTES)

    def test_the_byte_cap_counts_the_envelope(self):
        issue = [comment("maint", when=f"{i:05d}") for i in range(30)]
        api = FakeApi(issue=issue, maintainers={"maint"})
        for limit in range(100, 4000, 7):
            with mock.patch.object(fpc, "MAX_TOTAL_BYTES", limit):
                out = fpc.collect(api, REPO, PR)
            self.assertLessEqual(len(fpc.dumps(out)), limit)

    def test_the_description_is_capped_and_counted(self):
        issue = [comment("maint", body="y" * fpc.MAX_BODY)] * 80
        api = FakeApi(issue=issue, maintainers={"maint"})
        e = "\U0001f600"  # 12 bytes once escaped
        api.pull = {**api.pull, "title": e * 256, "body": e * 60000}
        out = fpc.collect(api, REPO, PR)
        self.assertEqual(out["title"], e * 256)
        self.assertTrue(out["body"].endswith(" [truncated]"))
        self.assertLessEqual(len(fpc.dumps(out)), fpc.MAX_TOTAL_BYTES)
        # Even a worst-case description leaves a quarter of the cap to comments.
        self.assertGreaterEqual(len(out["items"]), 5)


def ours(rid=50, body=fpc.MARKER + "\nverdict"):
    return {
        "id": rid,
        "user": user(fpc.REVIEW_AUTHOR, "Bot"),
        "body": body,
        "state": "CHANGES_REQUESTED",
        "submitted_at": "1",
    }


def finding(cid=10, rid=50, when="2026-01-01", body="finding", **extra):
    return comment(
        fpc.REVIEW_AUTHOR,
        body=body,
        when=when,
        kind="Bot",
        id=cid,
        pull_request_review_id=rid,
        path="a.py",
        line=3,
        **extra,
    )


def reaction(login, content, kind="User"):
    return {"user": user(login, kind), "content": content}


class TestEarlierAutomatedFindings(unittest.TestCase):
    def test_this_workflows_inline_findings_are_kept_with_their_replies(self):
        api = FakeApi(
            reviews=[ours(), ours(rid=51, body="not ours")],
            inline=[
                finding(),
                finding(cid=11, rid=51),
                comment("other-bot", kind="Bot", id=12, pull_request_review_id=50),
                comment(
                    "author", body="wrong", id=13, in_reply_to_id=10, when="2026-01-02"
                ),
            ],
        )
        out = fpc.collect(api, REPO, PR)
        self.assertEqual(
            [(i["id"], i["author"], i["role"]) for i in out["items"]],
            [
                (10, fpc.REVIEW_AUTHOR, fpc.AUTOMATED),
                (13, "author", "author"),
            ],
        )
        self.assertEqual(out["items"][1]["in_reply_to_id"], 10)
        # The review body itself stays out; only the inline findings come in.
        self.assertFalse([i for i in out["items"] if i["kind"] == "review"])
        self.assertFalse([c for c in api.calls if "/collaborators/" in c])

    def test_the_marker_and_author_match_post_review(self):
        import post_review

        self.assertEqual(fpc.MARKER, post_review.MARKER)
        self.assertEqual(fpc.REVIEW_AUTHOR, post_review.DEFAULT_AUTHOR)

    def test_people_are_kept_before_earlier_findings(self):
        issue = [
            comment("maint", body="y" * 1000, when=f"2026-01-{i:02d}")
            for i in range(1, 6)
        ]
        inline = [
            finding(cid=100 + i, when=f"2026-02-{i:02d}", body="z" * 1000)
            for i in range(1, 6)
        ]
        api = FakeApi(
            issue=issue, inline=inline, reviews=[ours()], maintainers={"maint"}
        )
        full = fpc.collect(api, REPO, PR)
        self.assertEqual(len(full["items"]), 10)
        with mock.patch.object(fpc, "MAX_TOTAL_BYTES", len(fpc.dumps(full)) - 500):
            out = fpc.collect(api, REPO, PR)
        roles = [i["role"] for i in out["items"]]
        self.assertEqual(roles.count("maintainer"), 5)
        self.assertEqual(roles.count(fpc.AUTOMATED), 4)
        # The oldest finding went, and what is left is still in time order.
        self.assertNotIn(101, [i["id"] for i in out["items"]])
        self.assertEqual(out["omitted_items"], 1)
        times = [i["created_at"] for i in out["items"]]
        self.assertEqual(times, sorted(times))

    def test_no_finding_is_kept_while_a_persons_comment_is_left_out(self):
        issue = [
            comment("maint", body="y" * fpc.MAX_BODY, when=f"2026-01-{i:02d}")
            for i in range(1, 31)
        ]
        api = FakeApi(
            issue=issue,
            inline=[finding(when="2026-02-01", body="short")],
            reviews=[ours()],
            maintainers={"maint"},
        )
        out = fpc.collect(api, REPO, PR)
        self.assertTrue(out["truncated"])
        self.assertFalse([i for i in out["items"] if i["role"] == fpc.AUTOMATED])


class TestReactions(unittest.TestCase):
    def test_reactions_of_the_author_and_maintainers_are_kept(self):
        api = FakeApi(
            reviews=[ours()],
            inline=[finding(reactions={"total_count": 4})],
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint"},
            reactions={
                "pulls/comments/10": [
                    reaction("author", "-1"),
                    reaction("maint", "+1"),
                    reaction("drive-by", "heart"),
                    reaction("pytorchbot", "eyes", "Bot"),
                ],
                "issues/comments/20": [reaction("author", "confused")],
            },
        )
        out = fpc.collect(api, REPO, PR)
        by_id = {i["id"]: i for i in out["items"]}
        self.assertEqual(
            by_id[10]["reactions"],
            [
                {"author": "author", "role": "author", "content": "-1"},
                {"author": "maint", "role": "maintainer", "content": "+1"},
            ],
        )
        self.assertEqual(
            by_id[20]["reactions"],
            [{"author": "author", "role": "author", "content": "confused"}],
        )

    def test_unreacted_comments_cost_no_call_and_carry_no_field(self):
        api = FakeApi(
            reviews=[ours()],
            inline=[finding(reactions={"total_count": 0})],
            issue=[comment("maint", id=20)],
            maintainers={"maint"},
        )
        out = fpc.collect(api, REPO, PR)
        self.assertFalse([c for c in api.calls if "/reactions" in c])
        self.assertFalse([i for i in out["items"] if "reactions" in i])

    def test_a_comment_deleted_since_listing_is_left_out(self):
        api = FakeApi(
            reviews=[ours()],
            inline=[finding(reactions={"total_count": 1})],
            issue=[comment("maint", id=20)],
            maintainers={"maint"},
        )
        api.deleted.add(f"repos/{REPO}/pulls/comments/10/reactions")
        out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [20])
        self.assertFalse(out["truncated"])

    def test_only_the_newest_reacted_comments_are_looked_up(self):
        issue = [
            comment(
                "maint", id=20 + n, when=f"2026-01-0{n}", reactions={"total_count": 1}
            )
            for n in range(1, 4)
        ]
        reactions = {
            f"issues/comments/{20 + n}": [reaction("author", "+1")] for n in range(1, 4)
        }
        api = FakeApi(issue=issue, maintainers={"maint"}, reactions=reactions)
        with mock.patch.object(fpc, "MAX_REACTION_LOOKUPS", 2):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"] if "reactions" in i], [22, 23])
        self.assertEqual(len([c for c in api.calls if "/reactions" in c]), 2)

    def test_only_drive_by_reactions_leave_no_field(self):
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint"},
            reactions={"issues/comments/20": [reaction("drive-by", "-1")]},
        )
        (item,) = fpc.collect(api, REPO, PR)["items"]
        self.assertNotIn("reactions", item)


class TestFailsClosed(unittest.TestCase):
    def test_an_api_error_fails_the_run(self):
        def broken(endpoint):
            raise RuntimeError("gh api failed: HTTP 502")

        with (
            tempfile.TemporaryDirectory() as td,
            mock.patch.object(fpc, "gh_api", broken),
        ):
            out = Path(td) / "c.json"
            with mock.patch("sys.stderr"):
                rc = fpc.main(["--repo", REPO, "--pr", str(PR), "--out", str(out)])
            self.assertEqual(rc, 1)
            self.assertFalse(out.exists())

    def test_a_malformed_permission_response_fails_the_run(self):
        api = FakeApi(issue=[comment("maint")])
        real = api.__call__

        def no_perms(endpoint):
            return {} if "/collaborators/" in endpoint else real(endpoint)

        with self.assertRaises(ValueError):
            fpc.collect(no_perms, REPO, PR)

    def test_main_writes_the_conversation(self):
        api = FakeApi(issue=[comment("maint")], maintainers={"maint"})
        with tempfile.TemporaryDirectory() as td, mock.patch.object(fpc, "gh_api", api):
            out = Path(td) / "c.json"
            with mock.patch("sys.stdout"):
                self.assertEqual(
                    fpc.main(["--repo", REPO, "--pr", str(PR), "--out", str(out)]), 0
                )
            self.assertEqual(json.loads(out.read_text())["items"][0]["author"], "maint")

    def test_main_writes_a_single_line(self):
        # The workflow hands the file over as a `name=value` job output.
        body = "a\nb\r\n"
        api = FakeApi(issue=[comment("maint", body=body)], maintainers={"maint"})
        with tempfile.TemporaryDirectory() as td, mock.patch.object(fpc, "gh_api", api):
            out = Path(td) / "c.json"
            with mock.patch("sys.stdout"):
                fpc.main(["--repo", REPO, "--pr", str(PR), "--out", str(out)])
            text = out.read_text()
            self.assertNotIn("\n", text)
            self.assertNotIn("\r", text)
            self.assertEqual(json.loads(text)["items"][0]["body"], body)

    def test_gh_404_and_other_errors_are_told_apart(self):
        with tempfile.TemporaryDirectory() as td:
            gh = Path(td) / "gh"
            gh.write_text(
                '#!/bin/bash\necho "gh: Not Found (HTTP $CODE)" >&2\nexit 1\n'
            )
            gh.chmod(0o755)
            path = f"{td}{os.pathsep}{os.environ['PATH']}"
            with mock.patch.dict(os.environ, {"PATH": path, "CODE": "404"}):
                with self.assertRaises(fpc.NotFound):
                    fpc.gh_api("repos/o/r/collaborators/x/permission")
            with mock.patch.dict(os.environ, {"PATH": path, "CODE": "502"}):
                with self.assertRaises(RuntimeError) as cm:
                    fpc.gh_api("repos/o/r/collaborators/x/permission")
                self.assertNotIsInstance(cm.exception, fpc.NotFound)


if __name__ == "__main__":
    run_this_suite()
