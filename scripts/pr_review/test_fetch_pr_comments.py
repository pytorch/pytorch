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
ARGV = ["--repo", REPO, "--pr", str(PR), "--head-sha", "abc"]


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
        empty = len(fpc.dumps(fpc.collect(FakeApi(), REPO, PR)))
        for limit in range(empty, 4000, 7):
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
            reviews=[
                {**ours(rid=49), "submitted_at": "0"},
                ours(),
                ours(rid=51, body="not ours"),
            ],
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
                (50, fpc.REVIEW_AUTHOR, fpc.AUTOMATED),
                (10, fpc.REVIEW_AUTHOR, fpc.AUTOMATED),
                (13, "author", "author"),
            ],
        )
        self.assertEqual(out["items"][2]["in_reply_to_id"], 10)
        # Of the review bodies, only the latest marked one comes in.
        self.assertEqual(out["items"][0]["kind"], "review")
        self.assertFalse([c for c in api.calls if "/collaborators/" in c])

    def test_the_marker_and_author_match_post_review(self):
        import post_review

        self.assertEqual(fpc.MARKER, post_review.MARKER)
        self.assertEqual(fpc.REVIEW_AUTHOR, post_review.DEFAULT_AUTHOR)

    def test_the_author_follows_post_reviews_override(self):
        app = "review-app[bot]"
        api = FakeApi(
            reviews=[ours(), {**ours(rid=60), "user": user(app, "Bot")}],
            inline=[
                finding(),
                comment(app, kind="Bot", id=11, pull_request_review_id=60),
            ],
        )
        with mock.patch.dict(os.environ, {"REVIEW_AUTHOR": app}):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual(
            [(i["id"], i["author"], i["role"]) for i in out["items"]],
            [(60, app, fpc.AUTOMATED), (11, app, fpc.AUTOMATED)],
        )

    def test_older_verdicts_stay_out_even_from_a_user_account(self):
        acct = "review-user"
        older = {**ours(rid=49), "user": user(acct), "submitted_at": "0"}
        api = FakeApi(
            reviews=[older, {**ours(), "user": user(acct)}], maintainers={acct}
        )
        with mock.patch.dict(os.environ, {"REVIEW_AUTHOR": acct}):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual(
            [(i["id"], i["role"]) for i in out["items"]], [(50, fpc.AUTOMATED)]
        )

    def test_inline_comments_say_which_commit_and_review_they_belong_to(self):
        api = FakeApi(
            reviews=[ours()],
            inline=[finding(original_commit_id="abc")],
        )
        api.pull = {**api.pull, "head": {"sha": "moved-since"}}
        out = fpc.collect(api, REPO, PR, "def")
        self.assertEqual(out["head_sha"], "def")
        (item,) = [i for i in out["items"] if i["kind"] == "review_comment"]
        self.assertEqual(
            (item["pull_request_review_id"], item["original_commit_id"]), (50, "abc")
        )

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
        self.assertEqual(len(full["items"]), 11)
        with mock.patch.object(fpc, "MAX_TOTAL_BYTES", len(fpc.dumps(full)) - 500):
            out = fpc.collect(api, REPO, PR)
        roles = [i["role"] for i in out["items"]]
        self.assertEqual(roles.count("maintainer"), 5)
        # The review body and the four newest inline findings.
        self.assertEqual(roles.count(fpc.AUTOMATED), 5)
        # The oldest finding went, and what is left is still in time order.
        self.assertNotIn(101, [i["id"] for i in out["items"]])
        self.assertEqual(out["omitted_items"], 1)
        times = [i["created_at"] for i in out["items"]]
        self.assertEqual(times, sorted(times))

    def test_answered_findings_are_kept_before_unanswered_ones(self):
        inline = [
            finding(cid=10, when="2026-01-01", body="old"),
            comment("author", id=11, in_reply_to_id=10, when="2026-01-02"),
            finding(cid=12, when="2026-01-03", reactions={"total_count": 1}),
        ] + [
            finding(cid=100 + i, when=f"2026-02-{i:02d}", body="z" * 1000)
            for i in range(1, 6)
        ]
        api = FakeApi(
            inline=inline,
            reviews=[ours()],
            reactions={"pulls/comments/12": [reaction("author", "-1")]},
        )
        full = fpc.collect(api, REPO, PR)
        with mock.patch.object(fpc, "MAX_TOTAL_BYTES", len(fpc.dumps(full)) - 2500):
            out = fpc.collect(api, REPO, PR)
        ids = [i["id"] for i in out["items"]]
        # The two oldest unanswered findings went; the answered ones stayed.
        self.assertEqual(ids, [50, 10, 11, 12, 103, 104, 105])

    def test_the_latest_review_body_has_its_own_cap(self):
        body = fpc.MARKER + "\n" + "x" * fpc.MAX_REVIEW_BODY
        api = FakeApi(reviews=[ours(body=body)])
        (item,) = fpc.collect(api, REPO, PR)["items"]
        self.assertEqual(item["role"], fpc.AUTOMATED)
        self.assertEqual(len(item["body"]), fpc.MAX_REVIEW_BODY + len(" [truncated]"))

    def test_a_kept_reply_keeps_the_finding_it_answers(self):
        issue = [
            comment("maint", body="y" * 3000, when=f"2026-01-{i:02d}")
            for i in range(1, 6)
        ]
        api = FakeApi(
            issue=issue,
            inline=[
                finding(cid=10, when="2025-12-01", body="z" * 3000),
                comment("author", id=11, in_reply_to_id=10, when="2026-02-01"),
            ],
            reviews=[ours()],
            maintainers={"maint"},
        )
        full = fpc.collect(api, REPO, PR)
        with mock.patch.object(fpc, "MAX_TOTAL_BYTES", len(fpc.dumps(full)) - 2000):
            out = fpc.collect(api, REPO, PR)
        ids = [i["id"] for i in out["items"]]
        self.assertIn(11, ids)
        self.assertIn(10, ids)
        self.assertTrue(out["truncated"])

    def test_a_finding_that_does_not_fit_does_not_hide_smaller_ones(self):
        big = fpc.MARKER + "\n" + "x" * 9000
        api = FakeApi(
            reviews=[ours(body=big)],
            inline=[
                finding(cid=10 + i, when=f"2026-01-0{i}", body="small")
                for i in range(1, 4)
            ],
        )
        full = fpc.collect(api, REPO, PR)
        with mock.patch.object(fpc, "MAX_TOTAL_BYTES", len(fpc.dumps(full)) - 5000):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [11, 12, 13])

    def test_an_older_reviews_answered_finding_is_kept(self):
        api = FakeApi(
            reviews=[{**ours(rid=49), "submitted_at": "0"}, ours()],
            inline=[
                finding(cid=10, rid=49, when="2026-01-01"),
                comment("author", id=11, in_reply_to_id=10, when="2026-01-02"),
            ],
        )
        ids = [i["id"] for i in fpc.collect(api, REPO, PR)["items"]]
        self.assertEqual(ids, [50, 10, 11])

    def test_a_drive_by_reply_does_not_make_a_finding_answered(self):
        inline = [
            finding(cid=10, when="2026-01-01", body="z" * 2000),
            comment("drive-by", id=11, in_reply_to_id=10, when="2026-01-02"),
            finding(cid=12, when="2026-01-03", body="z" * 2000),
        ]
        api = FakeApi(inline=inline, reviews=[ours()])
        full = fpc.collect(api, REPO, PR)
        with mock.patch.object(fpc, "MAX_TOTAL_BYTES", len(fpc.dumps(full)) - 1500):
            out = fpc.collect(api, REPO, PR)
        # Unanswered, so the newer finding is kept, not the one the stranger answered.
        self.assertEqual([i["id"] for i in out["items"]], [50, 12])

    def test_a_reply_whose_finding_does_not_fit_is_kept_alone(self):
        api = FakeApi(
            issue=[comment("maint", id=20, when="2026-01-05", body="older")],
            inline=[
                finding(cid=10, when="2025-12-01", body="z" * 3000),
                comment("maint", id=11, in_reply_to_id=10, when="2026-02-01"),
            ],
            reviews=[ours()],
            maintainers={"maint"},
        )
        full = fpc.collect(api, REPO, PR)
        with mock.patch.object(fpc, "MAX_TOTAL_BYTES", len(fpc.dumps(full)) - 2500):
            out = fpc.collect(api, REPO, PR)
        ids = [i["id"] for i in out["items"]]
        self.assertIn(11, ids)
        self.assertIn(20, ids)
        self.assertNotIn(10, ids)

    def test_the_latest_review_is_the_highest_id(self):
        api = FakeApi(
            reviews=[
                {**ours(rid=50, body=fpc.MARKER + "\nolder"), "submitted_at": "5"},
                {**ours(rid=51, body=fpc.MARKER + "\nnewer"), "submitted_at": "5"},
            ]
        )
        (item,) = fpc.collect(api, REPO, PR)["items"]
        self.assertEqual(item["id"], 51)

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
        self.assertFalse(out["reactions_truncated"])

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

    def test_a_failed_lookup_keeps_the_comment_with_unknown_reactions(self):
        api = FakeApi(
            reviews=[ours()],
            inline=[finding(reactions={"total_count": 1})],
            issue=[comment("maint", id=20)],
            maintainers={"maint"},
        )
        api.deleted.add(f"repos/{REPO}/pulls/comments/10/reactions")
        out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [50, 10, 20])
        self.assertNotIn("reactions", out["items"][1])
        self.assertTrue(out["reactions_truncated"])
        self.assertFalse(out["truncated"])

    def test_any_lookup_failure_keeps_the_comment_and_the_run(self):
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint"},
        )
        real = api.__call__

        def throttled(endpoint):
            if "/reactions" in endpoint:
                raise RuntimeError("gh api failed: HTTP 403 secondary rate limit")
            return real(endpoint)

        out = fpc.collect(throttled, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [20])
        self.assertTrue(out["reactions_truncated"])

    def test_a_failed_reactor_permission_check_keeps_the_comment(self):
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint"},
            reactions={"issues/comments/20": [reaction("stranger", "-1")]},
        )
        real = api.__call__

        def flaky(endpoint):
            if "/collaborators/stranger/" in endpoint:
                raise RuntimeError("gh api failed: HTTP 502")
            return real(endpoint)

        out = fpc.collect(flaky, REPO, PR)
        (item,) = out["items"]
        self.assertNotIn("reactions", item)
        self.assertTrue(out["reactions_truncated"])

    def test_a_failed_lookup_does_not_inherit_the_previous_reactions(self):
        api = FakeApi(
            issue=[
                comment(
                    "maint", id=20, when="2026-01-01", reactions={"total_count": 1}
                ),
                comment(
                    "maint", id=21, when="2026-01-02", reactions={"total_count": 1}
                ),
            ],
            maintainers={"maint"},
            reactions={"issues/comments/21": [reaction("author", "+1")]},
        )
        api.deleted.add(f"repos/{REPO}/issues/comments/20/reactions")
        by_id = {i["id"]: i for i in fpc.collect(api, REPO, PR)["items"]}
        self.assertIn("reactions", by_id[21])
        self.assertNotIn("reactions", by_id[20])

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
        self.assertTrue(out["reactions_truncated"])

    def test_permission_checks_for_new_reactors_are_bounded(self):
        crowd = [reaction(f"u{n}", "+1") for n in range(10)]
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 12})],
            maintainers={"maint", "u0", "u9"},
            reactions={
                "issues/comments/20": [
                    *crowd,
                    reaction("maint", "-1"),
                    reaction("author", "eyes"),
                ]
            },
        )
        with mock.patch.object(fpc, "MAX_REACTOR_CHECKS", 3):
            out = fpc.collect(api, REPO, PR)
        (item,) = out["items"]
        self.assertTrue(out["reactions_truncated"])
        checks = [c for c in api.calls if "/collaborators/" in c]
        # maint, from its comment, plus three reactors.
        self.assertEqual(len(checks), 4)
        self.assertEqual(
            [(r["author"], r["role"]) for r in item["reactions"]],
            [("u0", "maintainer"), ("maint", "maintainer"), ("author", "author")],
        )

    def test_only_drive_by_reactions_leave_no_field(self):
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint"},
            reactions={"issues/comments/20": [reaction("drive-by", "-1")]},
        )
        out = fpc.collect(api, REPO, PR)
        self.assertNotIn("reactions", out["items"][0])
        self.assertFalse(out["reactions_truncated"])

    def test_a_full_first_page_is_reported_missing(self):
        # The listing's count can be stale: a full page is what says more may exist.
        page = [reaction("author", "-1")] + [
            reaction(f"bot{n}", "+1", "Bot") for n in range(fpc.PER_PAGE - 1)
        ]
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint"},
            reactions={"issues/comments/20": page},
        )
        out = fpc.collect(api, REPO, PR)
        self.assertEqual(len(out["items"][0]["reactions"]), 1)
        self.assertTrue(out["reactions_truncated"])

    def reacted_comments(self, n):
        issue = [
            comment(
                "maint", id=20 + k, when=f"2026-01-0{k}", reactions={"total_count": 1}
            )
            for k in range(1, n + 1)
        ]
        reactions = {
            f"issues/comments/{20 + k}": [reaction("author", "+1")]
            for k in range(1, n + 1)
        }
        return FakeApi(issue=issue, maintainers={"maint"}, reactions=reactions)

    def test_exactly_the_lookup_allowance_is_not_reported(self):
        with mock.patch.object(fpc, "MAX_REACTION_LOOKUPS", 2):
            out = fpc.collect(self.reacted_comments(2), REPO, PR)
        self.assertFalse(out["reactions_truncated"])

    def test_an_unchecked_reactor_marks_only_its_own_comment(self):
        # The newest finding exhausts the checks and is then left out; the older
        # comment's author reaction needs no check, so nothing kept is partial.
        api = FakeApi(
            reviews=[ours()],
            inline=[finding(reactions={"total_count": 2})],
            issue=[comment("maint", id=20, when="2025", reactions={"total_count": 1})],
            maintainers={"maint"},
            reactions={
                "pulls/comments/10": [reaction("x1", "+1"), reaction("x2", "+1")],
                "issues/comments/20": [reaction("author", "+1")],
            },
        )
        with (
            mock.patch.object(fpc, "MAX_REACTOR_CHECKS", 1),
            mock.patch.object(fpc, "MAX_ITEMS", 1),
        ):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [20])
        self.assertFalse(out["reactions_truncated"])

    def test_a_partial_comment_left_out_is_not_reported(self):
        with (
            mock.patch.object(fpc, "MAX_REACTION_LOOKUPS", 2),
            mock.patch.object(fpc, "MAX_ITEMS", 2),
        ):
            out = fpc.collect(self.reacted_comments(3), REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [22, 23])
        self.assertFalse(out["reactions_truncated"])


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
                rc = fpc.main(ARGV + ["--out", str(out)])
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
                self.assertEqual(fpc.main(ARGV + ["--out", str(out)]), 0)
            written = json.loads(out.read_text())
            self.assertEqual(written["items"][0]["author"], "maint")
            self.assertEqual(written["head_sha"], "abc")

    def test_main_requires_the_commit_under_review(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            fpc.main(["--repo", REPO, "--pr", str(PR), "--out", "c.json"])

    def test_main_writes_a_single_line(self):
        # The workflow hands the file over as a `name=value` job output.
        body = "a\nb\r\n"
        api = FakeApi(issue=[comment("maint", body=body)], maintainers={"maint"})
        with tempfile.TemporaryDirectory() as td, mock.patch.object(fpc, "gh_api", api):
            out = Path(td) / "c.json"
            with mock.patch("sys.stdout"):
                fpc.main(ARGV + ["--out", str(out)])
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
