#!/usr/bin/env python3
"""Tests for the maintainer-conversation fetcher run by `prepare`.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import io
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
        }
        # Keyed like the REST routes: "issues/comments/20", "pulls/comments/10",
        # "reviews/50". `deleted` takes the same keys as full reaction routes.
        self.reactions = dict(reactions or {})
        self.graphql_ids = []
        self.maintainers = set(maintainers)
        self.unknown = set(unknown)
        self.deleted = set()
        self.calls = []

    def __call__(self, endpoint, body=None):
        self.calls.append(endpoint)
        if endpoint == "graphql":
            return self.graphql(body)
        if endpoint == f"repos/{REPO}/pulls/{PR}":
            return self.pull
        base, _, query = endpoint.partition("?")
        if base in self.deleted:
            raise fpc.NotFound(endpoint)
        if base in self.pages:
            page = int(query.split("&page=", 1)[1]) if "&page=" in query else 1
            items = self.pages[base]
            prefix = NODE_PREFIX[base]
            return [
                {"node_id": f"{prefix}{o.get('id')}", **o}
                for o in items[(page - 1) * fpc.PER_PAGE : page * fpc.PER_PAGE]
            ]
        login = base.split("/collaborators/", 1)[1].split("/", 1)[0]
        if login in self.unknown:
            raise fpc.NotFound(endpoint)
        perms = {p: False for p in ("admin", "maintain", "push", "triage", "pull")}
        perms["triage"] = login in self.maintainers
        return {"user": {"permissions": perms}}

    def graphql(self, body):
        self.assertions(body)
        ids = body["variables"]["ids"]
        self.graphql_ids.append(list(ids))
        nodes = []
        for nid in ids:
            prefix, _, num = nid.partition("_")
            key = f"{NODE_ROUTE[prefix + '_']}/{num}"
            if f"repos/{REPO}/{key}/reactions" in self.deleted:
                nodes.append(None)
                continue
            rs = self.reactions.get(key, [])
            nodes.append(
                {
                    "id": nid,
                    "reactions": {
                        "totalCount": len(rs),
                        "nodes": [
                            {
                                "content": GRAPHQL_CONTENT[r["content"]],
                                "user": {"login": r["user"]["login"]},
                            }
                            for r in rs[:100]
                        ],
                    },
                }
            )
        errors = [
            {"type": "NOT_FOUND", "message": "not found", "path": ["nodes", n]}
            for n, node in enumerate(nodes)
            if node is None
        ]
        return {"data": {"nodes": nodes}, **({"errors": errors} if errors else {})}

    @staticmethod
    def assertions(body):
        assert body["query"] == fpc.REACTIONS_QUERY
        assert 0 < len(body["variables"]["ids"]) <= 100


# REST listing -> node id prefix, and node id prefix -> REST route.
NODE_PREFIX = {
    f"repos/{REPO}/issues/{PR}/comments": "IC_",
    f"repos/{REPO}/pulls/{PR}/comments": "PRRC_",
    f"repos/{REPO}/pulls/{PR}/reviews": "PRR_",
}
NODE_ROUTE = {"IC_": "issues/comments", "PRRC_": "pulls/comments", "PRR_": "reviews"}
GRAPHQL_CONTENT = {v: k for k, v in fpc.CONTENT.items()}


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

    def test_the_commit_the_latest_review_saw_is_handed_on(self):
        old, new = "1" * 40, "2" * 40
        api = FakeApi(
            reviews=[{**ours(50), "commit_id": old}, {**ours(51), "commit_id": new}]
        )
        self.assertEqual(
            fpc.collect(api, REPO, PR, "3" * 40)["previous_review_sha"], new
        )
        # A re-run on the same commit, no earlier review, or a malformed id: nothing to compare.
        self.assertEqual(fpc.collect(api, REPO, PR, new)["previous_review_sha"], "")
        self.assertEqual(
            fpc.collect(FakeApi(), REPO, PR, new)["previous_review_sha"], ""
        )
        bad = FakeApi(reviews=[{**ours(), "commit_id": "not-a-sha"}])
        self.assertEqual(fpc.collect(bad, REPO, PR, new)["previous_review_sha"], "")

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

    def test_a_replys_finding_never_takes_a_persons_place(self):
        issue = [
            comment("maint", id=20 + i, body="y" * 3000, when=f"2026-01-{i:02d}")
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
        # Every person's comment; the reply's long finding is skipped and the
        # small latest-review body still fits.
        self.assertEqual(ids, [50, 21, 22, 23, 24, 25, 11])
        self.assertTrue(out["truncated"])

    def test_a_replied_to_finding_comes_first_among_findings(self):
        big = fpc.MARKER + "\n" + "x" * 3000
        api = FakeApi(
            inline=[
                finding(cid=10, when="2025-12-01", body="z" * 2500),
                comment("author", id=11, in_reply_to_id=10, when="2026-02-01"),
            ],
            reviews=[{**ours(body=big), "submitted_at": "2026-03-01"}],
        )
        full = fpc.collect(api, REPO, PR)
        # Room for the reply and either the finding or the body, not both.
        with mock.patch.object(fpc, "MAX_TOTAL_BYTES", len(fpc.dumps(full)) - 2000):
            out = fpc.collect(api, REPO, PR)
        # The newer latest-review body would come first in its group.
        self.assertEqual([i["id"] for i in out["items"]], [10, 11])

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
        # Only the latest review's body, whose REST listing has no count.
        self.assertEqual(api.graphql_ids, [["PRR_50"]])
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

        def throttled(endpoint, body=None):
            if endpoint == "graphql":
                raise RuntimeError("gh api failed: HTTP 403 secondary rate limit")
            return real(endpoint, body)

        with (
            mock.patch("sys.stderr", new_callable=io.StringIO) as err,
            mock.patch.object(fpc.time, "sleep") as sleep,
        ):
            out = fpc.collect(throttled, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [20])
        self.assertTrue(out["reactions_truncated"])
        self.assertEqual(err.getvalue().count("::warning::could not read reactions"), 3)
        self.assertEqual([c.args for c in sleep.call_args_list], [(1,), (2,)])

    def test_a_failed_reactions_call_is_retried(self):
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint"},
            reactions={"issues/comments/20": [reaction("author", "+1")]},
        )
        failures = [RuntimeError("gh api graphql failed: HTTP 502")]

        def flaky(endpoint, body=None):
            if endpoint == "graphql" and failures:
                raise failures.pop()
            return api(endpoint, body)

        with mock.patch("sys.stderr"), mock.patch.object(fpc.time, "sleep"):
            out = fpc.collect(flaky, REPO, PR)
        self.assertEqual(out["items"][0]["reactions"][0]["content"], "+1")
        self.assertFalse(out["reactions_truncated"])

    def test_the_latest_review_bodys_reactions_are_read(self):
        api = FakeApi(
            reviews=[ours(rid=40), ours(rid=50)],
            maintainers={"maint"},
            reactions={
                "reviews/40": [reaction("author", "+1")],
                "reviews/50": [reaction("maint", "-1"), reaction("drive-by", "+1")],
            },
        )
        (item,) = fpc.collect(api, REPO, PR)["items"]
        self.assertEqual(item["id"], 50)
        self.assertEqual(
            item["reactions"],
            [{"author": "maint", "role": "maintainer", "content": "-1"}],
        )
        self.assertEqual(api.graphql_ids, [["PRR_50"]])

    def test_app_reactions_are_not_people(self):
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint", "some-app[bot]"},
            reactions={"issues/comments/20": [reaction("some-app[bot]", "-1", "Bot")]},
        )
        out = fpc.collect(api, REPO, PR)
        self.assertNotIn("reactions", out["items"][0])
        self.assertFalse([c for c in api.calls if "some-app" in c])

    def test_a_failed_reactor_permission_check_keeps_the_comment(self):
        api = FakeApi(
            issue=[comment("maint", id=20, reactions={"total_count": 1})],
            maintainers={"maint"},
            reactions={"issues/comments/20": [reaction("stranger", "-1")]},
        )
        real = api.__call__

        def flaky(endpoint, body=None):
            if "/collaborators/stranger/" in endpoint:
                raise RuntimeError("gh api failed: HTTP 502")
            return real(endpoint, body)

        with mock.patch("sys.stderr"):
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

    def test_earlier_findings_get_the_reaction_lookups_first(self):
        issue = [
            comment(
                "maint", id=20 + n, when=f"2026-01-0{n}", reactions={"total_count": 1}
            )
            for n in range(1, 4)
        ]
        reactions = {
            f"issues/comments/{20 + n}": [reaction("author", "+1")] for n in range(1, 4)
        }
        reactions["pulls/comments/10"] = [reaction("author", "-1")]
        api = FakeApi(
            issue=issue,
            inline=[finding(when="2025-12-01", reactions={"total_count": 1})],
            reviews=[ours()],
            maintainers={"maint"},
            reactions=reactions,
        )
        # The finding and the latest review's body, then the newest comment.
        with mock.patch.object(fpc, "MAX_REACTION_LOOKUPS", 3):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"] if "reactions" in i], [10, 23])
        self.assertEqual(api.graphql_ids, [["PRRC_10", "PRR_50", "IC_23"]])

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
        self.assertEqual(api.graphql_ids, [["IC_23", "IC_22"]])
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

    def test_reactions_past_the_first_100_are_reported_missing(self):
        page = [reaction("author", "-1")] + [
            reaction(f"bot{n}", "+1", "Bot") for n in range(100)
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
        # The finding is looked up first, exhausts the checks, and is then left
        # out for the review body; the comment looked up after it has only the
        # author's reaction, which needs no check, so nothing kept is partial.
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
            mock.patch.object(fpc, "MAX_ITEMS", 2),
        ):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [50, 20])
        self.assertEqual(api.graphql_ids, [["PRRC_10", "PRR_50", "IC_20"]])
        self.assertFalse(out["reactions_truncated"])

    def test_a_partial_comment_left_out_is_not_reported(self):
        with (
            mock.patch.object(fpc, "MAX_REACTION_LOOKUPS", 2),
            mock.patch.object(fpc, "MAX_ITEMS", 2),
        ):
            out = fpc.collect(self.reacted_comments(3), REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [22, 23])
        self.assertFalse(out["reactions_truncated"])

    def test_reactions_are_read_100_comments_per_call(self):
        issue = [
            comment(
                "maint",
                id=1000 + k,
                when=f"2026-01-01T{k:05d}",
                reactions={"total_count": 1},
            )
            for k in range(150)
        ]
        api = FakeApi(issue=issue, maintainers={"maint"})
        out = fpc.collect(api, REPO, PR)
        self.assertEqual([len(ids) for ids in api.graphql_ids], [100, 50])
        self.assertFalse(out["reactions_truncated"])

    def test_a_finding_with_only_drive_by_reactions_is_selected_again(self):
        # Kept at first as possibly answered, the newer finding turns out to
        # have no reaction that counts, so the answered one takes its place.
        api = FakeApi(
            reviews=[ours()],
            inline=[
                finding(cid=11, when="2026-01-02", reactions={"total_count": 1}),
                finding(cid=12, when="2026-01-03", reactions={"total_count": 1}),
            ],
            reactions={
                "pulls/comments/11": [reaction("author", "-1")],
                "pulls/comments/12": [reaction("drive-by", "+1")],
            },
        )
        with mock.patch.object(fpc, "MAX_ITEMS", 2):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [50, 11])
        self.assertTrue(out["truncated"])
        self.assertFalse(out["reactions_truncated"])

    def test_findings_settle_before_people_spend_the_lookups(self):
        # The newer finding has only a drive-by reaction; finding that out must
        # free its place for the older, answered one before the remaining
        # lookup goes to a person's comment.
        api = FakeApi(
            reviews=[ours()],
            inline=[
                finding(cid=11, when="2026-01-02", reactions={"total_count": 1}),
                finding(cid=12, when="2026-01-03", reactions={"total_count": 1}),
            ],
            issue=[
                comment("maint", id=20, when="2026-01-04", reactions={"total_count": 1})
            ],
            maintainers={"maint"},
            reactions={
                "pulls/comments/11": [reaction("author", "-1")],
                "pulls/comments/12": [reaction("drive-by", "+1")],
                "issues/comments/20": [reaction("author", "+1")],
            },
        )
        with (
            mock.patch.object(fpc, "MAX_ITEMS", 2),
            mock.patch.object(fpc, "MAX_REACTION_LOOKUPS", 2),
        ):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [11, 20])
        self.assertIn("reactions", out["items"][0])
        self.assertTrue(out["reactions_truncated"])

    def test_each_lookup_is_followed_by_a_new_selection(self):
        # A is answered by a kept reply, B by the author's reaction, and C has
        # only a drive-by one. Once C turns out unanswered, B takes its place
        # and the last lookup goes to B, not to A, whose reply already counts.
        api = FakeApi(
            reviews=[ours()],
            inline=[
                finding(cid=11, when="2026-01-02", reactions={"total_count": 1}),
                finding(cid=12, when="2026-01-03", reactions={"total_count": 1}),
                finding(cid=13, when="2026-01-04", reactions={"total_count": 1}),
                comment("author", id=30, when="2026-01-05", in_reply_to_id=11),
            ],
            reactions={
                "pulls/comments/11": [reaction("author", "+1")],
                "pulls/comments/12": [reaction("author", "-1")],
                "pulls/comments/13": [reaction("drive-by", "+1")],
            },
        )
        with (
            mock.patch.object(fpc, "MAX_ITEMS", 3),
            mock.patch.object(fpc, "MAX_REACTION_LOOKUPS", 2),
        ):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [11, 12, 30])
        self.assertIn("reactions", out["items"][1])

    def test_omitted_findings_spend_no_reactor_checks(self):
        # People's comments overflow, so the finding is never a candidate and
        # its reactor is not checked; the one check left goes to the kept
        # comment's reactor.
        api = FakeApi(
            reviews=[ours()],
            inline=[finding(reactions={"total_count": 1})],
            issue=[
                comment("maint", id=20, when="2026-01-04"),
                comment(
                    "maint", id=21, when="2026-01-05", reactions={"total_count": 1}
                ),
            ],
            maintainers={"maint", "x1", "m2"},
            reactions={
                "pulls/comments/10": [reaction("x1", "+1")],
                "issues/comments/21": [reaction("m2", "-1")],
            },
        )
        with (
            mock.patch.object(fpc, "MAX_REACTOR_CHECKS", 1),
            mock.patch.object(fpc, "MAX_ITEMS", 1),
        ):
            out = fpc.collect(api, REPO, PR)
        (item,) = out["items"]
        self.assertEqual(item["id"], 21)
        self.assertEqual([r["author"] for r in item["reactions"]], ["m2"])
        self.assertFalse(out["reactions_truncated"])
        self.assertFalse(any(c.endswith("/x1/permission") for c in api.calls))

    def with_errors(self, api, errors, null_reaction=False):
        graphql = api.graphql

        def broken(body):
            resp = graphql(body)
            if null_reaction:
                resp["data"]["nodes"][0]["reactions"]["nodes"].append(None)
            return resp | {"errors": errors}

        api.graphql = broken
        return api

    def test_an_error_inside_a_node_marks_its_comment(self):
        api = self.with_errors(
            self.reacted_comments(2),
            [{"path": ["nodes", 0, "reactions", "nodes", 1, "user"]}],
            null_reaction=True,
        )
        with mock.patch.object(sys, "stderr", io.StringIO()):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([len(i["reactions"]) for i in out["items"]], [1, 1])
        self.assertTrue(out["reactions_truncated"])

    def test_an_error_in_an_omitted_comment_is_not_reported(self):
        # Node 1 is the older comment, which the item cap leaves out.
        api = self.with_errors(
            self.reacted_comments(2), [{"path": ["nodes", 1, "reactions"]}]
        )
        with (
            mock.patch.object(sys, "stderr", io.StringIO()),
            mock.patch.object(fpc, "MAX_ITEMS", 1),
        ):
            out = fpc.collect(api, REPO, PR)
        self.assertEqual([i["id"] for i in out["items"]], [22])
        self.assertFalse(out["reactions_truncated"])

    def test_an_error_tied_to_no_node_marks_every_comment(self):
        api = self.with_errors(self.reacted_comments(2), [{"message": "partial"}])
        with (
            mock.patch.object(sys, "stderr", io.StringIO()),
            mock.patch.object(fpc, "MAX_ITEMS", 1),
        ):
            out = fpc.collect(api, REPO, PR)
        self.assertTrue(out["reactions_truncated"])


class TestFailsClosed(unittest.TestCase):
    def test_an_api_error_fails_the_run(self):
        def broken(endpoint, body=None):
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

        def no_perms(endpoint, body=None):
            return {} if "/collaborators/" in endpoint else real(endpoint, body)

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

    def fake_gh(self, td, script):
        gh = Path(td) / "gh"
        gh.write_text("#!/bin/bash\n" + script)
        gh.chmod(0o755)
        return {"PATH": f"{td}{os.pathsep}{os.environ['PATH']}"}

    def test_a_graphql_call_sends_its_body_on_stdin(self):
        with tempfile.TemporaryDirectory() as td:
            env = self.fake_gh(td, 'echo "$@" > "$(dirname "$0")/args"; cat\n')
            with mock.patch.dict(os.environ, env):
                out = fpc.gh_api("graphql", {"query": "q", "variables": {"ids": ["a"]}})
            self.assertEqual(out, {"query": "q", "variables": {"ids": ["a"]}})
            self.assertEqual(
                (Path(td) / "args").read_text().split(),
                ["api", "graphql", "--method", "POST", "--input", "-"],
            )

    def test_graphql_data_beside_errors_is_returned(self):
        partial = '{"data":{"nodes":[null]},"errors":[{"message":"gone"}]}'
        with tempfile.TemporaryDirectory() as td:
            env = self.fake_gh(td, f"echo '{partial}'; echo 'gh: gone' >&2; exit 1\n")
            with mock.patch.dict(os.environ, env):
                out = fpc.gh_api("graphql", {"query": "q", "variables": {}})
            self.assertEqual(out["data"], {"nodes": [None]})
            env = self.fake_gh(td, "echo 'gh: HTTP 502' >&2; exit 1\n")
            with mock.patch.dict(os.environ, env), self.assertRaises(RuntimeError):
                fpc.gh_api("graphql", {"query": "q", "variables": {}})


if __name__ == "__main__":
    run_this_suite()
