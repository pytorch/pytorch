#!/usr/bin/env python3
"""Tests for posting the verdict as a native GitHub review.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import io
import json
import re
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parent))

import post_review  # noqa: E402

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import run_this_suite, TestTheSuiteIsWhole  # noqa: E402,F401
from extract_verdict import neutralize, neutralize_path  # noqa: E402
from post_review import (  # noqa: E402
    contained,
    defuse_bot_commands,
    GitHub,
    GitHubError,
    MARKER,
    run,
)


# Backfill writes pause between calls in production; not in tests.
post_review.BACKFILL_PAUSE = 0

SHA = "a" * 40
# The URL linter checks every URL literal in the tree; these resolve nowhere.
BLOB = f"https://github.com/o/r/blob/{SHA}"  # @lint-ignore
NEW_URL = "https://x/999"  # @lint-ignore
AUTHOR = "github-actions[bot]"
WORKFLOW = (
    Path(__file__).resolve().parents[2] / ".github/workflows/hardened-pr-review-run.yml"
)


def row(
    verdict="changes_requested", summary="One bug.", findings=None, status="succeeded"
):
    findings = (
        findings
        if findings is not None
        else [
            {
                "path": "torch/x.py",
                "line": 3,
                "severity": "major",
                "message": "Off by one.",
            }
        ]
    )
    return {
        "status": status,
        "verdict": verdict,
        "summary": summary,
        "extra": {"findings": json.dumps(findings)} if findings else {},
    }


def page(path):
    return int(re.search(r"[?&]page=(\d+)", path).group(1))


class FakeGitHub(GitHub):
    """Routes calls to canned state and records every write."""

    def __init__(
        self,
        head=SHA,
        labels=(),
        reviews=(),
        refuse_inline=False,
        comments=None,
    ):
        super().__init__("token", "o/r")
        self.head = head
        self.labels = [{"name": n} for n in labels]
        self.reviews = list(reviews)
        # One inline comment per review unless given: id = review id * 10.
        self.comments = (
            list(comments)
            if comments is not None
            else [comment(r["id"] * 10, r["id"]) for r in self.reviews]
        )
        self.refuse_inline = refuse_inline
        # Replies the REST listing never shows but the thread counts.
        self.hidden_replies: dict[int, int] = {}
        self.calls: list[tuple] = []

    def request(self, method, path, body=None, idempotent=None):
        self.calls.append((method, path, body))
        if method == "GET" and path == "/repos/o/r/pulls/1":
            return {"head": {"sha": self.head}}
        if method == "GET" and path.startswith("/repos/o/r/issues/1/labels"):
            return self.labels
        if method == "PUT" and path.endswith("/dismissals"):
            rid = int(path.split("/")[7])
            for r in self.reviews:
                if r["id"] == rid:
                    r["state"] = "DISMISSED"
            return {}
        if method == "GET" and path.startswith("/repos/o/r/pulls/1/comments"):
            return [dict(c) for c in self.comments] if page(path) == 1 else []
        if method == "DELETE" and path.startswith("/repos/o/r/pulls/comments/"):
            cid = int(path.split("/")[-1])
            if not any(c["id"] == cid for c in self.comments):
                raise GitHubError(404, "Not Found")
            self.comments = [c for c in self.comments if c["id"] != cid]
            return None
        if method == "PUT" and re.fullmatch(r"/repos/o/r/pulls/1/reviews/\d+", path):
            rid = int(path.split("/")[-1])
            for r in self.reviews:
                if r["id"] == rid:
                    r["body"] = body["body"]
            return {}
        if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews"):
            return self.reviews if page(path) == 1 else []
        if method == "POST" and path == "/repos/o/r/pulls/1/reviews":
            if self.refuse_inline and body.get("comments"):
                raise GitHubError(422, "line must be part of the diff")
            new = {
                "id": 999,
                "node_id": "N999",
                "html_url": NEW_URL,
                "user": {"login": AUTHOR},
                "body": body["body"],
                "state": {
                    "REQUEST_CHANGES": "CHANGES_REQUESTED",
                    "COMMENT": "COMMENTED",
                }[body["event"]],
            }
            self.reviews.append(new)
            self.comments += [
                {
                    **comment(9990 + i, 999, line=c["line"], path=c["path"]),
                    "body": c["body"],
                }
                for i, c in enumerate(body.get("comments") or [])
            ]
            return dict(new)  # a response, not the stored review
        if method == "POST" and path == "/graphql" and "reviewThreads" in body["query"]:
            nodes = [
                {
                    "id": f"T{c['id']}",
                    "isResolved": False,
                    "comments": {
                        "totalCount": 1
                        + self.hidden_replies.get(c["id"], 0)
                        + sum(
                            r.get("in_reply_to_id") == c["id"] for r in self.comments
                        ),
                        "nodes": [{"databaseId": c["id"]}],
                    },
                }
                for c in self.comments
                if not c.get("in_reply_to_id")
            ]
            return {
                "data": {
                    "repository": {
                        "pullRequest": {
                            "reviewThreads": {
                                "pageInfo": {"hasNextPage": False, "endCursor": None},
                                "nodes": nodes,
                            }
                        }
                    }
                }
            }
        if method == "POST" and path == "/graphql" and "isMinimized" in body["query"]:
            ids = body["variables"]["ids"]
            nodes = [
                {"id": r["node_id"], "isMinimized": r.get("minimized", False)}
                if not r.get("unreadable")
                else None
                for r in self.reviews
                if r["node_id"] in ids
            ]
            return {"data": {"nodes": nodes}}
        if (
            method == "POST"
            and path == "/graphql"
            and "minimizeComment" in body["query"]
        ):
            for r in self.reviews:
                if r["node_id"] == body["variables"]["id"]:
                    r["minimized"] = True
            return {"data": {}}
        if method == "POST" and path == "/graphql":
            return {"data": {}}
        return None

    def writes(self, method=None):
        return [
            c
            for c in self.calls
            if c[0] != "GET" and (method is None or c[0] == method)
        ]

    def posted(self):
        return [
            c[2] for c in self.calls if c[:2] == ("POST", "/repos/o/r/pulls/1/reviews")
        ]


def old_review(
    rid, author=AUTHOR, body=MARKER + "\nold", state="COMMENTED", commit="c" * 40
):
    return {
        "id": rid,
        "node_id": f"N{rid}",
        "user": {"login": author},
        "body": body,
        "state": state,
        "commit_id": commit,
    }


def comment(cid, review, author=AUTHOR, line=None, reply_to=None, path="torch/x.py"):
    return {
        "id": cid,
        "pull_request_review_id": review,
        "user": {"login": author},
        "body": "\U0001f534 Off by one." if reply_to is None else "No, it is right.",
        "path": path,
        "line": line,
        "in_reply_to_id": reply_to,
        "html_url": f"https://github.com/o/r/pull/1#discussion_r{cid}",  # @lint-ignore
    }


def deleted(gh):
    """Comment ids this run deleted, in order."""
    return [int(w[1].split("/")[-1]) for w in gh.writes("DELETE")]


def folds(gh):
    """Review ids whose body this run rewrote, in order."""
    return [
        int(w[1].split("/")[-1])
        for w in gh.writes("PUT")
        if re.fullmatch(r"/repos/o/r/pulls/1/reviews/\d+", w[1])
    ]


def minimized(gh):
    """Review node ids this run minimized, in order."""
    return [
        w[2]["variables"]["id"]
        for w in gh.writes()
        if w[1] == "/graphql" and "minimizeComment" in w[2]["query"]
    ]


def dismissed(gh):
    return [w[1].split("/")[7] for w in gh.writes() if w[1].endswith("/dismissals")]


def resolved(gh):
    """Thread ids this run resolved, in order."""
    return [
        w[2]["variables"]["id"]
        for w in gh.writes()
        if w[1] == "/graphql" and "resolveReviewThread" in w[2]["query"]
    ]


def go(gh, r=None):
    return run(gh, r if r is not None else row(), 1, SHA, "no automated review", AUTHOR)


class TestWhatIsPosted(unittest.TestCase):
    def test_findings_become_inline_comments_on_the_reviewed_commit(self):
        gh = FakeGitHub()
        go(gh)
        (body,) = gh.posted()
        self.assertEqual(body["commit_id"], SHA)
        self.assertEqual(body["event"], "REQUEST_CHANGES")
        self.assertTrue(body["body"].startswith(MARKER))
        (c,) = body["comments"]
        self.assertEqual((c["path"], c["line"], c["side"]), ("torch/x.py", 3, "RIGHT"))
        self.assertIn("Off by one.", c["body"])

    def test_inline_path_is_the_real_file_name(self):
        path = neutralize_path("node_modules/@babel/x_y.py")
        gh = FakeGitHub()
        go(
            gh,
            row(
                findings=[
                    {"path": path, "line": 1, "severity": "major", "message": "m"}
                ]
            ),
        )
        self.assertEqual(
            gh.posted()[0]["comments"][0]["path"], "node_modules/@babel/x_y.py"
        )

    def test_a_refused_anchor_falls_back_to_the_body(self):
        gh = FakeGitHub(refuse_inline=True)
        go(gh)
        first, second = gh.posted()
        self.assertIn("comments", first)
        self.assertNotIn("comments", second)
        self.assertIn(
            f"\n{BLOB}/torch/x.py#L3\n\n\U0001f534 Off by one.\n",
            second["body"],
        )

    def test_non_blocking_findings_fold_into_a_collapsed_section(self):
        findings = [
            {"path": "a.py", "line": 3, "severity": "major", "message": "Bug."},
            {"path": "b.py", "line": 7, "severity": "minor", "message": "Nit."},
            {"path": "c.py", "line": 9, "severity": "info", "message": "Note."},
        ]
        gh = FakeGitHub()
        go(gh, row(findings=findings))
        (body,) = gh.posted()
        self.assertEqual([c["path"] for c in body["comments"]], ["a.py"])
        text = body["body"]
        self.assertIn("\U0001f534 1 blocking finding is attached", text)
        details = text.split("<details>\n", 1)[1].split("</details>", 1)[0]
        self.assertTrue(
            details.startswith("<summary>\u26aa 2 non-blocking findings</summary>\n\n")
        )
        # Each permalink stands alone on its line, which is what makes GitHub
        # render the code it points at.
        self.assertIn(
            f"\n{BLOB}/b.py#L7\n\nNit.\n\n---\n\n{BLOB}/c.py#L9\n\nNote.\n",
            details,
        )
        self.assertTrue(details.endswith(f"/c.py#L9\n\nNote.\n\n"), details)
        self.assertNotIn("a.py", details)

    def test_only_non_blocking_findings_post_one_review_without_comments(self):
        minor = {"path": "b.py", "line": 7, "severity": "minor", "message": "Nit."}
        gh = FakeGitHub()
        go(gh, row(verdict="ready_for_human_review", findings=[minor]))
        (body,) = gh.posted()
        self.assertNotIn("comments", body)
        self.assertIn("<summary>\u26aa 1 non-blocking finding</summary>", body["body"])

    def test_changes_requested_on_minor_findings_alone_keeps_them_inline(self):
        minor = {"path": "b.py", "line": 7, "severity": "minor", "message": "Nit."}
        gh = FakeGitHub()
        go(gh, row(findings=[minor]))
        (body,) = gh.posted()
        self.assertEqual(body["event"], "REQUEST_CHANGES")
        self.assertEqual([c["path"] for c in body["comments"]], ["b.py"])
        self.assertNotIn("non-blocking", body["body"])

    def test_an_unclosed_fence_cannot_swallow_what_follows(self):
        fence = "x" + chr(10) + "```" + chr(10) + "y"
        findings = [
            {"path": "a.py", "line": 1, "severity": "minor", "message": fence},
            {"path": "b.py", "line": 2, "severity": "info", "message": "after"},
        ]
        gh = FakeGitHub()
        go(gh, row(verdict="ready_for_human_review", summary="s", findings=findings))
        text = gh.posted()[0]["body"]
        self.assertIn("x\n\\`\\`\\`\ny\n\n---\n", text)
        # Everything after the quote is outside it: the next permalink still
        # starts its own line, and the section still closes.
        self.assertIn(f"\n{BLOB}/b.py#L2\n", text)
        self.assertIn("\n</details>\n", text)

    def test_an_oversized_body_drops_findings_and_says_so(self):
        long = ("word " * 119 + "x")[:600]
        findings = [
            {"path": f"p{i}.py", "line": 1, "severity": "minor", "message": long}
            for i in range(25)
        ]
        with mock.patch("post_review.MAX_BODY", 5000):
            gh = FakeGitHub()
            go(gh, row(verdict="ready_for_human_review", findings=findings))
        text = gh.posted()[0]["body"]
        self.assertLessEqual(len(text), 5000)
        shown = text.count(BLOB)
        self.assertGreater(shown, 0)
        self.assertIn(f"{25 - shown} more findings did not fit", text)
        self.assertIn(f"<summary>\u26aa {shown} non-blocking findings</summary>", text)

    def test_the_footer_is_small_and_no_text_is_quoted(self):
        minor = {"path": "b.py", "line": 7, "severity": "minor", "message": "Nit."}
        gh = FakeGitHub()
        go(gh, row(verdict="ready_for_human_review", summary="Fine.", findings=[minor]))
        text = gh.posted()[0]["body"]
        self.assertTrue(text.endswith("</sub>"))
        self.assertIn("\n\nFine.\n\n", text)
        self.assertFalse([ln for ln in text.splitlines() if ln.startswith(">")])
        self.assertNotIn("**Minor**", text)

    def test_counts_are_pluralized(self):
        major = {"path": "a.py", "line": 1, "severity": "major", "message": "Bug."}
        gh = FakeGitHub()
        go(gh, row(findings=[major, dict(major, line=2)]))
        body = gh.posted()[0]
        self.assertIn("\U0001f534 2 blocking findings are attached", body["body"])
        self.assertEqual(body["comments"][0]["body"], "\U0001f534 Bug.")

    def test_the_mark_never_breaks_a_leading_block(self):
        table = "| a | b |" + chr(10) + "| --- | --- |"
        for message, first in (
            ("Plain.", "\U0001f534 Plain."),
            ("# Heading", "\U0001f534"),
            (table, "\U0001f534"),
            ("- item", "\U0001f534"),
            ("1. item", "\U0001f534"),
        ):
            f = {"path": "a.py", "line": 1, "severity": "major", "message": message}
            gh = FakeGitHub()
            go(gh, row(findings=[f]))
            lines = gh.posted()[0]["comments"][0]["body"].split(chr(10))
            self.assertEqual(lines[0], first, message)
            if first == "\U0001f534":
                self.assertEqual(lines[1:], ["", *message.split(chr(10))])

    def test_permalink_path_is_percent_encoded(self):
        path = neutralize_path("docs/a b/@x#1.md")
        minor = {"path": path, "line": 2, "severity": "minor", "message": "m"}
        gh = FakeGitHub()
        go(gh, row(verdict="ready_for_human_review", findings=[minor]))
        self.assertIn(
            f"\n{BLOB}/docs/a%20b/%40x%231.md#L2\n",
            gh.posted()[0]["body"],
        )

    def test_a_ready_verdict_is_a_comment_never_an_approval(self):
        gh = FakeGitHub()
        go(gh, row(verdict="ready_for_human_review"))
        self.assertEqual(gh.posted()[0]["event"], "COMMENT")

    def test_a_refused_request_changes_falls_back_to_a_comment(self):
        class NoRequestChanges(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if body and body.get("event") == "REQUEST_CHANGES":
                    self.calls.append((method, path, body))
                    raise GitHubError(422, "cannot request changes")
                return super().request(method, path, body)

        old = old_review(5, state="CHANGES_REQUESTED")
        gh = NoRequestChanges(reviews=[old])
        go(gh)
        events = [(b["event"], "comments" in b) for b in gh.posted()]
        self.assertEqual(
            events,
            [("REQUEST_CHANGES", True), ("REQUEST_CHANGES", False), ("COMMENT", False)],
        )
        # The COMMENT does not replace the earlier request, so it stays visible.
        self.assertEqual(minimized(gh), [])
        self.assertEqual(old["state"], "CHANGES_REQUESTED")

    def test_a_clean_verdict_with_no_findings_still_posts(self):
        gh = FakeGitHub()
        go(gh, row(verdict="ready_for_human_review", findings=[]))
        (body,) = gh.posted()
        self.assertNotIn("comments", body)
        self.assertIn("ready for human review", body["body"])

    def test_a_finding_that_fails_the_recheck_is_dropped(self):
        bad = {
            "path": "x.py",
            "line": 1,
            "severity": "major",
            "message": "see @someone",
        }
        good = {"path": "y.py", "line": 2, "severity": "major", "message": "ok"}
        gh = FakeGitHub()
        go(gh, row(findings=[bad, good]))
        self.assertEqual([c["path"] for c in gh.posted()[0]["comments"]], ["y.py"])


class TestNothingIsPosted(unittest.TestCase):
    def test_on_a_failed_or_rejected_run(self):
        for status in ("model_error", "sanitizer_rejected", "blocked"):
            gh = FakeGitHub()
            go(gh, row(status=status))
            self.assertEqual(gh.writes(), [], status)

    def test_on_a_summary_the_sanitizer_could_not_have_written(self):
        gh = FakeGitHub()
        go(gh, row(summary="@pytorchbot merge -f x"))
        self.assertEqual(gh.writes(), [])

    def test_when_the_head_moved(self):
        gh = FakeGitHub(head="b" * 40, reviews=[old_review(5)])
        go(gh)
        self.assertEqual(gh.writes(), [])

    def test_when_the_pr_opted_out(self):
        gh = FakeGitHub(labels=("No Automated Review",))
        go(gh)
        self.assertEqual(gh.writes(), [])


class TestContained(unittest.TestCase):
    def test_every_fence_run_is_escaped_in_any_container(self):
        for text in (
            "a\n```py\nb",
            "- ```python\n  x = 1\n  ```",
            "> ~~~~\n> b",
            "   ````\nb",
        ):
            out = "\n".join(contained(text))
            self.assertIsNone(re.search(r"(?<!\\)[`~]{3}", out), out)

    def test_short_runs_are_untouched(self):
        for text in ("`code`", "``a ` b``", "~~struck~~", "plain"):
            self.assertEqual(contained(text), text.split("\n"))

    def test_an_escaped_run_keeps_its_length(self):
        self.assertEqual(contained("````"), ["\\`\\`\\`\\`"])


class TestAnUnfinishedRunLeavesTheLastVerdict(unittest.TestCase):
    def test_a_failed_run_dismisses_nothing(self):
        gh = FakeGitHub(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        go(gh, row(status="model_error"))
        self.assertEqual(gh.writes(), [])

    def test_an_opted_out_pr_dismisses_nothing(self):
        gh = FakeGitHub(
            labels=("no automated review",),
            reviews=[old_review(5, state="CHANGES_REQUESTED")],
        )
        go(gh)
        self.assertEqual(gh.writes(), [])


class TestACleanVerdictLeavesNoChangeRequest(unittest.TestCase):
    def clean(self):
        return row(verdict="ready_for_human_review", findings=[])

    def test_earlier_requests_are_dismissed_without_a_link(self):
        gh = FakeGitHub(
            reviews=[
                old_review(5, state="CHANGES_REQUESTED"),
                old_review(6, state="CHANGES_REQUESTED", commit=SHA),
                old_review(7, state="CHANGES_REQUESTED", author="someone"),
            ]
        )
        self.assertEqual(go(gh, self.clean()), 0)
        self.assertEqual(dismissed(gh), ["5", "6"])
        for w in gh.writes("PUT"):
            if w[1].endswith("/dismissals"):
                self.assertNotIn("http", w[2]["message"])

    def test_a_request_that_cannot_be_dismissed_fails_the_step(self):
        class NeverDismiss(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if path.endswith("/dismissals"):
                    raise GitHubError(403, "forbidden")
                return super().request(method, path, body)

        gh = NeverDismiss(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        self.assertEqual(go(gh, self.clean()), 1)
        self.assertEqual(minimized(gh), [])

    def test_a_failed_post_still_withdraws_on_a_clean_verdict(self):
        class PostDown(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "POST" and path.endswith("/reviews"):
                    raise GitHubError(500, "down")
                return super().request(method, path, body)

        gh = PostDown(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        self.assertEqual(go(gh, self.clean()), 0)
        self.assertEqual(gh.reviews[0]["state"], "DISMISSED")

    def test_a_failed_post_and_failed_dismissal_still_fail_the_step(self):
        class AllDown(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "POST" and path.endswith("/reviews"):
                    raise GitHubError(500, "down")
                if path.endswith("/dismissals"):
                    raise GitHubError(403, "forbidden")
                return super().request(method, path, body)

        gh = AllDown(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        self.assertEqual(go(gh, self.clean()), 1)

    def test_a_newer_runs_request_on_the_same_commit_is_left_alone(self):
        newer = old_review(1000, state="CHANGES_REQUESTED", commit=SHA)
        gh = FakeGitHub(reviews=[newer])
        self.assertEqual(go(gh, self.clean()), 0)
        self.assertEqual(newer["state"], "CHANGES_REQUESTED")

    def test_a_head_that_moves_before_the_final_check_reports_nothing(self):
        class NoDismissThenMove(FakeGitHub):
            lists = 0

            def request(self, method, path, body=None, idempotent=None):
                if path.endswith("/dismissals"):
                    raise GitHubError(403, "forbidden")
                out = super().request(method, path, body)
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews?"):
                    self.lists += 1
                    if self.lists == 3:  # the final check's listing
                        self.head = "b" * 40
                return out

        gh = NoDismissThenMove(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        self.assertEqual(go(gh, self.clean()), 0)

    def test_a_request_is_hidden_only_after_it_is_dismissed(self):
        gh = FakeGitHub(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        self.assertEqual(go(gh, self.clean()), 0)
        steps = [
            "dismiss" if w[1].endswith("/dismissals") else "minimize"
            for w in gh.writes()
            if w[1].endswith("/dismissals")
            or (w[1] == "/graphql" and "minimizeComment" in w[2]["query"])
        ]
        self.assertEqual(steps, ["dismiss", "minimize"])

    def test_a_request_left_standing_by_a_moved_head_stays_visible(self):
        class MovesBeforeWithdraw(FakeGitHub):
            lists = 0

            def request(self, method, path, body=None, idempotent=None):
                out = super().request(method, path, body)
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews?"):
                    self.lists += 1
                    if self.lists == 2:  # withdraw's listing
                        self.head = "b" * 40
                return out

        old = old_review(5, state="CHANGES_REQUESTED")
        gh = MovesBeforeWithdraw(reviews=[old])
        self.assertEqual(go(gh, self.clean()), 0)
        self.assertEqual(old["state"], "CHANGES_REQUESTED")
        self.assertEqual(minimized(gh), [])

    def test_an_api_error_in_the_clean_check_fails_the_step(self):
        class FinalListingDown(FakeGitHub):
            lists = 0

            def request(self, method, path, body=None, idempotent=None):
                if path.endswith("/dismissals"):
                    raise GitHubError(403, "forbidden")
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews?"):
                    self.lists += 1
                    if self.lists == 3:  # the final check's listing
                        raise GitHubError(503, "unavailable")
                return super().request(method, path, body)

        old = old_review(5, state="CHANGES_REQUESTED")
        gh = FinalListingDown(reviews=[old])
        with mock.patch("post_review._sleep"):
            self.assertEqual(go(gh, self.clean()), 1)
        self.assertEqual(minimized(gh), [])

    def test_a_changes_requested_verdict_does_not_run_the_clean_check(self):
        gh = FakeGitHub(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        self.assertEqual(go(gh), 0)


class TestCleanupFailuresKeepTheNewReviewId(unittest.TestCase):
    def test_a_failed_listing_after_posting_still_clears_older_requests(self):
        class ListingDown(FakeGitHub):
            failures = 0

            def request(self, method, path, body=None, idempotent=None):
                if path == "/graphql" and "isMinimized" in body["query"]:
                    self.failures += 1
                    raise GitHubError(503, "unavailable")
                return super().request(method, path, body)

        old = old_review(5, state="CHANGES_REQUESTED", commit=SHA)
        gh = ListingDown(reviews=[old])
        clean = row(verdict="ready_for_human_review", findings=[])
        with mock.patch("post_review._sleep"):
            self.assertEqual(go(gh, clean), 0)
        self.assertGreater(gh.failures, 0)
        self.assertEqual(old["state"], "DISMISSED")


class TestEarlierReviewsAreDecluttered(unittest.TestCase):
    def test_unreplied_comments_are_deleted_and_replied_threads_resolved(self):
        comments = [
            comment(50, 5),
            comment(60, 6),
            comment(61, 7, author="tristan", reply_to=60),
        ]
        gh = FakeGitHub(reviews=[old_review(5), old_review(6)], comments=comments)
        go(gh)
        self.assertEqual(deleted(gh), [50])
        self.assertEqual(resolved(gh), ["T60"])
        self.assertEqual(sorted(c["id"] for c in gh.comments), [60, 61, 9990])
        self.assertEqual(minimized(gh), ["N6", "N5"])

    def test_the_earlier_body_is_folded_once_with_no_link(self):
        old = old_review(5, body=MARKER + "\n**Automated review** of c: x.\n\nOne bug.")
        gh = FakeGitHub(reviews=[old])
        go(gh)
        self.assertEqual(folds(gh), [5])
        body = old["body"]
        self.assertTrue(body.startswith(MARKER + "\n<details><summary>Outdated"))
        self.assertIn("One bug.", body)
        self.assertTrue(body.endswith("</details>"))
        self.assertNotIn("http", body)
        self.assertEqual(body.count(MARKER), 1)
        self.assertEqual(gh.writes("PATCH"), [])
        # Already folded: a later run leaves the body alone.
        old.pop("minimized")
        gh2 = FakeGitHub(reviews=[old], comments=[])
        go(gh2)
        self.assertEqual(folds(gh2), [])

    def test_a_standing_request_is_not_folded_or_emptied(self):
        gh = FakeGitHub(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        go(gh, row(verdict="ready_for_human_review", findings=[]))
        # Dismissed by the clean verdict first, then decluttered.
        self.assertEqual(dismissed(gh), ["5"])
        self.assertEqual(deleted(gh), [50])

        class NeverDismiss(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if path.endswith("/dismissals"):
                    raise GitHubError(403, "forbidden")
                return super().request(method, path, body)

        gh = NeverDismiss(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        go(gh, row(verdict="ready_for_human_review", findings=[]))
        self.assertEqual((deleted(gh), folds(gh), minimized(gh)), ([], [], []))

    def test_a_hidden_review_is_tidied_but_its_threads_are_left(self):
        old = old_review(7)
        old["minimized"] = True
        comments = [comment(70, 7), comment(71, 7), comment(72, 9, reply_to=71)]
        gh = FakeGitHub(reviews=[old], comments=comments)
        go(gh)
        self.assertEqual(deleted(gh), [70])
        self.assertEqual(folds(gh), [7])
        self.assertEqual(resolved(gh), [])
        self.assertEqual(minimized(gh), [])

    def test_a_hidden_request_left_standing_is_not_emptied(self):
        class NeverDismiss(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if path.endswith("/dismissals"):
                    raise GitHubError(403, "forbidden")
                return super().request(method, path, body)

        old = old_review(5, state="CHANGES_REQUESTED")
        old["minimized"] = True
        gh = NeverDismiss(reviews=[old])
        self.assertEqual(go(gh, row(verdict="ready_for_human_review", findings=[])), 1)
        self.assertEqual((deleted(gh), folds(gh)), ([], []))

    def test_a_reply_landing_after_the_first_listing_saves_the_comment(self):
        class ReplyArrives(FakeGitHub):
            listings = 0

            def request(self, method, path, body=None, idempotent=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/comments"):
                    self.listings += 1
                    if self.listings == 3:  # tidy's re-listing before deleting
                        self.comments.append(comment(51, 8, author="h", reply_to=50))
                return super().request(method, path, body)

        gh = ReplyArrives(reviews=[old_review(5)])
        go(gh)
        self.assertEqual(deleted(gh), [])
        self.assertEqual(resolved(gh), ["T50"])

    def test_a_reply_seen_by_either_listing_saves_the_comment(self):
        # The re-listing can lose a reply that shifted pages under a
        # concurrent delete; the first listing saw it.
        class ReplyVanishes(FakeGitHub):
            listings = 0

            def request(self, method, path, body=None, idempotent=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/comments"):
                    self.listings += 1
                    if self.listings == 3:
                        out = [c for c in self.comments if c["id"] != 51]
                        self.calls.append((method, path, body))
                        return [dict(c) for c in out]
                return super().request(method, path, body)

        comments = [comment(50, 5), comment(52, 5), comment(51, 8, "h", reply_to=50)]
        gh = ReplyVanishes(reviews=[old_review(5)], comments=comments)
        go(gh)
        self.assertEqual(deleted(gh), [52])

    def test_a_replacement_dismissed_meanwhile_spares_the_older_request(self):
        # Run A posted request 999; run B's clean verdict dismissed 999 but not
        # 5, which therefore still stands. A must not empty 5.
        class Overlapped(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                out = super().request(method, path, body)
                if method == "POST" and path == "/repos/o/r/pulls/1/reviews":
                    self.reviews[-1]["state"] = "DISMISSED"
                return out

        gh = Overlapped(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        go(gh)
        self.assertEqual((deleted(gh), folds(gh), minimized(gh)), ([], [], []))

    def test_backfill_runs_after_the_clean_dismissal_and_is_bounded(self):
        hidden = []
        comments = []
        for rid in range(1, 41):
            r = old_review(rid)
            r["minimized"] = True
            hidden.append(r)
            comments += [comment(rid * 100 + i, rid) for i in range(5)]
        standing = old_review(41, state="CHANGES_REQUESTED")
        gh = FakeGitHub(reviews=[*hidden, standing], comments=comments)
        self.assertEqual(go(gh, row(verdict="ready_for_human_review", findings=[])), 0)
        writes = gh.writes()
        dismissal = next(
            i for i, w in enumerate(writes) if w[1].endswith("/dismissals")
        )
        cleanup = [
            i
            for i, w in enumerate(writes)
            if w[0] == "DELETE" or (w[0] == "PUT" and not w[1].endswith("/dismissals"))
        ]
        self.assertLess(dismissal, cleanup[0])
        # 41 (no comments) is folded once dismissed; the backfill then spends
        # exactly its write budget, not all 240 writes.
        self.assertEqual(len(cleanup), 1 + post_review.MAX_BACKFILL)
        # A later run picks up where this one stopped.
        before = len(gh.comments)
        go(gh, row(verdict="ready_for_human_review", findings=[]))
        self.assertLess(len(gh.comments), before)

    def test_a_reply_the_takedown_saw_protects_the_comment_in_backfill(self):
        # Takedown keeps 50 for its reply; every later listing misses the reply.
        class ReplyLost(FakeGitHub):
            listings = 0

            def request(self, method, path, body=None, idempotent=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/comments"):
                    self.listings += 1
                    if self.listings > 2:
                        self.calls.append((method, path, body))
                        return [dict(c) for c in self.comments if c["id"] != 51]
                return super().request(method, path, body)

        comments = [comment(50, 5), comment(51, 8, "h", reply_to=50)]
        gh = ReplyLost(reviews=[old_review(5)], comments=comments)
        go(gh)
        self.assertEqual(deleted(gh), [])
        self.assertEqual(minimized(gh), ["N5"])

    def test_a_reply_seen_while_a_request_was_spared_still_protects_it(self):
        # hide_earlier skips standing request 5 but lists the reply; the clean
        # verdict dismisses 5, and every later listing misses the reply.
        class ReplyLost(FakeGitHub):
            listings = 0

            def request(self, method, path, body=None, idempotent=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/comments"):
                    self.listings += 1
                    if self.listings > 2:
                        self.calls.append((method, path, body))
                        return [dict(c) for c in self.comments if c["id"] != 51]
                return super().request(method, path, body)

        comments = [comment(50, 5), comment(51, 8, "h", reply_to=50)]
        gh = ReplyLost(
            reviews=[old_review(5, state="CHANGES_REQUESTED")], comments=comments
        )
        go(gh, row(verdict="ready_for_human_review", findings=[]))
        self.assertEqual(dismissed(gh), ["5"])
        self.assertEqual(deleted(gh), [])

    def test_backfill_lists_the_comments_once(self):
        hidden, comments = [], []
        for rid in range(1, 6):
            r = old_review(rid)
            r["minimized"] = True
            hidden.append(r)
            comments += [comment(rid * 100 + i, rid) for i in range(3)]
        gh = FakeGitHub(reviews=hidden, comments=comments)
        go(gh, row(verdict="ready_for_human_review", findings=[]))
        listings = [
            c for c in gh.calls if c[0] == "GET" and "/pulls/1/comments" in c[1]
        ]
        # One for the whole backfill: none for discussion links with no
        # findings, none per hidden review, none for an empty hide_earlier.
        self.assertEqual(len(listings), 1)
        self.assertEqual(len(deleted(gh)), 15)

    def test_a_reply_on_a_page_before_a_failed_page_is_remembered(self):
        class SecondPageDown(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/comments"):
                    if page(path) == 2:
                        raise GitHubError(503, "unavailable")
                    return [comment(1000 + i, 9) for i in range(99)] + [
                        comment(51, 8, "h", reply_to=50)
                    ]
                return super().request(method, path, body)

        gh = SecondPageDown()
        with mock.patch("post_review._sleep"):
            self.assertIsNone(post_review.pr_comments(gh, 1))
        self.assertEqual(gh.replies_seen, {50})

    def test_the_backfill_clock_stops_calls_and_retry_waits(self):
        gh = GitHub("token", "o/r")
        gh.deadline = 0.0  # long past
        with mock.patch.object(gh, "_once") as once:
            with self.assertRaises(GitHubError):
                gh.request("GET", "/x")
        once.assert_not_called()
        gh.deadline = post_review.time.monotonic() + 5
        throttled = GitHubError(429, "slow down", retry_after=60)
        with mock.patch.object(gh, "_once", side_effect=throttled) as once:
            with mock.patch("post_review._sleep") as sleep:
                with self.assertRaises(GitHubError):
                    gh.request("GET", "/x")
        self.assertEqual(once.call_count, 1)
        sleep.assert_not_called()

    def test_backfill_leaves_the_clock_unset_for_later_steps(self):
        hidden = old_review(3)
        hidden["minimized"] = True
        gh = FakeGitHub(reviews=[hidden])
        go(gh)
        self.assertIsNone(gh.deadline)
        self.assertEqual(deleted(gh), [30])

    def test_a_hidden_request_is_emptied_once_a_newer_request_replaces_it(self):
        old = old_review(5, state="CHANGES_REQUESTED")
        old["minimized"] = True
        gh = FakeGitHub(reviews=[old])
        go(gh)  # changes_requested: posts REQUEST_CHANGES
        self.assertEqual((deleted(gh), folds(gh)), ([50], [5]))

    def test_a_hidden_request_survives_a_request_downgraded_to_a_comment(self):
        class NoRequestChanges(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if body and body.get("event") == "REQUEST_CHANGES":
                    raise GitHubError(422, "cannot request changes")
                return super().request(method, path, body)

        old = old_review(5, state="CHANGES_REQUESTED")
        old["minimized"] = True
        gh = NoRequestChanges(reviews=[old])
        go(gh)
        # The COMMENT leaves 5 standing, so the backfill must not empty it.
        self.assertEqual((deleted(gh), folds(gh)), ([], []))

    def test_a_thread_with_a_reply_the_listing_missed_is_kept(self):
        # The REST listing never shows the reply (a page shifted under an
        # overlapping run's delete); the thread's comment count still has it.
        for hidden in (False, True):
            old = old_review(5)
            old["minimized"] = hidden
            gh = FakeGitHub(reviews=[old], comments=[comment(50, 5), comment(52, 5)])
            gh.hidden_replies = {50: 1}
            go(gh)
            self.assertEqual(deleted(gh), [52], f"hidden={hidden}")

    def test_unreadable_threads_delete_nothing(self):
        class NoThreads(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if path == "/graphql" and "reviewThreads" in body["query"]:
                    raise GitHubError(503, "unavailable")
                return super().request(method, path, body)

        gh = NoThreads(reviews=[old_review(5)])
        with mock.patch("post_review._sleep"):
            go(gh)
        self.assertEqual(deleted(gh), [])
        self.assertEqual(minimized(gh), ["N5"])

    def test_a_folded_body_is_defused_again(self):
        old = old_review(5, body=MARKER + "\n@pytorchbot merge -f")
        gh = FakeGitHub(reviews=[old])
        go(gh)
        self.assertNotIn("@pytorchbot", old["body"])
        self.assertIn("@ pytorchbot", old["body"])

    def test_a_retry_wait_past_the_clock_is_out_of_time(self):
        gh = GitHub("token", "o/r")
        gh.deadline = post_review.time.monotonic() + 5
        throttled = GitHubError(429, "slow down", retry_after=60)
        with mock.patch.object(gh, "_once", side_effect=throttled):
            with self.assertRaises(post_review.OutOfTime):
                gh.request("DELETE", "/x")

    def test_backfill_writes_are_paced(self):
        hidden = []
        for rid in (1, 2):
            r = old_review(rid)
            r["minimized"] = True
            hidden.append(r)
        gh = FakeGitHub(reviews=hidden)
        with mock.patch.object(post_review, "BACKFILL_PAUSE", 1.0):
            with mock.patch("post_review._sleep") as sleep:
                go(gh)
        writes = len(folds(gh)) + len(deleted(gh))
        self.assertEqual(writes, 4)
        self.assertEqual(
            [c.args for c in sleep.call_args_list], [(1.0,)] * (writes - 1)
        )

    def test_a_discussed_finding_survives_a_refused_inline_post(self):
        root = comment(50, 5, line=4)
        findings = [
            {"path": "torch/x.py", "line": 3, "severity": "major", "message": "A."},
            {"path": "torch/y.py", "line": 3, "severity": "major", "message": "B."},
        ]
        gh = FakeGitHub(
            reviews=[old_review(5)],
            comments=[root, comment(51, 8, "t", reply_to=50)],
            refuse_inline=True,
        )
        go(gh, row(findings=findings))
        body = gh.posted()[-1]["body"]
        self.assertIn("A.", body)
        self.assertIn("B.", body)
        self.assertEqual(body.count("discussion_r50"), 1)
        self.assertNotIn("attached to the code below", body)

    def test_a_comment_someone_reacted_to_is_kept(self):
        reacted = {**comment(50, 5), "reactions": {"total_count": 1}}
        gh = FakeGitHub(reviews=[old_review(5)], comments=[reacted, comment(52, 5)])
        go(gh)
        self.assertEqual(deleted(gh), [52])

    def test_the_backfill_stops_when_its_clock_runs_out(self):
        hidden, comments = [], []
        for rid in (1, 2, 3):
            r = old_review(rid)
            r["minimized"] = True
            hidden.append(r)
            comments += [comment(rid * 100 + i, rid) for i in range(3)]

        class ClockRunsOut(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method != "GET" and len(self.writes("DELETE")) == 3:
                    self.deadline = 0.0  # the first hidden review is done
                if self.deadline is not None and self.deadline <= 0.0:
                    raise post_review.OutOfTime()
                return super().request(method, path, body)

        gh = ClockRunsOut(reviews=hidden, comments=comments)
        out = io.StringIO()
        with redirect_stdout(out):
            go(gh)
        self.assertEqual(deleted(gh), [300, 301, 302])
        self.assertEqual(folds(gh), [3])
        self.assertNotIn("could not delete", out.getvalue())
        self.assertNotIn("could not fold", out.getvalue())
        self.assertIn("stopped tidying earlier reviews", out.getvalue())
        self.assertIsNone(gh.deadline)

    def test_graphql_retry_waits_honour_the_clock(self):
        gh = GitHub("token", "o/r")
        gh.deadline = post_review.time.monotonic() + 0.5
        with mock.patch.object(gh, "request", return_value={"errors": [{}]}):
            with mock.patch("post_review._sleep") as sleep:
                with self.assertRaises(post_review.OutOfTime):
                    gh.graphql("query", {})
        sleep.assert_not_called()

    def test_failed_backfill_deletes_still_spend_the_budget(self):
        class NoDelete(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "DELETE":
                    self.calls.append((method, path, body))
                    raise GitHubError(403, "forbidden")
                return super().request(method, path, body)

        hidden, comments = [], []
        for rid in (1, 2, 3):
            r = old_review(rid, body=post_review.folded(old_review(rid)))
            r["minimized"] = True
            hidden.append(r)
            comments += [comment(rid * 100 + i, rid) for i in range(40)]
        gh = NoDelete(reviews=hidden, comments=comments)
        go(gh)
        self.assertLessEqual(len(gh.writes("DELETE")), post_review.MAX_BACKFILL)

    def test_a_comment_already_deleted_by_an_overlapping_run_is_fine(self):
        class Raced(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "DELETE":
                    self.comments = []
                return super().request(method, path, body)

        gh = Raced(reviews=[old_review(5)])
        out = io.StringIO()
        with redirect_stdout(out):
            go(gh)
        self.assertNotIn("could not delete", out.getvalue())
        self.assertNotIn("still open", out.getvalue())
        self.assertEqual(minimized(gh), ["N5"])

    def test_a_refused_delete_keeps_the_comment_and_still_hides(self):
        class NoDelete(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "DELETE":
                    self.calls.append((method, path, body))
                    raise GitHubError(403, "forbidden")
                return super().request(method, path, body)

        gh = NoDelete(reviews=[old_review(5)])
        go(gh)
        self.assertEqual([c["id"] for c in gh.comments], [50, 9990])
        self.assertEqual(resolved(gh), ["T50"])
        self.assertEqual(minimized(gh), ["N5"])

    def test_a_refused_resolve_still_hides_the_review(self):
        # What pytorch/pytorch's GITHUB_TOKEN gets back: FORBIDDEN inside a 200.
        class NoResolve(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if path == "/graphql" and "resolveReviewThread" in body["query"]:
                    self.calls.append((method, path, body))
                    return {"errors": [{"type": "FORBIDDEN"}]}
                return super().request(method, path, body)

        replied = [comment(50, 5), comment(51, 8, author="h", reply_to=50)]
        gh = NoResolve(reviews=[old_review(5)], comments=replied)
        with mock.patch("post_review._sleep"):
            self.assertEqual(go(gh), 0)
        self.assertEqual(resolved(gh), ["T50"] * 3)
        self.assertEqual(minimized(gh), ["N5"])

    def test_a_failed_comment_listing_still_folds_and_hides_the_review(self):
        class NoComments(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/comments"):
                    raise GitHubError(503, "unavailable")
                return super().request(method, path, body)

        gh = NoComments(reviews=[old_review(5)])
        with mock.patch("post_review._sleep"):
            go(gh)
        self.assertEqual((deleted(gh), resolved(gh)), ([], []))
        self.assertEqual(folds(gh), [5])
        self.assertEqual(minimized(gh), ["N5"])


class TestADiscussedFindingIsNotRepeatedOnTheLine(unittest.TestCase):
    def discussed(self, line=4, path="torch/x.py", author=AUTHOR):
        root = comment(50, 5, author=author, line=line, path=path)
        return [root, comment(51, 8, author="tristan", reply_to=50)]

    def test_it_goes_in_the_body_with_a_link_to_the_discussion(self):
        gh = FakeGitHub(reviews=[old_review(5)], comments=self.discussed())
        go(gh)
        body = gh.posted()[0]
        self.assertNotIn("comments", body)
        self.assertIn(f"{BLOB}/torch/x.py#L3\n\n\U0001f534 Off by one.\n", body["body"])
        self.assertIn(
            "An earlier comment near this line has replies: "
            "https://github.com/o/r/pull/1#discussion_r50",  # @lint-ignore
            body["body"],
        )
        self.assertNotIn("attached to the code below", body["body"])
        self.assertEqual(body["event"], "REQUEST_CHANGES")
        # The discussion stays.
        self.assertEqual(deleted(gh), [])

    def test_other_findings_stay_inline_and_are_counted(self):
        findings = [
            {"path": "torch/x.py", "line": 3, "severity": "major", "message": "A."},
            {"path": "torch/y.py", "line": 3, "severity": "major", "message": "B."},
        ]
        gh = FakeGitHub(reviews=[old_review(5)], comments=self.discussed())
        go(gh, row(findings=findings))
        body = gh.posted()[0]
        self.assertEqual([c["path"] for c in body["comments"]], ["torch/y.py"])
        self.assertIn("1 blocking finding is attached", body["body"])
        self.assertIn("An earlier comment near this line has replies:", body["body"])

    def test_only_a_replied_nearby_current_comment_of_ours_counts(self):
        cases = {
            "far": self.discussed(line=10),
            "other file": self.discussed(path="torch/y.py"),
            "outdated": self.discussed(line=None),
            "a human's": [
                comment(50, 8, author="someone", line=3),
                comment(51, 9, author="t", reply_to=50),
            ],
            "another workflow's": [
                comment(50, 6, line=3),
                comment(51, 8, author="t", reply_to=50),
            ],
            "unreplied": [comment(50, 5, line=3)],
        }
        for name, comments in cases.items():
            other = old_review(6, body="another workflow")
            gh = FakeGitHub(reviews=[old_review(5), other], comments=comments)
            go(gh)
            self.assertIn("comments", gh.posted()[0], name)

    def test_a_failed_listing_posts_inline_as_before(self):
        class NoComments(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/comments"):
                    raise GitHubError(503, "unavailable")
                return super().request(method, path, body)

        gh = NoComments(comments=self.discussed())
        with mock.patch("post_review._sleep"):
            go(gh)
        self.assertIn("comments", gh.posted()[0])


class TestRetries(unittest.TestCase):
    def test_a_rate_limit_waits_as_asked_and_a_plain_403_does_not_retry(self):
        import email.message
        import urllib.error

        def http_error(code, headers):
            msg = email.message.Message()
            for k, v in headers.items():
                msg[k] = v
            return urllib.error.HTTPError("u", code, "x", msg, None)

        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = b"{}"
        gh = GitHub("token", "o/r")
        with mock.patch("post_review._sleep") as sleep:
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=[http_error(403, {"Retry-After": "30"}), ok],
            ):
                self.assertEqual(gh.request("PUT", "/x", {}), {})
            sleep.assert_called_once_with(30)
            with mock.patch(
                "urllib.request.urlopen", side_effect=[http_error(403, {}), ok]
            ) as urlopen:
                with self.assertRaises(GitHubError):
                    gh.request("PUT", "/x", {})
                self.assertEqual(urlopen.call_count, 1)

    def test_graphql_errors_inside_a_200_are_retried(self):
        gh = GitHub("token", "o/r")
        answers = [{"errors": [{"type": "INTERNAL"}]}, {"data": {"ok": 1}}]
        with mock.patch("post_review._sleep"):
            with mock.patch.object(gh, "request", side_effect=answers) as request:
                self.assertEqual(gh.graphql("query", {}), {"ok": 1})
        self.assertEqual(request.call_count, 2)

    def test_graphql_gives_up_after_the_retries(self):
        gh = GitHub("token", "o/r")
        with mock.patch("post_review._sleep"):
            with mock.patch.object(
                gh, "request", return_value={"errors": [{"type": "FORBIDDEN"}]}
            ) as request:
                with self.assertRaises(GitHubError):
                    gh.graphql("query", {})
        self.assertEqual(request.call_count, 3)

    def test_idempotent_calls_retry_and_review_posts_do_not(self):
        import urllib.error

        ok = mock.MagicMock()
        ok.__enter__.return_value.read.return_value = b"{}"
        gh = GitHub("token", "o/r")
        with mock.patch("post_review._sleep"):
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=[urllib.error.URLError("reset"), ok],
            ) as urlopen:
                self.assertEqual(gh.request("GET", "/x"), {})
                self.assertEqual(urlopen.call_count, 2)
            with mock.patch(
                "urllib.request.urlopen",
                side_effect=[urllib.error.URLError("reset"), ok],
            ) as urlopen:
                with self.assertRaises(GitHubError):
                    gh.request("POST", "/repos/o/r/pulls/1/reviews", {})
                self.assertEqual(urlopen.call_count, 1)


class TestTheEarlierReviewIsReplaced(unittest.TestCase):
    def test_earlier_review_is_hidden_after_the_new_one_posts(self):
        gh = FakeGitHub(reviews=[old_review(5)])
        go(gh)

        def step(w):
            if w[1].endswith("/pulls/1/reviews"):
                return "post"
            if w[0] in ("PUT", "DELETE"):
                return {"PUT": "fold", "DELETE": "delete"}[w[0]]
            return "minimize" if "minimizeComment" in w[2]["query"] else "read"

        steps = [step(w) for w in gh.writes()]
        # Reads: hidden state, thread sizes before deleting, then the
        # backfill's hidden-state and thread-size reads.
        self.assertEqual(
            steps,
            ["post", "read", "fold", "read", "delete", "minimize", "read", "read"],
        )
        self.assertEqual(minimized(gh), ["N5"])

    def test_a_newer_change_request_does_not_dismiss_the_older_one(self):
        gh = FakeGitHub(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        go(gh)
        self.assertEqual(dismissed(gh), [])
        self.assertEqual(gh.reviews[0]["state"], "CHANGES_REQUESTED")

    def test_a_comment_review_is_not_dismissed(self):
        gh = FakeGitHub(reviews=[old_review(5)])
        go(gh)
        self.assertFalse(any("dismissals" in w[1] for w in gh.writes()))

    def test_only_this_workflows_reviews_are_touched(self):
        gh = FakeGitHub(
            reviews=[
                old_review(5, author="someone"),  # a human pasting the marker
                old_review(6, body="looks good"),  # the bot, another workflow
            ]
        )
        go(gh)
        self.assertEqual(
            [w for w in gh.writes() if w[1] != "/repos/o/r/pulls/1/reviews"], []
        )

    def test_a_newer_review_from_an_overlapping_run_survives(self):
        gh = FakeGitHub(reviews=[old_review(1000)])
        go(gh)
        self.assertEqual(
            [w for w in gh.writes() if w[1] != "/repos/o/r/pulls/1/reviews"], []
        )

    def test_one_failed_takedown_does_not_stop_the_others(self):
        class Flaky(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                minimize = path == "/graphql" and "minimizeComment" in body["query"]
                if minimize and body["variables"]["id"] == "N8":
                    raise GitHubError(503, "unavailable")
                return super().request(method, path, body)

        gh = Flaky(reviews=[old_review(5), old_review(8)])
        with mock.patch("post_review._sleep"):
            go(gh)
        self.assertEqual(deleted(gh), [80, 50])
        self.assertIn("N5", minimized(gh))

    def test_a_head_that_moves_while_posting_retracts_this_review(self):
        class Moves(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                out = super().request(method, path, body)
                if method == "POST" and path == "/repos/o/r/pulls/1/reviews":
                    self.head = "b" * 40  # the newer commit's run already posted
                return out

        gh = Moves(reviews=[old_review(5)])
        go(gh)
        self.assertEqual(minimized(gh), ["N999"])
        # A newer run may already have finished with a clean verdict, so this
        # stale change request is withdrawn too, and nothing older is touched.
        self.assertEqual(dismissed(gh), ["999"])
        (put,) = [w for w in gh.writes("PUT") if w[1].endswith("/dismissals")]
        self.assertNotIn("http", put[2]["message"])
        # Only this review's own comment goes; nothing older is touched.
        self.assertEqual(deleted(gh), [9990])
        self.assertEqual(folds(gh), [999])
        self.assertIn("withdrawn: the PR changed", gh.reviews[-1]["body"])
        self.assertNotIn("newer one replaces", gh.reviews[-1]["body"])

    def test_a_retraction_that_cannot_dismiss_leaves_the_review_visible(self):
        class MovesNoDismiss(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if path.endswith("/dismissals"):
                    raise GitHubError(403, "forbidden")
                out = super().request(method, path, body)
                if method == "POST" and path == "/repos/o/r/pulls/1/reviews":
                    self.head = "b" * 40
                return out

        gh = MovesNoDismiss(reviews=[old_review(5)])
        go(gh)
        self.assertEqual(minimized(gh), [])

    def test_a_review_whose_state_did_not_come_back_is_left_alone(self):
        old = old_review(5)
        old["unreadable"] = True
        gh = FakeGitHub(reviews=[old])
        go(gh)
        self.assertEqual((deleted(gh), folds(gh), minimized(gh)), ([], [], []))

    def test_pagination_reaches_the_second_page(self):
        reviews = [old_review(i, author="someone") for i in range(1, 101)]
        reviews.append(old_review(101))

        class Paged(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews?"):
                    self.calls.append((method, path, body))
                    return reviews[(page(path) - 1) * 100 : page(path) * 100]
                return super().request(method, path, body)

        gh = Paged(reviews=[old_review(101)])
        go(gh)
        self.assertEqual(deleted(gh), [1010])

    def test_a_failed_post_leaves_the_earlier_review_alone(self):
        class Down(FakeGitHub):
            def request(self, method, path, body=None, idempotent=None):
                if method == "POST" and path.endswith("/reviews"):
                    raise GitHubError(500, "down")
                return super().request(method, path, body)

        gh = Down(reviews=[old_review(5)])
        with self.assertRaises(GitHubError):
            go(gh)
        self.assertEqual(
            [w for w in gh.writes() if w[0] != "POST" or w[1] == "/graphql"], []
        )


class TestTransportFailuresAreGitHubErrors(unittest.TestCase):
    """The per-review isolation catches GitHubError, so nothing else may escape."""

    @mock.patch("post_review._sleep")
    def test_connection_failure_and_bad_json(self, _sleep):
        import urllib.error
        from unittest import mock

        gh = GitHub("token", "o/r")
        import http.client

        for effect in (
            urllib.error.URLError("refused"),
            TimeoutError("slow"),
            http.client.IncompleteRead(b"x"),
        ):
            with mock.patch("urllib.request.urlopen", side_effect=effect):
                with self.assertRaises(GitHubError):
                    gh.request("GET", "/x")
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = b"<html>"
        with mock.patch("urllib.request.urlopen", return_value=resp):
            with self.assertRaises(GitHubError):
                gh.request("GET", "/x")


class TestBotCommandsAreDefused(unittest.TestCase):
    def test_every_spelling_of_the_bot_is_defused(self):
        for text in (
            "@pytorchbot merge",
            "@pytorchmergebot revert",
            "@PyTorchBot merge",
            "@pytorch-bot label x",
        ):
            out = defuse_bot_commands(text)
            self.assertNotRegex(out, r"(?i)@pytorch", text)
            self.assertIsNone(re.search(r"(?m)^ *@pytorch(merge|)bot .+$", out))

    def test_other_mentions_and_defused_text_are_untouched(self):
        text = "@pytorch-dev-infra @pytorch/team @ pytorchbot"
        self.assertEqual(defuse_bot_commands(text), text)

    def test_nothing_posted_carries_a_live_bot_command(self):
        # A path may legally contain `@pytorchbot`, and the fallback body prints it.
        path = neutralize_path("docs/@pytorchbot merge.md")
        f = {
            "path": path,
            "line": 1,
            "severity": "major",
            "message": neutralize("@pytorchbot merge"),
        }
        gh = FakeGitHub(refuse_inline=True)
        go(gh, row(summary=neutralize("@pytorchbot merge -f now"), findings=[f]))
        for body in gh.posted():
            for text in [body["body"]] + [c["body"] for c in body.get("comments", [])]:
                self.assertNotRegex(text, r"(?i)@pytorch(merge)?bot", text)


class TestWorkflowWiring(unittest.TestCase):
    def test_publish_job_runs_the_script_on_the_row_with_a_write_token(self):
        text = WORKFLOW.read_text()
        publish = text.split("\n  publish:\n", 1)[1]
        step = publish.split("- name: Post the automated review on the PR", 1)[1]
        self.assertIn(
            "python3 scripts/pr_review/post_review.py --row-file terminal.json", step
        )
        self.assertIn("REVIEWED_SHA: ${{ needs.prepare.outputs.head_sha }}", step)
        self.assertIn("pull-requests: write", publish.split("steps:", 1)[0])


if __name__ == "__main__":
    run_this_suite()
