#!/usr/bin/env python3
"""Tests for posting the verdict as a native GitHub review.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import run_this_suite, TestTheSuiteIsWhole  # noqa: E402,F401
from extract_verdict import neutralize, neutralize_path  # noqa: E402
from post_review import (  # noqa: E402
    defuse_bot_commands,
    GitHub,
    GitHubError,
    MARKER,
    run,
    SUPERSEDED_MARKER,
)


SHA = "a" * 40
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
    ):
        super().__init__("token", "o/r")
        self.head = head
        self.labels = [{"name": n} for n in labels]
        self.reviews = list(reviews)
        self.review_comments = {r["id"]: [{"id": r["id"] * 10}] for r in self.reviews}
        self.refuse_inline = refuse_inline
        self.calls: list[tuple] = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        if method == "GET" and path == "/repos/o/r/pulls/1":
            return {"head": {"sha": self.head}}
        if method == "GET" and path.startswith("/repos/o/r/issues/1/labels"):
            return self.labels
        if method == "GET" and re.match(
            r"/repos/o/r/pulls/1/reviews/\d+/comments", path
        ):
            rid = int(path.split("/")[7])
            return self.review_comments.get(rid, []) if page(path) == 1 else []
        if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews"):
            return self.reviews if page(path) == 1 else []
        if method == "POST" and path == "/repos/o/r/pulls/1/reviews":
            if self.refuse_inline and body.get("comments"):
                raise GitHubError(422, "line must be part of the diff")
            new = {
                "id": 999,
                "node_id": "N999",
                "html_url": "https://x/999",
                "user": {"login": AUTHOR},
                "body": body["body"],
            }
            self.reviews.append(new)
            return new
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
                    {"path": path, "line": 1, "severity": "minor", "message": "m"}
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
        self.assertIn("`torch/x.py` line 3 (Major): Off by one.", second["body"])

    def test_a_ready_verdict_is_a_comment_never_an_approval(self):
        gh = FakeGitHub()
        go(gh, row(verdict="ready_for_human_review"))
        self.assertEqual(gh.posted()[0]["event"], "COMMENT")

    def test_a_refused_request_changes_falls_back_to_a_comment(self):
        class NoRequestChanges(FakeGitHub):
            def request(self, method, path, body=None):
                if body and body.get("event") == "REQUEST_CHANGES":
                    self.calls.append((method, path, body))
                    raise GitHubError(422, "cannot request changes")
                return super().request(method, path, body)

        gh = NoRequestChanges()
        go(gh)
        events = [(b["event"], "comments" in b) for b in gh.posted()]
        self.assertEqual(
            events,
            [("REQUEST_CHANGES", True), ("REQUEST_CHANGES", False), ("COMMENT", False)],
        )

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
        good = {"path": "y.py", "line": 2, "severity": "info", "message": "ok"}
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


class TestStandingChangeRequestsAreWithdrawn(unittest.TestCase):
    def dismissed(self, gh):
        return [w[1].split("/")[7] for w in gh.writes() if w[1].endswith("/dismissals")]

    def test_a_failed_run_withdraws_a_request_on_an_earlier_commit(self):
        gh = FakeGitHub(
            reviews=[
                old_review(5, state="CHANGES_REQUESTED"),  # earlier commit
                old_review(6, state="CHANGES_REQUESTED", commit=SHA),  # this commit
                old_review(7, state="CHANGES_REQUESTED", author="someone"),
            ]
        )
        go(gh, row(status="model_error"))
        self.assertEqual(self.dismissed(gh), ["5"])
        self.assertEqual(gh.posted(), [])

    def test_a_failed_run_on_a_moved_head_touches_nothing(self):
        gh = FakeGitHub(
            head="b" * 40, reviews=[old_review(5, state="CHANGES_REQUESTED")]
        )
        go(gh, row(status="model_error"))
        self.assertEqual(gh.writes(), [])

    def test_a_newer_runs_request_is_not_withdrawn_by_a_failed_run(self):
        class NewerRunPosts(FakeGitHub):
            def request(self, method, path, body=None):
                out = super().request(method, path, body)
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews?"):
                    self.head = "b" * 40  # the newer head's review is in this list
                return out

        newer = old_review(6, state="CHANGES_REQUESTED", commit="b" * 40)
        gh = NewerRunPosts(reviews=[newer])
        go(gh, row(status="model_error"))
        self.assertEqual(gh.writes(), [])

    def test_opting_out_withdraws_every_standing_request(self):
        gh = FakeGitHub(
            labels=("no automated review",),
            reviews=[old_review(5, state="CHANGES_REQUESTED", commit=SHA)],
        )
        go(gh)
        self.assertEqual(self.dismissed(gh), ["5"])
        self.assertEqual(gh.posted(), [])


class TestTheEarlierReviewIsReplaced(unittest.TestCase):
    def test_earlier_review_is_taken_down_after_the_new_one_posts(self):
        gh = FakeGitHub(reviews=[old_review(5)])
        go(gh)
        writes = gh.writes()
        self.assertEqual(writes[0][:2], ("POST", "/repos/o/r/pulls/1/reviews"))
        self.assertEqual(
            [w[:2] for w in writes[1:]],
            [
                ("DELETE", "/repos/o/r/pulls/comments/50"),
                ("PUT", "/repos/o/r/pulls/1/reviews/5"),
                ("POST", "/graphql"),
            ],
        )
        put, minimize = writes[2][2], writes[3][2]
        self.assertTrue(put["body"].startswith(SUPERSEDED_MARKER))
        self.assertIn("https://x/999", put["body"])
        self.assertIn("minimizeComment", minimize["query"])
        self.assertEqual(minimize["variables"], {"id": "N5"})

    def test_a_changes_requested_review_is_dismissed_first(self):
        gh = FakeGitHub(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        go(gh)
        after_post = gh.writes()[1:]
        self.assertEqual(
            after_post[0][:2], ("PUT", "/repos/o/r/pulls/1/reviews/5/dismissals")
        )
        self.assertEqual(after_post[0][2]["event"], "DISMISS")
        self.assertIn("https://x/999", after_post[0][2]["message"])

    def test_a_failed_dismissal_leaves_the_review_whole(self):
        class NoDismiss(FakeGitHub):
            def request(self, method, path, body=None):
                if path.endswith("/dismissals"):
                    self.calls.append((method, path, body))
                    raise GitHubError(403, "forbidden")
                return super().request(method, path, body)

        gh = NoDismiss(reviews=[old_review(5, state="CHANGES_REQUESTED")])
        go(gh)
        self.assertEqual(
            [w[:2] for w in gh.writes()[1:]],
            [("PUT", "/repos/o/r/pulls/1/reviews/5/dismissals")],
        )

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

    def test_a_superseded_review_gets_its_leftover_comments_retried(self):
        gh = FakeGitHub(
            reviews=[old_review(7, body=SUPERSEDED_MARKER + "\nx")],
        )
        go(gh)
        self.assertIn(("DELETE", "/repos/o/r/pulls/comments/70", None), gh.writes())
        self.assertEqual(
            gh.writes("PUT"), [], "an already-superseded body is rewritten"
        )

    def test_one_failed_takedown_does_not_stop_the_others(self):
        class Flaky(FakeGitHub):
            def request(self, method, path, body=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews/8/"):
                    raise GitHubError(503, "unavailable")
                return super().request(method, path, body)

        gh = Flaky(reviews=[old_review(5), old_review(8)])
        go(gh)
        self.assertIn(("DELETE", "/repos/o/r/pulls/comments/50", None), gh.writes())

    def test_a_head_that_moves_while_posting_retracts_this_review(self):
        class Moves(FakeGitHub):
            def request(self, method, path, body=None):
                out = super().request(method, path, body)
                if method == "POST" and path == "/repos/o/r/pulls/1/reviews":
                    self.head = "b" * 40  # the newer commit's run already posted
                return out

        gh = Moves(reviews=[old_review(5)])
        go(gh)
        deleted = [w[2]["variables"] for w in gh.writes() if w[1] == "/graphql"]
        self.assertEqual(deleted, [{"id": "N999"}])

    def test_pagination_reaches_the_second_page(self):
        reviews = [old_review(i, author="someone") for i in range(1, 101)]
        reviews.append(old_review(101))

        class Paged(FakeGitHub):
            def request(self, method, path, body=None):
                if method == "GET" and path.startswith("/repos/o/r/pulls/1/reviews?"):
                    self.calls.append((method, path, body))
                    return reviews[(page(path) - 1) * 100 : page(path) * 100]
                return super().request(method, path, body)

        gh = Paged(reviews=[old_review(101)])
        go(gh)
        self.assertIn(("DELETE", "/repos/o/r/pulls/comments/1010", None), gh.writes())

    def test_a_failed_post_leaves_the_earlier_review_alone(self):
        class Down(FakeGitHub):
            def request(self, method, path, body=None):
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

    def test_connection_failure_and_bad_json(self):
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
