#!/usr/bin/env python3
# Owner(s): ["module: ci"]

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from typing import Any
from unittest import main, mock, TestCase


# RADAR helpers read the HMAC key from the environment at call time; set it
# unconditionally before importing so module-level constants and every helper
# see a non-empty key regardless of the ambient environment.
os.environ["RADAR_HMAC_KEY"] = "radar-unit-test-key"

import radar
from github_utils import GitHubComment
from trymerge import has_valid_radar_approval, radar_merger_has_write


ORG = "pytorch"
PROJECT = "pytorch"
PR_NUM = 1
HEAD_SHA = "a" * 40
OTHER_SHA = "b" * 40
ALLOWLISTED_FILES = ["test/test_radar_feature.py"]
NON_ALLOWLISTED_FILES = ["torch/csrc/foo.cpp"]


def _sig(sha: str, score: int, tier: str) -> str:
    key = radar.get_hmac_key()
    assert key is not None
    return radar.compute_marker_sig(key, ORG, PROJECT, PR_NUM, sha, score, tier)


def _marker(sha: str, score: int = 5, tier: str = "trivial") -> str:
    m = radar.RadarMarker(sha=sha, score=score, tier=tier, sig=_sig(sha, score, tier))
    return radar.render_marker(m) + "\napproved"


def _revoked(sha: str) -> str:
    key = radar.get_hmac_key()
    assert key is not None
    sig = radar.compute_revoked_sig(key, ORG, PROJECT, PR_NUM, sha)
    return radar.render_revoked_marker(radar.RadarRevocation(sha=sha, sig=sig))


def _comment(
    body: str,
    login: str = "github-actions",
    association: str = "MEMBER",
    editor: str | None = None,
) -> GitHubComment:
    # The RADAR marker is plain visible text, so it survives GitHub's bodyText
    # rendering; detection reads body_text, matching production.
    return GitHubComment(
        body_text=body,
        created_at="",
        author_login=login,
        author_url=None,
        author_association=association,
        editor_login=editor,
        database_id=1,
        url="",
    )


class FakePR:
    def __init__(
        self,
        comments: list[GitHubComment],
        head_sha: str = HEAD_SHA,
        changed_files: list[str] | None = None,
    ) -> None:
        self._comments = comments
        self._head_sha = head_sha
        self._changed_files = (
            list(ALLOWLISTED_FILES) if changed_files is None else changed_files
        )
        self.org = ORG
        self.project = PROJECT
        self.pr_num = PR_NUM

    def last_commit_sha(self, default: str | None = None) -> str:
        return self._head_sha

    def get_comments(self) -> list[GitHubComment]:
        return self._comments

    def get_comment_by_id(self, database_id: int) -> GitHubComment:
        return self._comments[-1]

    def get_changed_files(self) -> list[str]:
        return self._changed_files

    def is_base_repo_private(self) -> bool:
        return False


class TestRadarPure(TestCase):
    def test_parse_marker_roundtrip(self) -> None:
        m = radar.parse_radar_marker(_marker(HEAD_SHA, 7, "trivial"))
        assert m is not None
        self.assertEqual(m.sha, HEAD_SHA)
        self.assertEqual(m.score, 7)
        self.assertEqual(m.tier, "trivial")
        self.assertEqual(m.sig, _sig(HEAD_SHA, 7, "trivial"))

    def test_parse_marker_absent(self) -> None:
        self.assertIsNone(radar.parse_radar_marker("no marker here"))

    def test_parse_marker_requires_signature(self) -> None:
        # A marker with no sig=... must not parse, so unsigned/forged markers can
        # never be honored.
        unsigned = f"RADAR-APPROVED sha={HEAD_SHA} score=5 tier=trivial"
        self.assertIsNone(radar.parse_radar_marker(unsigned))

    def test_verify_good_signature(self) -> None:
        m = radar.parse_radar_marker(_marker(HEAD_SHA))
        assert m is not None
        self.assertTrue(radar.verify_marker_sig(m, ORG, PROJECT, PR_NUM))

    def test_verify_rejects_tampered_fields(self) -> None:
        m = radar.parse_radar_marker(_marker(HEAD_SHA, 5, "trivial"))
        assert m is not None
        # Same signature but a bumped score / different pr must not verify.
        tampered_score = m._replace(score=99)
        self.assertFalse(radar.verify_marker_sig(tampered_score, ORG, PROJECT, PR_NUM))
        self.assertFalse(radar.verify_marker_sig(m, ORG, PROJECT, 999))

    def test_verify_fails_closed_without_key(self) -> None:
        m = radar.parse_radar_marker(_marker(HEAD_SHA))
        assert m is not None
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RADAR_HMAC_KEY", None)
            self.assertIsNone(radar.get_hmac_key())
            self.assertFalse(radar.verify_marker_sig(m, ORG, PROJECT, PR_NUM))

    def test_trusted_login_normalizes_bot_suffix(self) -> None:
        self.assertTrue(radar.is_trusted_radar_login("github-actions[bot]"))
        self.assertTrue(radar.is_trusted_radar_login("github-actions"))
        self.assertFalse(radar.is_trusted_radar_login("some-random-user"))

    def test_low_risk_requires_tier_and_score(self) -> None:
        self.assertTrue(radar.is_low_risk("trivial", radar.RADAR_MAX_SCORE))
        self.assertFalse(radar.is_low_risk("low", 0))
        self.assertFalse(radar.is_low_risk("trivial", radar.RADAR_MAX_SCORE + 1))

    def test_write_access(self) -> None:
        for perm in ("write", "admin", "maintain"):
            self.assertTrue(radar.has_write_access(perm))
        for perm in ("read", "none", "triage"):
            self.assertFalse(radar.has_write_access(perm))

    def test_files_allowlisted(self) -> None:
        self.assertTrue(radar.files_allowlisted(["test/test_x.py"]))
        self.assertTrue(radar.files_allowlisted(["docs/source/foo.rst", "README.md"]))
        # .pyi.in is deliberately NOT allowlisted: it would admit the core
        # torch/_C/__init__.pyi.in and bypass its domain reviewers.
        self.assertFalse(radar.files_allowlisted(["torch/_C/__init__.pyi.in"]))
        # A single non-allowlisted file taints the whole set.
        self.assertFalse(
            radar.files_allowlisted(["test/test_x.py", "torch/csrc/foo.cpp"])
        )
        self.assertFalse(radar.files_allowlisted(["aten/src/ATen/native/foo.cpp"]))
        # Empty list fails closed.
        self.assertFalse(radar.files_allowlisted([]))
        # Denylist: CI-executing harness files are rejected even under test/.
        self.assertFalse(radar.files_allowlisted(["test/conftest.py"]))
        self.assertFalse(radar.files_allowlisted(["test/run_test.py"]))
        self.assertFalse(radar.files_allowlisted(["test/inductor/conftest.py"]))
        # Docs build config / Makefiles execute in CI: denied even under docs/.
        self.assertFalse(radar.files_allowlisted(["docs/source/conf.py"]))
        self.assertFalse(radar.files_allowlisted(["docs/Makefile"]))
        self.assertFalse(radar.files_allowlisted(["docs/conf.py"]))
        # One denied file taints the whole set.
        self.assertFalse(
            radar.files_allowlisted(["test/test_x.py", "test/conftest.py"])
        )
        # Basename-anchored globs must NOT over-match legitimate files.
        self.assertTrue(radar.files_allowlisted(["test/my_conftest.py"]))
        self.assertTrue(radar.files_allowlisted(["test/test_run_test.py"]))
        # RADAR_DENYLIST_GLOBS is tighten-only: it ADDS to the floor.
        with mock.patch.dict(os.environ, {"RADAR_DENYLIST_GLOBS": "*.md"}):
            self.assertFalse(radar.files_allowlisted(["README.md"]))
            self.assertFalse(radar.files_allowlisted(["test/conftest.py"]))

    def test_revoked_sign_verify_roundtrip(self) -> None:
        markers = radar.iter_revoked_markers(_revoked(HEAD_SHA))
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0].sha, HEAD_SHA)
        self.assertTrue(radar.verify_revoked_sig(markers[0], ORG, PROJECT, PR_NUM))

    def test_revoked_rejects_tampered_and_wrong_pr(self) -> None:
        rev = radar.iter_revoked_markers(_revoked(HEAD_SHA))[0]
        self.assertFalse(
            radar.verify_revoked_sig(rev._replace(sig="0" * 64), ORG, PROJECT, PR_NUM)
        )
        self.assertFalse(radar.verify_revoked_sig(rev, ORG, PROJECT, 999))

    def test_revoked_fails_closed_without_key(self) -> None:
        rev = radar.iter_revoked_markers(_revoked(HEAD_SHA))[0]
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RADAR_HMAC_KEY", None)
            self.assertFalse(radar.verify_revoked_sig(rev, ORG, PROJECT, PR_NUM))

    def test_approval_sig_is_not_a_valid_revocation(self) -> None:
        # The ":REVOKED" payload namespace means an approval signature can never
        # be replayed as a revocation for the same SHA.
        approval_sig = _sig(HEAD_SHA, 5, "trivial")
        rev = radar.RadarRevocation(sha=HEAD_SHA, sig=approval_sig)
        self.assertFalse(radar.verify_revoked_sig(rev, ORG, PROJECT, PR_NUM))

    def test_max_score_is_constant_not_env(self) -> None:
        # The score ceiling is a hardcoded constant, not an env read, so the
        # signer and the trymerge verifier cannot diverge. Reloading the module
        # under a bogus RADAR_MAX_SCORE env must NOT change it; this would fail
        # against a regression back to int(os.environ.get("RADAR_MAX_SCORE", ...)).
        self.assertEqual(radar.RADAR_MAX_SCORE, 10)
        reloaded = None
        try:
            with mock.patch.dict(os.environ, {"RADAR_MAX_SCORE": "999"}):
                importlib.reload(radar)
                reloaded = radar.RADAR_MAX_SCORE
        finally:
            # Restore the module under the ambient (unpatched) environment so
            # other tests see the normal constant regardless of the outcome above.
            importlib.reload(radar)
        self.assertEqual(reloaded, 10)
        self.assertEqual(radar.RADAR_MAX_SCORE, 10)


class TestRadarDetection(TestCase):
    def _valid(self, pr: Any) -> bool:
        return has_valid_radar_approval(pr)

    def test_valid_bot_approval_current_sha(self) -> None:
        pr = FakePR([_comment(_marker(HEAD_SHA))])
        self.assertTrue(self._valid(pr))

    def test_marker_is_visible_text_not_html_comment(self) -> None:
        # GitHub's bodyText (which detection reads) strips HTML comments, so the
        # marker must be plain visible text. Guards against regressing to an
        # HTML-comment marker that would silently never be detected.
        rendered = radar.render_marker(
            radar.RadarMarker(HEAD_SHA, 5, "trivial", _sig(HEAD_SHA, 5, "trivial"))
        )
        self.assertNotIn("<!--", rendered)
        self.assertIsNotNone(radar.parse_radar_marker(rendered))

    def test_stale_sha_rejected(self) -> None:
        pr = FakePR([_comment(_marker(OTHER_SHA))])
        self.assertFalse(self._valid(pr))

    def test_non_bot_author_rejected(self) -> None:
        pr = FakePR([_comment(_marker(HEAD_SHA), login="attacker")])
        self.assertFalse(self._valid(pr))

    def test_edited_comment_rejected(self) -> None:
        pr = FakePR([_comment(_marker(HEAD_SHA), editor="attacker")])
        self.assertFalse(self._valid(pr))

    def test_no_marker_rejected(self) -> None:
        pr = FakePR([_comment("just a normal bot comment")])
        self.assertFalse(self._valid(pr))

    def test_bad_signature_rejected(self) -> None:
        # Correctly-shaped marker whose signature does not match its fields
        # (e.g. reflected attacker text) must not validate.
        forged = radar.RadarMarker(sha=HEAD_SHA, score=1, tier="trivial", sig="0" * 64)
        body = radar.render_marker(forged)
        pr = FakePR([_comment(body)])
        self.assertFalse(self._valid(pr))

    def test_score_over_threshold_rejected(self) -> None:
        # Even a properly-signed marker is rejected at merge time if its score /
        # tier are above the auto-approve threshold.
        pr = FakePR([_comment(_marker(HEAD_SHA, 99, "high"))])
        self.assertFalse(self._valid(pr))

    def test_non_allowlisted_files_rejected(self) -> None:
        # A valid, signed, current-SHA marker is still rejected when the PR
        # touches files outside the allowlist.
        pr = FakePR(
            [_comment(_marker(HEAD_SHA))], changed_files=list(NON_ALLOWLISTED_FILES)
        )
        self.assertFalse(self._valid(pr))

    def test_fails_closed_without_key(self) -> None:
        pr = FakePR([_comment(_marker(HEAD_SHA))])
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RADAR_HMAC_KEY", None)
            self.assertFalse(self._valid(pr))

    def test_latest_valid_marker_wins(self) -> None:
        pr = FakePR([_comment(_marker(OTHER_SHA)), _comment(_marker(HEAD_SHA))])
        self.assertTrue(self._valid(pr))

    def test_shadow_marker_before_real_is_ignored(self) -> None:
        # A bogus marker-shaped string (bad sig) preceding the genuine marker in
        # the same comment must not shadow it: all markers are scanned.
        bogus = f"RADAR-APPROVED sha={HEAD_SHA} score=5 tier=trivial sig={'0' * 64}"
        body = bogus + "\n\n" + _marker(HEAD_SHA)
        self.assertTrue(self._valid(FakePR([_comment(body)])))

    def test_revocation_invalidates_prior_approval(self) -> None:
        # A signed revocation for the head SHA, newer than the approval, wins even
        # though the approval comment was not deleted (delete is best-effort).
        pr = FakePR([_comment(_marker(HEAD_SHA)), _comment(_revoked(HEAD_SHA))])
        self.assertFalse(self._valid(pr))

    def test_revocation_for_other_sha_ignored(self) -> None:
        # A revocation bound to a different SHA must not invalidate the current
        # head's approval.
        pr = FakePR([_comment(_marker(HEAD_SHA)), _comment(_revoked(OTHER_SHA))])
        self.assertTrue(self._valid(pr))

    def test_forged_revocation_sig_does_not_invalidate(self) -> None:
        # A revocation-shaped string with a bad signature cannot deny a genuine
        # approval (forgery is impossible in either direction).
        bogus = f"RADAR-REVOKED sha={HEAD_SHA} sig={'0' * 64}"
        pr = FakePR([_comment(_marker(HEAD_SHA)), _comment(bogus)])
        self.assertTrue(self._valid(pr))

    def test_reapproval_after_revocation_wins(self) -> None:
        # Newest decision wins: an approval posted after a revocation re-validates.
        pr = FakePR(
            [
                _comment(_marker(HEAD_SHA)),
                _comment(_revoked(HEAD_SHA)),
                _comment(_marker(HEAD_SHA)),
            ]
        )
        self.assertTrue(self._valid(pr))

    def test_revocation_from_untrusted_login_ignored(self) -> None:
        # Only a revocation under the trusted RADAR login counts; an attacker
        # cannot deny an approval from a non-bot comment.
        pr = FakePR(
            [
                _comment(_marker(HEAD_SHA)),
                _comment(_revoked(HEAD_SHA), login="attacker"),
            ]
        )
        self.assertTrue(self._valid(pr))

    def test_edited_revocation_still_revokes(self) -> None:
        # A revocation is deny-only and HMAC-signed, so it is honored even from an
        # EDITED comment. This closes the fail-open where editing the bot's
        # revocation comment would otherwise resurrect a lingering approval.
        pr = FakePR(
            [_comment(_marker(HEAD_SHA)), _comment(_revoked(HEAD_SHA), editor="x")]
        )
        self.assertFalse(self._valid(pr))

    def test_edited_approval_still_rejected(self) -> None:
        # The edited-comment skip still applies to approvals: an edited approval
        # marker must not authorize a merge.
        pr = FakePR([_comment(_marker(HEAD_SHA), editor="x")])
        self.assertFalse(self._valid(pr))


class TestRadarMergerHasWrite(TestCase):
    # The person half of authorization, reused per-PR across a ghstack. It gates
    # on the commenter's REAL repo permission, not their author_association.
    @mock.patch("trymerge._radar_commenter_repo_permission", return_value="write")
    def test_write_access_unedited(self, _perm: mock.MagicMock) -> None:
        pr = FakePR([_comment(_marker(HEAD_SHA))])
        self.assertTrue(radar_merger_has_write(pr, comment_id=1))

    @mock.patch("trymerge._radar_commenter_repo_permission", return_value="read")
    def test_read_only_rejected(self, _perm: mock.MagicMock) -> None:
        # An org MEMBER/COLLABORATOR association is not enough: read-only fails.
        pr = FakePR([_comment(_marker(HEAD_SHA), association="MEMBER")])
        self.assertFalse(radar_merger_has_write(pr, comment_id=1))

    @mock.patch("trymerge._radar_commenter_repo_permission", return_value="none")
    def test_no_permission_rejected(self, _perm: mock.MagicMock) -> None:
        pr = FakePR([_comment(_marker(HEAD_SHA), association="MEMBER")])
        self.assertFalse(radar_merger_has_write(pr, comment_id=1))

    @mock.patch("trymerge._radar_commenter_repo_permission", return_value="write")
    def test_edited_merge_comment_rejected(self, _perm: mock.MagicMock) -> None:
        pr = FakePR([_comment(_marker(HEAD_SHA), editor="x")])
        self.assertFalse(radar_merger_has_write(pr, comment_id=1))

    @mock.patch("trymerge._radar_commenter_repo_permission", return_value="write")
    def test_no_comment_id_rejected(self, _perm: mock.MagicMock) -> None:
        pr = FakePR([_comment(_marker(HEAD_SHA))])
        self.assertFalse(radar_merger_has_write(pr, comment_id=None))

    @mock.patch("trymerge.gh_fetch_json_dict", side_effect=RuntimeError("404"))
    def test_permission_fetch_failure_fails_closed(
        self, _fetch: mock.MagicMock
    ) -> None:
        # A failed/absent permission lookup (e.g. non-collaborator 404) must be
        # treated as no write access, not silently allowed.
        from trymerge import _radar_commenter_repo_permission

        pr = FakePR([_comment(_marker(HEAD_SHA))])
        self.assertEqual(_radar_commenter_repo_permission(pr, "someone"), "none")
        self.assertFalse(radar_merger_has_write(pr, comment_id=1))


class TestRadarMain(TestCase):
    # Exercises the composed approval DECISION in radar.main(): it posts an
    # approval + adds the label only when the change is low-risk, allowlisted,
    # authored by a write-access user, small/untruncated, and a signing key is
    # present; otherwise it prunes any stale approval and clears the label.
    def _run(
        self,
        tier: str,
        score: int,
        files: list[str],
        perm: str,
        additions: int = 3,
        deletions: int = 1,
        extra: tuple[str, ...] = (),
        include_line_counts: bool = True,
        changed_files_count: int | None = None,
        include_changed_files: bool = True,
        prior_comments: list[dict[str, Any]] | None = None,
    ) -> tuple[mock.MagicMock, mock.MagicMock, mock.MagicMock, mock.MagicMock]:
        d = tempfile.mkdtemp()
        vf, mf = os.path.join(d, "v.json"), os.path.join(d, "m.json")
        with open(vf, "w") as f:
            json.dump(
                {
                    "risk_score": score,
                    "risk_tier": tier,
                    "summary": "s",
                    "reasoning": "r",
                },
                f,
            )
        meta: dict[str, Any] = {"files": [{"path": p} for p in files]}
        if include_line_counts:
            meta["additions"] = additions
            meta["deletions"] = deletions
        if include_changed_files:
            meta["changedFiles"] = (
                len(files) if changed_files_count is None else changed_files_count
            )
        with open(mf, "w") as f:
            json.dump(meta, f)
        argv = [
            "radar.py",
            "--pr-num",
            str(PR_NUM),
            "--sha",
            HEAD_SHA,
            "--org",
            ORG,
            "--project",
            PROJECT,
            "--verdict-file",
            vf,
            "--meta-file",
            mf,
            "--author-permission",
            perm,
            *extra,
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch("radar.gh_fetch_json_list", return_value=prior_comments or []),
            mock.patch("radar.gh_delete_comment") as dele,
            mock.patch("radar.gh_post_pr_comment") as post,
            mock.patch("radar.gh_add_labels") as addl,
            mock.patch("radar.gh_remove_label") as reml,
        ):
            radar.main()
        return post, addl, reml, dele

    def test_approves_low_risk(self) -> None:
        post, addl, reml, _ = self._run("trivial", 5, ["test/test_x.py"], "write")
        post.assert_called_once()
        addl.assert_called_once()
        reml.assert_not_called()
        # The posted comment must carry a marker whose signature verifies.
        body = post.call_args[0][3]
        mk = radar.parse_radar_marker(body)
        assert mk is not None
        self.assertTrue(radar.verify_marker_sig(mk, ORG, PROJECT, PR_NUM))

    def test_rejects_high_score(self) -> None:
        post, addl, reml, _ = self._run("medium", 50, ["test/test_x.py"], "write")
        post.assert_not_called()
        addl.assert_not_called()
        reml.assert_called_once()

    def test_rejects_no_write(self) -> None:
        post, addl, _, _ = self._run("trivial", 5, ["test/test_x.py"], "read")
        post.assert_not_called()
        addl.assert_not_called()

    def test_rejects_non_allowlisted(self) -> None:
        post, addl, _, _ = self._run("trivial", 5, ["torch/csrc/foo.cpp"], "write")
        post.assert_not_called()
        addl.assert_not_called()

    def test_rejects_denylisted(self) -> None:
        post, addl, _, _ = self._run("trivial", 5, ["test/conftest.py"], "write")
        post.assert_not_called()
        addl.assert_not_called()

    def test_rejects_truncated_diff(self) -> None:
        post, addl, _, _ = self._run(
            "trivial", 5, ["test/test_x.py"], "write", extra=("--diff-truncated",)
        )
        post.assert_not_called()
        addl.assert_not_called()

    def test_rejects_huge_diff(self) -> None:
        post, addl, _, _ = self._run(
            "trivial", 5, ["test/test_x.py"], "write", additions=5000, deletions=0
        )
        post.assert_not_called()
        addl.assert_not_called()

    def test_rejects_missing_line_counts(self) -> None:
        # Absent additions/deletions must fail closed (treated as oversized).
        post, addl, _, _ = self._run(
            "trivial", 5, ["test/test_x.py"], "write", include_line_counts=False
        )
        post.assert_not_called()
        addl.assert_not_called()

    def test_rejects_no_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RADAR_HMAC_KEY", None)
            post, addl, _, _ = self._run("trivial", 5, ["test/test_x.py"], "write")
        post.assert_not_called()
        addl.assert_not_called()

    def test_rejects_truncated_file_list(self) -> None:
        # changedFiles > the (100-capped) files list means the list may hide a
        # non-allowlisted file, so RADAR must fail closed even if every visible
        # file is allowlisted.
        post, addl, _, _ = self._run(
            "trivial", 5, ["test/test_x.py"], "write", changed_files_count=150
        )
        post.assert_not_called()
        addl.assert_not_called()

    def test_rejects_missing_changed_files_count(self) -> None:
        # Absent changedFiles must fail closed: we cannot prove the list is whole.
        post, addl, _, _ = self._run(
            "trivial", 5, ["test/test_x.py"], "write", include_changed_files=False
        )
        post.assert_not_called()
        addl.assert_not_called()

    def _prior_approval_comment(self) -> dict[str, Any]:
        return {
            "user": {"login": "github-actions[bot]"},
            "body": _marker(HEAD_SHA),
            "id": 42,
        }

    def test_posts_signed_revocation_when_prior_approval_exists(self) -> None:
        # A PR that previously earned an approval but no longer qualifies must get
        # a SIGNED revocation bound to the head SHA, so withdrawal does not depend
        # on the prior approval comment being deleted.
        post, addl, reml, _ = self._run(
            "medium",
            50,
            ["test/test_x.py"],
            "write",
            prior_comments=[self._prior_approval_comment()],
        )
        post.assert_called_once()
        addl.assert_not_called()
        reml.assert_called_once()
        body = post.call_args[0][3]
        revs = radar.iter_revoked_markers(body)
        self.assertEqual(len(revs), 1)
        self.assertEqual(revs[0].sha, HEAD_SHA)
        self.assertTrue(radar.verify_revoked_sig(revs[0], ORG, PROJECT, PR_NUM))

    def test_no_revocation_when_no_prior_approval(self) -> None:
        # An ordinary non-qualifying PR that was never approved must not be spammed
        # with a revocation comment.
        post, _, reml, _ = self._run("medium", 50, ["test/test_x.py"], "write")
        post.assert_not_called()
        reml.assert_called_once()

    def _prior_revocation_comment(self) -> dict[str, Any]:
        return {
            "user": {"login": "github-actions[bot]"},
            "body": _revoked(HEAD_SHA),
            "id": 43,
        }

    def test_no_double_revocation_when_head_revocation_exists(self) -> None:
        # A standing signed revocation for the head already provides the
        # fail-closed withdrawal, so a re-run must not post a duplicate.
        post, _, _, dele = self._run(
            "medium",
            50,
            ["test/test_x.py"],
            "write",
            prior_comments=[
                self._prior_approval_comment(),
                self._prior_revocation_comment(),
            ],
        )
        post.assert_not_called()
        # The stale approval (42) is pruned; the standing head revocation (43) is
        # KEPT so the fail-closed withdrawal survives the prune.
        deleted = [c.args[2] for c in dele.call_args_list]
        self.assertIn(42, deleted)
        self.assertNotIn(43, deleted)

    def test_approval_posts_even_with_prior_revocation(self) -> None:
        # When a previously-revoked PR now qualifies again, the approval path posts
        # a fresh approval marker (newest decision wins).
        post, addl, _, dele = self._run(
            "trivial",
            5,
            ["test/test_x.py"],
            "write",
            prior_comments=[self._prior_revocation_comment()],
        )
        post.assert_called_once()
        addl.assert_called_once()
        self.assertIsNotNone(radar.parse_radar_marker(post.call_args[0][3]))
        # The approval path drops every prior banner, including the old revocation
        # (43), so the fresh approval is the sole, newest decision.
        self.assertIn(43, [c.args[2] for c in dele.call_args_list])


if __name__ == "__main__":
    main()
