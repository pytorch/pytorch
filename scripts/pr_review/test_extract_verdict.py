#!/usr/bin/env python3
"""Tests for the hardened PR review's output sanitizer.

This script is the egress boundary of a job that reads untrusted code while
holding Bedrock credentials, so the tests below are mostly about what must NOT
get through. Each hostile case is paired with a benign one that looks similar,
because a filter that rejects everything is as useless as one that rejects
nothing.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("extract_verdict.py")
sys.path.insert(0, str(SCRIPT.parent))

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import TestTheSuiteIsWhole  # noqa: E402,F401
from extract_verdict import (  # noqa: E402
    build,
    check_charset,
    check_no_encoded_blob,
    MAX_SUMMARY,
    neutralize,
    parse_diff,
    Rejected,
    sanitize_findings,
)


DIFF = """\
diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,5 @@ def handler():
     ctx = build()
     log.info("start")
+    token = os.environ["SECRET"]
+    send(token)
     return ctx
diff --git a/README.md b/README.md
index 3333333..4444444 100644
--- a/README.md
+++ b/README.md
@@ -1,1 +1,2 @@
 # Project
+A new line.
"""


def smuggled_secret() -> str:
    """A base64 blob the size and entropy of a real AWS session token (~1.3k chars)."""
    return base64.b64encode(bytes(range(256)) * 4).decode()


def ok_payload(**over):
    payload = {
        "verdict": "changes_requested",
        "summary": "Reads a secret from the environment and forwards it.",
        "findings": [
            {
                "path": "src/app.py",
                "line": 12,
                "severity": "major",
                "message": "Forwards a credential to send().",
            }
        ],
    }
    payload.update(over)
    return payload


class TestDiffParsing(unittest.TestCase):
    """The diff is the anchor every finding is validated against."""

    def test_records_added_and_context_lines_on_the_new_side(self):
        touched = parse_diff(DIFF)
        self.assertEqual(set(touched), {"src/app.py", "README.md"})
        # Hunk starts at new-side line 10: two context, two added, one context.
        self.assertEqual(touched["src/app.py"], {10, 11, 12, 13, 14})
        self.assertEqual(touched["README.md"], {1, 2})

    def test_ignores_deleted_files(self):
        deletion = (
            "diff --git a/gone.py b/gone.py\n"
            "--- a/gone.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-x\n-y\n"
        )
        self.assertEqual(parse_diff(deletion), {})


class TestCharset(unittest.TestCase):
    """Zero-width steganography is the one channel wide enough for a credential."""

    def test_accepts_ordinary_review_prose(self):
        check_charset("Line 12 leaks a token.\tSee send().\n", "t")

    def test_rejects_invisible_codepoints(self):
        for hostile in (
            "verdict​​​ready",  # zero-width space
            "text️︎",  # variation selectors
            "a‮b",  # RTL override
            "nul\x00byte",
        ):
            with self.subTest(hostile=repr(hostile)), self.assertRaises(Rejected):
                check_charset(hostile, "t")

    def test_rejects_non_ascii_prose_even_when_innocent(self):
        # Deliberate: we would rather lose an accented word than keep a channel
        # we cannot audit. A known cost, not an oversight.
        with self.assertRaises(Rejected):
            check_charset("naïve implementation", "t")


class TestEncodedBlobs(unittest.TestCase):
    def test_flags_a_smuggled_binary_secret(self):
        with self.assertRaises(Rejected):
            check_no_encoded_blob(f"looks fine but {smuggled_secret()} trailing", "t")

    def test_allows_a_commit_sha_and_long_identifiers(self):
        # Regression guard. A 40-char git Sha unhexlifies to 20 high-entropy
        # bytes, so an earlier 40-char floor rejected any review that cited one
        # — the guard was refusing legitimate output, not catching attacks.
        check_no_encoded_blob(
            "regressed in 4f2c1b9a8d3e5f60718293a4b5c6d7e8f9012345", "t"
        )
        check_no_encoded_blob(f"tree hash {'a1b2c3d4' * 8}", "t")  # 64-char sha256
        check_no_encoded_blob(
            "see MAX_CONCURRENT_UPLOAD_WORKERS_PER_SHARD_CONFIG_VALUE", "t"
        )

    def test_allows_base64_of_ordinary_text(self):
        printable = base64.b64encode(
            b"the quick brown fox jumps over the lazy dog"
        ).decode()
        check_no_encoded_blob(printable, "t")


class TestRenderingHazards(unittest.TestCase):
    def test_defuses_mentions_links_images_and_html(self):
        out = neutralize(
            "cc @pytorch-dev and @alice, see #1234 "
            "![x](http://evil/beacon.png) [click](http://evil) <img src=x onerror=1>"  # @lint-ignore
        )
        self.assertIn("@ pytorch-dev", out)
        self.assertIn("@ alice", out)
        self.assertIn("# 1234", out)
        self.assertNotIn("http://evil", out)  # @lint-ignore
        self.assertNotIn("<img", out)
        # The visible label of a markdown link survives; only the target goes.
        self.assertIn("click", out)
        # Defusing must not itself introduce the codepoints check_charset bans,
        # or the sanitizer would violate its own advertised invariant.
        check_charset(out, "neutralized")

    def test_keeps_ordinary_prose_intact(self):
        text = "The handler on line 12 calls send() with the token; scope the secret instead."
        self.assertEqual(neutralize(text), text)


class TestFindingAnchoring(unittest.TestCase):
    """Anchoring is the real bandwidth reducer on the free-text channel."""

    def test_finding_on_a_real_changed_line_survives(self):
        result = build(ok_payload(), parse_diff(DIFF))
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["verdict"], "changes_requested")
        self.assertEqual(len(result["findings"]), 1)
        self.assertEqual(result["findings_dropped"], 0)

    def test_finding_on_an_untouched_file_is_dropped(self):
        # ready_for_human_review, so the drop is observed on its own rather than
        # tripping the "objection with no evidence" refusal below.
        payload = ok_payload(
            verdict="ready_for_human_review",
            findings=[
                {
                    "path": "src/secrets.py",
                    "line": 3,
                    "severity": "major",
                    "message": "Unrelated file the PR never touched.",
                }
            ],
        )
        result = build(payload, parse_diff(DIFF))
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["findings_dropped"], 1)
        self.assertEqual(result["dropped_detail"][0]["reason"], "path_not_in_diff")

    def test_finding_outside_the_diff_hunk_is_dropped(self):
        payload = ok_payload(
            verdict="ready_for_human_review",
            findings=[
                {
                    "path": "src/app.py",
                    "line": 900,
                    "severity": "info",
                    "message": "Line the diff does not touch.",
                }
            ],
        )
        result = build(payload, parse_diff(DIFF))
        self.assertEqual(result["findings"], [])
        self.assertEqual(result["dropped_detail"][0]["reason"], "line_not_in_diff")

    def test_objection_with_every_finding_discarded_is_refused(self):
        # An attacker-steered model can emit changes_requested with findings
        # against files the PR never touched. Publishing the verdict once the
        # evidence is stripped would put an unsupported objection in front of a
        # reader, so the whole result is refused instead.
        payload = ok_payload(
            findings=[
                {
                    "path": "src/secrets.py",
                    "line": 3,
                    "severity": "major",
                    "message": "Unanchored.",
                }
            ]
        )
        with self.assertRaises(Rejected):
            build(payload, parse_diff(DIFF))

    def test_ready_verdict_with_no_findings_at_all_is_fine(self):
        result = build(
            ok_payload(verdict="ready_for_human_review", findings=[]), parse_diff(DIFF)
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["findings"], [])

    def test_finding_count_is_capped(self):
        payload = ok_payload(
            findings=[
                {
                    "path": "src/app.py",
                    "line": 12,
                    "severity": "info",
                    "message": f"n{i}",
                }
                for i in range(80)
            ]
        )
        result = build(payload, parse_diff(DIFF))
        self.assertEqual(len(result["findings"]), 25)
        self.assertEqual(result["findings_dropped"], 55)


class TestTypeRevalidation(unittest.TestCase):
    """The producer is attacker-influenced, so its schema is not a guarantee."""

    def base(self, **over):
        item = {"path": "src/app.py", "line": 12, "severity": "info", "message": "m"}
        item.update(over)
        return item

    def test_bool_line_is_not_accepted_as_line_one(self):
        # bool subclasses int, so a naive isinstance(line, int) lets True
        # through as line 1 — which happens to be a real line in most diffs.
        kept, dropped = sanitize_findings([self.base(line=True)], parse_diff(DIFF))
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["reason"], "line_not_an_integer")

    def test_float_line_is_not_truncated_to_an_integer(self):
        kept, dropped = sanitize_findings([self.base(line=12.9)], parse_diff(DIFF))
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["reason"], "line_not_an_integer")

    def test_non_string_message_is_not_stringified(self):
        kept, dropped = sanitize_findings(
            [self.base(message={"a": 1})], parse_diff(DIFF)
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["reason"], "path_or_message_not_a_string")

    def test_unknown_severity_is_dropped_not_coerced_to_info(self):
        kept, dropped = sanitize_findings(
            [self.base(severity="catastrophic")], parse_diff(DIFF)
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["reason"], "bad_severity")

    def test_extra_finding_keys_are_rejected(self):
        kept, dropped = sanitize_findings([self.base(evil="payload")], parse_diff(DIFF))
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["reason"], "unexpected_keys")

    def test_extra_top_level_keys_are_rejected(self):
        payload = ok_payload()
        payload["exfil"] = "x" * 200
        with self.assertRaises(Rejected):
            build(payload, parse_diff(DIFF))

    def test_findings_must_be_a_list(self):
        with self.assertRaises(Rejected):
            build(ok_payload(findings={"not": "a list"}), parse_diff(DIFF))


class TestRoundTwoRegressions(unittest.TestCase):
    """One test per defect the second adversarial review found."""

    def test_control_characters_hidden_by_json_escaping_are_caught(self):
        # json.dumps renders ESC as the printable text , so charset-checking
        # the SERIALIZED form passes a real control sequence through to whatever
        # decodes it. The check must see decoded values.
        #
        # Round three changed the RESPONSE from reject to strip: rejecting made
        # one codepoint discard the whole review, and a model emits an em dash
        # in ordinary prose. The property under test is unchanged — no control
        # sequence reaches the artifact — but the review now survives.
        result = build(ok_payload(summary="All good \x1b[2Jcleared"), parse_diff(DIFF))
        self.assertEqual(result["status"], "succeeded")
        self.assertNotIn("\x1b", result["summary"])
        self.assertIn("cleared", result["summary"])

    def test_ordinary_prose_punctuation_does_not_discard_the_review(self):
        """The reason the response changed. An em dash is not an attack."""
        result = build(
            ok_payload(summary="Looks fine — one nit, the caller’s guard."),
            parse_diff(DIFF),
        )
        self.assertEqual(result["status"], "succeeded")
        self.assertIn("one nit", result["summary"])

    def test_an_encoded_blob_still_rejects_the_whole_review(self):
        """Stripping is for formatting. A long run that decodes to binary is a
        smuggling signal, and the response to a signal is still refusal."""
        with self.assertRaises(Rejected):
            build(ok_payload(summary=f"All good {smuggled_secret()}"), parse_diff(DIFF))

    def test_added_line_cannot_forge_a_diff_file_header(self):
        # An added source line whose content is "++ fake.py" renders as
        # "+++ fake.py". Without pairing +++ to a preceding ---, the parser would
        # treat it as a header and let findings anchor to a file never touched.
        forged = DIFF + "+++ fake.py\n+something\n"
        touched = parse_diff(forged)
        self.assertNotIn("fake.py", touched)

    def test_html_separator_cannot_reassemble_a_url(self):
        # HTML must be stripped BEFORE the URL rule, or <i></i> acts as a
        # separator that survives it and then vanishes.
        self.assertNotIn(
            "https://evil", neutralize("https:<i></i>//evil/path")
        )  # @lint-ignore

    def test_html_separator_cannot_reassemble_an_encoded_blob(self):
        half = base64.b64encode(bytes(range(256)) * 4).decode()[:70]
        payload = ok_payload(summary=f"{half}<i></i>{half}")
        with self.assertRaises(Rejected):
            build(payload, parse_diff(DIFF))

    def test_dropped_detail_paths_are_capped_and_neutralized(self):
        # dropped_detail is published, so attacker-controlled paths in it must
        # get the same caps and neutralization as a real finding.
        payload = ok_payload(
            verdict="ready_for_human_review",
            findings=[
                {
                    # Dots break the run into sub-120-char pieces so this
                    # exercises the CAP, not the encoded-blob guard.
                    "path": "@evil " + "longdirname." * 400,
                    "line": 1,
                    "severity": "info",
                    "message": "m",
                }
            ],
        )
        result = build(payload, parse_diff(DIFF))
        detail = result["dropped_detail"][0]["path"]
        self.assertLessEqual(len(detail), 400)
        self.assertNotIn("@evil", detail)

    def test_sha512_hex_digest_does_not_reject_the_review(self):
        # A 128-char SHA-512 hex digest is valid base64 syntax and decodes to
        # high-entropy bytes, so without the digest exemption it rejected
        # every review that quoted one.
        check_no_encoded_blob(f"artifact digest {'ab12cd34' * 16}", "t")

    def test_changes_requested_with_no_findings_at_all_is_refused(self):
        with self.assertRaises(Rejected):
            build(ok_payload(findings=[]), parse_diff(DIFF))

    def test_message_that_neutralizes_to_nothing_is_dropped(self):
        kept, dropped = sanitize_findings(
            [
                {
                    "path": "src/app.py",
                    "line": 12,
                    "severity": "info",
                    "message": "<b></b>",
                }
            ],
            parse_diff(DIFF),
        )
        self.assertEqual(kept, [])
        self.assertEqual(dropped[0]["reason"], "message_empty_after_neutralize")

    def test_invalid_utf8_does_not_crash_the_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "s.json").write_bytes(b'{"verdict": "\xff\xfe bad bytes"}')
            (p / "d.txt").write_text(DIFF)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--structured-output-file",
                    str(p / "s.json"),
                    "--diff-file",
                    str(p / "d.txt"),
                    "--out",
                    str(p / "v.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((p / "v.json").read_text())
            self.assertIsNone(result["verdict"])

    def test_deeply_nested_json_does_not_crash_the_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "s.json").write_text("[" * 5000 + "]" * 5000)
            (p / "d.txt").write_text(DIFF)
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--structured-output-file",
                    str(p / "s.json"),
                    "--diff-file",
                    str(p / "d.txt"),
                    "--out",
                    str(p / "v.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((p / "v.json").read_text())
            self.assertEqual(result["status"], "schema_invalid")

    def test_unreadable_diff_blocks_rather_than_publishing_a_verdict(self):
        # An unreadable diff means anchoring never ran. Falling back to an empty
        # diff would publish a ready verdict for a review that did not happen.
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "s.json").write_text(json.dumps(ok_payload()))
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--structured-output-file",
                    str(p / "s.json"),
                    "--diff-file",
                    str(p / "does-not-exist.txt"),
                    "--out",
                    str(p / "v.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            result = json.loads((p / "v.json").read_text())
            self.assertEqual(result["status"], "blocked")
            self.assertIsNone(result["verdict"])

    def test_git_quoted_paths_are_decoded(self):
        quoted = (
            'diff --git a/x b/x\n--- a/dir/old name.py\n+++ "b/dir/new name.py"\n'
            "@@ -1,1 +1,2 @@\n unchanged\n+added\n"
        )
        self.assertIn("dir/new name.py", parse_diff(quoted))


class TestVerdictIntegrity(unittest.TestCase):
    def test_unknown_verdict_is_rejected_rather_than_coerced(self):
        with self.assertRaises(Rejected):
            build(ok_payload(verdict="approved_and_merged"), parse_diff(DIFF))

    def test_empty_summary_is_rejected(self):
        with self.assertRaises(Rejected):
            build(ok_payload(summary="   "), parse_diff(DIFF))


class TestEndToEnd(unittest.TestCase):
    """The script must always emit a well-formed file and exit 0."""

    def run_script(self, structured: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            so = tmp_path / "structured.json"
            so.write_text(structured)
            diff = tmp_path / "diff.txt"
            diff.write_text(DIFF)
            out = tmp_path / "verdict.json"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--structured-output-file",
                    str(so),
                    "--diff-file",
                    str(diff),
                    "--out",
                    str(out),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            return json.loads(out.read_text())

    def test_success(self):
        result = self.run_script(json.dumps(ok_payload()))
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["verdict"], "changes_requested")

    def test_empty_output_becomes_model_error_not_a_verdict(self):
        result = self.run_script("")
        self.assertEqual(result["status"], "model_error")
        # The critical property: a failed review must never look like an objection.
        self.assertIsNone(result["verdict"])

    def test_malformed_json_becomes_schema_invalid(self):
        result = self.run_script("{not json")
        self.assertEqual(result["status"], "schema_invalid")
        self.assertIsNone(result["verdict"])

    def test_hostile_output_becomes_sanitizer_rejected(self):
        result = self.run_script(
            json.dumps(ok_payload(summary=f"All good {smuggled_secret()}"))
        )
        self.assertEqual(result["status"], "sanitizer_rejected")
        self.assertIsNone(result["verdict"])


class TestRoundThreeRegressions(unittest.TestCase):
    """One test per defect found in the 2026-09-01 three-model review.

    Each pairs the hostile input with a benign lookalike, because every fix here
    tightened something that legitimate reviews also exercise.
    """

    # --- the forged file header, which the ---/+++ pairing rule did not stop ---

    def test_header_pair_inside_a_hunk_body_cannot_forge_a_file(self):
        """A MODIFICATION can emit a real ---/+++ pair inside the hunk body.

        The old rule honoured any `+++` directly preceded by `---`, reasoning
        that an ADDED line reading `++ x` has no `---` before it. Change a line
        whose old text starts `-- ` into new text starting `++ ` and git emits
        the pair for real; the deletion satisfies the pairing and the counter is
        never reset, so the forged file inherits the last hunk's line numbers.
        """
        forged = (
            "diff --git a/notes.md b/notes.md\n"
            "--- a/notes.md\n"
            "+++ b/notes.md\n"
            "@@ -1,4 +1,4 @@\n"
            " line1\n"
            "--- a/torch/secret.py\n"
            "+++ b/torch/secret.py\n"
            " line3\n"
            " line4\n"
        )
        touched = parse_diff(forged)
        self.assertNotIn("torch/secret.py", touched)
        # The forged lines are body, so they belong to the file being modified.
        self.assertEqual(touched, {"notes.md": {1, 2, 3, 4}})

    def test_a_real_multi_file_diff_still_parses(self):
        """The benign lookalike: budget accounting must not lose real files."""
        real = (
            "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n"
            "@@ -1,2 +1,3 @@\n x\n+y\n z\n"
            "diff --git a/b.py b/b.py\n--- a/b.py\n+++ b/b.py\n"
            "@@ -10,1 +10,2 @@\n+new\n ctx\n"
        )
        self.assertEqual(parse_diff(real), {"a.py": {1, 2, 3}, "b.py": {10, 11}})

    def test_a_unicode_line_separator_cannot_desynchronise_the_budget(self):
        """`str.splitlines()` splits on U+2028/U+2029/VT/FF/NEL; git counts only
        LF. Once line COUNTS are load-bearing that disagreement is a forgery
        primitive: one separator inside a context line and git's `@@ -1,2 +1,2`
        stays honest while Python sees an extra line, the budget closes early,
        and what follows is read as a file header outside any hunk.

        Found by cross-model review of the budget fix itself — the first version
        used `splitlines()`.
        """
        for sep in (" ", " ", "\x0b", "\x0c", "\x85"):
            with self.subTest(sep=repr(sep)):
                forged = (
                    "diff --git a/n.md b/n.md\n--- a/n.md\n+++ b/n.md\n"
                    "@@ -1,2 +1,2 @@\n"
                    f" keep{sep}split\n"
                    "--- old\n+++ b/forged.py\n@@ -0,0 +42 @@\n+x\n"
                )
                self.assertNotIn("forged.py", parse_diff(forged))

    def test_a_short_hunk_body_cannot_swallow_the_next_stanza(self):
        """A budget that outlives its body would attribute the FOLLOWING file's
        lines to this one and lose that file entirely. An unprefixed line cannot
        occur inside a real hunk, so it ends the hunk and is reprocessed."""
        malformed = (
            "diff --git a/real.py b/real.py\n--- a/real.py\n+++ b/real.py\n"
            "@@ -1,3 +1,3 @@\n"
            "diff --git a/next.py b/next.py\nindex 111..222 100644\n"
            "--- a/next.py\n+++ b/next.py\n@@ -7 +7 @@\n x\n"
        )
        touched = parse_diff(malformed)
        self.assertEqual(touched.get("next.py"), {7})
        self.assertNotIn(7, touched.get("real.py", set()))

    def test_a_header_pair_without_a_diff_git_stanza_names_no_file(self):
        """Real `git diff` always emits `diff --git a/X b/X` before the pair, so
        a pair appearing without one did not come from git. Without this guard,
        any way of ending a hunk early is a header position."""
        forged = (
            "diff --git a/real.py b/real.py\n--- a/real.py\n+++ b/real.py\n"
            "@@ -1 +1 @@\n ok\n"
            "--- old\n+++ b/forged.py\n@@ -10 +10 @@\n x\n"
        )
        self.assertNotIn("forged.py", parse_diff(forged))

    def test_new_and_deleted_file_stanzas_still_parse(self):
        created = (
            "diff --git a/n.py b/n.py\nnew file mode 100644\n"
            "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1,2 @@\n+a\n+b\n"
        )
        self.assertEqual(parse_diff(created), {"n.py": {1, 2}})
        removed = (
            "diff --git a/g.py b/g.py\ndeleted file mode 100644\n"
            "--- a/g.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-a\n-b\n"
        )
        self.assertEqual(parse_diff(removed), {})

    def test_no_newline_marker_spends_no_budget(self):
        diff = (
            "diff --git a/e.py b/e.py\n--- a/e.py\n+++ b/e.py\n@@ -1 +1 @@\n"
            "-old\n\\ No newline at end of file\n+new\n\\ No newline at end of file\n"
        )
        self.assertEqual(parse_diff(diff), {"e.py": {1}})

    def test_hunk_without_counts_defaults_to_one_line_each_side(self):
        one = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -5 +5 @@\n-old\n+new\n"
        self.assertEqual(parse_diff(one), {"a.py": {5}})

    # --- the published path, which was truncated but never neutralized ---

    def test_a_hostile_filename_is_neutralized_in_the_published_path(self):
        name = "@pytorchbot [rebase](evil.example.com) #1.py"
        diff = f"diff --git a/x b/x\n--- a/x\n+++ b/{name}\n@@ -0,0 +1,2 @@\n+a\n+b\n"
        touched = parse_diff(diff)
        kept, _ = sanitize_findings(
            [{"path": name, "line": 1, "severity": "info", "message": "m"}], touched
        )
        self.assertEqual(len(kept), 1)
        published = kept[0]["path"]
        # Every markdown-active character is backslash-escaped, so the path
        # renders as itself and activates nothing. Asserting the ESCAPED form
        # rather than absence, because the point is that the published path is
        # still the NAME of the file that was validated.
        for i, char in enumerate(published):
            if char in "[]()@#*_":
                self.assertEqual(published[i - 1 : i], "\\", f"unescaped {char!r}")
        self.assertEqual(published.replace("\\", ""), name)

    def test_a_legitimate_scoped_path_survives_intact(self):
        """The benign lookalike, and the reason a path gets its own escaper:
        `neutralize` rewrote `@babel` to `@ babel`, so the published path no
        longer named the file it had just been validated against."""
        from extract_verdict import neutralize_path

        for path in ("node_modules/@babel/core/x.js", "src/app.py", "a b/c-d_e.py"):
            with self.subTest(path=path):
                self.assertEqual(neutralize_path(path).replace("\\", ""), path)

    def test_paths_outside_the_repo_never_become_anchors(self):
        for target in ("/etc/shadow", "b/../../x"):
            with self.subTest(target=target):
                diff = (
                    f"diff --git a/x b/x\n--- a/x\n+++ {target}\n@@ -0,0 +1,1 @@\n+a\n"
                )
                self.assertEqual(parse_diff(diff), {})

    def test_a_tab_in_a_quoted_path_never_reaches_the_output(self):
        """git quotes such a name and unquote_git_path decodes it faithfully.

        A tab is a column separator wherever this is rendered as a cell, so the
        anchor is refused rather than published.
        """
        diff = 'diff --git a/x b/x\n--- a/x\n+++ "b/evil\\t| fake | row |"\n@@ -0,0 +1,1 @@\n+a\n'
        self.assertEqual(parse_diff(diff), {})

    def test_a_non_ascii_filename_drops_one_finding_not_the_whole_review(self):
        """Availability, not injection: the charset check rejects the ENTIRE
        result, so an anchor it would trip has to be refused earlier."""
        diff = (
            "diff --git a/x b/x\n--- a/x\n+++ b/tests/données.py\n@@ -0,0 +1,1 @@\n+a\n"
        )
        self.assertEqual(parse_diff(diff), {})
        result = build(ok_payload(), parse_diff(DIFF))
        self.assertEqual(result["status"], "succeeded")

    # --- the tag pattern, which had two bounds and both were holes ---

    def test_tags_with_newlines_or_over_200_chars_are_stripped(self):
        for hostile in (
            "ok\n<img\nsrc=x onerror=alert(1)>\n",
            '<img src=x onerror="' + "a=1;" * 60 + '">',
            "no issues\n</details\n>\n# SECRET",
        ):
            with self.subTest(hostile=hostile[:30]):
                out = neutralize(hostile)
                self.assertNotIn("<", out)
                self.assertNotIn(">", out)

    def test_an_unterminated_comment_is_escaped_rather_than_left_open(self):
        """No closing `>`, so no tag pattern can match it. Left alone it
        comments out everything a renderer shows after it."""
        out = neutralize("real finding here <!--\nmore text")
        self.assertNotIn("<!--", out)
        self.assertIn("&lt;!--", out)

    def test_a_comparison_in_prose_is_not_treated_as_a_tag(self):
        """The benign lookalike. Stripping `<...>` costs this, knowingly."""
        self.assertNotIn("<", neutralize("guard should be a &lt; b"))

    # --- links: one pass was not enough, and // targets were not covered ---

    def test_nested_markdown_links_are_resolved_to_a_fixed_point(self):
        out = neutralize("[[Sign in](x)](//evil.example.com/phish)")
        self.assertNotIn("evil.example.com", out)
        self.assertNotIn("](", out)

    def test_scheme_relative_targets_are_removed(self):
        for hostile in (
            "See [page][1].\n\n[1]: //evil.example.com/phish",
            "bare //evil.example.com/x",
        ):
            with self.subTest(hostile=hostile[:30]):
                self.assertNotIn("evil.example.com", neutralize(hostile))

    def test_a_path_with_slashes_is_not_mistaken_for_a_scheme_relative_url(self):
        self.assertIn("src/app.py", neutralize("see src/app.py line 12"))

    # --- the notifying forms the lookbehind exempted ---

    def test_cross_repo_reference_and_slash_prefixed_mention_are_defused(self):
        out = neutralize("see torch/@pytorch-dev-infra and pytorch/pytorch#12345")
        self.assertNotIn("/@pytorch-dev-infra", out)
        self.assertNotIn("pytorch#12345", out)

    def test_an_email_address_is_still_left_alone(self):
        """The benign lookalike the `\\w` lookbehind exists for."""
        self.assertIn("user@example.com", neutralize("reported by user@example.com"))

    # --- the quadratic blowup ---

    def test_input_is_capped_before_the_backtracking_passes_run(self):
        """`_MD_IMAGE`/`_MD_LINK` backtrack from every start position, so the
        cap has to apply to the INPUT. Applied to the RESULT — which is what
        every caller used to do — this scaled 4x per doubling: 6.78s at 64KB and
        ~106s at 256KB, so the job hit its timeout and produced no artifact at
        all, which the telemetry cannot tell apart from a dead runner.

        The bound is deliberately loose. It is not a performance target; it is
        the difference between 'capped' and 'quadratic in attacker input', and
        the pre-fix code missed it by more than a minute.
        """
        huge = "![" * 131072  # 256KB
        started = time.monotonic()
        out = neutralize(huge, 1500)
        elapsed = time.monotonic() - started
        self.assertLessEqual(len(out), 1500)
        self.assertLess(
            elapsed, 5.0, f"neutralize took {elapsed:.1f}s on 256KB of input"
        )

    def test_an_oversized_document_is_refused_before_it_is_parsed(self):
        from extract_verdict import load_structured, MAX_STRUCTURED_BYTES

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.json"
            path.write_text(" " * (MAX_STRUCTURED_BYTES + 1))
            parsed, problem = load_structured(path)
            self.assertIsNone(parsed)
            self.assertTrue(problem.startswith("schema_invalid:"))

    # --- found by cross-model review OF THE FIXES above ---

    def test_the_issue_ref_defusal_does_not_manufacture_a_heading(self):
        """`#123` is NOT an ATX heading — those need a space after the hashes —
        so the naive defusal `# 123` CREATED one at line start. The remedy was
        the hazard."""
        out = neutralize("#123 breaks on Windows")
        self.assertFalse(out.startswith("# "), f"defusal produced a heading: {out!r}")
        self.assertIn("123", out)

    def test_deeply_nested_links_are_peeled_to_convergence(self):
        """One pass peels exactly one level, so a fixed bound is a depth the
        attacker can simply exceed. Depth 9 defeated a bound of 8."""
        deep = (
            "[" * 9
            + "go"
            + "".join(f"]({n})" for n in range(1, 9))
            + "](//user@evil.example.com/)"
        )
        out = neutralize(deep)
        self.assertNotIn("evil.example.com", out)
        self.assertNotIn("](", out)

    @staticmethod
    def _unescaped_brackets(out: str) -> str:
        """What is left once every backslash-escaped character is dropped.

        The invariant `neutralize` establishes is that no BARE `[` or `]`
        reaches the output — that is what makes link syntax inert — so the
        assertion has to ignore the escaped ones it deliberately produces.
        """
        return "".join(c for c in re.sub(r"\\.", "", out) if c in "[]")

    def test_no_link_form_survives_with_a_live_bracket(self):
        """Peeling cannot be the control, because a regex cannot match a
        balanced-bracket label: `_MD_LINK`'s `[^\\]]*` stops at the first `]`,
        so `[[x]](/evil)` was never matched at all and came out untouched. Nor
        is `_URL` a backstop — it wants a scheme or `//host`, so a relative
        destination and `javascript:alert(1)` both walked through. Escaping the
        surviving brackets is what closes the whole class at once.
        """
        hostile = (
            "[[x]](/evil)",  # balanced label
            "[[[deep]]](/evil)",
            "[a][b]\n\n[b]: /evil",  # reference link
            "[a]\n\n[a]: /evil",  # shortcut reference
            "[a][b]\n\n[b]: javascript:alert(1)",  # scheme _URL misses
            "[click](/logout)",  # relative inline
            "[" * 200 + "click" + "](/x)" * 200,  # past any constant bound
        )
        for probe in hostile:
            with self.subTest(probe=probe[:40]):
                out = neutralize(probe)
                self.assertEqual(
                    self._unescaped_brackets(out),
                    "",
                    f"a live bracket survived: {out[:90]!r}",
                )
        # A relative destination stays in the text as inert characters — that is
        # the point, not a miss: escaping defuses the LINK, it does not censor
        # prose. An ABSOLUTE one is a different matter, and `_URL` still takes
        # it, brackets or no brackets.
        self.assertNotIn(
            "evil.example.com", neutralize("[[x]](//user@evil.example.com/)")
        )

    def test_ordinary_prose_keeps_its_meaning_through_the_escape(self):
        """`\\[0\\]` renders as `[0]`, so escaping costs appearance in a raw
        reader and nothing in a markdown one — the trade `neutralize_path`
        already makes for paths."""
        out = neutralize("the index x[0] is unchecked")
        self.assertIn("x\\[0\\]", out)
        self.assertEqual(self._unescaped_brackets(out), "")

    def test_the_peel_loop_is_not_bought_with_unbounded_work(self):
        """Many unmatched `[` before a distant `]` makes each pass quadratic.
        Measured at 627 ms before the loop bound went back to a constant; this
        pins that the cheapest-per-level and the costliest shapes both stay
        cheap."""
        deep = "[]()"
        for _ in range(124):
            deep = "[" + deep + "]()"
        for label, payload in (
            ("max nesting", deep),
            ("quadratic shape", (deep + "[" * 599 + "![]()" * 80 + "]")[:MAX_SUMMARY]),
        ):
            with self.subTest(shape=label):
                start = time.monotonic()
                out = neutralize(payload, MAX_SUMMARY)
                # 5s, not a tight bound on the 627 ms measured locally: the
                # property worth pinning is "not minutes", and a shared runner
                # would flake anything close to the measurement.
                self.assertLess(time.monotonic() - start, 5.0, f"{label} got expensive")
                self.assertEqual(self._unescaped_brackets(out), "")

    def test_the_replacement_marker_cannot_be_used_as_a_link_label(self):
        """With square brackets the marker was itself a reference-link label:
        `[link removed]: //evil` then `Click [link removed]` produced a live
        link wearing the sanitizer's own text."""
        out = neutralize(
            "[link removed]: //user@evil.example.com/\nClick [link removed] now."
        )
        self.assertNotIn("evil.example.com", out)

    def test_backslash_escaped_destinations_are_defused(self):
        """CommonMark honours escapes inside a link destination, so
        `\\/\\/host` renders as `//host` while containing no `//` to match."""
        self.assertNotIn(
            "evil.example.com", neutralize("[x][1]\n\n[1]: \\/\\/evil.example.com")
        )

    def test_url_covers_userinfo_and_bare_addresses(self):
        for hostile in (
            "//user@evil.example.com/x",
            "//203.0.113.9/x",
            "mailto:a@evil.example.com",
        ):
            with self.subTest(hostile=hostile):
                out = neutralize(f"see {hostile} now")
                self.assertNotIn("evil.example.com", out)
                self.assertNotIn("203.0.113.9", out)

    def test_url_does_not_eat_a_buck_label_or_a_comment(self):
        """The benign lookalikes. The host alternative demands a dot before the
        first slash, which is what keeps these out."""
        for benign in ("//caffe2/core:core", "a // b", "// TODO: fix", "src/app.py"):
            with self.subTest(benign=benign):
                self.assertIn(benign.strip("/ "), neutralize(f"see {benign} here"))

    def test_structural_characters_are_refused_in_a_path(self):
        """`|` is the natural spelling of the table-row forgery the no-tab rule
        was justified by; refusing tab and allowing the pipe closed only the
        awkward one."""
        from extract_verdict import _is_repo_path

        for hostile in (
            "evil| fake | row |.py",
            "~/.ssh/id_rsa",
            "back`tick.py",
            r"\\server\share\x",
            "a/./b",
            "./x",
            "a//b",
        ):
            with self.subTest(hostile=hostile):
                self.assertFalse(_is_repo_path(hostile))

    def test_ordinary_paths_are_still_accepted(self):
        from extract_verdict import _is_repo_path

        for benign in (
            "src/app.py",
            "node_modules/@babel/core/x.js",
            "a b/c-d_e.py",
            "docs/README.md",
        ):
            with self.subTest(benign=benign):
                self.assertTrue(_is_repo_path(benign))

    # --- the step-outcome downgrade, moved out of an inline python3 -c ---

    def test_a_clean_verdict_from_a_failed_step_is_downgraded(self):
        from extract_verdict import downgrade

        clean = build(ok_payload(), parse_diff(DIFF))
        self.assertEqual(clean["status"], "succeeded")
        for outcome, expected in (
            ("failure", "model_error"),
            ("cancelled", "blocked"),
            ("skipped", "blocked"),
        ):
            with self.subTest(outcome=outcome):
                down = downgrade(clean, outcome)
                self.assertEqual(down["status"], expected)
                self.assertIsNone(down["verdict"])
                self.assertIn(outcome, down["failure_detail"])


if __name__ == "__main__":
    unittest.main()
