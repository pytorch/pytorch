#!/usr/bin/env python3
"""Structural contracts on the two hardened-PR-review workflow files.

The sanitizer suite next door covers the Python. Nothing covered the YAML, and
that is where the two worst defects of this build lived: a job that reads the
PR through the REST API without the scope that endpoint needs, and a job whose
one write capability pointed at a host its own egress allowlist did not permit.
Both are silent — the first 403s only on a private repo, the second only under
`egress-policy: block` — so neither shows up in review by reading the diff.

These are text assertions, deliberately: PyYAML is not available to the runner
and pulling it in to lint two files is a worse trade than a narrow parser.

WHAT THESE TESTS DEFEND AGAINST, because it bounds how far the readers below
have to go. Both files under contract live ONLY on the default branch — a pull
request cannot edit them, and the one file a PR does control, Stage 1, is
authenticated by the `github.event.workflow.path` pin these tests check. So the
adversary here is a MAINTAINER'S ACCIDENT: a plausible edit or reformat that
quietly removes a control. It is not someone hand-crafting YAML to fool a text
reader, and hardening against that was tried — each round of it produced
another legal spelling and no convergence. The readers therefore aim to be
right on the spellings a maintainer writes, and to FAIL CLOSED with a message
naming the reader on ones they do not model. Where that trade is live it is
written down at the reader, and `TestTheReadersPremisesStillHold` pins the
spellings at one place so a reformat is a loud named failure rather than a
quiet mis-read.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import itertools
import json
import os
import posixpath
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import TestTheSuiteIsWhole  # noqa: E402,F401

# Imported, not restated: the rubric contract below has to compare against the
# set the sanitizer actually enforces, or the two drift apart silently.
from extract_verdict import SEVERITIES  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
STAGE1 = WORKFLOWS / "hardened-pr-review.yml"
STAGE2 = WORKFLOWS / "hardened-pr-review-run.yml"
SUITE_CI = WORKFLOWS / "pr-review-scripts-test.yml"
RUBRIC = REPO / ".claude" / "skills" / "pr-review-readiness" / "SKILL.md"
HOOK = REPO / ".claude" / "hooks" / "pr_review" / "restrict-write.sh"


def strip_comments(text: str) -> str:
    """Drop whole-line `#` comments so prose cannot satisfy a contract."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def job_block(text: str, name: str) -> str:
    """The lines of one job, from `  name:` to the next job key at that indent."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^  {re.escape(name)}:\s*$", ln):
            start = i
            break
    assert start is not None, f"job {name!r} not found"
    for j in range(start + 1, len(lines)):
        if re.match(r"^  [A-Za-z_][\w-]*:\s*$", lines[j]):
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


def _key_re(key: str) -> re.Pattern:
    """Match `key:` however YAML lets it be spelled, capturing the rest.

    Quoted (`"run":`) and block headers carrying an indentation indicator or a
    chomping marker (`|2`, `|-`, `>-`) are all the same key with the same
    scalar content, as is whitespace before the colon. Comparing the line to
    one exact string rejected every one of them — a false RED on a cosmetic
    edit, which is how a contract test earns itself a deletion.
    """
    return re.compile(rf"""^[ \t]*["']?{re.escape(key)}["']?[ \t]*:[ \t]*(.*)$""")


# A block-scalar header: `|`, `>`, either chomping marker, an optional explicit
# indentation indicator in either order, and an optional `!!str` tag — all of
# which name the same string. Shared so the readers below cannot drift into
# disagreeing about what a header looks like.
BLOCK_HEADER = re.compile(r"(?:!!str[ \t]+)?[|>](?:[-+]?\d*|\d*[-+]?)")


def uncommented(raw: str) -> str:
    """Drop a trailing YAML comment — one introduced by WHITESPACE then `#`.

    A bare `str.split("#")` is not that rule, and the difference is a real
    path: `#` with no space before it is part of the scalar, so `trusted#/..`
    is ONE value that normalizes to `.`, while splitting on `#` reported a
    harmless `trusted` and validated that instead of the value.

    QUOTE-AWARE, because `#` inside a quoted scalar is content: truncating
    `"a # b"` to `"a` both corrupts the value and leaves the quotes unbalanced,
    and doing it to a gate expression red a workflow YAML had not changed.
    """
    quote, i = "", 0
    while i < len(raw):
        c = raw[i]
        if quote:
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c == "#" and (i == 0 or raw[i - 1] in " \t"):
            return raw[:i].strip()
        i += 1
    return raw.strip()


def plain_scalar(raw: str) -> str:
    """A scalar's VALUE, refusing any spelling this reader cannot decode.

    `str.strip("\\"'")` is not unquoting: it eats characters from both ends
    until neither is a quote, so `"'x'"` — whose value really is `'x'`, with
    the apostrophes — came back as `x` and matched a required entry the file
    does not contain. And a DOUBLE-quoted scalar processes escapes, so
    `"\\u0021**"` is the exclusion `!**` while the raw text starts with a
    backslash and passes every check written against the literal.
    """
    v = uncommented(raw)
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        inner = v[1:-1]
        assert v[0] not in inner, f"nested or doubled quote in the scalar {v!r}"
        assert not (v[0] == '"' and "\\" in inner), (
            f"the double-quoted scalar {v!r} carries a backslash escape, which "
            "YAML decodes and this reader does not — teach it, or unquote."
        )
        return inner
    return v


# Any mapping key, and the same with a block-scalar header as its value.
_ANY_KEY = re.compile(r"""^([ \t]*)["']?[\w.-]+["']?[ \t]*:(?:[ \t].*|)$""")
_ANY_BLOCK_KEY = re.compile(r"""^[ \t]*["']?([\w.-]+)["']?[ \t]*:[ \t]*(\S.*)$""")


def block_scalars(text: str, key: str) -> list[tuple[str, list[str]]]:
    """Every TOP-LEVEL `key:` block scalar in `text`, as (header, body lines).

    Three rules, each closing a hole a simpler scan left open:

    * Step over every block scalar's BODY. A scanner that walked all lines read
      a step's own shell as YAML — a heredoc whose payload contains `run: |`
      looked like a second key.
    * Step over EVERY scalar's body, not just this key's. An `env:` value
      spelled `EXAMPLE: |` holding an indented `run: |` and a script under it
      impersonated the step's command.
    * Match only at the OUTERMOST key indentation in `text`. Skipping bodies
      does not establish OWNERSHIP: a block scalar genuinely named `run:` but
      nested inside the step's `env:` mapping is not the step's command, and
      with the real `run: ":"` left as a plain scalar the reader extracted the
      decoy and the executable tests passed against a script Actions never
      runs.
    """
    lines = text.splitlines()

    def walk():
        """(indent, key, header, body) for each block scalar, bodies skipped."""
        i = 0
        while i < len(lines):
            m = _ANY_BLOCK_KEY.match(lines[i])
            header = uncommented(m.group(2)) if m else ""
            if not header or not BLOCK_HEADER.fullmatch(header):
                i += 1
                continue
            indent = len(lines[i]) - len(lines[i].lstrip())
            name, body, i = m.group(1), [], i + 1
            while i < len(lines):
                if (
                    lines[i].strip()
                    and len(lines[i]) - len(lines[i].lstrip()) <= indent
                ):
                    break
                body.append(lines[i])
                i += 1
            yield indent, name, header, body

    found = list(walk())
    # The outermost mapping level, measured the same way — over key lines only,
    # with block bodies skipped, so shell text spelled `foo: bar` cannot lower
    # it and hide a nested decoy.
    keys, i = [], 0
    while i < len(lines):
        m = _ANY_BLOCK_KEY.match(lines[i])
        header = uncommented(m.group(2)) if m else ""
        if (k := _ANY_KEY.match(lines[i])) is not None:
            keys.append(len(k.group(1)))
        if not header or not BLOCK_HEADER.fullmatch(header):
            i += 1
            continue
        indent = len(lines[i]) - len(lines[i].lstrip())
        i += 1
        while i < len(lines):
            if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= indent:
                break
            i += 1
    top = min(keys) if keys else 0
    return [(h, b) for indent, name, h, b in found if name == key and indent == top]


def scalar_block(text: str, key: str) -> str:
    """The body of the one block scalar introduced by `key:`.

    NOT ended on the literal next key, and NOT split on an exact `key: |\\n`.
    `str.split` on a separator that is not present returns the whole remainder
    with no error — or, with the newline attached, raises IndexError — so the
    legal `run: |-`, `run: |2` and `run: | # why` each broke a reader that
    matched one spelling.
    """
    blocks = block_scalars(text, key)
    assert len(blocks) == 1, f"premise changed: {len(blocks)} `{key}` block scalars"
    assert blocks[0][1], f"premise changed: the `{key}` block is empty"
    return "\n".join(blocks[0][1])


def _scopes(mapping: str) -> str:
    """`scope: level` pairs, decoded and normalized, as `a: x, b: y`.

    Comparing the RAW text meant `{contents: "read"}` — the very permission
    required, quoted — did not equal `contents: read`, and so did quoting the
    key. Both spellings of the mapping, inline and block, arrive here so the
    comparison sees values rather than punctuation.
    """
    pairs = []
    for part in mapping.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition(":")
        pairs.append(f"{plain_scalar(k)}: {plain_scalar(v)}")
    return ", ".join(pairs)


def with_block(step: str) -> list[str]:
    """The lines of a step's `with:` mapping — the action's INPUTS, only.

    `path:` anywhere in the step is not the checkout's `path` input. Moving it
    into the step's `env:` mapping leaves the same key, the same plausible
    value and the same one match, while `actions/checkout` receives no path at
    all and lands on the workspace root — the exact condition two tests here
    exist to forbid.
    """
    lines = step.splitlines()
    hits = [
        i
        for i, ln in enumerate(lines)
        if (m := _key_re("with").match(ln)) and not uncommented(m.group(1))
    ]
    assert len(hits) == 1, f"expected one `with:` mapping in the step, got {len(hits)}"
    i = hits[0]
    indent = len(lines[i]) - len(lines[i].lstrip())
    out = []
    for nxt in lines[i + 1 :]:
        # A COMMENT DOES NOT END A MAPPING, whatever column it sits in. Ending
        # on one truncated `with:` at an ordinary explanatory comment written
        # level with the key, dropping the `path:` under it and reddening two
        # tests over a file YAML reads exactly as before.
        if nxt.lstrip().startswith("#"):
            continue
        if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
            break
        out.append(nxt)
    assert out, "the step's `with:` mapping is empty"
    return out


def indented_block(text: str, key: str) -> list[str]:
    """Values of a block scalar or list introduced by `key:`, as raw lines."""
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith(key):
            indent = len(ln) - len(ln.lstrip())
            out = []
            for nxt in lines[i + 1 :]:
                if not nxt.strip():
                    continue
                if len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                out.append(nxt)
            return out
    raise AssertionError(f"{key!r} not found")


class TestPreparePermissions(unittest.TestCase):
    """`prepare` calls `gh api repos/.../pulls/N`, which needs the PR scope.

    `contents: read` does not cover the pulls endpoint, and declaring any
    permissions block defaults every unlisted scope to `none`. A public repo
    masks the miss because that endpoint serves public data unauthenticated;
    A private repo would 403 and take the whole run with it; declare it regardless.
    """

    def test_prepare_reads_prs_and_declares_the_scope(self):
        stage2 = STAGE2.read_text()
        prepare = job_block(stage2, "prepare")
        self.assertIn(
            "pulls/",
            strip_comments(prepare),
            "premise changed: prepare no longer reads the PR API",
        )
        perms = indented_block(prepare, "permissions:")
        self.assertIn("pull-requests: read", "\n".join(perms))

    def test_review_job_holds_no_pr_scope(self):
        """The converse: the job that reads untrusted code must NOT gain it."""
        perms = indented_block(job_block(STAGE2.read_text(), "review"), "permissions:")
        joined = strip_comments("\n".join(perms))
        self.assertNotIn("pull-requests", joined)
        self.assertNotIn("write", joined.replace("id-token: write", ""))


class TestEgressAllowlistCoversItsOwnWrites(unittest.TestCase):
    """Every host the review job actually talks to must be allowlisted.

    The trace upload is this job's single write capability. It is best-effort
    (`|| echo ::warning::`), so a blocked endpoint costs a silent, permanent
    loss of every transcript plus a reasoning_uri pointing at nothing.
    """

    def setUp(self):
        self.text = STAGE2.read_text()
        self.review = job_block(self.text, "review")
        self.allowlist = " ".join(indented_block(self.review, "allowed-endpoints:"))

    def test_block_scalar_holds_no_comment_lines(self):
        """`>` makes indented lines content — a `#` line becomes a bogus entry."""
        for ln in indented_block(self.review, "allowed-endpoints:"):
            self.assertFalse(
                ln.lstrip().startswith("#"), f"comment inside block scalar: {ln!r}"
            )

    def test_every_s3_bucket_written_is_allowlisted(self):
        bucket = re.search(r"^\s*S3_BUCKET:\s*(\S+)", self.text, re.M)
        self.assertIsNotNone(bucket, "S3_BUCKET env not found")
        name = bucket.group(1)
        writes = re.findall(r"s3://\$\{S3_BUCKET\}/", strip_comments(self.review))
        self.assertTrue(writes, "premise changed: review job no longer writes to S3")
        self.assertIn(
            f"{name}.s3.", self.allowlist, f"{name} is written but not allowlisted"
        )

    def test_egress_policy_is_block(self):
        self.assertIn("egress-policy: block", self.review)


class TestCredentialDuration(unittest.TestCase):
    """900s is the STS minimum and the stated policy; the role ceiling is 1h.

    Omitting the line does not fail anything — it silently widens a stolen
    credential from 15 minutes to an hour, which is exactly why a comment in
    this workflow calls the line load-bearing.
    """

    def test_every_role_assumption_requests_the_minimum(self):
        text = strip_comments(STAGE2.read_text())
        steps = text.split("uses: aws-actions/configure-aws-credentials")
        self.assertGreaterEqual(len(steps) - 1, 3, "expected three role assumptions")
        for i, step in enumerate(steps[1:], 1):
            head = step.split("- name:")[0]
            self.assertIn(
                "role-duration-seconds: 900",
                head,
                f"assumption #{i} omits the 15-minute bound",
            )


class TestTriggerTypesAgreeAcrossStages(unittest.TestCase):
    """Stage 2 re-validates `trigger_event` against a literal allowlist.

    It is an allowlist, so drift does not fail open — a type Stage 1 emits and
    Stage 2 does not know fails every run of that type at the regex. That is
    safe and completely invisible until someone uses the trigger.
    """

    def test_stage2_accepts_every_type_stage1_emits(self):
        types = re.search(r"types:\s*\[([^\]]*)\]", strip_comments(STAGE1.read_text()))
        self.assertIsNotNone(types)
        emitted = {t.strip() for t in types.group(1).split(",") if t.strip()}
        pattern = re.search(r'TRIGGER"\s*=~\s*\^\(([^)]*)\)\$', STAGE2.read_text())
        self.assertIsNotNone(pattern, "trigger_event validation regex not found")
        accepted = set(pattern.group(1).split("|"))
        self.assertEqual(
            emitted - accepted, set(), "Stage 1 emits types Stage 2 rejects"
        )

    def test_ready_for_review_is_covered(self):
        """A PR labelled while draft gets no review until this event exists."""
        self.assertIn("ready_for_review", strip_comments(STAGE1.read_text()))


class TestTheSuiteActuallyRunsInCI(unittest.TestCase):
    """Every other contract in this file assumes something executes it.

    The suite was unwired for the whole of this build: no workflow ran
    `unittest discover`, so each invariant below could have been deleted in a
    green pull request. This class pins the wiring itself — including the paths
    filter, because the contract tests assert on workflow TEXT and would
    otherwise not run for a change that touches only YAML.
    """

    REQUIRED_PATHS = (
        "scripts/pr_review/**",
        ".github/workflows/hardened-pr-review.yml",
        ".github/workflows/hardened-pr-review-run.yml",
        ".claude/hooks/pr_review/**",
        ".claude/skills/pr-review-readiness/**",
        # This file's OWN path. Without it, an edit that rewires or weakens the
        # wiring is the one change that schedules no run of the suite checking
        # the wiring.
        ".github/workflows/pr-review-scripts-test.yml",
    )

    def setUp(self):
        self.assertTrue(
            SUITE_CI.is_file(), f"{SUITE_CI.name} is missing — the suite runs nowhere"
        )
        self.text = SUITE_CI.read_text()

    def test_it_runs_the_discovery_form_over_every_module(self):
        """Naming one module skips the other two, silently and greenly."""
        self.assertIn(
            "python3 -m unittest discover -s scripts/pr_review -t .",
            strip_comments(self.text),
        )

    def test_it_refuses_a_discovery_suppressing_package_hook(self):
        """The one guard that cannot live inside the suite it guards.

        `load_tests` on `scripts/pr_review/__init__.py` stops discovery before a
        single test is collected, so every in-suite check is silent about it.
        This step runs outside the subtree; without it the suite's own
        completeness guarantee is void.

        It also covers the emptied suite, whose exit code is interpreter-
        dependent, so neither of the two "the suite never ran" cases rests on a
        test inside the suite.

        Pinned on the REJECTION, not just the words: a step that evaluates
        `hasattr` and discards the answer would satisfy a substring check while
        guarding nothing, so each condition is pinned together with the
        `raise SystemExit` that acts on it.
        """
        body = strip_comments(self.text)
        step = "\n".join(
            indented_block(body, "- name: Refuse a suppressed or emptied suite")
        )
        self.assertTrue(
            step.strip(), "the preflight step is gone; nothing checks either case"
        )
        self.assertIn("import scripts.pr_review as p", step)
        self.assertIn("if hasattr(p, 'load_tests'):", step)
        self.assertIn("EXPECTED_MODULES", step)
        self.assertIn("if missing:", step)
        # Two conditions, and each must actually reject. `assert` would not: -O
        # or a PYTHONOPTIMIZE in the environment strips it and the step passes.
        self.assertEqual(
            step.count("raise SystemExit"), 2, "a checked condition does not reject"
        )

    def test_the_preflight_runs_before_the_suite(self):
        """It is worthless after the step whose result it is meant to qualify."""
        body = strip_comments(self.text)
        self.assertLess(
            body.index("- name: Refuse a suppressed or emptied suite"),
            body.index("- name: Run the pr_review suite"),
            "the preflight runs after the suite it is supposed to gate",
        )

    def _halves(self) -> tuple[str, str]:
        """(push half, pull_request half) of `on:`, split on the PR key.

        It assumes `push:` precedes `pull_request:`, so the order is ASSERTED
        below rather than relied on — reordering them used to make the PR
        assertions silently read the push list instead.
        """
        body = strip_comments(self.text)
        halves = body.split("pull_request:", 1)
        self.assertEqual(len(halves), 2, "no pull_request trigger: nothing gates a PR")
        return halves[0], halves[1]

    def test_the_two_triggers_are_in_the_order_the_slicing_assumes(self):
        """The premise `_halves` rests on, asserted instead of assumed."""
        body = strip_comments(self.text)
        self.assertLess(
            body.index("push:"),
            body.index("pull_request:"),
            "the triggers were reordered; the path assertions below now read "
            "the wrong filter and would pass while one of them is empty",
        )

    @staticmethod
    def _path_items(half: str) -> list[str]:
        """The items of the `paths:` list — that exact key — each a whole entry.

        Scoped to the key and its indented block, not "every `- x` in the
        half": renaming `paths:` to `paths-ignore:` INVERTS the filter while
        leaving every entry where a whole-half scan would still find it.

        Whole entries, because substring membership is not the same test and
        the difference defeats exactly the coverage this filter provides: both
        `"!.claude/hooks/pr_review/**"` — an EXCLUSION — and
        `".claude/hooks/pr_review/**/*.py"` — a NARROWING that misses
        `restrict-write.sh` — contain the required string while the filter no
        longer schedules the run.
        """
        lines = half.splitlines()
        pat = _key_re("paths")  # `"paths":` is the same key, quoted
        starts = [
            i
            for i, ln in enumerate(lines)
            if (m := pat.match(ln)) and not m.group(1).strip()
        ]
        assert len(starts) == 1, f"expected one `paths:` key, found {len(starts)}"
        i = starts[0]
        key_indent = len(lines[i]) - len(lines[i].lstrip())
        items, seq_indent = [], None
        for nxt in lines[i + 1 :]:
            if not nxt.strip():
                continue
            indent = len(nxt) - len(nxt.lstrip())
            entry = re.fullmatch(r"-[ \t]+(.*)", nxt.strip())
            if seq_indent is None:
                # A sequence may be indented level with its key — legal YAML,
                # and breaking on `<= key_indent` returned an EMPTY list for
                # it, which passes nothing and fails everything.
                if indent < key_indent or (indent == key_indent and not entry):
                    break
                assert entry, f"a `paths:` list starts with a non-item line: {nxt!r}"
                seq_indent = indent
            elif indent < seq_indent or (
                # A SIBLING KEY ends the list. With the sequence indented level
                # with its own key, a following `branches: [main]` sits at the
                # entries' indentation, and stopping only on a SMALLER indent
                # folded it into the last pattern.
                indent == seq_indent and not entry and _ANY_KEY.match(nxt)
            ):
                break
            # THE SEQUENCE'S OWN INDENTATION decides what is an entry. Matching
            # `- …` on the stripped line counted a MORE-INDENTED `- x` as a
            # sixth entry, when YAML folds it — hyphen and all — into the
            # entry above; five required patterns were reported present while
            # the filter held one folded pattern matching none of them.
            if indent == seq_indent and entry:
                raw = entry.group(1)
                assert not BLOCK_HEADER.fullmatch(uncommented(raw)), (
                    f"a `paths:` entry is a block scalar ({uncommented(raw)!r}); "
                    "this reader does not decode one, and `>- !**` would read "
                    "as a harmless string while excluding every path."
                )
                items.append(plain_scalar(raw))
                continue
            # A plain scalar CONTINUES onto a more-indented line and YAML FOLDS
            # the two with a space, so `- scripts/pr_review/**` over an
            # indented `/elsewhere` is the ONE pattern `scripts/pr_review/**
            # /elsewhere`, which matches nothing the first line does. Folded
            # here for the same reason, or the membership test below would see
            # a required entry that the filter does not actually contain.
            assert items, f"a `paths:` list starts with a continuation: {nxt!r}"
            items[-1] = f"{items[-1]} {uncommented(nxt.strip())}".strip()
        return items

    def _assert_covers(self, half: str, which: str) -> None:
        items = self._path_items(half)
        for path in self.REQUIRED_PATHS:
            self.assertIn(
                path,
                items,
                f"{path} is not a whole entry in the {which} paths filter; "
                f"entries are {items}",
            )
        # An exclusion later in the list wins over every positive entry above
        # it, so a single `!**` disables the filter while leaving each required
        # entry exactly where this test looks for it.
        self.assertEqual(
            [e for e in items if e.startswith("!")],
            [],
            f"the {which} paths filter carries an EXCLUSION, which overrides "
            f"the entries above it: {items}",
        )

    def test_pull_request_triggers_it_for_python_and_for_yaml(self):
        _, pr = self._halves()
        self._assert_covers(pr, "pull_request")

    def test_push_to_main_triggers_it_for_the_same_paths(self):
        """BOTH filters, because only one of them was ever checked.

        This file carries two independent `paths:` lists. The test above reads
        the text AFTER `pull_request:`, so a path silently dropped from the
        `push:` list changed nothing it could see — and that is the list that
        runs the suite on `main`. A regression merged to main would then never
        be caught there, only on the next PR that happens to touch a path still
        in the other filter.
        """
        push, _ = self._halves()
        # The BRANCH, not merely the presence of a `branches:` key: changing
        # `[main]` to `[release]` disables the main-branch run entirely while
        # leaving every path assertion satisfied. `!main` is an EXCLUSION, so
        # the `!` is kept rather than tokenized away. The one-line flow-list
        # spelling this reads is pinned by TestTheReadersPremisesStillHold.
        m = next(
            (mm for ln in push.splitlines() if (mm := _key_re("branches").match(ln))),
            None,
        )
        self.assertIsNotNone(m, "premise changed: push has no branches filter")
        listed = re.findall(r"!?[\w./*-]+", uncommented(m.group(1)))
        self.assertTrue(
            listed,
            "the push `branches:` filter is no longer an inline list; this "
            "reader is line-at-a-time — see TestTheReadersPremisesStillHold."
            "test_the_suite_ci_branch_filter_is_a_single_line_flow_list",
        )
        self.assertIn(
            "main", listed, f"the push trigger does not run on main: {listed}"
        )
        self.assertNotIn("!main", listed, "the push trigger EXCLUDES main")
        self._assert_covers(push, "push")

    def test_it_holds_no_more_capability_than_the_workflows_it_tests(self):
        """It executes PR-authored Python, so it must stay a read-only
        `pull_request` job — never `pull_request_target`, never a write scope,
        never a secret."""
        body = strip_comments(self.text)
        self.assertNotIn("pull_request_target", body)
        # BOTH ways to reach the context. `secrets.` alone missed the equally
        # ordinary `secrets['CI_TOKEN']`, which a maintainer wiring up a test
        # dependency would plausibly write, and which hands the value to
        # PR-authored Python.
        self.assertNotRegex(body, r"secrets\s*[.\[]")
        self.assertNotIn("id-token", body)
        # EVERY `permissions:` block, not the first one. A job-level block
        # OVERRIDES the workflow-level one, so `jobs.test.permissions.contents:
        # write` left the top-level `contents: read` exactly where a
        # first-match reader looks for it while the job ran with write.
        lines = body.splitlines()
        blocks = []
        for i, ln in enumerate(lines):
            m = _key_re("permissions").match(ln)
            if not m:
                continue
            # BOTH SPELLINGS. Skipping keys that carry a value read the INLINE
            # flow mapping `permissions: {contents: write}` as "not a
            # permissions block", so a job-level override sat one line away
            # from an assertion that was still passing on the workflow-level
            # `contents: read` above it.
            inline = uncommented(m.group(1))
            if inline:
                blocks.append(_scopes(inline.strip().strip("{}")))
                continue
            # Read the block from THIS line's indentation. Handing the key back
            # to `indented_block`, which matches a literal `permissions:`,
            # meant the two readers disagreed about what the key is: `_key_re`
            # accepts `"permissions":` and `permissions :`, and either
            # cosmetic edit then raised "not found" from the second reader.
            indent = len(lines[i]) - len(lines[i].lstrip())
            out = []
            for nxt in lines[i + 1 :]:
                if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                if nxt.strip():
                    out.append(nxt.strip())
            blocks.append(_scopes(",".join(out)))
        self.assertTrue(blocks, "the workflow declares no permissions block")
        for perms in blocks:
            self.assertEqual(perms, "contents: read")


class TestReviewLabelAgreesAcrossStages(unittest.TestCase):
    """The gating label is spelled in three places and nothing keeps them level.

    Stage 1 carries it as `env.REVIEW_LABEL` and again as a literal in its
    job-level `if:` — unavoidable, because the `env` context is not available
    there — and Stage 2 carries the trusted copy that decides eligibility. Move
    one and the pipeline does not break loudly: Stage 1 keeps firing while
    Stage 2 reads the PR's labels for a string nobody applies, calls every run
    ineligible, and writes no rows. A review that silently stops happening is
    the exact failure class this file exists for.
    """

    @staticmethod
    def _env_label(text: str) -> str:
        m = re.search(r"^\s*REVIEW_LABEL:\s*(.+?)\s*$", strip_comments(text), re.M)
        assert m is not None, "REVIEW_LABEL env not found"
        return m.group(1).strip().strip("\"'")

    def test_both_stages_declare_the_same_label(self):
        self.assertEqual(
            self._env_label(STAGE1.read_text()), self._env_label(STAGE2.read_text())
        )

    def test_the_job_if_literals_match_the_declared_label(self):
        """The `if:` cannot read `env`, so these two are hand-copied."""
        label = self._env_label(STAGE1.read_text())
        body = strip_comments(STAGE1.read_text())
        added = re.search(r"github\.event\.label\.name\s*==\s*'([^']*)'", body)
        carried = re.search(
            r"contains\(github\.event\.pull_request\.labels\.\*\.name,\s*'([^']*)'\)",
            body,
        )
        self.assertIsNotNone(
            added, "the `labeled` branch of the job `if:` was not found"
        )
        self.assertIsNotNone(
            carried, "the already-labelled branch of the job `if:` was not found"
        )
        self.assertEqual(
            added.group(1), label, "the `labeled` literal drifted from REVIEW_LABEL"
        )
        self.assertEqual(
            carried.group(1), label, "the `contains` literal drifted from REVIEW_LABEL"
        )


class TestCorroboratedValuesAreTheOnlyOnesOffered(unittest.TestCase):
    """`base_sha` and `is_fork` must reach a row only from the trusted API.

    Both arrive first in the Stage-1 artifact, where the PR author writes them:
    an unverified `base_sha` selects what the model is shown, and an unverified
    `is_fork` mislabels every row. `prepare` re-derives both from the REST API,
    and used to export the artifact's copies alongside — dead, but one
    autocomplete away from being picked, with only a comment against it.
    """

    def test_the_artifact_copies_are_not_exported_at_all(self):
        prepare = strip_comments(job_block(STAGE2.read_text(), "prepare"))
        coords = prepare.split("id: freshness", 1)[0]
        for field in ("base_sha", "is_fork"):
            self.assertNotIn(
                f'echo "{field}=',
                coords,
                f"the coords step exports an unverified {field}",
            )

    def test_every_consumer_reads_the_freshness_copy(self):
        text = strip_comments(STAGE2.read_text())
        for field in ("base_sha", "is_fork"):
            self.assertNotIn(
                f"steps.coords.outputs.{field}",
                text,
                f"a consumer reads the unverified {field}",
            )
            self.assertIn(f"steps.freshness.outputs.{field}", text)


class TestReviewJobConfigDiscoveryIsClosed(unittest.TestCase):
    """`--setting-sources user` is the boundary, and nothing else enforces it.

    Claude Code loads CLAUDE.md, .claude/settings.json and .mcp.json from the
    working tree and its ancestors. A hook in that settings file is arbitrary
    command execution that runs BEFORE any --allowedTools policy applies, so
    the read-only-tools story is void without this flag.

    THE VALUE IS LOAD-BEARING IN BOTH DIRECTIONS, and this test used to demand
    the one spelling that is wrong on both counts. It required
    `--setting-sources ""`, which was measured against the CLI directly — but
    the deployment goes through claude-code-action, whose parser records a flag
    with a falsy next token as a boolean `null`
    (base-action/src/parse-sdk-options.ts), after which its own fallback loads
    `user,project,local`. So the empty form left discovery fully ON while this
    test passed, because it greps YAML rather than the parser.

    And the empty form must not start working either: the action installs the
    `settings:` block — our three hooks — into ~/.claude/settings.json, the
    `user` source. `[]` would disable restrict-write.sh, the only fence on the
    bare `Write` grant. `user` is the single value that excludes the PR tree and
    keeps the hooks.
    """

    @staticmethod
    def _claude_args() -> str:
        """Just the block scalar — prompt prose must not be able to satisfy these."""
        review = strip_comments(job_block(STAGE2.read_text(), "review"))
        body = review.split("claude_args: |", 1)
        assert len(body) == 2, "premise changed: no claude_args block"
        return body[1].split("\n          prompt:", 1)[0]

    def test_review_step_scopes_setting_sources_to_user(self):
        args = self._claude_args()
        self.assertRegex(
            args,
            r"--setting-sources\s+user(\s|$)",
            "review job no longer scopes setting discovery to `user`",
        )
        self.assertEqual(
            len(re.findall(r"--setting-sources", args)),
            1,
            "more than one --setting-sources: a later one silently wins",
        )

    def test_the_empty_form_the_action_ignores_is_not_used(self):
        args = self._claude_args()
        self.assertNotRegex(
            args,
            r'--setting-sources[\s=]+""',
            "the empty form parses to a boolean and re-enables project/local",
        )

    def test_only_one_add_dir_because_the_flag_does_not_accumulate(self):
        # `add-dir` is absent from the action's ACCUMULATING_FLAGS, so a second
        # occurrence overwrites the first instead of adding to it. A review
        # silently loses the directory it writes its verdict into.
        args = self._claude_args()
        self.assertEqual(
            len(re.findall(r"--add-dir", args)),
            1,
            "more than one --add-dir: this action keeps only the last",
        )
        self.assertRegex(args, r"--add-dir\s+\$\{\{\s*runner\.temp\s*\}\}")


class TestTheFindingsPathIsPinned(unittest.TestCase):
    """The tool rule, the prompt and the hook must name one findings path.

    The workflow step that claims to check this compares two values it declares
    itself. Nothing compared the literal inside `--allowedTools`, which is the
    one an edit would actually drift.
    """

    def setUp(self):
        self.review = strip_comments(job_block(STAGE2.read_text(), "review"))

    # One expected value, and every site that must carry it named explicitly.
    # A regex over "whatever follows runner.temp" cannot see the drift that
    # matters: retargeting the prompt to `...findings.txt`, or the hook env to
    # `nested/...findings.json`, stops matching and so stops being checked,
    # while the remaining references keep any aggregate assertion green.
    EXPECTED = "${{ runner.temp }}/pr-review-findings.json"

    def test_the_read_grant_names_the_findings_file(self):
        self.assertIn(f"Read(/{self.EXPECTED})", self.review)

    def test_the_prompt_tells_the_model_to_write_that_exact_path(self):
        # WHOLE-TOKEN, not `assertIn`: a containment check takes the expected
        # path as a PREFIX, so retargeting the prompt to `...findings.jsonl`
        # passed while directing a write the hook then refuses.
        prompt = self.review.split("prompt: |", 1)[1]
        output = prompt.split("OUTPUT.", 1)
        self.assertEqual(len(output), 2, "premise changed: no OUTPUT section")
        section = output[1].split("\n\n", 1)[0]
        targets = [
            t.rstrip(".,") for t in re.findall(r"\$\{\{ runner\.temp \}\}/\S+", section)
        ]
        self.assertEqual(
            targets,
            [self.EXPECTED],
            f"the prompt names a different verdict path: {targets}",
        )

    def test_every_env_var_that_names_it_is_present_and_agrees(self):
        # Presence asserted FIRST. Iterating matches alone made DELETING a
        # binding pass vacuously, after which the hooks fall back to their own
        # /tmp default and silently disagree with the prompt.
        for var in ("FINDINGS_FILE", "PR_REVIEW_FINDINGS_FILE"):
            values = re.findall(rf"^\s+{var}: (.+)$", self.review, re.M)
            self.assertTrue(values, f"{var} is no longer set in the review job")
            for value in values:
                self.assertEqual(
                    value.strip(), self.EXPECTED, f"{var} drifted to {value!r}"
                )

    def test_the_hooks_default_agrees_with_the_workflow(self):
        # The hook falls back to its own literal when the env var is unset, and
        # the executable tests override that var — so nothing compared it.
        self.assertIn("pr-review-findings.json", HOOK.read_text())

    def test_the_grant_is_read_not_write(self):
        # `Write` is deliberately bare and fenced by the hook; the findings file
        # carries a READ grant so the model can re-read what it wrote.
        self.assertIn("Read(/${{ runner.temp }}/pr-review-findings.json)", self.review)


class TestBothReviewCheckoutsStayOutOfTheWorkspaceRoot(unittest.TestCase):
    """`path:` on both checkouts is what keeps PR config undiscoverable.

    Two separate controls depend on the workspace root staying empty. Project
    and local setting discovery resolve against the action's cwd, which is
    $GITHUB_WORKSPACE; and claude-code-action's agent mode runs
    `configureGitAuth` unconditionally, which would write a token into a
    .git/config at that root — the file `persist-credentials: false` exists to
    prevent. Today both throw or find nothing because the root is not a
    checkout. Losing either `path:` is a one-token edit that changes nothing
    observable in a passing run.
    """

    def test_every_checkout_in_the_review_job_declares_a_path(self):
        review = job_block(STAGE2.read_text(), "review")
        steps = review.split("- name:")
        checkouts = [s for s in steps if "actions/checkout@" in s]
        self.assertEqual(len(checkouts), 2, "premise changed: not two checkouts")
        for step in checkouts:
            # SINGLE LINE, and the value may be empty. `\s+` after `path:`
            # crosses newlines, so a valueless `path:` captured the next line's
            # key and a `path: # why` captured the `#` — both pass a
            # non-emptiness check while leaving the input unset, which is the
            # workspace-root checkout this test exists to forbid.
            # THE `with:` MAPPING, not the step: `path:` under `env:` is not an
            # action input, and the checkout would land at the workspace root.
            lines = with_block(step)
            pat = _key_re("path")  # quoted key and space-before-colon too
            hit = next(
                ((i, m) for i, ln in enumerate(lines) if (m := pat.match(ln))), None
            )
            self.assertIsNotNone(
                hit,
                f"a review-job checkout declares no path: {step[:60]!r}",
            )
            i, m = hit
            value = plain_scalar(m.group(1))
            self.assertNotEqual(
                value,
                "",
                f"a review-job checkout declares an EMPTY path: {step[:60]!r}",
            )
            # A bare block header (`path: |`) with no body is an EMPTY string
            # to YAML, while the literal `|` normalizes to a plausible-looking
            # directory name — so read the body when there is a header, rather
            # than rejecting the header outright (which reds the legal
            # `path: |-` followed by an indented `trusted`).
            #
            # The body STOPS AT THE FIRST DEDENT. Scanning the rest of the step
            # for any deeper-indented line walks past the scalar into a later
            # comment, which then stands in as the path.
            if BLOCK_HEADER.fullmatch(value):
                indent = len(lines[i]) - len(lines[i].lstrip())
                body = []
                for nxt in lines[i + 1 :]:
                    if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= indent:
                        break
                    if nxt.strip():
                        body.append(nxt.strip())
                value = body[0] if body else ""
                self.assertNotEqual(
                    value,
                    "",
                    "a review-job checkout's path is an empty block scalar; "
                    "YAML resolves that to an empty path and the checkout "
                    "lands at the workspace root",
                )
            # posixpath.normpath, not a trailing-slash strip: `././` survived
            # that as `./.` and still resolves to the workspace root.
            declared = posixpath.normpath(value)
            self.assertNotIn(
                declared,
                {"", "."},
                "this path IS the workspace root; the checkout must land beside it",
            )

    def test_review_step_keeps_mcp_config_strict(self):
        """`.mcp.json` is the OTHER discovery channel, closed by a DIFFERENT flag.

        `--setting-sources ""` does not cover it. Dropping `--strict-mcp-config`
        reopens `pr/.mcp.json`, which is a server definition and therefore
        command execution.
        """
        args = strip_comments(job_block(STAGE2.read_text(), "review"))
        self.assertIn("--strict-mcp-config", args)

    def test_escaping_symlinks_are_removed_before_the_model_runs(self):
        """Path-scoped Read matches the PATH, not where it RESOLVES.

        Without this step a fork commits `pr/leak -> /proc/self/environ`, the
        read is inside the allowed glob, and the step's AWS session key, secret
        and token are handed to the model. Verified live against CLI 2.1.201.
        """
        review = job_block(STAGE2.read_text(), "review")
        names = re.findall(r"^\s*- name: (.+)$", review, re.M)
        self.assertIn("Remove symlinks that escape the PR tree", names)
        idx = {n: i for i, n in enumerate(names)}
        self.assertLess(
            idx["Remove symlinks that escape the PR tree"],
            idx["Run the PR review"],
            "symlink removal must run BEFORE the model does",
        )
        self.assertIn(
            "-type l", strip_comments(review), "the step no longer looks for symlinks"
        )

    def test_no_step_deletes_agent_config_from_the_pr_tree(self):
        """Deleting agent config out of `pr/` mutates the tree under review.

        A PR whose actual change is to `CLAUDE.md` would then be reviewed with
        that change removed. Symlink removal above is deliberately narrower: it
        touches only links resolving OUTSIDE `pr/`, so it cannot alter a file
        the diff contains.
        """
        review = strip_comments(job_block(STAGE2.read_text(), "review"))
        self.assertNotIn("CONFIG_NAMES", review, "the agent-config scrub is back")
        self.assertNotIn(
            "rm -rf", review, "a step recursively deletes from the reviewed tree"
        )


class TestForkCheckoutPreconditionsHold(unittest.TestCase):
    """`allow-unsafe-pr-checkout: true` is only safe while four things stay true.

    actions/checkout refuses fork code in a `workflow_run` workflow because such
    a job carries the base repo's secrets, GITHUB_TOKEN, default-branch cache
    scope and runner access. We opted in on 2026-09-11 after checking each
    against the review job. The comment beside that line records the reasoning;
    these assertions are what stop the reasoning going stale, because every one
    of the four is a property a later edit could quietly remove.

    This is deliberately about the REVIEW job only. prepare and publish never
    execute pull-request content, so the guard's premise does not apply to them.
    """

    def setUp(self):
        self.review = job_block(STAGE2.read_text(), "review")
        self.code = strip_comments(self.review)

    def test_opt_in_is_present_and_on_the_untrusted_checkout(self):
        """Without it, fork PRs fail at checkout and no review ever runs."""
        self.assertIn("allow-unsafe-pr-checkout: true", self.code)
        untrusted = self.code.split("path: pr", 1)
        self.assertEqual(len(untrusted), 2, "premise changed: no `path: pr` checkout")
        self.assertIn(
            "allow-unsafe-pr-checkout: true",
            untrusted[1].split("- name:", 1)[0],
            "the opt-in is not on the checkout that fetches PR code",
        )

    def test_review_job_uses_no_secrets(self):
        """A secret here would be readable by a process running fork code."""
        found = sorted(set(re.findall(r"secrets\.([A-Za-z_]\w*)", self.code)))
        self.assertEqual(found, [], f"review job now references secrets: {found}")

    def test_review_job_caches_nothing(self):
        """Cache writes land in the default-branch scope — poisonable from a fork."""
        self.assertNotIn("actions/cache", self.code)
        self.assertNotIn("cache:", self.code)

    def test_review_job_runs_on_an_ephemeral_github_runner(self):
        """A self-hosted runner would give fork code a persistent host."""
        runners = re.findall(r"runs-on:\s*(\S+)", self.code)
        self.assertEqual(runners, ["ubuntu-latest"], f"runner changed: {runners}")

    def test_untrusted_checkout_does_not_persist_credentials(self):
        """Otherwise the token lands in pr/.git/config, which the model may read."""
        after = self.code.split("path: pr", 1)[1].split("- name:", 1)[0]
        self.assertIn("persist-credentials: false", after)


class TestReviewRoleSeparationIsPinned(unittest.TestCase):
    """The review job's `environment:` and REVIEW_ROLE must stay one pair.

    The separation is real and it hangs
    entirely on these two lines agreeing. `gha_workflow_claude_untrusted`
    trusts `sub` values of the form `…:environment:claude-untrusted` (this repo
    and pytorch/pytorch), and GitHub mints such a subject only for a job that
    DECLARES that environment.

    Two distinct half-edits, with different consequences:

    * `environment: bedrock` + the untrusted role. The configured assume fails
      loudly — the role does not trust the publisher's subject. But the job's
      TOKEN is again eligible for the publisher role, so code running in the
      job could request credentials for it explicitly. Loud in the happy path,
      and a real widening in the adversarial one.
    * `environment: bedrock` + the shared role — i.e. reverting both together
      to the old interim state. Nothing fails; the untrusted job silently
      regains PutObject on `pr_review_verdicts/`, the forgery path this design
      exists to remove. This is the dangerous one, and asserting the exact
      expected pair below is what catches it.

    Dropping the environment line entirely fails closed and is merely broken.

    The denials were verified end-to-end in pytorch/ciforge, where this originated.
    """

    EXPECTED_ENV = "claude-untrusted"
    EXPECTED_ROLE = "arn:aws:iam::308535385114:role/gha_workflow_claude_untrusted"
    SHARED_ROLE = "arn:aws:iam::308535385114:role/gha_workflow_claude_code"

    def setUp(self):
        self.text = STAGE2.read_text()
        self.review = strip_comments(job_block(self.text, "review"))
        role = re.search(r"^\s*REVIEW_ROLE:\s*(\S+)", self.text, re.M)
        self.assertIsNotNone(role, "REVIEW_ROLE env not found")
        self.review_role = role.group(1)
        env = re.search(r"^\s*environment:\s*(\S+)\s*$", self.review, re.M)
        self.review_env = env.group(1) if env else None

    def test_review_role_is_the_untrusted_role(self):
        self.assertEqual(
            self.review_role,
            self.EXPECTED_ROLE,
            "the review job must assume the untrusted role; the shared role "
            "carries PutObject on every ossci-raw-job-status prefix including "
            "pr_review_verdicts",
        )

    def test_review_job_declares_the_untrusted_environment(self):
        self.assertEqual(
            self.review_env,
            self.EXPECTED_ENV,
            "the review job must declare `environment: claude-untrusted` — that "
            "declaration is the only thing that mints the OIDC `sub` the "
            "untrusted role's trust policy accepts",
        )

    def test_the_two_never_disagree(self):
        """The pairing, asserted as a pair so a half-edit cannot pass."""
        self.assertEqual(
            (self.review_env, self.review_role),
            (self.EXPECTED_ENV, self.EXPECTED_ROLE),
            "the review job's `environment:` and REVIEW_ROLE must flip together",
        )

    def test_review_job_never_names_the_publisher_environment(self):
        """`environment: bedrock` here would hand the untrusted job the publisher's sub."""
        self.assertIsNone(
            re.search(r"^\s*environment:\s*bedrock\s*$", self.review, re.M),
            "the review job declares the publisher's environment",
        )

    def test_review_job_never_names_the_shared_role(self):
        self.assertNotIn(self.SHARED_ROLE, self.review)

    def test_review_job_assumes_review_role_indirectly(self):
        self.assertIn("role-to-assume: ${{ env.REVIEW_ROLE }}", self.review)


class TestStage1CollapseGroupIsJobLevel(unittest.TestCase):
    """A WORKFLOW-level group on Stage 1 lets an ignored run cancel a real one.

    GitHub claims a workflow-level concurrency group before evaluating the job
    `if:`, so a run this workflow is going to skip still joins the group and,
    with `cancel-in-progress`, kills a review already in flight.

    Observed in pytorch/ciforge: the CLA bot's `cla signed` label created a
    run one second after the `in progress` label created ours. The bot's run
    skipped its own `capture` and cancelled ours one second in — no artifact,
    and both Stage 2 runs skipped. Moving the group onto `capture` fixes it,
    because a job rejected by `if:` never enters the group.

    This repo REQUIRES a workflow-level cancelling group, so the group cannot
    simply be absent. The reconciliation is the suffix: the required prefix is
    followed by the event's label, so two runs triggered by DIFFERENT labels
    land in different groups and cannot cancel each other. Drop the suffix and
    the incident above comes straight back.

    Reverting either half is a small edit that looks tidier and silently
    restores the failure, which is why both are pinned rather than commented.
    """

    def setUp(self):
        self.text = STAGE1.read_text()

    def test_workflow_level_group_is_differentiated_by_label(self):
        """Column 0 == workflow level. Present is fine; undifferentiated is not."""
        m = re.search(r"^concurrency:\n(?:[ \t]+.*\n)+", self.text, re.M)
        self.assertIsNotNone(m, "this repo requires a workflow-level group")
        group = re.search(r"^\s+group:\s*(.+)$", m.group(0), re.M)
        self.assertIsNotNone(group, "workflow-level concurrency has no group")
        self.assertIn(
            "github.event.label.name",
            group.group(1),
            "the workflow-level group is not differentiated by label — a run "
            "this workflow skips will cancel a live review, as it did in "
            "pytorch/ciforge",
        )

    def test_capture_job_keeps_a_collapse_group(self):
        """The group must not be dropped either — bursts should still collapse."""
        capture = job_block(self.text, "capture")
        block = strip_comments(capture)
        # re.M matters: assertRegex uses re.search, so a bare `^` would only
        # anchor at the start of the whole block and never match the indented key.
        self.assertIsNotNone(
            re.search(r"^\s+concurrency:", block, re.M),
            "capture lost its concurrency group",
        )
        self.assertIn("cancel-in-progress: true", block)


class TestSymlinkScrubIsNulSafe(unittest.TestCase):
    """A newline in a path component must not split one entry into two.

    The variable round-trip this replaces aimed `rm -f` at the workspace root,
    where the TRUSTED checkout sits, and the fail-closed re-scan used the same
    split so it reported clean afterwards.
    """

    def setUp(self):
        self.step = strip_comments(job_block(STAGE2.read_text(), "review"))
        self.scrub = self.step[self.step.index("Remove symlinks that escape") :]
        self.scrub = self.scrub[: self.scrub.index("- name: Build the diff")]

    def test_the_list_is_never_joined_with_newlines(self):
        self.assertNotIn("printf '%s\\n' \"$ESCAPED\"", self.scrub)
        self.assertNotIn("ESCAPED=$(", self.scrub)
        self.assertNotIn("LEFT=$(", self.scrub)

    def test_both_the_delete_and_the_rescan_read_nul_separated(self):
        self.assertEqual(self.scrub.count("printf '%s\\0'"), 1)
        self.assertEqual(self.scrub.count("read -r -d ''"), 3)

    def test_the_delete_uses_a_double_dash(self):
        self.assertIn('rm -f -- "$link"', self.scrub)

    def test_attacker_chosen_names_are_quoted_into_the_log(self):
        # %s would let a newline in the name forge a `::error::` workflow command.
        self.assertNotIn("printf '  %s\\n' \"$link\"", self.scrub)
        self.assertIn("printf '  %q\\n' \"$link\"", self.scrub)

    def test_it_still_fails_closed(self):
        self.assertIn("::error::escaping symlink still present", self.scrub)
        self.assertIn("exit 1", self.scrub)


class TestLabelMoveCannotContradictTheRow(unittest.TestCase):
    def setUp(self):
        self.publish = strip_comments(job_block(STAGE2.read_text(), "publish"))

    def test_the_label_step_reads_the_effective_status_not_the_raw_claim(self):
        self.assertIn("steps.row.outputs.effective_status", self.publish)
        # The claim alone must not be what gates the label.
        self.assertNotIn("STATUS=$(jq -r '.status", self.publish)

    def test_the_row_step_exports_that_status(self):
        self.assertIn("effective_status=", self.publish)

    def test_the_head_is_rechecked_before_the_label_moves(self):
        self.assertIn("CURRENT_SHA", self.publish)
        self.assertIn("REVIEWED_SHA", self.publish)
        self.assertIn('"$CURRENT_SHA" != "$REVIEWED_SHA"', self.publish)

    def test_a_failed_removal_is_not_reported_as_a_move(self):
        self.assertIn("could not remove", self.publish)
        self.assertNotIn(">/dev/null 2>&1 || true", self.publish)


class TestLabelComparisonsAreCaseInsensitive(unittest.TestCase):
    """pytorch/pytorch spells the marker `Ready for Review`; `==` never matched."""

    def test_both_label_checks_downcase_both_sides(self):
        prepare = strip_comments(job_block(STAGE2.read_text(), "prepare"))
        self.assertEqual(prepare.count("ascii_downcase == ($l | ascii_downcase)"), 2)
        self.assertNotIn("any(.labels[]?.name; . == $l)", prepare)


class TestRubricSpeaksTheSchemaSeverities(unittest.TestCase):
    """The rubric and the schema in the prompt must name the same severities.

    `extract_verdict.SEVERITIES` drops a finding whose severity is outside the
    set, as `bad_severity`, before anyone reads it. The rubric used to prescribe
    `blocking` — and to prescribe it specifically for reporting a prompt-injection
    attempt, so the one finding the rubric most wants surfaced was the one
    guaranteed to be discarded.
    """

    def setUp(self):
        self.rubric = RUBRIC.read_text()
        self.stage2 = STAGE2.read_text()

    def test_the_rubric_names_no_severity_the_sanitizer_would_drop(self):
        # Every emphasised or code-quoted word the rubric uses as a severity,
        # however the sentence is phrased. An earlier version keyed on the
        # literal " finding" suffix and went blind the moment the wording
        # changed — while still passing, because one other occurrence matched.
        # No vocabulary filter — enumerating the words we already know about is
        # how `critical` walks in unnoticed — but scoped to the sentences that
        # DECLARE a severity, so ordinary prose citing a schema field is not
        # read as one. Sweeping the whole Report section and subtracting a
        # denylist did that: adding "with `path` and `line`" to the anchoring
        # instruction failed the test for no reason.
        declarations = [
            ln
            for ln in self.rubric.splitlines()
            if re.search(
                r"Report (a finding as|such an attempt as a)|`severity` must be", ln
            )
        ]
        self.assertTrue(
            declarations, "premise changed: the rubric declares no severity"
        )
        named = {
            m.lower()
            for ln in declarations
            for pair in re.findall(r"\*\*`?(\w+)`?\*\*|`(\w+)`", ln)
            for m in pair
            if m and m != "severity"
        }
        self.assertTrue(named, "premise changed: the rubric names no severity at all")
        self.assertEqual(
            named - SEVERITIES,
            set(),
            f"rubric prescribes severities the sanitizer drops: {sorted(named - SEVERITIES)}",
        )

    def test_the_prompt_schema_offers_exactly_the_sanitizer_set(self):
        line = next(
            ln for ln in self.stage2.splitlines() if '"severity":' in ln and "|" in ln
        )
        self.assertEqual(set(re.findall(r'"(\w+)"', line)) - {"severity"}, SEVERITIES)

    def test_the_injection_report_uses_a_severity_that_survives(self):
        sentence = next(
            ln for ln in self.rubric.splitlines() if "Report such an attempt" in ln
        )
        # `major` specifically, not merely a surviving severity: `minor` would
        # pass a membership check and would no longer force changes_requested.
        self.assertIn(
            "`major`",
            sentence,
            f"an injection attempt is no longer reported as major: {sentence}",
        )


# A GitHub Actions `if:` expression, tokenized. Single-quoted strings first, so
# an operator inside a literal cannot be mistaken for one: `'&&'` is a value.
_EXPR_TOKEN = re.compile(
    r"'(?:[^']|'')*'"  # single-quoted string; '' escapes a quote
    r"|&&|\|\||==|!=|<=|>="  # two-character operators
    r"|[()!<>,]"  # grouping / call args / unary not / comparisons
    r"|[^\s'()!<>,&|=]+"  # a bare term: identifier, number, dotted context path
)


def _lex_expr(expr: str) -> list[str]:
    """Tokens of `expr`, refusing anything the grammar below cannot see.

    The gaps between matches are checked rather than assumed: a lone `&` or an
    unbalanced quote would otherwise be dropped silently and the expression
    would be judged on a truncated reading of itself.
    """
    toks, pos = [], 0
    for m in _EXPR_TOKEN.finditer(expr):
        gap = expr[pos : m.start()]
        assert not gap.strip(), f"unlexable text in the `if:` expression: {gap!r}"
        toks.append(m.group(0))
        pos = m.end()
    tail = expr[pos:]
    assert not tail.strip(), f"unlexable trailing text in the `if:`: {tail!r}"
    # INDEX SYNTAX IS REFUSED, not absorbed. `github[0 && pin && 0]` lexed as
    # the two bare terms `github[0` and `0]`, which reads as `(!A) && pin && B`
    # — apparently binding — while GitHub evaluates the bracket first, finds a
    # missing property and negates a null. Brackets are legal and this parser
    # does not model their precedence, so it fails closed.
    #
    # AFTER tokenizing, and never inside a string: a bracket in a LITERAL is
    # just a character, and rejecting the raw text red the perfectly ordinary
    # `!contains(…, '[skip review]')`.
    for t in toks:
        if t.startswith("'"):
            continue
        assert "[" not in t and "]" not in t, (
            f"the `if:` expression uses index syntax, whose precedence this "
            f"parser does not model ({t!r}). Teach it, then allow this."
        )
        # GitHub Actions quotes strings with `'`. A `"` OUTSIDE a literal is an
        # undecoded YAML quote, which would be absorbed into a condition. One
        # INSIDE a literal is an ordinary character, and refusing it red the
        # legitimate `!startsWith(…, 'Revert "')`.
        assert '"' not in t, (
            f"the `if:` expression carries a double quote this reader did not "
            f"decode ({t!r}). Unquote it, or teach the reader."
        )
    return toks


_CMP_OPS = ("==", "!=", "<=", ">=", "<", ">")


def parse_bool_expr(expr: str):
    """`expr` as a propositional tree whose leaves are opaque truth values.

    `&&`, `||`, `!` and GROUPING parens are interpreted; a comparison, a context
    reference and a `contains(a, b)` call are each ONE opaque leaf. That is why
    this needs no model of GitHub's value semantics and no idea what any context
    holds — but it does have to agree with GitHub about PRECEDENCE, and the one
    that matters is that **`!` binds tighter than a comparison**. `!a == b` is
    `(!a) == b` in GitHub, not `!(a == b)`, and reading it the second way makes
    a double negation look like the identity while GitHub evaluates it to a
    constant `true`. So `!` is resolved below the comparison level and a
    negated term is folded into the comparison's own text.

    A `(` is grouping only where a term is expected; inside a term it is a call
    and is depth-tracked, so `'(' == '('` cannot corrupt the nesting the way a
    paren-counting heuristic did.

    Refuses rather than guesses. A comparison over a GROUPED sub-expression
    (`(a || b) == true`) is legal and binds, but its truth is not a function of
    the group's truth under any assignment this model can enumerate, so it is
    rejected with a message naming the limitation instead of silently read as
    something else.
    """
    toks = _lex_expr(expr)

    def term(i):
        """One value: a literal, a context path, or a `name(...)` call."""
        assert i < len(toks), "the `if:` expression ends where a term was expected"
        t = toks[i]
        assert t not in ("&&", "||", "!", ")", ",") and t not in _CMP_OPS, (
            f"the `if:` expression has {t!r} where a term was expected"
        )
        assert t != "(", "internal: a grouping paren reached term()"
        start, i = i, i + 1
        if i < len(toks) and toks[i] == "(":  # a call: absorb its balanced args
            depth = 0
            while i < len(toks):
                if toks[i] == "(":
                    depth += 1
                elif toks[i] == ")":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            else:
                raise AssertionError("unbalanced call parens in the `if:`")
        return " ".join(toks[start:i]), i

    def unary(i):
        """(node, next index, grouped?) — `!` over a term or over a group."""
        if toks[i] == "!":
            node, i, grouped = unary(i + 1)
            return ("not", node), i, grouped
        if toks[i] == "(":
            node, i = or_(i + 1)
            assert i < len(toks) and toks[i] == ")", "unbalanced parens in the `if:`"
            return node, i + 1, True
        text, i = term(i)
        return ("atom", text), i, False

    def cmp(i):
        start = i
        lhs, i, lhs_grouped = unary(i)
        if i < len(toks) and toks[i] in _CMP_OPS:
            op = toks[i]
            _rhs, j, rhs_grouped = unary(i + 1)
            assert not (lhs_grouped or rhs_grouped), (
                f"the `if:` compares a PARENTHESIZED sub-expression ({op!r} at "
                f"token {i}). That is legal, but this reader models a "
                "comparison as one opaque condition and cannot see the "
                "conditions inside it — teach it, then update this test."
            )
            # The whole comparison, INCLUDING any `!` on either operand, is one
            # opaque condition: `!a == b` is not the negation of `a == b`.
            return ("atom", " ".join(toks[start:j])), j
        return lhs, i

    def and_(i):
        node, i = cmp(i)
        while i < len(toks) and toks[i] == "&&":
            rhs, i = cmp(i + 1)
            node = ("and", node, rhs)
        return node, i

    def or_(i):
        node, i = and_(i)
        while i < len(toks) and toks[i] == "||":
            rhs, i = and_(i + 1)
            node = ("or", node, rhs)
        return node, i

    assert toks, "the `if:` expression is empty"
    tree, i = or_(0)
    assert i == len(toks), f"trailing tokens in the `if:`: {toks[i:]}"
    return tree


def expr_atoms(node) -> set[str]:
    if node[0] == "atom":
        return {node[1]}
    if node[0] == "not":
        return expr_atoms(node[1])
    return expr_atoms(node[1]) | expr_atoms(node[2])


def eval_expr(node, env: dict) -> bool:
    if node[0] == "atom":
        # `true` and `false` are the literals, not free variables — otherwise a
        # gate wired to `false &&` would look satisfiable.
        return {"true": True, "false": False}.get(node[1], env.get(node[1], False))
    if node[0] == "not":
        return not eval_expr(node[1], env)
    if node[0] == "and":
        return eval_expr(node[1], env) and eval_expr(node[2], env)
    return eval_expr(node[1], env) or eval_expr(node[2], env)


def job_if(job: str) -> str:
    """The JOB-level `if:` expression — block or plain, `${{ }}` unwrapped.

    Indent-scoped to the job's own keys, so a step's `if:` cannot stand in for
    the gate, and asserted unique rather than taking the first match.

    A PLAIN scalar continues onto following more-indented lines and YAML folds
    them with a space, so `if: <pin>` with an indented `|| true` under it is the
    single value `<pin> || true`. Reading the key's own line only returned the
    pin and reported a gate that does not exist.
    """
    lines = job.splitlines()
    pat = _key_re("if")
    hits = [
        (i, m)
        for i, ln in enumerate(lines)
        if (m := pat.match(ln)) and len(ln) - len(ln.lstrip()) == 4
    ]
    assert len(hits) == 1, f"premise changed: {len(hits)} job-level `if:` keys"
    i, m = hits[0]
    header = BLOCK_HEADER.fullmatch(uncommented(m.group(1)))
    body = []
    for nxt in lines[i + 1 :]:
        if nxt.strip() and len(nxt) - len(nxt.lstrip()) <= 4:
            break
        if nxt.strip():
            body.append(nxt.rstrip())
    # A GHA string literal may not SPAN lines here. The reader joins the
    # scalar's lines with a space, which is right for a plain scalar and wrong
    # inside a literal block, where YAML keeps the newline as part of the
    # string — so `'a<newline>b' == 'a b'` is false to GitHub and true to a
    # reader that folded it. Refused rather than mis-read.
    #
    # BODY LINES ONLY. The key's own line carries no expression content when it
    # is a block header, and counting its apostrophes red the file over an
    # ordinary comment: `if: | # Don't weaken this guard`.
    for ln in body:
        assert ln.count("'") % 2 == 0, (
            f"a string literal in the job-level `if:` spans lines ({ln.strip()!r}); "
            "this reader folds them and would change what the string contains."
        )
    if header:
        # Inside a LITERAL block, `#` is content, not a comment.
        assert body, "the job-level `if:` block is empty"
        value = " ".join(ln.strip() for ln in body)
    else:
        # FOLD FIRST, THEN DECODE — one scalar, one decoding. Decoding each
        # line on its own unquoted a continuation's own GHA string literal, so
        # the truthy `'false'` in `pin || 'false'` became the FALSE literal and
        # a gate that admits every path read as binding.
        #
        # A plain scalar's trailing ` #` IS a comment, dropped by the
        # quote-aware `uncommented`; `plain_scalar` also unwraps a quoted whole
        # gate, which YAML decodes away and which otherwise left every
        # condition unrecognizable.
        value = plain_scalar(" ".join([m.group(1)] + [ln.strip() for ln in body]))
    # An undecoded YAML double quote is refused in `_lex_expr`, per TOKEN, so
    # that a `"` inside a GHA string literal stays the ordinary character it is.
    wrapped = re.fullmatch(r"\$\{\{(.*)\}\}", value.strip(), re.S)
    return (wrapped.group(1) if wrapped else value).strip()


class TestStage2CannotBeTriggeredByALookalikeWorkflow(unittest.TestCase):
    """`workflows:` matches by NAME, and a pull request can claim a name.

    Stage 1's file comes from the PR's own ref, so a PR may add a second
    workflow also called "Hardened PR Review". Without a path condition that
    lookalike triggers Stage 2 and supplies the request artifact.
    """

    PIN = (
        r"github\.event\.workflow\.path\s*==\s*"
        r"'\.github/workflows/hardened-pr-review\.yml'"
    )

    def test_prepare_pins_the_triggering_workflow_path(self):
        # Comments stripped and the whole equality matched: asserting the two
        # strings separately passed on `!=`, and on prose in a comment.
        #
        # The WHOLE prepare job, not "the text before `runs-on:`": narrowing
        # the region made the test depend on job-key ORDER, so moving an
        # unchanged `runs-on:` above `if:` emptied the region and red it.
        prepare = strip_comments(job_block(STAGE2.read_text(), "prepare"))
        self.assertRegex(
            prepare,
            self.PIN,
            "prepare no longer pins which workflow file may trigger it",
        )

    def test_the_pin_is_binding_and_not_merely_present(self):
        """Present is not the same contract, and the gap is the whole control.

        A text search is satisfied by a pin that decides nothing: demote it
        behind a top-level `||`, negate it as `!(pin)`, or move it out of `if:`
        into a quoted job `name:`, and the assertion above stays green while
        the gate admits a lookalike workflow. Two earlier attempts to close
        this SYNTACTICALLY each red a legal edit instead — a paren-counting
        rule broke on `'(' == '('`, and a no-`||` rule rejected the perfectly
        good `(a || b) && pin`.

        So this EVALUATES the gate rather than pattern-matching it. The `if:`
        is read as a propositional formula whose conditions are opaque, and the
        contract is: no combination of the other conditions can let a run
        through unless THIS COMPARISON holds. Extra conjuncts, grouped
        alternatives and a repeated pin all satisfy that and stay green.

        TWO limits, stated because neither is closed here. The guarantee is
        only as strong as the comparison it depends on, and GitHub's `==`
        IGNORES CASE (docs.github.com, "Evaluate expressions": *"GitHub ignores
        case when comparing strings"*), so a workflow added at a path differing
        only in case satisfies the gate — a defect in the workflow YAML, not in
        this test, and not one the expression language can fix. And the parser
        FAILS CLOSED rather than completely: a handful of legal spellings it
        does not model (index syntax, a comparison over a parenthesized group)
        red with a message naming the limitation instead of being read wrong.
        """
        expr = job_if(job_block(STAGE2.read_text(), "prepare"))
        tree = parse_bool_expr(expr)
        atoms = expr_atoms(tree)
        pinned = [a for a in atoms if re.fullmatch(self.PIN, a)]
        # FAIL CLOSED on a spelling this reader cannot model. Without this, a
        # workflow-path comparison folded into something unrecognizable — `!pin
        # == x`, a comparison over a group — reports "no pin" with a message
        # about the gate being gone, which reads like a different defect.
        if not pinned:
            self.assertNotRegex(
                expr,
                self.PIN,
                f"prepare's gate mentions the workflow path but not as a "
                f"condition this reader can evaluate: {expr!r}. Either the pin "
                f"no longer binds, or it is spelled in a way the parser above "
                f"does not model — establish which, then teach the parser.",
            )
        self.assertEqual(
            len(pinned),
            1,
            f"expected exactly one workflow-path condition in prepare's `if:`, "
            f"found {pinned} in {expr!r}. Two of them means a second file may "
            f"trigger this workflow; none means the gate is elsewhere or gone.",
        )
        pin = pinned[0]
        others = sorted(atoms - {pin, "true", "false"})
        self.assertLessEqual(
            len(others), 12, f"too many conditions to enumerate: {others}"
        )
        cases = list(itertools.product([False, True], repeat=len(others)))
        for bits in cases:
            env = dict(zip(others, bits))
            env[pin] = False
            self.assertFalse(
                eval_expr(tree, env),
                f"prepare's gate admits a run whose workflow path is NOT "
                f"{'.github/workflows/hardened-pr-review.yml'!r}, given "
                f"{env}. The pin is present but not binding, so a pull "
                f"request can add a second workflow with the same NAME and "
                f"drive Stage 2 with an artifact of its choosing.",
            )
        # The converse, to the limited extent opaque atoms can establish it: a
        # gate wired to a literal `false` can never run, and a review that
        # silently stops happening looks exactly like a clean one. This catches
        # the CONSTANT, not the semantically impossible — `&& (1 == 2)` is one
        # opaque atom here and passes. It is a liveness backstop, not a proof.
        self.assertTrue(
            any(
                eval_expr(tree, {**dict(zip(others, bits)), pin: True})
                for bits in cases
            ),
            f"prepare's gate ({expr!r}) is unsatisfiable on its literals alone "
            f"— the job can never run",
        )


class TestTheStage1ArtifactDownloadRetries(unittest.TestCase):
    """A miss here skips every downstream step: no review AND no terminal row.

    `actions/download-artifact@v4` carries no retry and intermittently fails to
    find a cross-run artifact that exists. `claude-issue-triage-run.yml` already
    documents that failure in this repository.
    """

    def setUp(self):
        self.prepare = job_block(STAGE2.read_text(), "prepare")

    def _step(self) -> str:
        step = self.prepare.split("Download Stage-1 artifact", 1)[1]
        return step.split("- name:", 1)[0]

    def test_the_download_actually_retries_more_than_once(self):
        # Asserting only that a `for` loop exists is not enough: `for attempt
        # in 1` satisfies that and retries nothing. Read the bound.
        step = self._step()
        attempts = re.search(r"for attempt in ([\d ]+); do", step)
        self.assertIsNotNone(attempts, "the download is not a retry loop")
        bounds = attempts.group(1).split()
        self.assertGreaterEqual(
            len(bounds), 3, f"only {len(bounds)} attempt(s); that is not a retry"
        )
        self.assertIn("gh run download", step)
        self.assertIn("sleep", step)

    def test_an_exhausted_retry_fails_the_job_rather_than_continuing(self):
        step = self._step()
        attempts = re.search(r"for attempt in ([\d ]+); do", step).group(1).split()
        # The give-up branch must fire on the LAST attempt. Off by one and the
        # loop falls through silently with no artifact and no row.
        self.assertRegex(step, rf'\[ "\$attempt" -eq {attempts[-1]} \]')
        self.assertIn("::error::", step)
        self.assertIn("exit 1", step)

    # Everything above is structural, and structure cannot see control flow:
    # moving the `echo` and the `break` below the `fi` keeps every assertion
    # above green while the loop makes exactly one attempt and exits 0. So the
    # two below RUN the extracted shell against a stubbed `gh`.

    def _run_script(self, fail_first: int) -> tuple[int, int]:
        """Execute the step with a `gh` that fails its first `fail_first` calls.

        Returns (exit status, number of `gh run download` attempts made).
        """
        # Tolerant of the block header's spelling AND of a trailing comment:
        # `run: |-`, `run: |2-` and `run: | # why` all change formatting, not
        # the shell, and an exact `"run: |\n"` split raised IndexError before
        # running anything. That the header is LITERAL rather than folded is
        # pinned by TestTheReadersPremisesStillHold.
        script = textwrap.dedent(scalar_block(self._step(), "run"))
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "count").write_text("0")
            gh = d / "gh"
            gh.write_text(
                "#!/bin/bash\n"
                f"n=$(( $(cat {d}/count) + 1 )); echo $n > {d}/count\n"
                f"[ $n -gt {fail_first} ]\n"
            )
            gh.chmod(0o755)
            # `sleep` stubbed too, or the backoff makes this test take a minute.
            sleep = d / "sleep"
            sleep.write_text("#!/bin/bash\nexit 0\n")
            sleep.chmod(0o755)
            proc = subprocess.run(
                ["bash", "-c", script],
                cwd=td,
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "PATH": f"{d}:{os.environ['PATH']}",
                    "GH_TOKEN": "x",
                    "RUN_ID": "1",
                    "REPO": "o/r",
                },
            )
            return proc.returncode, int((d / "count").read_text())

    def test_a_transient_failure_is_actually_retried(self):
        rc, tries = self._run_script(fail_first=2)
        self.assertEqual(rc, 0, "a download that succeeds on attempt 3 failed the step")
        self.assertEqual(tries, 3, f"expected 3 attempts, made {tries}")

    def test_a_persistent_failure_exhausts_the_budget_then_fails(self):
        rc, tries = self._run_script(fail_first=99)
        self.assertNotEqual(rc, 0, "an undownloadable artifact did not fail the step")
        self.assertEqual(tries, 5, f"expected 5 attempts, made {tries}")

    def test_the_unretried_action_is_gone(self):
        # Comments stripped: the retry loop's own comment names the action it
        # replaced, and that prose must not be able to fail this.
        self.assertNotIn("actions/download-artifact", strip_comments(self.prepare))


class TestTheHookLogCannotForgeAWorkflowCommand(unittest.TestCase):
    """The refused write target is model-controlled and reaches the job log.

    The runner parses `::command::` at the start of a line, so a target holding
    a newline would reach that parser even though the write was denied. Denying
    the filesystem operation does not close the output channel. This EXECUTES
    the hook rather than grepping it, because the neutralization has to hold on
    the value, not in the source text.
    """

    def _run(self, file_path: str, log: Path) -> str:
        payload = json.dumps(
            {"tool_name": "Write", "tool_input": {"file_path": file_path}}
        )
        subprocess.run(
            ["bash", str(HOOK)],
            input=payload,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PR_REVIEW_HOOK_LOG": str(log),
                "PR_REVIEW_FINDINGS_FILE": "/tmp/allowed.json",
            },
            check=False,
        )
        return log.read_text() if log.exists() else ""

    def test_a_newline_in_the_refused_path_cannot_start_a_log_line(self):
        # The V2 form. actions/runner Runner.Common/ActionCommand.cs
        # TryParseV2 accepts `::cmd::` only when the line starts with it after
        # TrimStart, so this models the parser exactly.
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "hooks.log"
            written = self._run("/tmp/x\n::add-mask::secret", log)
        self.assertTrue(written, "the hook logged nothing; premise changed")
        for line in written.splitlines():
            self.assertFalse(
                line.lstrip().startswith("::"),
                f"model-controlled text began a workflow command: {line!r}",
            )

    def test_the_legacy_command_form_is_neutralized_anywhere_in_the_line(self):
        # The V1 form, and the reason position is not a defence. The same file's
        # TryParse uses `IndexOf("##[")`, which matches ANYWHERE in the line, so
        # a `DENY `-prefixed line is no protection against it.
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "hooks.log"
            written = self._run("/tmp/x ##[error]forged", log)
        self.assertTrue(written, "the hook logged nothing; premise changed")
        self.assertNotIn("##[", written, "a legacy workflow command survived")

    def test_the_v2_introducer_is_neutralized_mid_line_too(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "hooks.log"
            written = self._run("/tmp/x ::add-mask::secret", log)
        self.assertNotIn("::", written, "a `::` introducer survived mid-line")

    def test_the_refused_path_is_still_reported_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "hooks.log"
            written = self._run("/tmp/somewhere-else.json", log)
        self.assertIn("DENY", written)
        self.assertIn("somewhere-else.json", written)

    def test_the_logged_path_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "hooks.log"
            written = self._run("/tmp/" + "a" * 5000, log)
        self.assertTrue(written, "the hook logged nothing; the bound is vacuous")
        self.assertTrue(
            all(len(ln) < 600 for ln in written.splitlines()), written[:200]
        )


class TestTheSizeGateShortCircuitsBeforeTheRunner(unittest.TestCase):
    """The file-count gate is decided in `prepare`, off the API's own count.

    It used to be evaluated five steps into `review`, after a runner, two
    checkouts, two symlink walks and two diffs — all discarded to publish a
    100-byte `skipped_too_large` verdict. `prepare` already holds the PR
    representation that carries `changed_files`.
    """

    def setUp(self):
        self.text = STAGE2.read_text()
        self.stripped = strip_comments(self.text)
        self.prepare = strip_comments(job_block(self.text, "prepare"))

    def test_prepare_reads_changed_files_and_publishes_the_decision(self):
        self.assertIn("changed_files", self.prepare)
        self.assertRegex(self.prepare, r"too_large=true")
        self.assertRegex(self.prepare, r"too_large=false")
        # Exported, or no downstream job can see it.
        self.assertRegex(
            self.prepare, r"too_large:\s*\$\{\{\s*steps\.freshness\.outputs\.too_large"
        )

    def test_the_gate_compares_against_the_same_constant_the_review_uses(self):
        """One constant. Two gates that could disagree are worse than one gate."""
        self.assertIn("MAX_CHANGED_FILES", self.prepare)
        review = strip_comments(job_block(self.text, "review"))
        self.assertIn("MAX_CHANGED_FILES", review)
        # It is a workflow-level env, so both jobs resolve the same value.
        header = self.stripped.split("jobs:", 1)[0]
        self.assertTrue(
            re.search(r"^\s*MAX_CHANGED_FILES:\s*\d+\s*$", header, re.M),
            "MAX_CHANGED_FILES is not a workflow-level env; the two gates could drift",
        )

    def test_both_downstream_jobs_stand_down_when_prepare_says_too_large(self):
        """`publish` too: it is `always()`, so skipping `review` is not enough."""
        for job in ("review", "publish"):
            block = strip_comments(job_block(self.text, job))
            cond = block.split("runs-on:", 1)[0]
            self.assertIn(
                "needs.prepare.outputs.too_large != 'true'",
                " ".join(cond.split()),
                f"{job} would still run on the oversized path",
            )

    def test_prepare_writes_the_terminal_row_on_that_path(self):
        """Otherwise the attempt is a `started` with no terminal — a runner death.

        Scoped to the oversized step's own block. Asserting `--phase terminal`
        against the whole job passed by borrowing it from the stale close-out,
        which is the "test cannot fail" shape this suite exists to refuse.
        """
        # Bounded by the next STEP or the next JOB — this is the last step in
        # `prepare`, so stopping only at `- name:` ran on into `review:`.
        block = job_block(self.text, "prepare")
        i = block.index("Close out an oversized request")
        rest = block[i:]
        nxt = rest.find("\n      - name:", 1)
        block = rest if nxt == -1 else rest[:nxt]
        body = textwrap.dedent(scalar_block(block, "run"))
        # The runner expands `${{ ... }}` before bash ever sees it; bash reads
        # it as a bad substitution. Render them the way Actions would.
        body = re.sub(r"\$\{\{[^}]*\}\}", "CTX", body)
        with tempfile.TemporaryDirectory() as td:
            bin_dir = Path(td) / "bin"
            bin_dir.mkdir()
            log = Path(td) / "argv.log"
            # Record what the emitter and the uploader were actually asked for.
            # An assertion on the TEXT passes when the flag is moved into a
            # trailing `#` comment, because `strip_comments` drops whole-line
            # comments only. Running it does not.
            (bin_dir / "python3").write_text(
                f'#!/bin/bash\nprintf "%s\\n" "$*" >> {log}\n'
                'for a in "$@"; do case "$a" in --out) n=1 ;; '
                'esac; done\necho "{}" > terminal.json\nexit 0\n'
            )
            (bin_dir / "aws").write_text(
                f'#!/bin/bash\nprintf "aws %s\\n" "$*" >> {log}\nexit 0\n'
            )
            for f in ("python3", "aws"):
                (bin_dir / f).chmod(0o755)
            proc = subprocess.run(
                ["bash", "-c", body],
                capture_output=True,
                text=True,
                cwd=td,
                env={
                    "PATH": f"{bin_dir}:/usr/bin:/bin",
                    "S3_BUCKET": "b",
                    "S3_PREFIX": "p",
                },
            )
            recorded = log.read_text() if log.exists() else ""
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("--phase terminal", recorded)
        self.assertIn("--status skipped_too_large", recorded)
        self.assertNotIn("--status skipped_stale", recorded)
        self.assertRegex(recorded, r"aws s3 cp terminal\.json")


class TestThePrCheckoutIsNotAFullClone(unittest.TestCase):
    """`fetch-depth: 0` on pytorch/pytorch is several GB, fetched every review.

    What the review needs from history is one commit — the fork point. It is
    resolved by the API in `prepare`, on complete history, and fetched here at
    depth 1. The review job must never compute it itself: `git merge-base` in a
    shallow clone can return an older ancestor with exit status 0, and a
    wrong-but-successful fork point yields a wrong diff rather than an error.
    """

    def setUp(self):
        self.text = STAGE2.read_text()
        self.review = job_block(self.text, "review")
        self.stripped = strip_comments(self.review)
        self.prepare = strip_comments(job_block(self.text, "prepare"))

    def _step(self, marker: str, hay: str | None = None) -> str:
        hay = self.stripped if hay is None else hay
        i = hay.index(marker)
        rest = hay[i:]
        nxt = rest.find("\n      - name:", 1)
        return rest if nxt == -1 else rest[:nxt]

    def test_the_untrusted_checkout_is_shallow(self):
        block = self._step("Untrusted checkout")
        self.assertRegex(block, r"fetch-depth:\s*1\b")
        self.assertNotRegex(block, r"fetch-depth:\s*0\b")

    def test_no_checkout_in_the_review_job_asks_for_full_history(self):
        self.assertNotRegex(self.stripped, r"fetch-depth:\s*0\b")

    def test_the_review_job_never_computes_the_merge_base_itself(self):
        """The whole point of resolving it upstream; a shallow one can be wrong."""
        self.assertNotIn("merge-base", self.stripped)
        self.assertNotIn("--deepen", self.stripped)
        self.assertNotIn("--unshallow", self.stripped)

    def test_prepare_resolves_it_from_the_compare_endpoint_and_validates_it(self):
        self.assertIn("merge_base_commit.sha", self.prepare)
        self.assertRegex(self.prepare, r"/compare/")
        self.assertRegex(
            self.prepare, r'"merge_base_sha=\$MERGE_BASE"\s*>>\s*"\$GITHUB_OUTPUT"'
        )

    def _run_corroboration(self, merge_base: str, td: str):
        """Execute the corroboration step against a stubbed `gh`."""
        i = self.text.index("Corroborate the claimed PR against the trusted API")
        rest = self.text[i:]
        nxt = rest.find("\n      - name:", 1)
        body = textwrap.dedent(scalar_block(rest if nxt == -1 else rest[:nxt], "run"))
        sha = "a" * 40
        pr_json = json.dumps(
            {
                "head": {"sha": sha, "repo": {"full_name": "o/r"}, "ref": "b"},
                "base": {"sha": "b" * 40},
                "draft": False,
                "labels": [{"name": "in progress"}],
                "changed_files": 3,
            }
        )
        bin_dir = Path(td) / "bin"
        bin_dir.mkdir()
        (bin_dir / "gh").write_text(
            "#!/bin/bash\n"
            f"case \"$2\" in\n  *compare*) printf '%s' {json.dumps(merge_base)} ;;\n"
            f"  *) printf '%s' {json.dumps(pr_json)} ;;\nesac\n"
        )
        (bin_dir / "gh").chmod(0o755)
        out = Path(td) / "gh_out"
        out.write_text("")
        return subprocess.run(
            ["bash", "-c", body],
            capture_output=True,
            text=True,
            cwd=td,
            env={
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "GITHUB_OUTPUT": str(out),
                "PR_NUMBER": "1",
                "HEAD_SHA": sha,
                "TRIGGER_EVENT": "labeled",
                "REPO": "o/r",
                "EVENT_HEAD_REPO": "o/r",
                "EVENT_HEAD_BRANCH": "b",
                "REVIEW_LABEL": "in progress",
                "DONE_LABEL": "ready for review",
                "MAX_CHANGED_FILES": "100",
            },
        ), out

    def test_a_malformed_merge_base_aborts_the_run(self):
        """Executed, not matched: a `#`-commented-out guard passes a text test."""
        for bad in ("", "null", "not-a-sha", "a" * 39):
            with tempfile.TemporaryDirectory() as td:
                proc, out = self._run_corroboration(bad, td)
            self.assertNotEqual(proc.returncode, 0, f"merge base {bad!r} was accepted")
            self.assertIn("no usable merge base", proc.stdout + proc.stderr)

    def test_a_well_formed_merge_base_is_exported(self):
        with tempfile.TemporaryDirectory() as td:
            proc, out = self._run_corroboration("c" * 40, td)
            written = out.read_text()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn(f"merge_base_sha={'c' * 40}", written)

    def test_the_merge_base_reaches_the_review_job_as_an_output(self):
        self.assertRegex(
            self.stripped,
            r"MERGE_BASE_SHA:\s*\$\{\{\s*needs\.prepare\.outputs\.merge_base_sha\s*\}\}",
        )

    def test_the_diff_is_taken_against_the_resolved_merge_base(self):
        """Two-dot against the fork point — three-dot would recompute it here."""
        diff = self._step("Build the diff and apply the size gate")
        self.assertIn('git diff "${MERGE_BASE_SHA}" HEAD', diff)
        self.assertIn('git diff --name-only -z "${MERGE_BASE_SHA}" HEAD', diff)
        self.assertNotIn("...HEAD", diff)

    def test_the_fetch_step_runs_before_the_diff_and_before_any_credential(self):
        fetch = self.stripped.find("Fetch the merge base")
        diff = self.stripped.find("Build the diff and apply the size gate")
        creds = self.stripped.find("Configure AWS credentials via OIDC")
        self.assertNotEqual(fetch, -1, "the merge-base fetch step is gone")
        self.assertLess(fetch, diff)
        self.assertLess(fetch, creds)

    def test_the_fetch_proves_the_object_arrived(self):
        """A partial fetch would otherwise surface as a confusing diff error."""
        block = self._step("Fetch the merge base")
        self.assertIn("--depth=1", block)
        self.assertIn("git cat-file -e", block)

    def test_the_merge_base_guard_is_executable_not_a_comment(self):
        """`strip_comments` drops whole-line comments only, so assert by running.

        Round 1 of the cross-model review defeated the earlier version of this
        by appending `# no merge base between` to a `|| true`. The fix is to
        execute the extracted shell rather than to match its text.
        """
        block = self._step("Fetch the merge base", hay=self.review)
        body = textwrap.dedent(scalar_block(block, "run"))
        with tempfile.TemporaryDirectory() as td:
            bin_dir = Path(td) / "bin"
            bin_dir.mkdir()
            # `git fetch` succeeds, `git cat-file -e` fails: the object did not
            # arrive. The step must exit non-zero.
            (bin_dir / "git").write_text(
                '#!/bin/bash\ncase "$1" in\n'
                "  fetch) exit 0 ;;\n"
                "  cat-file) exit 1 ;;\n"
                "  *) exit 0 ;;\nesac\n"
            )
            (bin_dir / "git").chmod(0o755)
            proc = subprocess.run(
                ["bash", "-c", body],
                capture_output=True,
                text=True,
                cwd=td,
                env={
                    "PATH": f"{bin_dir}:/usr/bin:/bin",
                    "MERGE_BASE_SHA": "0" * 40,
                },
            )
        self.assertNotEqual(
            proc.returncode, 0, "a merge base that never arrived was accepted"
        )
        self.assertIn("did not arrive", proc.stdout + proc.stderr)


class TestTheReadersPremisesStillHold(unittest.TestCase):
    """PyYAML is not available to the runner, so this suite reads YAML as TEXT.

    That is a sound way to pin a file whose spelling you also control, and an
    unsound way to parse YAML in general — the language has many spellings for
    one value, and a text reader silently mis-reads the ones it was not written
    for. A `path: !!str ''` reads as a non-empty path; a `path: |-` body of two
    lines reads as its first line only; a `run: >` yields shell that FOLDING
    would have joined into something bash rejects; a branch list's inline
    comment tokenizes as a branch name.

    Chasing those one spelling at a time does not converge. This class is the
    alternative: it pins the SPELLINGS the readers are written for, at ONE
    place. Reformatting the workflow is allowed; doing it without updating the
    reader is what must not pass silently. So a mis-read becomes a loud failure
    HERE, naming the reader to fix, instead of a quiet false green elsewhere.

    If you are here because you reformatted a workflow: nothing is wrong with
    your YAML. Update the reader this test names, then update this test.
    """

    def test_both_review_checkout_paths_are_plain_scalars(self):
        """The reader in TestBothReviewCheckoutsStayOutOfTheWorkspaceRoot."""
        review = job_block(STAGE2.read_text(), "review")
        steps = [s for s in review.split("- name:") if "actions/checkout@" in s]
        self.assertEqual(len(steps), 2, "premise changed: not two checkouts")
        for step in steps:
            lines = with_block(step)
            hits = [
                (i, uncommented(mm.group(1)))
                for i, ln in enumerate(lines)
                if (mm := _key_re("path").match(ln))
            ]
            self.assertEqual(len(hits), 1, f"expected one `path:` key, got {hits}")
            i, value = hits[0]
            # A plain scalar CONTINUES onto a following more-indented line, and
            # YAML folds the two with a space — so `path: trusted` followed by
            # an indented `/..` is the single value `trusted /..`, which
            # normalizes to the workspace root while a line-at-a-time reader
            # sees only `trusted`. A comment line is NOT a continuation.
            indent = len(lines[i]) - len(lines[i].lstrip())
            nxt = next(
                (
                    ln
                    for ln in lines[i + 1 :]
                    if ln.strip() and not ln.lstrip().startswith("#")
                ),
                "",
            )
            self.assertFalse(
                nxt and len(nxt) - len(nxt.lstrip()) > indent,
                f"the checkout `path:` scalar continues onto the next line "
                f"({nxt.strip()!r}); the reader is line-at-a-time and would "
                "validate only the first line of the folded value.",
            )
            # POSITIVE grammar, not a blacklist of the spellings seen so far. A
            # blacklist of `|>!!&*` prefixes accepts `path: "."`, which YAML
            # decodes to the workspace root while the reader — which only
            # strips quotes and normalizes the UNDECODED text — sees a harmless
            # name. Anything outside this grammar is a spelling the reader
            # cannot be trusted on, whatever it decodes to.
            self.assertRegex(
                value,
                r"^[A-Za-z0-9_][A-Za-z0-9._/-]*$",
                f"a checkout `path:` is not a plain unquoted scalar ({value!r}). "
                "The reader in TestBothReviewCheckoutsStayOutOfTheWorkspaceRoot."
                "test_every_checkout_in_the_review_job_declares_a_path does not "
                "decode YAML — it cannot see through quoting or an escape, so "
                "teach it this spelling, then widen this grammar.",
            )
            # Shape is not resolution: `null`, `Null`, `NULL` and `~` all match
            # the grammar above yet resolve to an empty value — the
            # workspace-root checkout again — and the reader normalizes the
            # literal string "null" and finds nothing wrong with it.
            self.assertNotIn(
                value,
                {"null", "Null", "NULL", "~", "Yes", "No", "On", "Off"},
                f"a checkout `path:` is the YAML scalar {value!r}, which "
                "resolves to an empty or boolean value rather than the "
                "directory name it looks like",
            )

    # The four steps this suite EXECUTES. Each is extracted with `scalar_block`
    # and handed to bash, so each must be a LITERAL block.
    EXECUTED_STEPS = (
        ("prepare", "Download Stage-1 artifact"),
        ("prepare", "Close out an oversized request"),
        ("prepare", "Corroborate the claimed PR against the trusted API"),
        ("review", "Fetch the merge base"),
    )

    def test_every_executed_step_uses_a_literal_block_scalar(self):
        """The `scalar_block` readers that run the extracted shell.

        `|` keeps newlines; `>` FOLDS them. Extracting a folded scalar as if it
        were literal yields shell the runner would never execute — a `for` and
        the comments above it join into one line — so the executable tests
        would pass against code that is not what runs.
        """
        text = STAGE2.read_text()
        for job, marker in self.EXECUTED_STEPS:
            block = job_block(text, job)
            self.assertIn(marker, block, f"premise changed: {marker!r} is not in {job}")
            step = block.split(marker, 1)[1].split("- name:", 1)[0]
            # The step's OWN `run:`, found the same way the readers find it —
            # stepping over every neighbouring block scalar rather than taking
            # the first textual `run:`, which an `env:` value can supply.
            runs = block_scalars(step, "run")
            self.assertEqual(
                len(runs),
                1,
                f"{marker!r} does not have exactly one `run:` block scalar "
                f"({len(runs)} found); the readers extract one and would run "
                "something other than the step's command.",
            )
            header, body = runs[0]
            # `|` or `|-`/`|+`, and NO explicit indentation indicator. `>` folds
            # lines, so the extracted script would differ from what Actions
            # runs. `|1` is subtler: YAML keeps one leading space on every body
            # line and `textwrap.dedent` removes it, which is invisible in
            # ordinary shell and silently re-indents a heredoc terminator.
            self.assertRegex(
                header.lstrip("!").replace("str", "").strip(),
                r"^\|[-+]?$",
                f"{marker!r} no longer runs a plain LITERAL block ({header!r}); "
                "the reader extracts it verbatim and dedents it, which is only "
                "faithful for `|`, `|-` and `|+` — teach the reader this "
                "spelling, then widen this test.",
            )
            # SPACES ONLY in the indentation. A tab after the common space
            # prefix is scalar CONTENT to YAML, and `textwrap.dedent` — which
            # strips the longest common whitespace prefix, tabs included —
            # removes it. Invisible in ordinary shell, and enough to move a
            # heredoc terminator so the real script stops terminating it.
            tabbed = [ln for ln in body if "\t" in ln[: len(ln) - len(ln.lstrip())]]
            self.assertEqual(
                tabbed,
                [],
                f"{marker!r} indents its shell with tabs ({tabbed[:1]}); the "
                "reader dedents, which would strip content YAML preserves.",
            )

    def test_the_suite_ci_branch_filter_is_a_single_line_flow_list(self):
        """The branch reader in TestTheSuiteActuallyRunsInCI."""
        inline = next(
            (
                uncommented(mm.group(1))
                for ln in strip_comments(SUITE_CI.read_text()).splitlines()
                if (mm := _key_re("branches").match(ln))
            ),
            None,
        )
        self.assertIsNotNone(inline, "premise changed: no `branches:` key")
        # THE WHOLE LIST, matched as one grammar, before any splitting. A
        # reader that splits first and validates the fragments accepts
        # `['main[ab]x']` — which matches `mainax`, never `main` — as the three
        # acceptable pieces `main`, `ab`, `x`, and the single quoted name
        # `'release,main'` as two. Brackets and commas inside a quoted item
        # must not be able to disappear.
        #
        # LITERAL NAMES ONLY, no glob and no exclusion: the reader asks "is
        # `main` listed", which is a sound reading of the filter only when
        # GitHub's ordered pattern matching cannot disagree with membership.
        name = r"[A-Za-z0-9_][A-Za-z0-9._/-]*"
        self.assertRegex(
            inline,
            rf"^\[[ \t]*{name}(?:[ \t]*,[ \t]*{name})*[ \t]*\]$",
            f"the push branch filter ({inline!r}) is not a one-line flow list "
            "of unquoted literal branch names. The reader in "
            "TestTheSuiteActuallyRunsInCI.test_push_to_main_triggers_it_for_"
            "the_same_paths does a membership test where GitHub does ordered "
            "pattern matching, and cannot see quoting — teach it, then widen "
            "this grammar.",
        )


if __name__ == "__main__":
    unittest.main()
