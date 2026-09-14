#!/usr/bin/env python3
"""Run the symlink-scrub step's ACTUAL shell against hostile trees.

The other contract tests for this step are structural — they assert that the
YAML contains `printf '%s\\0'` and no newline-joined variable. Measured, not
assumed: replacing both `scan_escaping > "$SCAN"` calls with `: > "$SCAN"`
turns the step into a no-op that every structural assertion still passes,
because the text they grep for is all still present. (Deleting the step
outright DOES fail them, on their `.index()` calls — the gap is a neutered
step, not a missing one.) This module extracts the step's `run:` block from
the workflow and EXECUTES it, so that mutation has somewhere to land.

The tree it builds mirrors the job: a trusted checkout beside the untrusted
`pr/`, and the assertion is that a sentinel under `trusted/` is still there
afterwards.

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from _suite_manifest import TestTheSuiteIsWhole  # noqa: E402,F401


WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "hardened-pr-review-run.yml"
)
STEP_NAME = "Remove symlinks that escape the PR tree"


def scrub_script() -> str:
    """Pull the step's `run:` body out of the workflow and dedent it.

    Hand-slicing YAML is how a test ends up asserting against a FRAGMENT and
    passing for the wrong reason, so this refuses rather than guesses: the step
    name must occur exactly once, `run:` must be plain `|` (a `|2` explicit
    indentation indicator would make the indentation heuristic below cut the
    block short), and the extracted text must contain every landmark the step
    is supposed to end with. A rename or a reindent fails loudly here instead
    of silently shrinking what these tests cover.
    """
    text = WORKFLOW.read_text()
    marker = f"- name: {STEP_NAME}"
    if text.count(marker) != 1:
        raise AssertionError(
            f"expected exactly one {marker!r}, found {text.count(marker)}"
        )
    start = text.index(marker)
    # Bound to THIS step before looking for `run:`. Slicing to the end of the
    # file and taking the first `run:` lets a later -- even a disabled -- step
    # supply the script, which would execute text the workflow never runs.
    step_indent = len(text[:start].rsplit("\n", 1)[-1])
    rest = text[start + len(marker) :]
    # Comment lines are skipped when looking for the boundary: a `#` at or
    # below the step indent is not the next step, and treating it as one cut
    # the slice short and failed for the wrong reason.
    nxt = None
    for m in re.finditer(rf"^ {{0,{step_indent}}}(?!#)(-|\S)", rest, re.M):
        nxt = m
        break
    body = rest[: nxt.start()] if nxt else rest
    # `run` must be an IMMEDIATE property of the step. Without the indent
    # anchor, a `uses:` step carrying the whole script under `with: run: |`
    # satisfies every landmark below, and these tests would then execute a
    # value the action never runs as shell.
    prop_indent = step_indent + 2
    if re.search(rf"^ {{{prop_indent}}}uses:", body, re.M):
        raise AssertionError(
            "the named step is a `uses:` step; it has no shell body to run"
        )
    header = re.search(rf"^ {{{prop_indent}}}run: (\|[-+0-9]*)\s*$", body, re.M)
    if header is None:
        raise AssertionError("no `run:` block scalar found under the step")
    if header.group(1) != "|":
        raise AssertionError(
            f"run: uses the block indicator {header.group(1)!r}; the indentation "
            "heuristic here is only valid for a bare `|`"
        )
    lines = body[header.end() :].split("\n")[1:]
    indent = len(lines[0]) - len(lines[0].lstrip())
    kept: list[str] = []
    for line in lines:
        if line.strip() and (len(line) - len(line.lstrip())) < indent:
            break
        kept.append(line)
    script = textwrap.dedent("\n".join(kept))
    # Landmarks spanning the WHOLE step, so a truncated slice cannot pass.
    for landmark in (
        "scan_escaping() {",
        "rm -f --",
        "::error::escaping symlink still present",
        "exit 1",
    ):
        if landmark not in script:
            raise AssertionError(
                f"extracted block is truncated or wrong — missing {landmark!r}"
            )
    if script.count('scan_escaping > "$SCAN"') != 2:
        raise AssertionError(
            "expected both the delete scan and the fail-closed re-scan"
        )
    return script


class ScrubHarness(unittest.TestCase):
    def run_scrub(self, build_tree) -> subprocess.CompletedProcess:
        # addCleanup, not a single self.root: a test that calls this twice
        # would otherwise overwrite the attribute and leak the first workspace.
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.root)], check=False)
        ws = self.root / "workspace"
        (ws / "trusted" / "scripts" / "pr_review").mkdir(parents=True)
        (ws / "pr").mkdir()
        self.sentinel = ws / "trusted" / "scripts" / "pr_review" / "extract_verdict.py"
        self.sentinel.write_text("TRUSTED SENTINEL\n")
        (ws / "trusted" / "restrict-write.sh").write_text("GUARD SENTINEL\n")
        self.ws = ws
        build_tree(ws / "pr")
        # stderr INTO stdout, not captured separately. The runner reads one
        # interleaved stream; concatenating two captures puts every stdout line
        # before every stderr line, which can make an unfenced hostile line
        # look fenced purely as an artefact of the reordering.
        return subprocess.run(
            ["bash", "-c", scrub_script()],
            cwd=ws,
            env={**os.environ, "PR_DIR": str(ws / "pr")},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )


def unfenced_occurrences(output: str, marker: str) -> list[str]:
    """Lines carrying `marker` that the RUNNER would still parse as commands.

    The oracle deliberately recognises the resume command MORE loosely than it
    is emitted: `::<token>::` anywhere in a line ends the fence, not only as
    the whole stripped line. The runner is the lenient party, so an oracle that
    demanded an exact line would call a name fenced that the runner had already
    resumed on — which is the direction that hides a regression.
    """
    active: str | None = None
    leaked: list[str] = []
    for line in output.splitlines():
        if active is None:
            stripped = line.strip()
            if stripped.startswith("::stop-commands::"):
                active = stripped.split("::")[2]
                continue
        elif _resumes(line, active):
            active = None
            continue
        if marker in line and active is None:
            leaked.append(line)
    return leaked


def _resumes(line: str, token: str) -> bool:
    """Every spelling the runner accepts for the resume command.

    Deliberately broader than what this workflow emits. The runner also honours
    the legacy `##[name]` syntax, matches the command name case-INsensitively,
    and allows properties after the name — so `##[tok]`, `::TOK::` and
    `::tok x=y::` all resume. An oracle that only knew `::tok::` would call a
    name fenced that the runner had already resumed on, which is the direction
    that hides a regression.
    """
    low = line.lower()
    tok = token.lower()
    return bool(
        re.search(rf"::{re.escape(tok)}(\s[^:]*)?::", low)
        or re.search(rf"##\[{re.escape(tok)}(\s[^\]]*)?\]", low)
    )


class TestANewlineInAPathCannotReachTheTrustedCheckout(ScrubHarness):
    """The blocking defect: a newline-joined list aimed `rm -f` at the workspace root."""

    def build(self, pr: Path):
        # A directory whose NAME contains a newline. Joining the escaping-link
        # list with newlines splits this into a second path, resolved against
        # the workspace root, where `trusted/` sits.
        attack = pr / "x\ntrusted/scripts/pr_review"
        attack.mkdir(parents=True)
        (attack / "extract_verdict.py").symlink_to("/etc/hostname")

    def test_the_trusted_file_survives(self):
        proc = self.run_scrub(self.build)
        self.assertTrue(
            self.sentinel.exists(),
            f"trusted sentinel was deleted\nstdout={proc.stdout}\nstderr={proc.stderr}",
        )

    def test_the_escaping_link_is_removed_and_the_step_passes(self):
        proc = self.run_scrub(self.build)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        remaining = list((self.ws / "pr").rglob("*"))
        self.assertFalse([p for p in remaining if p.is_symlink()], remaining)

    def forging_tree(self, pr: Path):
        # The name itself carries a newline followed by a workflow command. If
        # the name is printed with %s it becomes its own line and GitHub acts
        # on it; %q renders it as `$'...\n...'` on one line.
        attack = pr / "x\n::error::FORGED_BY_THE_PR"
        attack.mkdir()
        (attack / "link").symlink_to("/etc/hostname")

    def test_a_name_cannot_forge_its_own_workflow_command(self):
        """Shell quoting is not workflow-command quoting: the runner acts on
        `::command::` in the log whatever the shell did to it. The fence is
        what makes the name inert, so assert the FENCE, not the spelling."""
        proc = self.run_scrub(self.forging_tree)
        lines = proc.stdout.splitlines()
        self.assertTrue(
            any("FORGED_BY_THE_PR" in ln for ln in lines),
            f"the name was never reported at all: {proc.stdout!r}",
        )
        self.assertEqual(
            unfenced_occurrences(proc.stdout, "FORGED_BY_THE_PR"),
            [],
            f"a name reached the runner unfenced: {proc.stdout!r}",
        )
        opens = [ln for ln in lines if ln.strip().startswith("::stop-commands::")]
        self.assertTrue(opens, f"no stop-commands fence was emitted: {proc.stdout!r}")
        for ln in opens:
            self.assertGreaterEqual(
                len(ln.strip().split("::")[2]), 16, f"fence token is guessable: {ln!r}"
            )

    def test_the_fence_token_is_fresh_per_run(self):
        # A fixed token would be in the PR author's reach: they could name a
        # directory `::<token>::` and close the fence from inside it.
        first = self.run_scrub(self.forging_tree).stdout
        second = self.run_scrub(self.forging_tree).stdout

        def token_of(out):
            for ln in out.splitlines():
                if ln.startswith("::stop-commands::"):
                    return ln.split("::")[2]
            return None

        a, b = token_of(first), token_of(second)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertNotEqual(a, b, "the fence token is reused across runs")


class TestOrdinarySymlinksAreLeftAlone(ScrubHarness):
    def build(self, pr: Path):
        (pr / "real.txt").write_text("hello\n")
        (pr / "link-to-real").symlink_to("real.txt")
        (pr / "sub").mkdir()
        (pr / "sub" / "up").symlink_to("../real.txt")

    def test_in_repo_links_survive(self):
        proc = self.run_scrub(self.build)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue((self.ws / "pr" / "link-to-real").is_symlink())
        self.assertTrue((self.ws / "pr" / "sub" / "up").is_symlink())


class TestATrailingNewlineInTheTargetCannotForgeContainment(ScrubHarness):
    """`$(readlink -f ...)` strips trailing newlines, so a sibling `pr<newline>`
    directory resolved to the same string as the PR tree and was accepted."""

    def build(self, pr: Path):
        # The link must resolve TO the sibling directory, not to a file inside
        # it: only then does the newline land at the END of the resolved path,
        # where command substitution eats it and the result compares equal to
        # $REAL_PR. A link to `pr<newline>/loot` puts the newline in the middle
        # and is caught even by the unfixed code.
        sibling = pr.parent / "pr\n"
        sibling.mkdir()
        (sibling / "loot").write_text("OUTSIDE\n")
        (pr / "escape").symlink_to(sibling)

    def test_the_link_to_the_newline_sibling_is_removed(self):
        proc = self.run_scrub(self.build)
        self.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")
        self.assertFalse(
            (self.ws / "pr" / "escape").is_symlink(),
            "a link to the `pr<newline>` sibling was accepted as inside the tree",
        )
        self.assertTrue(
            (self.ws / "pr\n" / "loot").exists(), "the sibling itself must survive"
        )


class TestEscapingDirectoryLinksAndDanglingLinks(ScrubHarness):
    def build(self, pr: Path):
        (pr / "outdir").symlink_to("/etc")
        (pr / "dangling").symlink_to("/nonexistent/nowhere")

    def test_both_are_removed(self):
        proc = self.run_scrub(self.build)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse((pr := self.ws / "pr" / "outdir").is_symlink(), pr)
        self.assertFalse((self.ws / "pr" / "dangling").is_symlink())


class TestTheStepFailsClosedWhenALinkCannotBeRemoved(ScrubHarness):
    def build(self, pr: Path):
        locked = pr / "locked"
        locked.mkdir()
        (locked / "escape").symlink_to("/etc/hostname")
        os.chmod(locked, 0o555)  # deletion inside will fail

    def tearDown(self):
        # `self.ws` only exists once run_scrub() has built the tree. A skip
        # before that still runs tearDown, and reaching for it there raises
        # AttributeError — which is an ERROR, not a skip, so the guard that was
        # meant to keep the suite green under root turned it red instead.
        ws = getattr(self, "ws", None)
        if ws is not None:
            try:
                os.chmod(ws / "pr" / "locked", 0o755)
            except OSError:
                pass
        super().tearDown()

    def test_it_exits_nonzero_rather_than_reporting_clean(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores the directory mode")
        proc = self.run_scrub(self.build)
        self.assertNotEqual(proc.returncode, 0, f"reported clean: {proc.stdout}")
        self.assertIn("::error::escaping symlink still present", proc.stdout)


class TestUtilityStderrIsFencedToo(ScrubHarness):
    """`rm` and `find` quote the offending path in their OWN diagnostics, so a
    name that is a workflow command reaches the runner through stderr even
    while every name we print ourselves sits inside the fence."""

    def build(self, pr: Path):
        locked = pr / "::error::FORGED_VIA_STDERR"
        locked.mkdir()
        (locked / "escape").symlink_to("/etc/hostname")
        os.chmod(locked, 0o555)  # makes `rm` fail and name the path

    def tearDown(self):
        ws = getattr(self, "ws", None)
        if ws is not None:
            try:
                os.chmod(ws / "pr" / "::error::FORGED_VIA_STDERR", 0o755)
            except OSError:
                pass
        super().tearDown()

    def test_no_marker_reaches_the_runner_outside_a_fence(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores the directory mode")
        proc = self.run_scrub(self.build)
        self.assertIn("FORGED_VIA_STDERR", proc.stdout, "the fixture never fired")
        self.assertEqual(
            unfenced_occurrences(proc.stdout, "FORGED_VIA_STDERR"),
            [],
            f"attacker-chosen name escaped the fence: {proc.stdout!r}",
        )

    def test_the_diagnostic_is_still_reported_rather_than_swallowed(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores the directory mode")
        proc = self.run_scrub(self.build)
        self.assertIn("Permission denied", proc.stdout)
        self.assertNotEqual(proc.returncode, 0)


class TestAFailingScanSaysWhy(ScrubHarness):
    """Under `set -e` a failing `find` aborted the step before any diagnostic
    was emitted, and the EXIT trap then deleted the file holding it."""

    def build(self, pr: Path):
        blind = pr / "unreadable"
        blind.mkdir()
        (blind / "escape").symlink_to("/etc/hostname")
        os.chmod(blind, 0o000)  # find cannot descend

    def tearDown(self):
        ws = getattr(self, "ws", None)
        if ws is not None:
            try:
                os.chmod(ws / "pr" / "unreadable", 0o755)
            except OSError:
                pass
        super().tearDown()

    def test_it_fails_closed_and_reports_the_reason(self):
        if os.geteuid() == 0:
            self.skipTest("root ignores the directory mode")
        proc = self.run_scrub(self.build)
        self.assertNotEqual(proc.returncode, 0, f"reported clean: {proc.stdout}")
        self.assertIn("Permission denied", proc.stdout, "the diagnostic was lost")
        self.assertIn("refusing to continue", proc.stdout)


class TestTheFenceOracleItself(unittest.TestCase):
    """The oracle is the thing every other fence test trusts, so pin it."""

    def test_a_plain_resume_ends_the_fence(self):
        out = "::stop-commands::abc123\nHIT\n::abc123::\nHIT"
        self.assertEqual(unfenced_occurrences(out, "HIT"), ["HIT"])

    def test_the_legacy_bracket_syntax_also_ends_it(self):
        out = "::stop-commands::abc123\n##[abc123]\nHIT"
        self.assertEqual(unfenced_occurrences(out, "HIT"), ["HIT"])

    def test_the_name_is_matched_case_insensitively(self):
        out = "::stop-commands::abc123\n::ABC123::\nHIT"
        self.assertEqual(unfenced_occurrences(out, "HIT"), ["HIT"])

    def test_properties_after_the_name_still_resume(self):
        out = "::stop-commands::abc123\n::abc123 x=y::\nHIT"
        self.assertEqual(unfenced_occurrences(out, "HIT"), ["HIT"])

    def test_a_resume_embedded_after_a_prefix_still_counts(self):
        out = "::stop-commands::abc123\nprefix ::abc123::\nHIT"
        self.assertEqual(unfenced_occurrences(out, "HIT"), ["HIT"])

    def test_a_genuinely_fenced_name_is_not_reported(self):
        out = "::stop-commands::abc123\nHIT\n::abc123::"
        self.assertEqual(unfenced_occurrences(out, "HIT"), [])

    def test_a_different_token_does_not_resume(self):
        out = "::stop-commands::abc123\n::deadbeef::\nHIT\n::abc123::"
        self.assertEqual(unfenced_occurrences(out, "HIT"), [])


class TestAFailedScanWriteCannotReportClean(ScrubHarness):
    """errexit is off inside a function called on the left of `||`, and the
    `while` returns its LAST iteration's status -- so a failed write followed by
    an ordinary link used to return success with an empty scan."""

    def build(self, pr: Path):
        (pr / "a-escape").symlink_to("/etc/hostname")  # written first
        (pr / "real.txt").write_text("x\n")
        (pr / "z-ok").symlink_to("real.txt")  # ordinary, examined last

    def test_a_write_failure_is_not_swallowed_by_a_later_ok_link(self):
        # /dev/full accepts the open and fails every write with ENOSPC, having
        # written zero bytes -- exactly the shape of the fail-open.
        script = scrub_script().replace("SCAN=$(mktemp)", "SCAN=/dev/full", 1)
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(subprocess.run, ["rm", "-rf", str(self.root)], check=False)
        ws = self.root / "workspace"
        (ws / "pr").mkdir(parents=True)
        self.ws = ws
        self.build(ws / "pr")
        proc = subprocess.run(
            ["bash", "-c", script],
            cwd=ws,
            env={**os.environ, "PR_DIR": str(ws / "pr")},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # The EXIT CODE ALONE CANNOT DECIDE THIS, and asserting on it made this
        # test vacuous when it was first written. `/dev/full` cannot be removed
        # by the EXIT trap, and a trap's last command status replaces the
        # script's, so the unfixed version also exits 1 -- for an unrelated
        # reason. Only the fixed version reaches the scan's `||` handler, so the
        # handler's own message is the discriminator.
        self.assertIn(
            "refusing to continue",
            proc.stdout,
            f"a failed scan write did not take the fail-closed branch: {proc.stdout!r}",
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertTrue(
            (ws / "pr" / "a-escape").is_symlink(), "should not have been reached"
        )


if __name__ == "__main__":
    unittest.main()
