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

Run: python3 -m unittest discover -s scripts/pr_review -t .
"""

from __future__ import annotations

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

    def test_pull_request_triggers_it_for_python_and_for_yaml(self):
        body = strip_comments(self.text)
        halves = body.split("pull_request:", 1)
        self.assertEqual(len(halves), 2, "no pull_request trigger: nothing gates a PR")
        for path in self.REQUIRED_PATHS:
            self.assertIn(
                path, halves[1], f"{path} is not in the pull_request paths filter"
            )

    def test_it_holds_no_more_capability_than_the_workflows_it_tests(self):
        """It executes PR-authored Python, so it must stay a read-only
        `pull_request` job — never `pull_request_target`, never a write scope,
        never a secret."""
        body = strip_comments(self.text)
        self.assertNotIn("pull_request_target", body)
        self.assertNotIn("secrets.", body)
        self.assertNotIn("id-token", body)
        perms = "\n".join(indented_block(body, "permissions:"))
        self.assertEqual(perms.strip(), "contents: read")


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
            found = re.search(r"\n\s+path:\s+(\S+)", step)
            self.assertIsNotNone(
                found,
                f"a review-job checkout declares no path: {step[:60]!r}",
            )
            # posixpath.normpath, not a trailing-slash strip: `././` survived
            # that as `./.` and still resolves to the workspace root.
            declared = posixpath.normpath(found.group(1).strip().strip("\"'") or ".")
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


class TestStage2CannotBeTriggeredByALookalikeWorkflow(unittest.TestCase):
    """`workflows:` matches by NAME, and a pull request can claim a name.

    Stage 1's file comes from the PR's own ref, so a PR may add a second
    workflow also called "Hardened PR Review". Without a path condition that
    lookalike triggers Stage 2 and supplies the request artifact.
    """

    def test_prepare_pins_the_triggering_workflow_path(self):
        # Comments stripped and the whole equality matched: asserting the two
        # strings separately passed on `!=`, and on prose in a comment.
        prepare = strip_comments(job_block(STAGE2.read_text(), "prepare"))
        gate = prepare.split("runs-on:", 1)[0]
        self.assertRegex(
            gate,
            r"github\.event\.workflow\.path\s*==\s*'\.github/workflows/hardened-pr-review\.yml'",
            "prepare no longer pins which workflow file may trigger it",
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
        script = textwrap.dedent(self._step().split("run: |\n", 1)[1])
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


if __name__ == "__main__":
    unittest.main()
