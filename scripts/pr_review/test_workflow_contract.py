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

import re
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Collected as a test of THIS module, so the suite notices if this file is the
# one deleted. See _suite_manifest for why the guard is shared, not copied.
from _suite_manifest import TestTheSuiteIsWhole  # noqa: E402,F401


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
STAGE1 = WORKFLOWS / "hardened-pr-review.yml"
STAGE2 = WORKFLOWS / "hardened-pr-review-run.yml"
SUITE_CI = WORKFLOWS / "pr-review-scripts-test.yml"


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
    """`--setting-sources ""` is the boundary, and nothing else enforces it.

    Claude Code loads CLAUDE.md, .claude/settings.json and .mcp.json from the
    working tree and its ancestors. A hook in that settings file is arbitrary
    command execution that runs BEFORE any --allowedTools policy applies, so
    the read-only-tools story is void without this flag. Measured on CLI
    2.1.201: with the default sources a PreToolUse hook planted in
    `pr/.claude/settings.json` executed; with `--setting-sources ""` it did not.

    Deleting the flag is a one-token edit that reopens command execution and
    changes nothing observable in a passing run — the shape that needs a test
    rather than a comment.
    """

    def test_review_step_disables_project_setting_sources(self):
        args = strip_comments(job_block(STAGE2.read_text(), "review"))
        self.assertRegex(
            args,
            r'--setting-sources\s+""',
            "review job no longer disables project/local setting discovery",
        )
        self.assertEqual(
            len(re.findall(r"--setting-sources", args)),
            1,
            "more than one --setting-sources: a later one silently wins",
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

    Reverting this is a two-line edit that looks tidier and silently restores
    the failure, which is why it is pinned rather than only commented.
    """

    def setUp(self):
        self.text = STAGE1.read_text()

    def test_stage1_has_no_workflow_level_concurrency(self):
        # Column 0 == workflow level; the job's own block is indented.
        self.assertIsNone(
            re.search(r"^concurrency:", self.text, re.M),
            "Stage 1 has a workflow-level concurrency group again — an ignored "
            "run will cancel a live review",
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


if __name__ == "__main__":
    unittest.main()


class TestFindingsFileWiringAgreesEverywhere(unittest.TestCase):
    """The findings path is written in four places and must be one path.

    `${{ env.FINDINGS_FILE }}` is NOT usable everywhere it would read naturally.
    In `--allowedTools` an expression that resolved empty would silently produce
    `Write(/)` — a grant over the whole filesystem, from a typo, with no error.
    So the tool rules and the prompt carry the literal and this pins them to the
    `env:` value that the shell steps and hooks actually use.
    """

    # The path is now per-job (`runner.temp`), so what can be pinned is the
    # EXPRESSION, identically in all three places.
    LITERAL = "${{ runner.temp }}/pr-review-findings.json"

    def setUp(self):
        self.text = STAGE2.read_text()
        # Comments stripped: an explanatory comment further up quotes a sample
        # `--allowedTools "Read(<workspace>/pr/**)"`, and a search over the raw
        # text finds THAT and reports the real rule list as empty.
        self.review = strip_comments(job_block(self.text, "review"))

    def test_the_env_declares_the_basename(self):
        m = re.search(r"^\s*FINDINGS_BASENAME:\s*(.+?)\s*$", self.text, re.M)
        self.assertIsNotNone(m, "FINDINGS_BASENAME is not declared in the workflow env")
        self.assertEqual(m.group(1), "pr-review-findings.json")
        self.assertTrue(self.LITERAL.endswith("/" + m.group(1)))

    def test_write_is_bare_and_the_hook_is_the_boundary(self):
        """A path-qualified `Write(...)` rule depends on how the CLI globs a
        path, and when it does not match the call is refused with no reason
        recorded anywhere — three runs were lost to that. The grant is bare and
        `restrict-write.sh` does the restricting, where it is a string compare
        whose outcome is logged."""
        rules = re.search(r'--allowedTools "([^"]*)"', self.review)
        self.assertIsNotNone(rules, "the review job has no --allowedTools")
        entries = rules.group(1).split(",")
        self.assertIn("Write", entries, "Write must be granted")
        self.assertEqual(
            [r for r in entries if r.startswith("Write(")],
            [],
            "no path-qualified Write rule; the hook is the boundary",
        )
        self.assertIn("restrict-write.sh", self.review)

    def test_the_write_restricting_hook_is_a_pretooluse_deny(self):
        """PostToolUse cannot stop a write: the file is already on disk."""
        self.assertIn('"PreToolUse"', self.review)
        hook = (
            Path(__file__).resolve().parents[2]
            / ".claude/hooks/pr_review/restrict-write.sh"
        ).read_text()
        self.assertIn('"deny"', hook)
        self.assertIn("PR_REVIEW_FINDINGS_FILE", hook)

    def test_every_file_mutating_tool_is_covered_by_that_matcher(self):
        """MultiEdit was in neither the allow nor the deny list."""
        matchers = re.findall(r'"matcher":\s*"([^"]*)"', self.review)
        self.assertTrue(matchers, "no hook matchers found at all")
        covered = set()
        for m in matchers:
            covered |= set(m.split("|"))
        self.assertTrue(
            {"Write", "Edit", "MultiEdit", "NotebookEdit"} <= covered,
            f"file-mutating tools not all covered by a hook matcher: {sorted(covered)}",
        )

    def test_no_tool_rule_reads_a_workflow_env_var(self):
        """`runner.*`/`github.*` are runner-provided; a workflow `env` var is
        ours to mistype, and an empty expansion here is a total grant."""
        rules = re.search(r'--allowedTools "([^"]*)"', self.review)
        self.assertNotIn("env.", rules.group(1))

    def test_the_resolved_path_is_asserted_before_the_model_runs(self):
        """The empty-expansion failure is silent, so something must check it."""
        self.assertIn("Check the findings path resolved", self.text)
        self.assertIn("findings path did not resolve", self.text)
        self.assertIn("exists before the model has run", self.text)

    def test_the_findings_file_is_not_a_fixed_tmp_path(self):
        """A predictable name is pre-creatable as a symlink on a reused runner."""
        self.assertNotIn("/tmp/pr-review-findings.json", self.text)
        self.assertIn("runner.temp", self.review)

    def test_the_findings_directory_is_in_the_session_scope(self):
        """A permission rule grants a tool over a path; it does not make the
        path reachable. Without an --add-dir covering it, the Write is refused
        before the rule is read — silently, as a permission denial."""
        self.assertIn("--add-dir ${{ runner.temp }}", self.review)

    def test_bash_stays_refused(self):
        """This job holds live AWS credentials while reading untrusted code."""
        disallowed = re.search(r'--disallowedTools "([^"]*)"', self.review)
        self.assertIsNotNone(disallowed)
        self.assertIn("Bash", disallowed.group(1).split(","))

    def test_the_prompt_names_the_same_literal(self):
        self.assertIn(f"Write your verdict to {self.LITERAL}", self.review)

    def test_the_sanitizer_reads_the_model_file_not_the_action_output(self):
        """Two sources of truth is the failure this replaced."""
        self.assertIn('--structured-output-file "$FINDINGS_FILE"', self.review)
        self.assertNotIn("steps.claude.outputs.structured_output", self.text)

    def test_no_json_schema_flag_survives(self):
        """It would be a rival source of truth for the same verdict."""
        self.assertNotIn("--json-schema", strip_comments(self.review))


class TestWorkflowEnvUsesOnlyContextsItHas(unittest.TestCase):
    """A workflow-level `env:` may not reference `runner`, `steps`, `job`,
    `needs`, `matrix` or `env` itself.

    This is not a style rule. GitHub rejects the WHOLE FILE at validation, so
    the run completes as `failure` having created ZERO jobs — there is no job to
    open, no annotation on the commit, and no log line naming the key. Observed
    live: `FINDINGS_FILE: ${{ runner.temp }}/...` at workflow level, run
    33814337391.
    """

    UNAVAILABLE = ("runner", "steps", "job", "needs", "matrix", "env")

    def test_no_workflow_level_env_value_uses_an_unavailable_context(self):
        for wf in (STAGE1, STAGE2):
            text = wf.read_text()
            m = re.search(r"^env:\n(.*?)^\w", text, re.S | re.M)
            if not m:
                continue
            for line in m.group(1).splitlines():
                if line.lstrip().startswith("#"):
                    continue
                for ctx in self.UNAVAILABLE:
                    with self.subTest(workflow=wf.name, context=ctx, line=line.strip()):
                        self.assertNotRegex(line, r"\$\{\{\s*" + ctx + r"\.")


class TestValidationHooksArePresentAndReal(unittest.TestCase):
    """Hooks are supplied through the action's trusted `settings` input.

    Not through a settings FILE in the repo: `--setting-sources ""` exists to
    stop config discovery from the checked-out PR, which is attacker-authored.
    """

    def setUp(self):
        self.review = job_block(STAGE2.read_text(), "review")
        self.repo_root = Path(__file__).resolve().parents[2]

    def test_both_hooks_are_registered(self):
        self.assertIn('"PostToolUse"', self.review)
        self.assertIn('"Stop"', self.review)
        self.assertIn("validate-post-write.sh", self.review)
        self.assertIn("validate-on-stop.sh", self.review)

    def test_every_registered_hook_script_exists_and_is_executable(self):
        """A path typo disables validation silently — the hook just never fires."""
        referenced = set(
            re.findall(r'"command":\s*"[^"]*?/(\.claude/hooks/[^"]+)"', self.review)
        )
        self.assertTrue(referenced, "no hook commands found in the settings blob")
        for rel in referenced:
            script = self.repo_root / rel
            with self.subTest(script=rel):
                self.assertTrue(script.is_file(), f"{rel} does not exist")
                self.assertTrue(
                    script.stat().st_mode & 0o111, f"{rel} is not executable"
                )

    def test_setting_sources_discovery_is_still_closed(self):
        self.assertIn('--setting-sources ""', self.review)

    def test_the_scripts_dir_handed_to_the_hooks_is_the_trusted_checkout(self):
        """Pointing it at the PR checkout would let the PR define `valid`."""
        m = re.search(r"PR_REVIEW_SCRIPTS_DIR:\s*(\S.*)$", self.review, re.M)
        self.assertIsNotNone(m)
        self.assertIn("/trusted/scripts/pr_review", m.group(1))


class TestTerminalLabelStopsReReview(unittest.TestCase):
    """`ready for review` must gate the passive branch and NOT the explicit one.

    Gating both makes a reviewed PR permanently unreviewable; gating neither
    means every push re-reviews forever. The asymmetry is the design.
    """

    def setUp(self):
        self.stage1 = STAGE1.read_text()
        self.stage2 = STAGE2.read_text()

    def test_stage2_declares_the_terminal_label(self):
        m = re.search(r'^\s*DONE_LABEL:\s*"([^"]+)"', self.stage2, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "ready for review")

    def test_stage1_excludes_it_on_the_passive_branch(self):
        gate = job_block(self.stage1, "capture").split("concurrency:")[0]
        gate = strip_comments(gate)
        self.assertIn(
            "!contains(github.event.pull_request.labels.*.name, 'ready for review')",
            gate,
        )

    def test_stage1_still_honours_an_explicit_relabel(self):
        """Otherwise a reviewed PR can never be re-reviewed by anyone."""
        gate = strip_comments(
            job_block(self.stage1, "capture").split("concurrency:")[0]
        )
        self.assertIn(
            "github.event.action == 'labeled' && github.event.label.name == 'in progress'",
            gate,
        )
        labeled_branch = gate.split("||")[0]
        self.assertNotIn("ready for review", labeled_branch)

    def test_stage2_re_derives_the_same_rule_from_the_api(self):
        """Stage 1's `if:` is attacker-writable, so it decides nothing."""
        prepare = strip_comments(job_block(self.stage2, "prepare"))
        self.assertIn("IS_DONE=", prepare)
        self.assertIn('"$TRIGGER_EVENT" != "labeled"', prepare)

    def test_stage2_reads_the_trigger_through_a_step_output(self):
        """A shell variable does not survive a step boundary; under `set -u`
        the stale spelling would abort the step instead of gating."""
        self.assertIn(
            "TRIGGER_EVENT: ${{ steps.coords.outputs.trigger_event }}", self.stage2
        )

    def test_publish_can_move_labels_and_review_cannot(self):
        publish = job_block(self.stage2, "publish")
        review = job_block(self.stage2, "review")
        self.assertIn("issues: write", publish)
        self.assertNotIn("issues: write", review)

    def test_a_label_permission_gap_does_not_red_the_job(self):
        """`issues: write` is declared and the token still 403s -- the org caps
        GITHUB_TOKEN below it. Failing here would red publish on every clean
        review while losing nothing, since the row is already in S3 and
        `in progress` is still on."""
        publish = strip_comments(job_block(self.stage2, "publish"))
        add = publish.index("labels[]=${DONE_LABEL}")
        tail = publish[add : add + 600]
        self.assertIn("::warning::", tail)
        self.assertNotIn("exit 1", tail)

    def test_the_marker_label_is_self_provisioned(self):
        """Adding a label a repo lacks is a 422. `in progress` had to be
        hand-created before this workflow could fire at all, and nothing
        surfaces a missing label until a run needs one."""
        publish = strip_comments(job_block(self.stage2, "publish"))
        self.assertIn("repos/${REPO}/labels", publish)
        self.assertIn("name=${DONE_LABEL}", publish)

    def test_the_marker_is_add_only(self):
        """Ivan's call: adding it is the workflow's job, removing it is not."""
        publish = strip_comments(job_block(self.stage2, "publish"))
        self.assertNotIn('rm_label "$DONE_LABEL"', publish)
        self.assertIn("labels[]=${DONE_LABEL}", publish)

    def test_the_marker_is_added_before_the_gating_label_is_removed(self):
        """The other order can leave a PR with neither label, which nothing
        recovers from: no marker, and no way to make it reviewable again."""
        publish = strip_comments(job_block(self.stage2, "publish"))
        add = publish.index("labels[]=${DONE_LABEL}")
        remove = publish.index('rm_label "$REVIEW_LABEL"')
        self.assertLess(add, remove)

    def test_publish_labels_only_a_clean_positive_verdict(self):
        """A blocked or errored run has reviewed nothing; marking it done would
        be false AND would stop every future attempt."""
        publish = strip_comments(job_block(self.stage2, "publish"))
        self.assertIn('"$VERDICT" != "ready_for_human_review"', publish)
        self.assertIn('"$STATUS" != "succeeded"', publish)
        # Only the gating label is ever removed, and only on a clean verdict.
        self.assertEqual(publish.count("rm_label "), 1)

    def test_publish_reads_the_sanitized_verdict_not_the_model_file(self):
        publish = strip_comments(job_block(self.stage2, "publish"))
        self.assertIn("out/verdict.json", publish)
