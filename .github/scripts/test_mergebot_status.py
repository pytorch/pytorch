"""Tests for mergebot_status.py, and guards that keep mergebot's replies tagged."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any
from unittest import main, mock, TestCase

import comment_on_pr
import yaml
from mergebot_status import (
    MERGE_FAILURE,
    MERGE_SUCCESS,
    parse_comment_id,
    parse_status,
    parse_tag,
    REVERT_FAILURE,
    REVERT_SUCCESS,
    Tag,
    with_status,
)


SCRIPTS = Path(__file__).resolve().parent
WORKFLOWS = SCRIPTS.parent / "workflows"


class TestStatusTag(TestCase):
    def test_round_trip(self) -> None:
        for status in (MERGE_SUCCESS, MERGE_FAILURE, REVERT_SUCCESS, REVERT_FAILURE):
            with self.subTest(status=status):
                self.assertEqual(parse_status(with_status("body", status)), status)

    def test_tag_format(self) -> None:
        self.assertEqual(
            with_status("## Merge failed\n**Reason**: boom\n", MERGE_FAILURE),
            "## Merge failed\n**Reason**: boom\n\n<!-- mergebot-status: merge failure -->\n",
        )

    def test_comment_id_round_trip(self) -> None:
        body = with_status("body", REVERT_SUCCESS, 4567)
        self.assertTrue(
            body.endswith("<!-- mergebot-status: revert success comment=4567 -->\n")
        )
        self.assertEqual(parse_tag(body), Tag(REVERT_SUCCESS, 4567))
        self.assertEqual(parse_status(body), REVERT_SUCCESS)
        self.assertEqual(
            parse_tag(with_status("body", MERGE_FAILURE)), Tag(MERGE_FAILURE, None)
        )

    def test_an_invalid_comment_id_is_left_out(self) -> None:
        # Losing the id is better than losing the reply
        for comment_id in (0, -5):
            with self.subTest(comment_id=comment_id):
                self.assertEqual(
                    with_status("body", MERGE_FAILURE, comment_id),
                    with_status("body", MERGE_FAILURE),
                )

    def test_an_oversized_comment_id_is_left_out(self) -> None:
        # int() refuses digit strings over 4300 long, and str() such ints
        for comment_id in (2**63, 10**5000):
            with self.subTest(bits=comment_id.bit_length()):
                self.assertEqual(
                    with_status("body", MERGE_FAILURE, comment_id),
                    with_status("body", MERGE_FAILURE),
                )
        self.assertIsNone(parse_comment_id("1" * 5000))
        self.assertIsNone(parse_comment_id(str(2**63)))
        self.assertEqual(parse_comment_id(str(2**63 - 1)), 2**63 - 1)

    def test_malformed_comment_ids_do_not_parse(self) -> None:
        for suffix in (
            "comment=",
            "comment=0",
            "comment=-1",
            "comment=12a",
            "comment=1 x=2",
            f"comment={2**63}",
            "comment=" + "1" * 5000,
        ):
            with self.subTest(suffix=suffix):
                body = f"body\n\n<!-- mergebot-status: merge failure {suffix} -->\n"
                self.assertIsNone(parse_tag(body))

    def test_none_leaves_the_body_unchanged(self) -> None:
        self.assertEqual(with_status("notice", None), "notice")
        self.assertIsNone(parse_status("notice"))

    def test_tag_keeps_the_text_readers_match(self) -> None:
        """Existing readers match how a reply starts or a substring of it."""
        started = with_status(
            "### Merge started\nYour change will be merged", MERGE_SUCCESS
        )
        self.assertTrue(started.startswith("### Merge started"))
        self.assertIn("Merge started", started)
        self.assertNotIn("@pytorchbot", MERGE_SUCCESS.tag())

    def test_only_the_appended_tag_counts(self) -> None:
        # A tag quoted anywhere but the end doesn't count, so it is left as it is
        quoted = f"Error: `{MERGE_SUCCESS.tag()}`"
        self.assertEqual(with_status(quoted, None), quoted)
        self.assertIsNone(parse_status(quoted))
        tagged = with_status(quoted, MERGE_FAILURE)
        self.assertTrue(tagged.startswith(f"{quoted}\n"))
        self.assertEqual(parse_status(tagged), MERGE_FAILURE)
        # One ending a reply that gets no tag would count, so it is escaped
        for end in ("", "\n", "  \r\n"):
            ending = f"Error:\n{MERGE_SUCCESS.tag()}{end}"
            self.assertIsNotNone(parse_status(ending))
            escaped = with_status(ending, None)
            self.assertIsNone(parse_status(escaped))
            self.assertEqual(escaped, ending.replace("<", "&lt;"))
        self.assertEqual(
            parse_status(with_status(with_status("x", MERGE_SUCCESS), MERGE_FAILURE)),
            MERGE_FAILURE,
        )

    def test_an_open_code_fence_is_closed_before_the_tag(self) -> None:
        body = with_status("## Merge failed\n```\nboom", MERGE_FAILURE)
        self.assertEqual(
            body, "## Merge failed\n```\nboom\n```\n\n" + MERGE_FAILURE.tag() + "\n"
        )
        closed = "## Merge failed\n```\nboom\n```"
        self.assertEqual(
            with_status(closed, MERGE_FAILURE), f"{closed}\n\n{MERGE_FAILURE.tag()}\n"
        )

    def test_long_and_tilde_fences_are_closed_with_their_own_marker(self) -> None:
        for fence in ("````", "~~~"):
            with self.subTest(fence=fence):
                body = with_status(f"boom\n{fence}\nx\n```", MERGE_FAILURE)
                self.assertTrue(body.startswith(f"boom\n{fence}\nx\n```\n{fence}\n\n"))

    def test_inline_code_is_not_a_fence(self) -> None:
        body = "boom\n```inline code```"
        self.assertEqual(
            with_status(body, MERGE_FAILURE), f"{body}\n\n{MERGE_FAILURE.tag()}\n"
        )

    def test_crlf_line_endings_still_parse(self) -> None:
        body = with_status("Merge started", MERGE_SUCCESS).replace("\n", "\r\n")
        self.assertEqual(parse_status(body), MERGE_SUCCESS)


def workflow_jobs(workflow: str) -> Any:
    """A workflow's jobs, with each action's pinned ref removed so that bumping a
    version changes nothing here."""
    jobs = yaml.safe_load((WORKFLOWS / workflow).read_text())["jobs"]
    for job in jobs.values():
        for step in job.get("steps", []):
            if "uses" in step:
                step["uses"] = step["uses"].split("@")[0]
    return jobs


def workflow_steps(workflow: str) -> list[dict[str, Any]]:
    """A workflow's steps without their names, inputs (`with:`) and environment
    other than COMMENT_ID, and with the commands they run stripped of comments
    and blank lines. Runners, permissions and secrets are left out too: none of
    these decide what replies to a command."""
    steps = []
    for job in workflow_jobs(workflow).values():
        for step in job.get("steps", []):
            pinned = {k: v for k, v in step.items() if k not in ("name", "with", "env")}
            if "run" in step:
                lines = [line.strip() for line in step["run"].split("\n")]
                pinned["run"] = [x for x in lines if x and not x.startswith("#")]
            env = job.get("env", {}) | step.get("env", {})
            if "COMMENT_ID" in env:
                pinned["COMMENT_ID"] = env["COMMENT_ID"]
            steps.append(pinned)
    return steps


COMMENT_ID = "${{ github.event.client_payload.comment_id }}"
CANCELED = "${{ cancelled() && steps.checkout.outcome == 'success' }}"
SETUP_STEPS = [
    {"id": "checkout", "uses": "actions/checkout"},
    {"uses": "actions/setup-python"},
    {"run": ["pip install pyyaml==6.0.2"]},
    {
        "run": [
            'git config --global user.email "pytorchmergebot@users.noreply.github.com"',
            'git config --global user.name "PyTorch MergeBot"',
        ]
    },
]


def scripts_run(workflow: str) -> list[str]:
    """The names of the .py files a workflow's jobs mention, by any path."""
    return re.findall(r"\b(\w+)\.py\b", json.dumps(workflow_jobs(workflow)))


def trymerge_workflows() -> list[str]:
    """Every workflow that runs trymerge.py. Workflows that only import from it
    (cherry_pick.py) post no merge or revert replies, so they are not listed."""
    return sorted(
        path.name
        for path in WORKFLOWS.iterdir()
        if path.suffix in (".yml", ".yaml") and "trymerge" in scripts_run(path.name)
    )


class TestMergeWorkflowSteps(TestCase):
    """The merge and revert workflows reply to commands only through code that
    tags those replies: trymerge.py, and comment_on_pr.py with the command it
    reports (tryrebase.py reports rebases, which carry no status). Any change to
    what they run fails this test, so whoever makes it checks that this still
    holds, then updates the steps below. A new workflow that runs trymerge.py
    fails it too, until it is listed here."""

    STEPS = {
        "trymerge.yml": [
            *SETUP_STEPS,
            {
                "shell": "bash",
                "COMMENT_ID": COMMENT_ID,
                "run": [
                    "set -x",
                    'if [ -n "${REBASE}" ]; then',
                    'if ! python3 .github/scripts/tryrebase.py "${PR_NUM}" --branch "${REBASE}"; then',
                    'python3 .github/scripts/comment_on_pr.py "${PR_NUM}" "merge"',
                    "exit 0",
                    "fi",
                    "git checkout main",
                    "git fetch -p",
                    "sleep 60",
                    "fi",
                    'if [ -z "${COMMENT_ID}" ]; then',
                    'echo "Error: merge requires COMMENT_ID to be specified"',
                    "exit 1",
                    "fi",
                    'if [ -n "${FORCE}" ]; then',
                    'python3 .github/scripts/trymerge.py --force --comment-id "${COMMENT_ID}" "${PR_NUM}"',
                    'elif [ -n "${IGNORE_CURRENT}" ]; then',
                    'python3 .github/scripts/trymerge.py --ignore-current --comment-id "${COMMENT_ID}" "${PR_NUM}"',
                    "else",
                    'python3 .github/scripts/trymerge.py --comment-id "${COMMENT_ID}" "${PR_NUM}"',
                    "fi",
                ],
            },
            {
                "if": CANCELED,
                "continue-on-error": True,
                "COMMENT_ID": COMMENT_ID,
                "run": [
                    "set -x",
                    'python3 .github/scripts/comment_on_pr.py "${PR_NUM}" "merge"',
                ],
            },
            {
                "uses": "aws-actions/configure-aws-credentials",
                "continue-on-error": True,
            },
            {
                "uses": "seemethere/upload-artifact-s3",
                "if": "always()",
                "continue-on-error": True,
            },
        ],
        "revert.yml": [
            *SETUP_STEPS,
            {
                "COMMENT_ID": COMMENT_ID,
                "run": [
                    "set -ex",
                    'if [ -n "${COMMENT_ID}" ]; then',
                    'if [ -n "${REASON}" ]; then',
                    'python3 .github/scripts/trymerge.py --revert --comment-id "${COMMENT_ID}" --reason "${REASON}" "${PR_NUM}"',
                    "else",
                    'python3 .github/scripts/trymerge.py --revert --comment-id "${COMMENT_ID}" "${PR_NUM}"',
                    "fi",
                    "else",
                    'if [ -n "${REASON}" ]; then',
                    'python3 .github/scripts/trymerge.py --revert --reason "${REASON}" "${PR_NUM}"',
                    "else",
                    'python3 .github/scripts/trymerge.py --revert "${PR_NUM}"',
                    "fi",
                    "fi",
                ],
            },
            {
                "if": CANCELED,
                "continue-on-error": True,
                "COMMENT_ID": COMMENT_ID,
                "run": [
                    "set -ex",
                    'python3 .github/scripts/comment_on_pr.py "${PR_NUM}" "revert"',
                ],
            },
        ],
    }

    # Workflows that run trymerge.py with no command to answer, so their replies
    # carry no tag (TestMainStatusTags checks trymerge.py's side)
    NO_COMMAND = {
        "check_mergeability_ghstack.yml": [
            {"uses": "actions/checkout"},
            {
                "shell": "bash",
                "run": [
                    'git config --global user.email "pytorchmergebot@users.noreply.github.com"',
                    'git config --global user.name "PyTorch MergeBot"',
                    "git fetch origin main:main",
                ],
            },
            {
                "shell": "bash",
                "run": [
                    'BRANCH="${{ github.base_ref }}"',
                    'echo "$BRANCH"',
                    'BRANCH="${BRANCH%/base}/orig"',
                    'echo "$BRANCH"',
                    "WAIT_SECONDS=300",
                    "END_WAIT=$((SECONDS+WAIT_SECONDS))",
                    "BRANCH_EXISTS=0",
                    "while [ $SECONDS -lt $END_WAIT ]; do",
                    'git fetch --prune origin "${BRANCH}" || true',
                    'if git rev-parse --verify "origin/${BRANCH}"; then',
                    "BRANCH_EXISTS=1",
                    "break",
                    "fi",
                    'echo "Waiting for branch ${BRANCH} to exist..."',
                    "sleep 30  # Wait for 30 seconds before retrying",
                    "done",
                    "if [ $BRANCH_EXISTS -eq 0 ]; then",
                    'echo "Branch ${BRANCH} not found after ${WAIT_SECONDS} seconds."',
                    'echo "Mergeability check failed for infrastructure reasons."',
                    "exit 1",
                    "fi",
                ],
            },
            {"uses": "actions/setup-python"},
            {"shell": "bash", "run": ["pip install pyyaml==6.0.2"]},
            {
                "shell": "bash",
                "run": [
                    "set -ex",
                    'python3 .github/scripts/trymerge.py --check-mergeability "${PR_NUM}"',
                ],
            },
            {
                "if": "failure()",
                "shell": "bash",
                "run": [
                    "{",
                    'echo "# PR $PR_NUM is not mergeable into main"',
                    'echo "To debug, run the diagnostic workflow:"',
                    'echo "https://github.com/pytorch/test-infra/actions/workflows/pr-dependencies-check.yml"',
                    '} >> "$GITHUB_STEP_SUMMARY"',
                ],
            },
        ],
    }

    def test_steps_are_reviewed(self) -> None:
        self.maxDiff = None
        for workflow, steps in (self.STEPS | self.NO_COMMAND).items():
            with self.subTest(workflow=workflow):
                self.assertEqual(workflow_steps(workflow), steps)

    def test_every_workflow_running_trymerge_is_reviewed(self) -> None:
        self.assertEqual(trymerge_workflows(), sorted(self.STEPS | self.NO_COMMAND))


def module_path(name: str) -> Path | None:
    base = SCRIPTS.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.exists():
            return path
    return None


def workflow_modules() -> dict[str, ast.Module]:
    """The scripts every workflow that runs trymerge.py runs, and every module or
    package under .github/scripts they import, transitively; github_utils, where
    the comment posters are defined, is left out."""
    todo = []
    for workflow in trymerge_workflows():
        todo += scripts_run(workflow)
    found: dict[str, ast.Module] = {}
    while todo:
        name = todo.pop()
        path = module_path(name)
        if name in found or path is None:
            continue
        found[name] = tree = ast.parse(path.read_text())
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        parts = name.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]  # parents
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = package.split(".") if package else []
                base = base[: len(base) - node.level + 1] if node.level else []
                module = ".".join(base + ([node.module] if node.module else []))
                if module:
                    todo.append(module)
                todo += [f"{module}.{a.name}".lstrip(".") for a in node.names]
            elif isinstance(node, ast.Import):
                todo += [alias.name for alias in node.names]
    found.pop("github_utils", None)
    return found


def calls_to(tree: ast.AST, names: set[str]) -> list[tuple[ast.Call, str | None]]:
    """Calls in `tree` of a function named in `names` (called directly or as an
    attribute), each with the function that makes it."""
    found: list[tuple[ast.Call, str | None]] = []

    def visit(node: ast.AST, function: str | None) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function = node.name
        if isinstance(node, ast.Call):
            func = node.func
            called = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            )
            if called in names:
                found.append((node, function))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(tree, None)
    return found


def comment_arg(call: ast.Call) -> str:
    for keyword in call.keywords:
        if keyword.arg == "comment":
            return ast.unparse(keyword.value)
    return ast.unparse(call.args[3]) if len(call.args) > 3 else "?"


def renamed_imports(tree: ast.Module, names: set[str]) -> list[str]:
    """Imports of `names` under another name. A helper module that does this can
    hand the function on under a name the scans below don't follow, so each one
    has to be listed as reviewed."""
    return [
        f"import {a.name} as {a.asname}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for a in node.names
        if a.name in names and a.asname
    ]


def bound_names(tree: ast.Module, names: set[str]) -> set[str]:
    """`names` plus whatever a module imports them as."""
    bound = set(names)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            bound |= {a.asname for a in node.names if a.name in names and a.asname}
    return bound


def stray_references(tree: ast.Module, names: set[str]) -> list[str]:
    """References to `names` that aren't calls, such as `post = post_pr_comment`,
    which would hide the calls made through them."""
    called = {id(call.func) for call, _ in calls_to(tree, names)}
    return [
        ast.unparse(node)
        for node in ast.walk(tree)
        if id(node) not in called
        and (
            (isinstance(node, ast.Name) and node.id in names)
            or (isinstance(node, ast.Attribute) and node.attr in names)
        )
    ]


RAW_POSTERS = {"gh_post_pr_comment", "_gh_post_comment"}


def raw_posts(name: str, tree: ast.Module) -> list[tuple[str, str | None, str]]:
    """Every use of github_utils' comment posters in a module, as
    (module, function, posted expression), plus any other way it could reach
    GitHub's comment API: a reference to a poster that isn't a call, or a
    literal comments URL."""
    aliases = bound_names(tree, RAW_POSTERS)
    found = [
        (name, function, comment_arg(call))
        for call, function in calls_to(tree, aliases)
    ]
    found += [
        (name, "<module>", f"reference to {ref}")
        for ref in stray_references(tree, aliases)
    ]
    found += [
        (name, "<module>", rename) for rename in renamed_imports(tree, RAW_POSTERS)
    ]
    found += [
        (name, "<module>", f"literal {node.value!r}")
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and "/comments" in str(node.value)
    ]
    return found


class TestWorkflowScriptsPostThroughTheTaggingPath(TestCase):
    """Across every module the workflows that run trymerge.py run, comments are
    posted only from the reviewed places below: trymerge.py's post_pr_comment,
    comment_on_pr.py's one reply (tagged, see TestCanceledJobComment), and
    tryrebase.py's rebase replies. A new posting path, such as a helper module
    that calls gh_post_pr_comment, fails this test: route a merge or revert reply
    through post_pr_comment or with_status, then list it here."""

    modules = workflow_modules()

    REVIEWED = [
        ("comment_on_pr", "main", "msg"),
        ("trymerge", "post_pr_comment", "with_status(comment, status, comment_id)"),
    ]
    # Modules whose replies report no merge or revert outcome, with the functions
    # that post them; their text doesn't matter here
    UNTAGGED = {"tryrebase"}
    REVIEWED_UNTAGGED = [
        ("tryrebase", "<module>"),
        ("tryrebase", "main"),
        ("tryrebase", "main"),
        ("tryrebase", "main"),
        ("tryrebase", "post_already_uptodate"),
        ("tryrebase", "rebase_ghstack_onto"),
        ("tryrebase", "rebase_ghstack_onto"),
        ("tryrebase", "rebase_native_stack_onto"),
        ("tryrebase", "rebase_onto"),
    ]

    def test_posting_paths_are_reviewed(self) -> None:
        found = sorted(
            post
            for name, tree in self.modules.items()
            for post in raw_posts(name, tree)
        )
        self.assertEqual(
            [post for post in found if post[0] not in self.UNTAGGED], self.REVIEWED
        )
        self.assertEqual(
            [post[:2] for post in found if post[0] in self.UNTAGGED],
            self.REVIEWED_UNTAGGED,
        )


class TestEveryCommentDeclaresItsStatus(TestCase):
    """post_pr_comment's `status` has no default, so every reply has to say
    whether it reports how a merge or revert command went; if it does, it gets
    the tag. These tests cover every call of it in the workflows' modules."""

    modules = workflow_modules()
    posts = [
        (name, call, function)
        for name, tree in modules.items()
        for call, function in calls_to(tree, bound_names(tree, {"post_pr_comment"}))
    ]

    def test_post_pr_comment_is_only_called(self) -> None:
        for name, tree in self.modules.items():
            with self.subTest(module=name):
                names = bound_names(tree, {"post_pr_comment"})
                self.assertEqual(stray_references(tree, names), [])
                self.assertEqual(renamed_imports(tree, {"post_pr_comment"}), [])

    OUTCOMES = {"MERGE_SUCCESS", "MERGE_FAILURE", "REVERT_SUCCESS", "REVERT_FAILURE"}

    def is_status(self, node: ast.expr) -> bool:
        if isinstance(node, ast.IfExp):
            return self.is_status(node.body) and self.is_status(node.orelse)
        return self.is_none(node) or (
            isinstance(node, ast.Name) and node.id in self.OUTCOMES
        )

    @staticmethod
    def is_none(node: ast.expr) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    def may_be_none(self, node: ast.expr) -> bool:
        if isinstance(node, ast.IfExp):
            return self.may_be_none(node.body) or self.may_be_none(node.orelse)
        return self.is_none(node)

    def test_every_comment_passes_an_outcome_or_none(self) -> None:
        """A status is one of the outcome constants or a literal None, or a choice
        between them, so the reviewed-notices test below sees every post that can
        go untagged."""
        self.assertGreater(len(self.posts), 10)
        for name, call, function in self.posts:
            with self.subTest(module=name, function=function, line=call.lineno):
                status = [k.value for k in call.keywords if k.arg == "status"]
                self.assertEqual(len(status), 1)
                self.assertTrue(self.is_status(status[0]), ast.unparse(status[0]))

    def test_every_tagged_comment_names_its_command(self) -> None:
        """A tagged reply passes the id of the comment it answers (None when there
        is none), so a reader can tell it from the reply to an earlier command."""
        for name, call, function in self.posts:
            status = [k.value for k in call.keywords if k.arg == "status"]
            if status and self.is_none(status[0]):
                continue
            with self.subTest(module=name, function=function, line=call.lineno):
                passed = [
                    ast.unparse(k.value) for k in call.keywords if k.arg == "comment_id"
                ]
                self.assertIn(passed, (["comment_id"], ["args.comment_id"]))

    # Every comment that can be posted without a status, with the module and
    # function that post it. Only notices that report no merge or revert outcome
    # belong here, and refusals that are untagged only when no command was
    # issued.
    # If you add or reword one, check it still reports none: a reply saying a
    # merge or revert started, landed, failed or was refused passes its status
    # instead. `result.comment` and `msg` are written by greenlight_guard.py and
    # native_stack.branch_update_comments.
    REVIEWED_NOTICES = [
        (
            "trymerge",
            "ensure_mergeable_labels",
            "f'Adding the `{NOT_USER_FACING_LABEL}` label automatically: this co-dev "
            "PR is being merged after its internal diff landed, without a "
            "release-notes label. If this change is user facing, please replace the "
            "label with the appropriate `release notes: ...` one.'",
        ),
        ("trymerge", "check_greenlight_reviewed_head_sha", "result.comment"),
        (
            "trymerge",
            "post_starting_merge_comment",
            "f'Starting merge as part of PR stack under #{pr.pr_num}'",
        ),
        (
            "trymerge",
            "manually_close_merged_pr",
            "f'This PR (#{additional_pr.pr_num}) was merged as part of PR #{pr.pr_num} "
            "in the stack under {merge_commit_sha} but it is still open, likely due to "
            "a Github bug, so mergebot is closing it manually. If you think this is a "
            "mistake, please feel free to reopen and contact Dev Infra.'",
        ),
        ("trymerge", "reopen_reverted_prs", "revert_message"),
        ("trymerge", "refresh_reverted_branches", "msg"),
        (
            "trymerge",
            "refresh_reverted_branches",
            "f'The branches of the reverted PRs could not be updated, so they may not "
            "show only their own changes. To update them, comment `@pytorchbot rebase "
            "-b {default_branch}` on the topmost reverted PR. Error: {e}'",
        ),
        # A failure after the merge or revert landed (LandedError)
        (
            "trymerge",
            "handle_exception",
            "format_failure_comment(e.__cause__, title, run_url)",
        ),
        # Refusals, untagged only in a --check-mergeability run
        ("trymerge", "main", 'f"Can\'t merge closed PR #{args.pr_num}"'),
        ("trymerge", "main", "'Cross-repo ghstack merges are not supported'"),
        ("trymerge", "main", "refusal"),
    ]

    def test_untagged_comments_are_the_reviewed_notices(self) -> None:
        untagged = [
            (name, function, comment_arg(call))
            for name, call, function in self.posts
            if any(
                k.arg == "status" and self.may_be_none(k.value) for k in call.keywords
            )
        ]
        self.assertEqual(untagged, self.REVIEWED_NOTICES)


class TestCanceledJobComment(TestCase):
    def post(self, action: str, comment_id: str | None = None) -> str:
        env = {} if comment_id is None else {"COMMENT_ID": comment_id}
        args = mock.MagicMock(pr_num=123, action=action)
        repo = mock.MagicMock()
        repo.gh_owner_and_name.return_value = ("pytorch", "pytorch")
        with (
            mock.patch.object(comment_on_pr, "parse_args", return_value=args),
            mock.patch.object(comment_on_pr, "GitRepo", return_value=repo),
            mock.patch.object(comment_on_pr, "get_git_repo_dir"),
            mock.patch.object(comment_on_pr, "get_git_remote_name"),
            mock.patch.object(comment_on_pr, "gh_post_pr_comment") as post,
            mock.patch.dict(comment_on_pr.os.environ, env),
        ):
            if comment_id is None:
                comment_on_pr.os.environ.pop("COMMENT_ID", None)
            comment_on_pr.main()
        post.assert_called_once()
        body: Any = post.call_args.args[3]
        self.assertIn("was canceled or timed out", body)
        return body

    def test_merge_and_revert_are_tagged_failure(self) -> None:
        self.assertEqual(parse_status(self.post("merge")), MERGE_FAILURE)
        self.assertEqual(parse_status(self.post("revert")), REVERT_FAILURE)

    def test_the_tag_names_the_command_the_job_ran_for(self) -> None:
        # A second merge command cancels the first one's job, so this reply can
        # land after the second command
        self.assertEqual(parse_tag(self.post("merge", "777")), Tag(MERGE_FAILURE, 777))
        self.assertEqual(parse_tag(self.post("merge")), Tag(MERGE_FAILURE, None))
        # An empty value is what the workflow passes for a revert dispatched
        # without a comment
        self.assertEqual(parse_tag(self.post("revert", "")), Tag(REVERT_FAILURE, None))
        # A value that isn't a comment id never costs the reply
        self.assertEqual(
            parse_tag(self.post("merge", "1" * 5000)), Tag(MERGE_FAILURE, None)
        )

    def test_rebase_is_not_tagged(self) -> None:
        self.assertIsNone(parse_status(self.post("rebase")))


if __name__ == "__main__":
    main()
