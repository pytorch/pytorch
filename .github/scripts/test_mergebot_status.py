"""Tests for mergebot_status.py, and guards that keep mergebot's replies tagged."""

from __future__ import annotations

import ast
import json
import re
from hashlib import sha256
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
        quoted = f"Error: `{MERGE_SUCCESS.tag()}`"
        self.assertIsNone(parse_status(with_status(quoted, None)))
        self.assertIsNone(
            parse_status(with_status(f"Error:\n{MERGE_SUCCESS.tag()}", None))
        )
        self.assertEqual(
            parse_status(with_status(quoted, MERGE_FAILURE)), MERGE_FAILURE
        )
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
        for step in job["steps"]:
            if "uses" in step:
                step["uses"] = step["uses"].split("@")[0]
    return jobs


class TestMergeWorkflowSteps(TestCase):
    """Any change to what the merge and revert workflows run fails this test, so
    whoever makes it checks that the workflows still reply to merge and revert
    commands only through code that tags those replies (trymerge.py and
    comment_on_pr.py; tryrebase.py reports rebases, which carry no status). Once
    that holds, update the hash. Action version bumps don't change it."""

    PINNED = {
        "trymerge.yml": "c2c010959270dfb408a5dcde0183e57fb2d633d215ec5e3bc8f75f668647fb8f",
        "revert.yml": "c62cca424cdf4081d6723ec3d7362a6698f5e0e9edac2e87a90f506fdbdd8068",
    }

    def test_jobs_are_reviewed(self) -> None:
        for workflow, pinned in self.PINNED.items():
            with self.subTest(workflow=workflow):
                jobs = json.dumps(workflow_jobs(workflow), sort_keys=True)
                self.assertEqual(sha256(jobs.encode()).hexdigest(), pinned)


def module_path(name: str) -> Path | None:
    base = SCRIPTS.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.exists():
            return path
    return None


def workflow_modules() -> dict[str, ast.Module]:
    """The scripts the merge and revert workflows run, and every module or
    package under .github/scripts they import, transitively; github_utils, where
    the comment posters are defined, is left out."""
    todo = []
    for workflow in ("trymerge.yml", "revert.yml"):
        text = json.dumps(workflow_jobs(workflow))
        todo += re.findall(r"\.github/scripts/(\w+)\.py", text)
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
    """Across every module the merge and revert workflows run, comments are
    posted only from the reviewed places below: trymerge.py's post_pr_comment,
    comment_on_pr.py's one reply (tagged, see TestCanceledJobComment), and
    tryrebase.py's rebase replies. A new posting path, such as a helper module
    that calls gh_post_pr_comment, fails this test: route a merge or revert reply
    through post_pr_comment or with_status, then list it here."""

    modules = workflow_modules()

    REVIEWED = [
        ("comment_on_pr", "main", "msg"),
        ("trymerge", "post_pr_comment", "with_status(comment, status, comment_id)"),
        ("tryrebase", "<module>", "import gh_post_pr_comment as gh_post_comment"),
        ("tryrebase", "main", 'f"PR #{args.pr_num} is closed, won\'t rebase"'),
        ("tryrebase", "main", "msg"),
        ("tryrebase", "main", "msg"),
        ("tryrebase", "post_already_uptodate", "msg"),
        (
            "tryrebase",
            "rebase_ghstack_onto",
            "f'Rebased `{orig_ref}` onto `{onto_branch}` because #{pr.pr_num} was rebased, please pull locally before adding more changes (for example, via `ghstack ' + f'checkout https://github.com/{org}/{project}/pull/{pr_num}`)'",
        ),
        (
            "tryrebase",
            "rebase_ghstack_onto",
            "f'Successfully rebased `{orig_ref}` onto `{onto_branch}`, please pull locally ' + 'before adding more changes (for example, via `ghstack ' + f'checkout https://github.com/{org}/{project}/pull/{pr.pr_num}`)'",
        ),
        ("tryrebase", "rebase_native_stack_onto", "msg"),
        (
            "tryrebase",
            "rebase_onto",
            "f'Successfully rebased `{pr.head_ref()}` onto `{onto_branch}`, please pull locally ' + f'before adding more changes (for example, via `git checkout {pr.head_ref()} && ' + 'git pull --rebase`)'",
        ),
    ]

    def test_posting_paths_are_reviewed(self) -> None:
        found = sorted(
            post
            for name, tree in self.modules.items()
            for post in raw_posts(name, tree)
        )
        self.assertEqual(found, self.REVIEWED)


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

    def is_outcome(self, node: ast.expr) -> bool:
        if isinstance(node, ast.IfExp):
            return self.is_outcome(node.body) and self.is_outcome(node.orelse)
        return isinstance(node, ast.Name) and node.id in self.OUTCOMES

    @staticmethod
    def is_none(node: ast.expr) -> bool:
        return isinstance(node, ast.Constant) and node.value is None

    def test_every_comment_passes_an_outcome_or_none(self) -> None:
        """A status is one of the outcome constants (or a choice between them) or a
        literal None, so the reviewed-notices test below sees every untagged post."""
        self.assertGreater(len(self.posts), 10)
        for name, call, function in self.posts:
            with self.subTest(module=name, function=function, line=call.lineno):
                status = [k.value for k in call.keywords if k.arg == "status"]
                self.assertEqual(len(status), 1)
                self.assertTrue(
                    self.is_outcome(status[0]) or self.is_none(status[0]),
                    ast.unparse(status[0]),
                )

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

    # Every comment posted without a status, with the module and function that
    # post it. Only notices that report no merge or revert outcome belong here.
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
    ]

    def test_untagged_comments_are_the_reviewed_notices(self) -> None:
        untagged = [
            (name, function, comment_arg(call))
            for name, call, function in self.posts
            if any(k.arg == "status" and self.is_none(k.value) for k in call.keywords)
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
