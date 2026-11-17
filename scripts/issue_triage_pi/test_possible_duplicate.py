"""Possible-duplicate suggestions through the trusted apply entry point."""

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import apply_plan


REPO = "pytorch/pytorch"
SOURCE = 203001
TARGET = 203002


def issue(number, **extra):
    return {
        "number": number,
        "html_url": f"https://github.com/{REPO}/issues/{number}",
        "state": "open",
        "labels": [],
        **extra,
    }


class IssueAPI:
    def __init__(self, target=TARGET):
        self.source = issue(SOURCE)
        self.target = issue(target)
        self.target_number = target
        self.comments = []
        self.writes = []
        self.fail_labels = False
        self.target_reads = 0
        self.change_target = False
        self.source_reads = 0
        self.close_source = False

    def __call__(self, args, **kwargs):
        source = f"repos/{REPO}/issues/{SOURCE}"
        if args[0] == "-X":
            method, endpoint = args[1:3]
            self.writes.append((list(args), kwargs))
            if method == "POST" and endpoint == f"{source}/comments":
                self.comments.append(
                    {
                        "user": {"login": apply_plan.TRIAGE_BOT},
                        "body": args[4].removeprefix("body="),
                    }
                )
            elif method == "POST" and endpoint == f"{source}/labels":
                if self.fail_labels:
                    self.fail_labels = False
                    raise RuntimeError("label write failed")
                names = {label["name"] for label in self.source["labels"]}
                names.update(arg.removeprefix("labels[]=") for arg in args[4::2])
                self.source["labels"] = [{"name": name} for name in sorted(names)]
            else:
                raise AssertionError(f"unexpected write: {args}")
            return "{}"
        if args[0] == "--paginate":
            if args[2] != f"{source}/comments?per_page=100":
                raise AssertionError(f"unexpected paginated read: {args}")
            return json.dumps([self.comments])
        if args[0] == source:
            self.source_reads += 1
            if self.close_source and self.source_reads > 1:
                self.source["state"] = "closed"
            return json.dumps(self.source)
        if args[0] == f"{source}/comments?per_page=100":
            return json.dumps(self.comments)
        if args[0] == f"repos/{REPO}/issues/{self.target_number}":
            self.target_reads += 1
            if self.change_target and self.target_reads > 1:
                self.target["html_url"] = (
                    f"https://github.com/other/repo/issues/{TARGET}"
                )
            if self.target is None:
                raise RuntimeError("target unavailable")
            return json.dumps(self.target)
        raise AssertionError(f"unexpected read: {args}")


class PossibleDuplicateTest(unittest.TestCase):
    def run_plan(self, api, *, apply=True, **extra):
        plan = {
            "decision": "label",
            "labels": ["module: nn"],
            "templates": [],
            "reasoning": "Same underlying problem after reading the candidate.",
            "possible_duplicate": TARGET,
            **extra,
        }
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "plan.json"
            filename.write_text(json.dumps(plan))
            argv = ["apply_plan", REPO, str(SOURCE), str(filename)]
            if apply:
                argv.append("--apply")
            with (
                patch("sys.argv", argv),
                patch("apply_plan.gh_api", side_effect=api),
                patch("apply_plan.publish") as publish,
                patch("sys.stdout", new_callable=StringIO),
            ):
                apply_plan.main()
            return publish.call_args.args[0]

    def test_suggestion_posts_fixed_source_comment_and_never_closes(self):
        api = IssueAPI()
        summary = self.run_plan(api)
        self.assertEqual(len(api.comments), 1)
        reference = f"[{REPO}#{TARGET}](https://github.com/{REPO}/issues/{TARGET})"
        expected = (
            f"This looks similar to {reference} and may be the same underlying issue. "
            "Could you check whether that report covers your case?\n\n"
            "If it does, please add any new reproduction details there "
            "and close this issue "
            "so we can track the discussion in one place. If yours differs—or still "
            "reproduces after the linked issue’s fix—please explain the difference here."
        )
        marker = f"<!-- issue-triage-pi:possible-duplicate:{REPO}#{TARGET} -->"
        self.assertEqual(api.comments[0]["body"], f"{expected}\n\n{marker}")
        self.assertEqual(api.source["state"], "open")
        self.assertEqual(api.target["state"], "open")
        self.assertIn(f"**Possible duplicate:** {REPO}#{TARGET}", summary)
        self.assertIn("**Close:** False", summary)
        self.assertEqual(len(api.writes), 2)
        self.assertFalse(api.writes[0][1]["retry"])

    def test_dry_run_reports_suggestion_without_writes(self):
        api = IssueAPI()
        summary = self.run_plan(api, apply=False)
        self.assertIn(f"**Possible duplicate:** {REPO}#{TARGET}", summary)
        self.assertIn("**Close:** False", summary)
        self.assertEqual(api.writes, [])

    def test_failed_label_write_retry_does_not_repeat_comment(self):
        api = IssueAPI()
        api.fail_labels = True
        with self.assertRaisesRegex(RuntimeError, "label write failed"):
            self.run_plan(api)
        self.assertEqual(len(api.comments), 1)
        self.assertEqual(api.source["labels"], [])
        api.writes.clear()
        self.run_plan(api)
        self.assertEqual(len(api.comments), 1)
        self.assertEqual(len(api.writes), 1)
        self.assertTrue(api.writes[0][0][2].endswith("/labels"))
        api.writes.clear()
        self.run_plan(api)
        self.assertEqual(api.writes, [])
        self.assertEqual(api.source["state"], "open")

    def test_only_matching_bot_marker_suppresses_suggestion(self):
        for author, target, expected in (
            (apply_plan.TRIAGE_BOT, TARGET, 1),
            ("reporter", TARGET, 2),
            (apply_plan.TRIAGE_BOT, TARGET + 1, 2),
        ):
            with self.subTest(author=author, target=target):
                api = IssueAPI()
                api.comments = [
                    {
                        "user": {"login": author},
                        "body": f"<!-- issue-triage-pi:possible-duplicate:{REPO}#{target} -->",
                    }
                ]
                self.run_plan(api)
                self.assertEqual(len(api.comments), expected)

    def test_invalid_or_catalog_targets_never_get_duplicate_suggestions(self):
        for number in (True, 0, -1, SOURCE, "203002", 195547, 198017):
            with self.subTest(number=number):
                api = IssueAPI(target=number)
                summary = self.run_plan(api, possible_duplicate=number)
                self.assertEqual(api.comments, [])
                self.assertIn("**Close:** False", summary)
                self.assertIn(
                    apply_plan.TRIAGE_REVIEW,
                    [label["name"] for label in api.source["labels"]],
                )

    def test_unavailable_foreign_or_pr_target_never_gets_a_comment(self):
        for target in (
            None,
            issue(TARGET, html_url=f"https://github.com/other/repo/issues/{TARGET}"),
            issue(TARGET, pull_request={}),
            issue(TARGET + 1),
            issue(TARGET, state="unknown"),
        ):
            with self.subTest(target=target):
                api = IssueAPI()
                api.target = target
                self.run_plan(api)
                self.assertEqual(api.comments, [])
                self.assertEqual(api.source["state"], "open")

    def test_closed_candidate_can_be_suggested_without_mutating_it(self):
        api = IssueAPI()
        api.target = issue(TARGET, state="closed")
        self.run_plan(api)
        self.assertEqual(len(api.comments), 1)
        self.assertEqual(api.target["state"], "closed")
        self.assertEqual(api.source["state"], "open")

    def test_duplicate_annotation_cannot_be_combined_with_closure(self):
        for extra in (
            {"decision": "close_question", "templates": ["redirect_to_forum"]},
            {
                "decision": "close_expected_behavior",
                "templates": ["numerical_accuracy"],
            },
            {"umbrella": {"issue": 195547, "match": "covered"}},
        ):
            with self.subTest(extra=extra):
                api = IssueAPI()
                summary = self.run_plan(api, **extra)
                self.assertEqual(api.comments, [])
                self.assertEqual(api.source["state"], "open")
                self.assertIn("**Close:** False", summary)
                self.assertIn(
                    apply_plan.TRIAGE_REVIEW,
                    [label["name"] for label in api.source["labels"]],
                )

    def test_source_and_target_are_revalidated_before_writes(self):
        api = IssueAPI()
        api.change_target = True
        self.run_plan(api)
        self.assertEqual(api.comments, [])
        for source in (
            issue(SOURCE, state="closed"),
            issue(SOURCE, pull_request={}),
            issue(SOURCE, html_url=f"https://github.com/other/repo/issues/{SOURCE}"),
            issue(SOURCE, labels=[{"name": "oncall: pt2"}]),
        ):
            with self.subTest(source=source):
                api = IssueAPI()
                api.source = source
                self.run_plan(api)
                self.assertEqual(api.writes, [])

    def test_skip_fallback_does_not_mutate_a_source_closed_before_apply(self):
        api = IssueAPI()
        api.close_source = True
        self.run_plan(api, decision="skip_already_routed")
        self.assertEqual(api.source["state"], "closed")
        self.assertEqual(api.writes, [])

    def test_suggestion_effects_require_the_plan_for_revalidation(self):
        with self.assertRaisesRegex(ValueError, "trusted revalidation"):
            apply_plan.apply_effects(
                REPO, SOURCE, apply_plan.Effects(possible_duplicate=TARGET)
            )


if __name__ == "__main__":
    unittest.main()
