"""Approved umbrella policies at the model-free issue apply boundary."""

import json
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import patch

import apply_plan


REPO = "pytorch/pytorch"
SOURCE = 203001
TARGET = 195547
VALID = apply_plan.load_valid_labels()
TEMPLATES = apply_plan.load_templates(apply_plan.TRIAGE_SKILL)


def issue(number, *, labels=(), **extra):
    return {
        "id": number + 1_000_000,
        "number": number,
        "html_url": f"https://github.com/{REPO}/issues/{number}",
        "state": "open",
        "labels": [{"name": label} for label in labels],
        **extra,
    }


def plan(number=TARGET, *, match="covered", decision="label", labels=(), templates=()):
    return {
        "decision": decision,
        "labels": ["module: nn", *labels],
        "templates": list(templates),
        "reasoning": "This is a missing-validation hardening report.",
        "umbrella": {"issue": number, "match": match},
    }


class GitHub:
    def __init__(
        self,
        number: int = TARGET,
        *,
        labels: Iterable[str] = (),
        comments: Iterable[dict] = (),
        close_failures: int = 0,
    ) -> None:
        self.source = issue(SOURCE, labels=labels)
        self.target = issue(number)
        self.comments = list(comments)
        self.writes = []
        self.close_failures = close_failures
        self.target_error = False
        self.after_labels = None

    def __call__(self, args: list[str], *, retry: bool = True) -> str:
        source = f"repos/{REPO}/issues/{SOURCE}"
        comments = source + "/comments?per_page=100"
        if args[0] != "-X":
            if args[0] == source:
                return json.dumps(self.source)
            if args == ["--paginate", "--slurp", comments]:
                return json.dumps([self.comments[:1], self.comments[1:]])
            if args[0] == comments:
                return json.dumps(self.comments[:1])
            if args[0] in {
                f"repos/{REPO}/issues/{n}" for n in (TARGET, 198017, 197554)
            }:
                if retry:
                    raise AssertionError(
                        "target lookup must not retry a stale snapshot"
                    )
                if self.target_error:
                    raise RuntimeError("target unavailable")
                return json.dumps(self.target)
            raise AssertionError(f"unexpected read: {args}")
        self.writes.append(args)
        if args[1:3] == ["POST", source + "/comments"]:
            if retry:
                raise AssertionError("comment must not retry a stale snapshot")
            self.comments.append(
                {
                    "body": args[-1].removeprefix("body="),
                    "user": {"login": apply_plan.TRIAGE_BOT},
                }
            )
        elif args[1:3] == ["POST", source + "/labels"]:
            names = {label["name"] for label in self.source["labels"]}
            names.update(arg.removeprefix("labels[]=") for arg in args[4::2])
            self.source["labels"] = [{"name": name} for name in sorted(names)]
            if self.after_labels:
                self.after_labels(self)
        elif args[1:3] == ["PATCH", source] and "state=closed" in args:
            if retry:
                raise AssertionError("close must not retry a stale snapshot")
            if self.close_failures:
                self.close_failures -= 1
                raise RuntimeError("close failed")
            self.source["state"] = "closed"
        else:
            raise AssertionError(f"unexpected write: {args}")
        return "{}"


class UmbrellaPlanTest(unittest.TestCase):
    def evaluate(self, proposal, *, source=None, target=None, comments=(), repo=REPO):
        source = issue(SOURCE) if source is None else source
        target = issue(proposal["umbrella"]["issue"]) if target is None else target
        return apply_plan.plan_effects(
            proposal,
            {label["name"] for label in source["labels"]},
            VALID,
            TEMPLATES,
            list(comments),
            repo=repo,
            issue=SOURCE,
            source=source,
            umbrella_target=target,
        )

    def apply(self, api, proposal, preview=None):
        if preview is None:
            preview = self.evaluate(proposal, source=api.source, target=api.target)
        with patch("apply_plan.gh_api", side_effect=api):
            return apply_plan.apply_effects(REPO, SOURCE, preview, plan=proposal)

    def test_catalog_actions_through_main_and_dry_run(self):
        for number, writes, labels in (
            (TARGET, False, []),
            (TARGET, True, []),
            (198017, True, []),
            (197554, True, []),
            (198017, True, ["oncall: pt2"]),
        ):
            with (
                self.subTest(number=number, apply=writes, labels=labels),
                tempfile.TemporaryDirectory() as temp,
            ):
                api = GitHub(number)
                filename = Path(temp) / "plan.json"
                filename.write_text(json.dumps(plan(number, labels=labels)))
                argv = ["apply_plan", REPO, str(SOURCE), str(filename)]
                if writes:
                    argv.append("--apply")
                with (
                    patch("sys.argv", argv),
                    patch("apply_plan.gh_api", side_effect=api),
                    patch("apply_plan.publish") as publish,
                ):
                    self.assertEqual(apply_plan.main(), 0)
                close = number == TARGET
                summary = publish.call_args.args[0]
                self.assertIn(f"**Umbrella:** {'close' if close else 'link'}", summary)
                self.assertIn(f"{REPO}#{number}", summary)
                self.assertIn(f"**Close:** {close}", summary)
                expected_state = "closed" if writes and close else "open"
                self.assertEqual(api.source["state"], expected_state)
                if not writes:
                    self.assertEqual(api.writes, [])
                    continue
                reference = (
                    f"[{REPO}#{number}](https://github.com/{REPO}/issues/{number})"
                )
                if close:
                    body = f"Duplicate of {reference}. This report is covered by the umbrella's scope."  # noqa: B950
                else:
                    body = f"Related tracking issue: {reference}. Linking for context; this issue remains open for its own investigation."  # noqa: B950
                expected = (
                    f"{body}\n\n<!-- issue-triage-pi:umbrella:{REPO}#{number} -->"
                )
                self.assertEqual([c["body"] for c in api.comments], [expected])
                if close:
                    self.assertIn("state_reason=duplicate", api.writes[-1])
                    self.assertEqual(
                        api.writes[-1][-2:],
                        ["-F", f"duplicate_issue_id={api.target['id']}"],
                    )
                else:
                    self.assertFalse(any(args[1] == "PATCH" for args in api.writes))

    def test_incomplete_reports_link_without_closing(self):
        cases = (
            (plan(match="uncertain"), [], True, None),
            (plan(decision="triage_review"), [], True, None),
            (plan(), ["triage review"], True, None),
            (plan(labels=["triage review"]), [], True, None),
            (plan(labels=["high priority"]), [], True, None),
            (plan(match="uncertain", labels=["oncall: pt2"]), [], True, None),
            (plan(labels=["oncall: pt2", "triage review"]), [], True, None),
            (plan(), ["high priority", "triaged"], False, None),
            (plan(), ["needs reproduction"], False, None),
            (plan(decision="request_info"), [], False, "request_more_info"),
            (
                plan(decision="request_reproduction"),
                [],
                False,
                "request_self_contained_reproduction",
            ),
            (plan(templates=["request_more_info"]), [], False, "request_more_info"),
        )
        for proposal, existing, review, template in cases:
            with self.subTest(plan=proposal, existing=existing):
                result = self.evaluate(proposal, source=issue(SOURCE, labels=existing))
                self.assertEqual(result.umbrella_action, "link")
                self.assertFalse(result.close)
                self.assertIn("remains open", result.comment)
                self.assertNotIn("umbrella-pt2-close:", result.comment)
                self.assertNotIn("triaged", result.add_labels)
                resulting_labels = set(result.add_labels) | set(existing)
                self.assertEqual("triage review" in resulting_labels, review)
                if template:
                    self.assertIn(template, result.comment_templates)
                self.assertEqual(
                    result.redact_body, proposal["decision"] == "request_reproduction"
                )

    def test_invalid_targets_and_conflicting_decisions_require_review(self):
        cases = [(plan(n), issue(n)) for n in (SOURCE, 69512)]
        cases += [
            (plan(), value)
            for value in (
                {},
                issue(TARGET, id=None),
                issue(TARGET, id=True),
                issue(TARGET, state="closed"),
                issue(TARGET, pull_request={}),
                issue(
                    TARGET, html_url=f"https://github.com/other/repo/issues/{TARGET}"
                ),
                issue(198017),
            )
        ]
        cases += [
            (plan(decision=decision), issue(TARGET))
            for decision in ("close_question", "close_expected_behavior", "transfer")
        ]
        for proposal, target in cases:
            with self.subTest(plan=proposal, target=target):
                result = self.evaluate(proposal, target=target)
                self.assertEqual(result.umbrella_action, "review")
                self.assertFalse(result.close)
                self.assertEqual(result.comment, "")
                self.assertIn("triage review", result.add_labels)
                self.assertNotIn("triaged", result.add_labels)

    def test_model_cannot_supply_umbrella_actions_or_malformed_targets(self):
        for annotation in (
            None,
            {},
            {"issue": TARGET},
            {"issue": TARGET, "match": "related"},
            *(
                {"issue": n, "match": "covered"}
                for n in (True, float(TARGET), 0, -1, "195547")
            ),
            {"issue": TARGET, "match": "covered", "action": "close"},
            {"issue": TARGET, "match": "covered", "comment": "model-written prose"},
        ):
            with self.subTest(annotation=annotation):
                proposal = {**plan(), "umbrella": annotation}
                result = self.evaluate(proposal, target=issue(TARGET))
                self.assertFalse(result.close)
                self.assertEqual(result.comment, "")
                self.assertIn("triage review", result.add_labels)
        source = issue(
            SOURCE, html_url=f"https://github.com/other/repo/issues/{SOURCE}"
        )
        target = issue(
            TARGET, html_url=f"https://github.com/other/repo/issues/{TARGET}"
        )
        result = self.evaluate(plan(), repo="other/repo", source=source, target=target)
        self.assertEqual(result.umbrella_action, "review")
        self.assertEqual(result.comment, "")

    def test_routed_or_nonissue_sources_are_not_mutated(self):
        for source in (
            issue(SOURCE, labels=["oncall: pt2"]),
            issue(SOURCE, labels=["oncall: distributed", "actionable"]),
            issue(SOURCE, state="closed"),
            issue(SOURCE, pull_request={}),
        ):
            with self.subTest(source=source):
                self.assertFalse(self.evaluate(plan(), source=source).mutates)
        for label in ("oncall: distributed", "oncall: cpu inductor"):
            result = self.evaluate(plan(labels=[label]))
            self.assertFalse(result.close)
            self.assertEqual(result.comment, "")
            self.assertNotIn("triaged", result.add_labels)

    def test_redirects_skip_uncertain_or_invalid_umbrellas_without_review(self):
        for match, target in (("uncertain", issue(TARGET)), ("covered", {})):
            with self.subTest(match=match, target=target):
                proposal = plan(match=match, labels=["oncall: jit"])
                result = self.evaluate(proposal, target=target)
                self.assertEqual(result.umbrella_action, "skip")
                self.assertIn("oncall: jit", result.add_labels)
                self.assertNotIn("triage review", result.add_labels)
                self.assertEqual(result.comment, "")
                self.assertFalse(result.close)

    def test_apply_revalidates_stale_previews_and_unavailable_targets(self):
        for labels, target, error, action in (
            (["high priority"], issue(TARGET), False, "link"),
            (["triage review"], issue(TARGET), False, "link"),
            ([], issue(TARGET, state="closed"), False, "review"),
            ([], issue(TARGET), True, "review"),
        ):
            with self.subTest(labels=labels, target=target, error=error):
                proposal = plan()
                preview = self.evaluate(proposal)
                api = GitHub(labels=labels)
                api.target, api.target_error = target, error
                result = self.apply(api, proposal, preview)
                self.assertEqual(result.umbrella_action, action)
                self.assertEqual(api.source["state"], "open")
                self.assertFalse(any(args[1] == "PATCH" for args in api.writes))
                self.assertEqual(len(api.comments), int(action == "link"))

    def test_final_recheck_cancels_close_and_marks_review(self):
        api = GitHub()
        api.after_labels = lambda api: api.target.update(state="closed")
        result = self.apply(api, plan())
        self.assertFalse(result.close)
        self.assertEqual(api.source["state"], "open")
        self.assertIn({"name": "triage review"}, api.source["labels"])
        self.assertFalse(any(args[1] == "PATCH" for args in api.writes))

    def test_close_and_failed_close_retry_with_optional_pt2_routing(self):
        for labels, failures in (
            ([], 0),
            ([], 1),
            (["oncall: pt2"], 0),
            (["oncall: pt2"], 1),
        ):
            with self.subTest(labels=labels, failures=failures):
                proposal, api = plan(labels=labels), GitHub(close_failures=failures)
                if failures:
                    with self.assertRaisesRegex(RuntimeError, "close failed"):
                        self.apply(api, proposal)
                    self.assertEqual(api.source["state"], "open")
                result = self.apply(api, proposal)
                self.assertTrue(result.close)
                self.assertEqual(api.source["state"], "closed")
                self.assertEqual(len(api.comments), 1)
                if failures:
                    self.assertEqual(result.comment, "")
                    self.assertEqual(result.add_labels, [])
                if labels:
                    self.assertIn({"name": "oncall: pt2"}, api.source["labels"])
                    receipt = f"<!-- issue-triage-pi:umbrella-pt2-close:{REPO}#{TARGET}:{api.target['id']} -->"  # noqa: B950
                    self.assertIn(receipt, api.comments[0]["body"].splitlines())
                self.assertEqual(
                    [args[1] for args in api.writes],
                    ["POST", "POST", "PATCH"] + ["PATCH"] * failures,
                )
                api.writes.clear()
                self.assertFalse(self.apply(api, proposal).mutates)
                self.assertEqual(api.writes, [])

    def test_completed_pt2_receipt_does_not_close_a_reopened_issue(self):
        proposal, api = plan(labels=["oncall: pt2"]), GitHub()
        self.assertTrue(self.apply(api, proposal).close)
        api.source.update(state="open", state_reason="reopened")
        api.writes.clear()
        self.assertFalse(self.apply(api, proposal).mutates)
        self.assertEqual(api.source["state"], "open")
        self.assertEqual(api.writes, [])

    def test_only_bot_close_receipts_can_resume_preexisting_pt2_routing(self):
        source = issue(SOURCE, labels=["high priority"])
        link = self.evaluate(plan(), source=source).comment
        receipt = f"<!-- issue-triage-pi:umbrella-pt2-close:{REPO}#{TARGET}:{issue(TARGET)['id']} -->"  # noqa: B950
        for comments in (
            [],
            [{"body": link, "user": {"login": apply_plan.TRIAGE_BOT}}],
            [{"body": receipt, "user": {"login": "reporter"}}],
        ):
            with self.subTest(comments=comments):
                api = GitHub(labels=["oncall: pt2"], comments=comments)
                self.assertFalse(self.apply(api, plan(labels=["oncall: pt2"])).mutates)
                self.assertEqual(api.writes, [])

    def test_new_human_stops_cancel_close_before_patch(self):
        for stop in (
            "oncall: distributed",
            "triage review",
            "high priority",
            "needs reproduction",
        ):
            with self.subTest(stop=stop):
                api = GitHub()
                api.after_labels = lambda api, stop=stop: api.source["labels"].append(
                    {"name": stop}
                )
                result = self.apply(api, plan(labels=["oncall: pt2"]))
                self.assertFalse(result.close)
                self.assertEqual(api.source["state"], "open")
                self.assertEqual(len(api.comments), 1)
                self.assertFalse(any(args[1] == "PATCH" for args in api.writes))

    def test_pending_receipt_cannot_override_new_stops_or_changed_intent(self):
        cases = [
            ("source", {"labels": [{"name": "oncall: pt2"}, {"name": stop}]}, False)
            for stop in (
                "oncall: distributed",
                "triage review",
                "high priority",
                "needs reproduction",
            )
        ]
        cases += [
            ("target", {"id": issue(TARGET)["id"] + 1}, False),
            ("annotation", {"issue": 198017}, False),
            ("plan", {"labels": ["module: nn"]}, False),
            ("plan", {"umbrella": None}, False),
            ("annotation", {"match": "uncertain"}, False),
            ("plan", {"decision": "triage_review"}, False),
            ("plan", {"templates": ["request_more_info"]}, False),
            ("plan", {"templates": ["request_self_contained_reproduction"]}, False),
            ("source", {"state": "closed"}, False),
            ("source", {"pull_request": {}}, False),
            (
                "source",
                {"html_url": f"https://github.com/other/repo/issues/{SOURCE}"},
                False,
            ),
            ("target", {"state": "closed"}, True),
            ("target", {"pull_request": {}}, True),
            ("target", {"number": TARGET + 1}, True),
        ]
        for owner, changes, review in cases:
            with self.subTest(owner=owner, changes=changes):
                proposal = plan(labels=["oncall: pt2"])
                api = GitHub(close_failures=1)
                with self.assertRaisesRegex(RuntimeError, "close failed"):
                    self.apply(api, proposal)
                owners = {
                    "source": api.source,
                    "target": api.target,
                    "plan": proposal,
                    "annotation": proposal["umbrella"],
                }
                owners[owner].update(changes)
                api.writes.clear()
                result = self.apply(api, proposal)
                self.assertFalse(result.close)
                self.assertEqual(len(api.comments), 1)
                if review:
                    self.assertIn({"name": "triage review"}, api.source["labels"])
                else:
                    self.assertEqual(api.writes, [])
                self.assertFalse(any(args[1] == "PATCH" for args in api.writes))

    def test_existing_link_gets_only_a_close_receipt_when_adding_pt2(self):
        source = issue(SOURCE, labels=["high priority"])
        link = self.evaluate(plan(), source=source).comment
        api = GitHub(
            comments=[{"body": link, "user": {"login": apply_plan.TRIAGE_BOT}}]
        )
        proposal = plan(labels=["oncall: pt2"])
        self.assertTrue(self.apply(api, proposal).close)
        receipt = f"<!-- issue-triage-pi:umbrella-pt2-close:{REPO}#{TARGET}:{api.target['id']} -->"  # noqa: B950
        self.assertEqual([c["body"] for c in api.comments], [link, receipt])
        api.writes.clear()
        self.assertFalse(self.apply(api, proposal).mutates)
        self.assertEqual(api.writes, [])

    def test_link_retries_use_all_pages_and_only_matching_bot_markers(self):
        for author, number, added in (
            (apply_plan.TRIAGE_BOT, 198017, 0),
            ("reporter", 198017, 1),
            (apply_plan.TRIAGE_BOT, 197554, 1),
        ):
            with self.subTest(author=author, number=number):
                marker = f"<!-- issue-triage-pi:umbrella:{REPO}#{number} -->"
                api = GitHub(
                    198017,
                    comments=[
                        {"body": "earlier", "user": {"login": apply_plan.TRIAGE_BOT}},
                        {"body": marker, "user": {"login": author}},
                    ],
                )
                proposal = plan(198017)
                self.apply(api, proposal)
                self.assertEqual(len(api.comments), 2 + added)
                self.assertIn({"name": "bot-triaged"}, api.source["labels"])
                api.writes.clear()
                self.assertFalse(self.apply(api, proposal).mutates)
                self.assertEqual(api.writes, [])


if __name__ == "__main__":
    unittest.main()
