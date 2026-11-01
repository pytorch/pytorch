"""Approved umbrella policies at the model-free issue apply boundary."""

import json
import tempfile
import unittest
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import call, patch

import apply_plan


REPO = "pytorch/pytorch"
SOURCE = 203001
TARGET = 195547


def issue(number, *, state="open", labels=(), **extra):
    return {
        "id": number + 1_000_000,
        "number": number,
        "html_url": f"https://github.com/{REPO}/issues/{number}",
        "state": state,
        "labels": [{"name": label} for label in labels],
        **extra,
    }


class StatefulUmbrellaAPI:
    def __init__(
        self,
        *,
        labels: Iterable[str] = (),
        comments: Iterable[dict] = (),
        close_failures: int = 0,
        after_labels: Iterable[str] = (),
    ) -> None:
        self.source = issue(SOURCE, labels=labels)
        self.target = issue(TARGET)
        self.comments = list(comments)
        self.writes = []
        self.close_failures = close_failures
        self.after_labels = after_labels

    def __call__(self, args: list[str], *, retry: bool = True) -> str:
        source_endpoint = f"repos/{REPO}/issues/{SOURCE}"
        if args[0] != "-X":
            if args[0] == source_endpoint:
                return json.dumps(self.source)
            if args[0] in (
                f"repos/{REPO}/issues/{TARGET}",
                f"repos/{REPO}/issues/198017",
            ):
                return json.dumps(self.target)
            comments_endpoint = source_endpoint + "/comments?per_page=100"
            if args == ["--paginate", "--slurp", comments_endpoint]:
                return json.dumps([self.comments])
            raise AssertionError(f"unexpected read: {args}")
        self.writes.append(args)
        if args[1:3] == ["POST", source_endpoint + "/comments"]:
            self.comments.append(
                {
                    "body": args[-1].removeprefix("body="),
                    "user": {"login": apply_plan.TRIAGE_BOT},
                }
            )
        elif args[1:3] == ["POST", source_endpoint + "/labels"]:
            names = {
                value.removeprefix("labels[]=")
                for value in args
                if value.startswith("labels[]=")
            }
            names.update(self.after_labels)
            existing = {label["name"] for label in self.source["labels"]}
            self.source["labels"].extend(
                {"name": name} for name in sorted(names - existing)
            )
        elif args[1:3] == ["PATCH", source_endpoint] and "state=closed" in args:
            if retry:
                raise AssertionError("umbrella close must not retry a stale snapshot")
            if self.close_failures:
                self.close_failures -= 1
                raise RuntimeError("close failed after comment and labels succeeded")
            self.source["state"] = "closed"
        else:
            raise AssertionError(f"unexpected write: {args}")
        return "{}"


class UmbrellaPlanTest(unittest.TestCase):
    def run_plan(self, plan, *, source=None, target=None, comments=(), apply=False):
        source = issue(SOURCE) if source is None else source
        target = issue(plan["umbrella"]["issue"]) if target is None else target
        targets = iter(target) if isinstance(target, list) else None
        sources = iter(source) if isinstance(source, list) else None
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "plan.json"
            filename.write_text(json.dumps(plan))
            argv = ["apply_plan", REPO, str(SOURCE), str(filename)]
            if apply:
                argv.append("--apply")

            def api(args, **kwargs):
                if args[0] == f"repos/{REPO}/issues/{plan['umbrella']['issue']}":
                    value = next(targets) if targets is not None else target
                    if isinstance(value, RuntimeError):
                        raise value
                    return json.dumps(value)
                return "{}"

            def fetch(*args, **kwargs):
                value = next(sources) if sources is not None else source
                return value, list(comments)

            with (
                patch("sys.argv", argv),
                patch("apply_plan.fetch_issue", side_effect=fetch),
                patch("apply_plan.gh_api", side_effect=api) as calls,
                patch("apply_plan.publish") as publish,
            ):
                self.assertEqual(apply_plan.main(), 0)
            return publish.call_args.args[0], calls.call_args_list

    def plan(self, target=TARGET, match="covered", decision="label", labels=()):
        return {
            "decision": decision,
            "labels": ["module: nn", *labels],
            "templates": [],
            "reasoning": "This is a missing-validation hardening report.",
            "umbrella": {"issue": target, "match": match},
        }

    def test_dry_run_reports_configured_close_and_target_without_writes(self):
        summary, calls = self.run_plan(self.plan())
        self.assertIn("**Umbrella:** close", summary)
        self.assertIn(f"{REPO}#{TARGET}", summary)
        self.assertIn("**Close:** True", summary)
        self.assertTrue(all(call.args[0][0] != "-X" for call in calls))

    def evaluate(self, plan, *, source=None, target=None, comments=(), repo=REPO):
        source = issue(SOURCE) if source is None else source
        target = issue(plan["umbrella"]["issue"]) if target is None else target
        return apply_plan.plan_effects(
            plan,
            {label["name"] for label in source["labels"]},
            apply_plan.load_valid_labels(),
            apply_plan.load_templates(apply_plan.TRIAGE_SKILL),
            list(comments),
            repo=repo,
            issue=SOURCE,
            source=source,
            umbrella_target=target,
        )

    def test_catalog_action_is_enforced_and_writes_only_the_new_issue(self):
        for number, action in ((195547, "close"), (198017, "link"), (197554, "link")):
            with self.subTest(number=number, action=action):
                summary, calls = self.run_plan(self.plan(number), apply=True)
                self.assertIn(f"**Umbrella:** {action}", summary)
                writes = [entry for entry in calls if entry.args[0][0] == "-X"]
                self.assertTrue(writes)
                for entry in writes:
                    endpoint = entry.args[0][2]
                    self.assertIn(f"repos/{REPO}/issues/{SOURCE}", endpoint)
                    self.assertNotIn(str(number), endpoint)
                posted = [
                    entry.args[0][-1]
                    for entry in writes
                    if "/comments" in entry.args[0][2]
                ]
                self.assertEqual(len(posted), 1)
                self.assertIn(f"https://github.com/{REPO}/issues/{number}", posted[0])
                self.assertIn(f"issue-triage-pi:umbrella:{REPO}#{number}", posted[0])
                closures = [
                    entry for entry in writes if "state=closed" in entry.args[0]
                ]
                if action == "close":
                    self.assertEqual(
                        closures,
                        [
                            call(
                                [
                                    "-X",
                                    "PATCH",
                                    f"repos/{REPO}/issues/{SOURCE}",
                                    "-f",
                                    "state=closed",
                                    "-f",
                                    "state_reason=duplicate",
                                    "-F",
                                    f"duplicate_issue_id={number + 1_000_000}",
                                ],
                                retry=False,
                            )
                        ],
                    )
                    self.assertIn("Duplicate of", posted[0])
                    self.assertNotIn("Closing", posted[0])
                else:
                    self.assertEqual(closures, [])
                    self.assertIn("remains open", posted[0])

    def test_uncertain_and_existing_review_or_priority_only_link(self):
        for match, decision, existing, requested in (
            ("uncertain", "label", [], []),
            ("covered", "triage_review", [], []),
            ("covered", "label", ["triage review"], []),
            ("covered", "label", [], ["triage review"]),
            ("covered", "label", [], ["high priority"]),
        ):
            with self.subTest(
                match=match, decision=decision, existing=existing, requested=requested
            ):
                plan = self.plan(match=match, decision=decision, labels=requested)
                source = issue(SOURCE, labels=existing)
                result = self.evaluate(plan, source=source)
                self.assertEqual(result.umbrella_action, "link")
                self.assertFalse(result.close)
                self.assertIn("triage review", set(result.add_labels) | set(existing))
                self.assertNotIn("triaged", result.add_labels)
                self.assertIn("remains open", result.comment)
                _, calls = self.run_plan(plan, source=source, apply=True)
                self.assertFalse(
                    any("state=closed" in entry.args[0] for entry in calls)
                )

    def test_existing_priority_links_without_requeueing(self):
        source = issue(SOURCE, labels=["high priority", "triaged"])
        for target in (195547, 198017):
            with self.subTest(target=target):
                result = self.evaluate(self.plan(target), source=source)
                self.assertEqual(result.umbrella_action, "link")
                self.assertFalse(result.close)
                self.assertNotIn("triage review", result.add_labels)
                self.assertNotIn("triaged", result.add_labels)

    def test_info_and_reproduction_requests_keep_open_with_their_templates(self):
        for decision, template in (
            ("request_info", "request_more_info"),
            ("request_reproduction", "request_self_contained_reproduction"),
        ):
            with self.subTest(decision=decision):
                result = self.evaluate(self.plan(decision=decision))
                self.assertFalse(result.close)
                self.assertEqual(result.umbrella_action, "link")
                self.assertNotIn("triaged", result.add_labels)
                self.assertIn(template, result.comment_templates)
                self.assertIn("remains open", result.comment)
                self.assertEqual(result.redact_body, decision == "request_reproduction")
        plan = self.plan()
        plan["templates"] = ["request_more_info"]
        self.assertFalse(self.evaluate(plan).close)
        pending = self.evaluate(
            self.plan(), source=issue(SOURCE, labels=["needs reproduction"])
        )
        self.assertFalse(pending.close)
        self.assertEqual(pending.umbrella_action, "link")
        self.assertNotIn("triaged", pending.add_labels)
        self.assertNotIn("triage review", pending.add_labels)

    def test_close_or_transfer_decisions_with_annotation_fall_back_to_review(self):
        for decision in ("close_question", "close_expected_behavior", "transfer"):
            with self.subTest(decision=decision):
                result = self.evaluate(self.plan(decision=decision))
                self.assertFalse(result.close)
                self.assertEqual(result.umbrella_action, "review")
                self.assertEqual(result.comment, "")
                self.assertIn("triage review", result.add_labels)
                self.assertNotIn("triaged", result.add_labels)

    def test_unconfigured_invalid_or_stale_targets_never_link_or_close(self):
        for number, target in (
            (69512, issue(69512)),
            (TARGET, {}),
            (TARGET, issue(TARGET, id=None)),
            (TARGET, issue(TARGET, id=True)),
            (TARGET, issue(TARGET, state="closed")),
            (TARGET, issue(TARGET, pull_request={})),
            (
                TARGET,
                issue(
                    TARGET, html_url=f"https://github.com/other/repo/issues/{TARGET}"
                ),
            ),
            (TARGET, issue(198017)),
            (SOURCE, issue(SOURCE)),
        ):
            with self.subTest(number=number, target=target):
                result = self.evaluate(self.plan(number), target=target)
                self.assertFalse(result.close)
                self.assertEqual(result.comment, "")
                self.assertEqual(result.umbrella_action, "review")
                self.assertIn("triage review", result.add_labels)

    def test_other_repository_cannot_use_catalog_even_with_matching_issue_number(self):
        source = issue(
            SOURCE, html_url=f"https://github.com/other/repo/issues/{SOURCE}"
        )
        target = issue(
            TARGET, html_url=f"https://github.com/other/repo/issues/{TARGET}"
        )
        result = self.evaluate(
            self.plan(), repo="other/repo", source=source, target=target
        )
        self.assertEqual(result.umbrella_action, "review")
        self.assertFalse(result.close)
        self.assertEqual(result.comment, "")

    def test_missing_target_read_falls_back_to_review_without_retry(self):
        summary, calls = self.run_plan(
            self.plan(), target=RuntimeError("not found"), apply=True
        )
        self.assertIn("**Umbrella:** review", summary)
        self.assertIn("triage review", summary)
        self.assertFalse(any("state=closed" in entry.args[0] for entry in calls))
        reads = [entry for entry in calls if entry.args[0][0] != "-X"]
        self.assertTrue(all(entry.kwargs == {"retry": False} for entry in reads))
        self.assertFalse(
            any(
                "/comments" in entry.args[0][2]
                for entry in calls
                if entry.args[0][0] == "-X"
            )
        )

    def test_apply_rechecks_target_and_source_instead_of_trusting_dry_run(self):
        for source, targets, expected in (
            (
                [issue(SOURCE), issue(SOURCE)],
                [issue(TARGET), issue(TARGET, state="closed")],
                "review",
            ),
            (
                [issue(SOURCE), issue(SOURCE, labels=["high priority"])],
                [issue(TARGET), issue(TARGET)],
                "link",
            ),
            (
                [issue(SOURCE), issue(SOURCE, labels=["triage review"])],
                [issue(TARGET), issue(TARGET)],
                "link",
            ),
        ):
            with self.subTest(expected=expected, source=source, targets=targets):
                summary, calls = self.run_plan(
                    self.plan(), source=source, target=targets, apply=True
                )
                self.assertIn(f"**Umbrella:** {expected}", summary)
                self.assertFalse(
                    any("state=closed" in entry.args[0] for entry in calls)
                )

    def test_final_close_recheck_cancels_stale_target_and_adds_review(self):
        summary, calls = self.run_plan(
            self.plan(),
            target=[issue(TARGET), issue(TARGET), issue(TARGET, state="closed")],
            apply=True,
        )
        self.assertIn("closure cancelled", summary)
        self.assertIn("**Close:** False", summary)
        self.assertIn("triage review", summary)
        self.assertFalse(any("state=closed" in entry.args[0] for entry in calls))
        self.assertTrue(
            any("labels[]=triage review" in entry.args[0] for entry in calls)
        )

    def test_retry_does_not_repeat_reference_and_completed_link_is_noop(self):
        for number in (TARGET, 198017):
            with self.subTest(number=number):
                plan = self.plan(number)
                comment = self.evaluate(plan).comment
                result = self.evaluate(plan, comments=[comment])
                self.assertEqual(result.comment, "")
                self.assertIn("bot-triaged", result.add_labels)
                completed = issue(
                    SOURCE, labels=["module: nn", "triaged", "bot-triaged"]
                )
                result = self.evaluate(plan, source=completed, comments=[comment])
                self.assertEqual(result.add_labels, [])
                summary, calls = self.run_plan(
                    plan, source=completed, comments=[comment], apply=True
                )
                self.assertFalse(
                    any(
                        "/comments" in entry.args[0][2]
                        for entry in calls
                        if entry.args[0][0] == "-X"
                    )
                )
                if number == 198017:
                    self.assertFalse(result.mutates)
                    self.assertTrue(all(entry.args[0][0] != "-X" for entry in calls))
                self.assertIn("already posted umbrella reference", summary)

    def test_failed_close_retry_posts_once_then_only_retries_source_closure(self):
        plan = self.plan()
        source = issue(SOURCE)
        comments = []
        writes = []
        close_attempts = 0

        def api(args, *, retry=True):
            nonlocal close_attempts
            if args[0] == f"repos/{REPO}/issues/{TARGET}":
                return json.dumps(issue(TARGET))
            writes.append(args)
            endpoint = args[2]
            if endpoint.endswith("/comments"):
                comments.append(args[-1].removeprefix("body="))
            elif endpoint.endswith("/labels"):
                source["labels"].extend(
                    {"name": value.removeprefix("labels[]=")}
                    for value in args
                    if value.startswith("labels[]=")
                )
            elif "state=closed" in args:
                self.assertFalse(retry)
                close_attempts += 1
                if close_attempts == 1:
                    raise RuntimeError(
                        "close failed after comment and labels succeeded"
                    )
                source["state"] = "closed"
            return "{}"

        with (
            patch(
                "apply_plan.fetch_issue",
                side_effect=lambda *a, **kw: (source, comments),
            ),
            patch("apply_plan.gh_api", side_effect=api),
        ):
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                apply_plan.apply_effects(REPO, SOURCE, self.evaluate(plan), plan=plan)
            self.assertEqual(source["state"], "open")
            retried = apply_plan.apply_effects(
                REPO, SOURCE, self.evaluate(plan), plan=plan
            )
            self.assertEqual(source["state"], "closed")
            self.assertEqual(retried.comment, "")
            self.assertEqual(retried.add_labels, [])
            self.assertTrue(retried.close)
            done = apply_plan.apply_effects(
                REPO, SOURCE, self.evaluate(plan), plan=plan
            )
            self.assertFalse(done.mutates)
        self.assertEqual(len(comments), 1)
        self.assertEqual(
            [(args[1], args[2]) for args in writes],
            [
                ("POST", f"repos/{REPO}/issues/{SOURCE}/comments"),
                ("POST", f"repos/{REPO}/issues/{SOURCE}/labels"),
                ("PATCH", f"repos/{REPO}/issues/{SOURCE}"),
                ("PATCH", f"repos/{REPO}/issues/{SOURCE}"),
            ],
        )

    def test_new_pt2_routing_closes_with_one_comment_and_completed_retry_is_noop(self):
        plan = self.plan(labels=["oncall: pt2"])
        api = StatefulUmbrellaAPI()
        with patch("apply_plan.gh_api", side_effect=api):
            result = apply_plan.apply_effects(
                REPO, SOURCE, self.evaluate(plan), plan=plan
            )
            self.assertTrue(result.close)
            self.assertEqual(api.source["state"], "closed")
            self.assertIn({"name": "oncall: pt2"}, api.source["labels"])
            self.assertEqual(len(api.comments), 1)
            self.assertIn("Duplicate of", api.comments[0]["body"])
            self.assertIn(
                f"<!-- issue-triage-pi:umbrella-pt2-close:{REPO}#{TARGET}:"
                f"{api.target['id']} -->",
                api.comments[0]["body"].splitlines(),
            )
            self.assertEqual(
                api.writes[-1],
                [
                    "-X",
                    "PATCH",
                    f"repos/{REPO}/issues/{SOURCE}",
                    "-f",
                    "state=closed",
                    "-f",
                    "state_reason=duplicate",
                    "-F",
                    f"duplicate_issue_id={api.target['id']}",
                ],
            )
            writes_before = len(api.writes)
            done = apply_plan.apply_effects(
                REPO, SOURCE, self.evaluate(plan), plan=plan
            )
            self.assertFalse(done.mutates)
            self.assertEqual(len(api.writes), writes_before)

    def test_failed_pt2_close_retry_repairs_without_reposting_or_relabeling(self):
        plan = self.plan(labels=["oncall: pt2"])
        api = StatefulUmbrellaAPI(close_failures=1)
        with patch("apply_plan.gh_api", side_effect=api):
            with self.assertRaisesRegex(RuntimeError, "close failed"):
                apply_plan.apply_effects(REPO, SOURCE, self.evaluate(plan), plan=plan)
            self.assertEqual(api.source["state"], "open")
            retried = apply_plan.apply_effects(
                REPO, SOURCE, self.evaluate(plan), plan=plan
            )
            self.assertTrue(retried.close)
            self.assertEqual(api.source["state"], "closed")
            self.assertEqual(retried.comment, "")
            self.assertEqual(retried.add_labels, [])
            writes_before = len(api.writes)
            done = apply_plan.apply_effects(
                REPO, SOURCE, self.evaluate(plan), plan=plan
            )
            self.assertFalse(done.mutates)
            self.assertEqual(len(api.writes), writes_before)
        self.assertEqual(len(api.comments), 1)
        self.assertEqual(
            [args[1] for args in api.writes], ["POST", "POST", "PATCH", "PATCH"]
        )

    def test_preexisting_pt2_without_a_bot_close_receipt_is_never_mutated(self):
        link = self.evaluate(
            self.plan(), source=issue(SOURCE, labels=["high priority"])
        ).comment
        receipt = (
            f"<!-- issue-triage-pi:umbrella-pt2-close:{REPO}#{TARGET}:"
            f"{issue(TARGET)['id']} -->"
        )
        for comments in (
            [],
            [{"body": link, "user": {"login": apply_plan.TRIAGE_BOT}}],
            [{"body": receipt, "user": {"login": "reporter"}}],
        ):
            with self.subTest(comments=comments):
                plan = self.plan(labels=["oncall: pt2"])
                api = StatefulUmbrellaAPI(labels=["oncall: pt2"], comments=comments)
                with patch("apply_plan.gh_api", side_effect=api):
                    result = apply_plan.apply_effects(
                        REPO, SOURCE, self.evaluate(plan), plan=plan
                    )
                self.assertFalse(result.mutates)
                self.assertEqual(api.writes, [])
                self.assertEqual(api.source["state"], "open")

    def test_new_oncall_or_human_stop_after_labels_blocks_pt2_close(self):
        for stop in (
            "oncall: distributed",
            "triage review",
            "high priority",
            "needs reproduction",
        ):
            with self.subTest(stop=stop):
                plan = self.plan(labels=["oncall: pt2"])
                api = StatefulUmbrellaAPI(after_labels=[stop])
                with patch("apply_plan.gh_api", side_effect=api):
                    result = apply_plan.apply_effects(
                        REPO, SOURCE, self.evaluate(plan), plan=plan
                    )
                self.assertFalse(result.close)
                self.assertEqual(api.source["state"], "open")
                self.assertIn({"name": stop}, api.source["labels"])
                self.assertEqual(len(api.comments), 1)
                self.assertFalse(any("state=closed" in args for args in api.writes))

    def test_pending_pt2_close_retry_still_honors_new_oncall_or_human_stops(self):
        for stop in (
            "oncall: distributed",
            "triage review",
            "high priority",
            "needs reproduction",
        ):
            with self.subTest(stop=stop):
                plan = self.plan(labels=["oncall: pt2"])
                api = StatefulUmbrellaAPI(close_failures=1)
                with patch("apply_plan.gh_api", side_effect=api):
                    with self.assertRaisesRegex(RuntimeError, "close failed"):
                        apply_plan.apply_effects(
                            REPO, SOURCE, self.evaluate(plan), plan=plan
                        )
                    api.source["labels"].append({"name": stop})
                    writes_before = len(api.writes)
                    result = apply_plan.apply_effects(
                        REPO, SOURCE, self.evaluate(plan), plan=plan
                    )
                self.assertFalse(result.mutates)
                self.assertEqual(len(api.writes), writes_before)
                self.assertEqual(api.source["state"], "open")
                self.assertEqual(len(api.comments), 1)

    def test_pt2_receipt_requires_the_same_target_id_and_current_close_plan(self):
        cases = (
            "changed_id",
            "different_target",
            "no_pt2",
            "no_umbrella",
            "uncertain",
            "review",
            "info_template",
            "repro_template",
        )
        for case in cases:
            with self.subTest(case=case):
                plan = self.plan(labels=["oncall: pt2"])
                api = StatefulUmbrellaAPI(close_failures=1)
                with patch("apply_plan.gh_api", side_effect=api):
                    with self.assertRaisesRegex(RuntimeError, "close failed"):
                        apply_plan.apply_effects(
                            REPO, SOURCE, self.evaluate(plan), plan=plan
                        )
                    match case:
                        case "changed_id":
                            api.target["id"] += 1
                        case "different_target":
                            plan["umbrella"]["issue"] = 198017
                            api.target = issue(198017)
                        case "no_pt2":
                            plan["labels"] = ["module: nn"]
                        case "no_umbrella":
                            del plan["umbrella"]
                        case "uncertain":
                            plan["umbrella"]["match"] = "uncertain"
                        case "review":
                            plan["decision"] = "triage_review"
                        case "info_template":
                            plan["templates"] = ["request_more_info"]
                        case "repro_template":
                            plan["templates"] = ["request_self_contained_reproduction"]
                    writes_before = len(api.writes)
                    result = apply_plan.apply_effects(
                        REPO,
                        SOURCE,
                        self.evaluate(plan, source=api.source, target=api.target),
                        plan=plan,
                    )
                self.assertFalse(result.mutates)
                self.assertEqual(len(api.writes), writes_before)
                self.assertEqual(api.source["state"], "open")

    def test_receipt_does_not_bypass_source_or_target_validation(self):
        for endpoint, extra in (
            ("source", {"state": "closed"}),
            ("source", {"pull_request": {}}),
            ("source", {"html_url": "https://github.com/other/repo/issues/1"}),
            ("target", {"state": "closed"}),
            ("target", {"pull_request": {}}),
            ("target", {"number": TARGET + 1}),
        ):
            with self.subTest(endpoint=endpoint, extra=extra):
                plan = self.plan(labels=["oncall: pt2"])
                api = StatefulUmbrellaAPI(close_failures=1)
                with patch("apply_plan.gh_api", side_effect=api):
                    with self.assertRaisesRegex(RuntimeError, "close failed"):
                        apply_plan.apply_effects(
                            REPO, SOURCE, self.evaluate(plan), plan=plan
                        )
                    current = api.source if endpoint == "source" else api.target
                    current.update(extra)
                    writes_before = len(api.writes)
                    result = apply_plan.apply_effects(
                        REPO, SOURCE, self.evaluate(plan), plan=plan
                    )
                self.assertFalse(result.close)
                self.assertFalse(
                    any("state=closed" in args for args in api.writes[writes_before:])
                )
                self.assertEqual(len(api.comments), 1)

    def test_receipt_does_not_change_ordinary_close_or_link_comments(self):
        for number in (TARGET, 198017, 197554):
            with self.subTest(number=number):
                reference = (
                    f"[{REPO}#{number}](https://github.com/{REPO}/issues/{number})"
                )
                marker = f"<!-- issue-triage-pi:umbrella:{REPO}#{number} -->"
                if number == TARGET:
                    body = (
                        f"Duplicate of {reference}. "
                        "This report is covered by the umbrella's scope."
                    )
                else:
                    body = (
                        f"Related tracking issue: {reference}. Linking for context; "
                        "this issue remains open for its own investigation."
                    )
                result = self.evaluate(self.plan(number))
                self.assertEqual(result.comment, f"{body}\n\n{marker}")
        for plan in (
            self.plan(198017, labels=["oncall: pt2"]),
            self.plan(match="uncertain", labels=["oncall: pt2"]),
            self.plan(labels=["oncall: pt2", "triage review"]),
        ):
            with self.subTest(plan=plan):
                result = self.evaluate(plan)
                self.assertFalse(result.close)
                self.assertNotIn("umbrella-pt2-close:", result.comment)

    def test_existing_link_reference_gets_only_the_new_pt2_close_receipt(self):
        plan = self.plan(labels=["oncall: pt2"])
        link = self.evaluate(
            self.plan(), source=issue(SOURCE, labels=["high priority"])
        ).comment
        api = StatefulUmbrellaAPI(
            comments=[{"body": link, "user": {"login": apply_plan.TRIAGE_BOT}}]
        )
        with patch("apply_plan.gh_api", side_effect=api):
            result = apply_plan.apply_effects(
                REPO, SOURCE, self.evaluate(plan), plan=plan
            )
            self.assertTrue(result.close)
            self.assertEqual(api.source["state"], "closed")
            self.assertEqual(len(api.comments), 2)
            self.assertEqual(api.comments[0]["body"], link)
            self.assertEqual(
                api.comments[1]["body"],
                f"<!-- issue-triage-pi:umbrella-pt2-close:{REPO}#{TARGET}:"
                f"{api.target['id']} -->",
            )
            writes_before = len(api.writes)
            done = apply_plan.apply_effects(
                REPO, SOURCE, self.evaluate(plan), plan=plan
            )
            self.assertFalse(done.mutates)
            self.assertEqual(len(api.writes), writes_before)

    def test_retry_marker_is_specific_to_target(self):
        comment = self.evaluate(self.plan(198017)).comment
        result = self.evaluate(self.plan(197554), comments=[comment])
        self.assertIn("197554", result.comment)
        self.assertNotIn("198017", result.comment)

    def test_existing_oncall_and_nonissue_sources_are_never_mutated(self):
        for source in (
            issue(SOURCE, labels=["oncall: pt2"]),
            issue(SOURCE, labels=["oncall: distributed", "actionable"]),
            issue(SOURCE, state="closed"),
            issue(SOURCE, pull_request={}),
        ):
            with self.subTest(source=source):
                result = self.evaluate(self.plan(), source=source)
                self.assertFalse(result.mutates)
        for labels in (["oncall: distributed"], ["oncall: cpu inductor"]):
            with self.subTest(labels=labels):
                result = self.evaluate(self.plan(labels=labels))
                self.assertFalse(result.close)
                self.assertEqual(result.comment, "")
                self.assertNotIn("triaged", result.add_labels)

    def test_schema_match_choices_and_runtime_validation_align(self):
        schema = json.loads(
            (
                apply_plan.TRIAGE_SKILL.parents[2]
                / ".github/pi/schemas/issue-triage-plan.json"
            ).read_text()
        )["properties"]["umbrella"]
        self.assertEqual(set(schema["required"]), {"issue", "match"})
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["properties"]["issue"]["type"], "integer")
        self.assertEqual(schema["properties"]["issue"]["minimum"], 1)
        self.assertEqual(
            set(schema["properties"]["match"]["enum"]), {"covered", "uncertain"}
        )
        for match in schema["properties"]["match"]["enum"]:
            self.assertIsNotNone(
                apply_plan.umbrella_policy(REPO, {"issue": TARGET, "match": match})
            )
        for annotation in (
            None,
            {},
            {"issue": TARGET},
            {"issue": TARGET, "match": "related"},
            {"issue": True, "match": "covered"},
            {"issue": float(TARGET), "match": "covered"},
            {"issue": 0, "match": "covered"},
            {"issue": -1, "match": "covered"},
            {"issue": f"https://github.com/{REPO}/issues/{TARGET}", "match": "covered"},
            {"issue": TARGET, "match": "covered", "action": "close"},
            {"issue": TARGET, "match": "covered", "comment": "model-written prose"},
        ):
            with self.subTest(annotation=annotation):
                self.assertIsNone(apply_plan.umbrella_policy(REPO, annotation))
                plan = self.plan()
                plan["umbrella"] = annotation
                result = self.evaluate(plan, target=issue(TARGET))
                self.assertFalse(result.close)
                self.assertEqual(result.comment, "")
                self.assertIn("triage review", result.add_labels)


class UmbrellaCommentHistoryTest(unittest.TestCase):
    def test_retry_history_checks_all_pages_and_only_bot_comments(self):
        marker = f"<!-- issue-triage-pi:umbrella:{REPO}#{TARGET} -->"
        pages = [
            [{"body": "earlier", "user": {"login": apply_plan.TRIAGE_BOT}}],
            [
                {"body": marker, "user": {"login": apply_plan.TRIAGE_BOT}},
                {"body": "human " + marker, "user": {"login": "reporter"}},
            ],
        ]
        with patch(
            "apply_plan.gh_api",
            side_effect=[json.dumps(issue(SOURCE)), json.dumps(pages)],
        ) as api:
            _, comments = apply_plan.fetch_issue(REPO, SOURCE, all_comments=True)
        self.assertEqual(comments, ["earlier", marker])
        self.assertEqual(
            api.call_args.args[0],
            [
                "--paginate",
                "--slurp",
                f"repos/{REPO}/issues/{SOURCE}/comments?per_page=100",
            ],
        )


if __name__ == "__main__":
    unittest.main()
