from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

from schemas import IntakeResult, LLMInput, LLMResult, OwnershipResult
from tests.stage_fixtures import (
    BYPASS_CRITERIA,
    changed_file,
    concern,
    intake_result,
    llm_input,
    llm_result,
    make_ownership_result,
    PATCH,
)
from validate_ownership import (
    build_ownership_result,
    load_action_execution,
    log_analysis_record,
    main as ownership_result_main,
    validate_result,
    write_json,
)


def codepath_owner_concern(
    *, owners: list[str], file: str = "torch/file.py"
) -> dict[str, object]:
    return {
        "concern": concern(
            file=file, description="Changes behavior the codepath owners review."
        ),
        "codepath_owners": owners,
        "reason": "The codepath owners already review this API surface.",
    }


def validate(*, prepared: LLMInput, result: dict[str, Any]) -> list[str]:
    return validate_result(llm_input=prepared, result=LLMResult.from_dict(result))


def normalize(
    *, prepared: LLMInput, result: dict[str, Any] | None = None, **kwargs: Any
) -> OwnershipResult:
    parsed = None if result is None else LLMResult.from_dict(result)
    return build_ownership_result(llm_input=prepared, llm_result=parsed, **kwargs)


def ownership_result_argv(*, root: Path, execution: Path) -> list[str]:
    return [
        "validate_ownership.py",
        "--output-dir",
        str(root),
        "--execution-file",
        str(execution),
        "--github-step-summary",
        str(root / "github-step-summary"),
    ]


class OwnershipResultMainTest(unittest.TestCase):
    def run_processor(
        self,
        *,
        prepared: LLMInput,
        result: dict[str, object],
        intake: IntakeResult | None = None,
        **execution_metadata: object,
    ) -> tuple[dict[str, object], OwnershipResult, str]:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(
                path=root / "intake.json", value=(intake or intake_result()).to_dict()
            )
            write_json(path=root / "llm_input.json", value=prepared.to_dict())
            execution = root / "execution.json"
            execution.write_text(
                json.dumps(
                    [
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "structured_output": result,
                            **execution_metadata,
                        }
                    ]
                )
            )
            argv = ownership_result_argv(root=root, execution=execution)
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                self.assertEqual(ownership_result_main(), 0)
            result_record = json.loads((root / "result.json").read_text())
            ownership = OwnershipResult.from_json((root / "ownership.json").read_text())
            step_summary = (root / "github-step-summary").read_text()
        return result_record, ownership, step_summary

    def test_completed_result_is_normalized_for_planning(self) -> None:
        prepared = llm_input(
            team_rosters={"owner": ["@owner"], "extra": ["@extra"]},
        )
        result = llm_result(additional_owners=["extra"])
        record, ownership, step_summary = self.run_processor(
            prepared=prepared,
            result=result,
            num_turns=1,
            total_cost_usd=0.01,
        )

        self.assertEqual(
            ownership,
            make_ownership_result(
                codepath_owners=["@pytorch/baseline"],
                additional_owners=["extra"],
            ),
        )
        self.assertEqual(record["ownership_result"], ownership.to_dict())
        self.assertEqual(record["llm_result"], result)
        self.assertEqual(record["validation_errors"], [])
        self.assertEqual(record["llm_metadata"]["num_turns"], 1)
        self.assertIn("| Open non-draft PR against main | true |", step_summary)
        self.assertIn("| Already handled | false |", step_summary)
        self.assertIn("| Author has triage permission | false |", step_summary)
        self.assertIn("| Actionable issue linked | true |", step_summary)
        self.assertIn("| Maintainer activity | false |", step_summary)
        self.assertIn("| LLM run status | succeeded |", step_summary)
        self.assertIn("| Has uncovered concerns | false |", step_summary)
        self.assertIn("| Codepath owners | @pytorch/baseline |", step_summary)
        self.assertIn("| Additional owners | extra |", step_summary)
        self.assertIn("| Bypass intake owners | none |", step_summary)
        self.assertIn(
            "- Additional owner `extra` (accepted, high confidence): "
            "extra owns a distinct changed contract. Evidence: torch/file.py",
            step_summary,
        )

    def test_uncovered_concern_is_published_as_completed(self) -> None:
        prepared = llm_input(
            team_rosters={"owner": ["@owner"], "extra": ["@extra"]},
        )
        result = llm_result(
            additional_owners=["extra"],
            uncovered_concerns=[
                {
                    "description": "A material contract has no configured owner.",
                    "reason": "The available metadata does not describe this contract.",
                    "files": ["torch/file.py"],
                }
            ],
        )

        record, normalized, step_summary = self.run_processor(
            prepared=prepared, result=result
        )

        self.assertEqual(normalized.additional_owners, ("extra",))
        self.assertEqual(
            normalized.uncovered_concerns,
            LLMResult.from_dict(result).uncovered_concerns,
        )
        self.assertTrue(normalized.has_uncovered_concerns)
        self.assertEqual(record["llm_result"], result)
        self.assertIn("| LLM run status | succeeded |", step_summary)
        self.assertIn("| Has uncovered concerns | true |", step_summary)
        self.assertIn(
            "- Uncovered concern: A material contract has no configured owner. "
            "Why uncovered: The available metadata does not describe this contract.",
            step_summary,
        )
        self.assertIn("material contract", normalized.to_json())

    def test_step_summary_lists_codepath_owner_concerns(self) -> None:
        result = llm_result(
            codepath_owner_concerns=[
                codepath_owner_concern(owners=["@pytorch/baseline"])
            ]
        )

        _, normalized, step_summary = self.run_processor(
            prepared=llm_input(), result=result
        )

        self.assertEqual(normalized.llm_run_status, "succeeded")
        self.assertIn(
            "- Codepath owner concern (`@pytorch/baseline`): Changes behavior the "
            "codepath owners review. Why covered: The codepath owners already review "
            "this API surface. Evidence: torch/file.py\n",
            step_summary,
        )

    def test_step_summary_renders_llm_text_inertly(self) -> None:
        result = llm_result(
            uncovered_concerns=[
                {
                    "description": "![x](x.png) <img src=x>\n## Heading",
                    "reason": "`code` | cell",
                    "files": ["torch/file.py"],
                }
            ],
        )

        _, _, step_summary = self.run_processor(prepared=llm_input(), result=result)

        self.assertIn(
            "- Uncovered concern: !\\[x\\](x.png) \\<img src=x\\> "
            "\\#\\# Heading Why uncovered: \\`code\\` \\| cell "
            "Evidence: torch/file.py\n",
            step_summary,
        )

    def test_step_summary_omits_llm_concerns_when_incomplete(self) -> None:
        result = llm_result(additional_owners=["unknown"])

        _, normalized, step_summary = self.run_processor(
            prepared=llm_input(), result=result
        )

        self.assertEqual(normalized.llm_run_status, "failed")
        self.assertNotIn("LLM concerns", step_summary)

    def test_inactive_or_handled_pr_gets_skipped(self) -> None:
        cases = {
            "outside_active_target": intake_result(
                is_open_non_draft_pr_against_main=False,
                has_actionable_linked_issue=False,
            ),
            "already_handled": intake_result(is_already_handled=True),
        }
        for name, intake in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_json(path=root / "intake.json", value=intake.to_dict())
                argv = ownership_result_argv(
                    root=root, execution=root / "missing-execution.json"
                )
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch("builtins.print"),
                ):
                    self.assertEqual(ownership_result_main(), 0)

                record = json.loads((root / "result.json").read_text())
                ownership = OwnershipResult.from_json(
                    (root / "ownership.json").read_text()
                )

            self.assertEqual(ownership, make_ownership_result(llm_run_status="skipped"))
            self.assertEqual(record["ownership_result"], ownership.to_dict())
            self.assertEqual(record["llm_metadata"]["reason"], "inactive_or_handled")
            self.assertIsNone(record["llm_result"])

    def test_bypass_for_a_pr_that_fails_intake_reaches_the_result(self) -> None:
        prepared = llm_input(bypass={"owner": "PRs that fix incorrect gradients."})
        result = llm_result(additional_owners=["owner"], bypass=["owner"])

        _, ownership, step_summary = self.run_processor(
            prepared=prepared,
            result=result,
            intake=intake_result(has_actionable_linked_issue=False),
        )

        self.assertEqual(ownership.bypass_intake_matches, ("owner",))
        self.assertIn("| Bypass intake owners | owner |", step_summary)
        self.assertIn("(accepted, high confidence, bypass intake)", step_summary)
        self.assertIn(
            '  - Matches bypass intake "PRs that fix incorrect gradients.": '
            "The change fixes a gradient formula the team asked to see.\n",
            step_summary,
        )

    def test_bypass_claim_without_criteria_makes_the_result_incomplete(self) -> None:
        result = llm_result(additional_owners=["owner"], bypass=["owner"])

        record, ownership, _ = self.run_processor(prepared=llm_input(), result=result)

        self.assertEqual(ownership.llm_run_status, "failed")
        self.assertEqual(ownership.bypass_intake_matches, ())
        self.assertEqual(
            record["validation_errors"],
            ["bypass claimed without criteria: ['owner']"],
        )

    def test_maintainer_activity_runs_analysis_and_preserves_owners(self) -> None:
        intake = intake_result(
            has_actionable_linked_issue=False, has_maintainer_activity=True
        )
        result = llm_result(additional_owners=["owner"])

        record, ownership, step_summary = self.run_processor(
            prepared=llm_input(), result=result, intake=intake
        )

        self.assertEqual(
            ownership,
            make_ownership_result(
                codepath_owners=["@pytorch/baseline"], additional_owners=["owner"]
            ),
        )
        self.assertEqual(record["llm_result"], result)
        self.assertIn("| Maintainer activity | true |", step_summary)
        self.assertIn("| LLM run status | succeeded |", step_summary)

    def test_result_carries_accepted_owner_concerns(self) -> None:
        answer = llm_result(additional_owners=["owner"])
        record, ownership, _ = self.run_processor(prepared=llm_input(), result=answer)

        self.assertEqual(
            ownership.codepath_owners, {"@pytorch/baseline": ("torch/file.py",)}
        )
        self.assertEqual(
            ownership.additional_owner_concerns,
            LLMResult.from_dict(answer).additional_owner_concerns,
        )
        self.assertEqual(record["ownership_result"], ownership.to_dict())

    def test_validation_failure_preserves_codepath_and_discards_additional(
        self,
    ) -> None:
        invalid = llm_result(additional_owners=["unknown"])

        record, ownership, step_summary = self.run_processor(
            prepared=llm_input(), result=invalid
        )

        self.assertEqual(
            ownership,
            make_ownership_result(
                llm_run_status="failed",
                codepath_owners=["@pytorch/baseline"],
            ),
        )
        self.assertEqual(record["llm_result"], invalid)
        self.assertEqual(
            record["validation_errors"],
            ["extra ownership metadata contains unknown owners: ['unknown']"],
        )
        self.assertIn("| LLM run status | failed |", step_summary)

    def test_low_confidence_and_truncated_results_filter_additional_owners(
        self,
    ) -> None:
        cases = [
            (
                "low-confidence",
                llm_input(),
                llm_result(additional_owners=["owner"], confidence="low"),
            ),
            (
                "truncated",
                llm_input(files=(changed_file(truncated=True),)),
                llm_result(additional_owners=["owner"]),
            ),
        ]
        for name, prepared, result in cases:
            with self.subTest(name=name):
                record, ownership, _ = self.run_processor(
                    prepared=prepared, result=result
                )

            self.assertEqual(ownership.additional_owner_concerns, ())
            self.assertEqual(
                ownership.discarded_additional_owner_concerns,
                LLMResult.from_dict(result).additional_owner_concerns,
            )
            self.assertEqual(record["llm_result"], result)
            self.assertEqual(record["validation_errors"], [])

    def test_execution_failure_preserves_codepath_owners(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(path=root / "intake.json", value=intake_result().to_dict())
            write_json(path=root / "llm_input.json", value=llm_input().to_dict())
            execution = root / "execution.json"
            execution.write_text("{not-json")
            argv = ownership_result_argv(root=root, execution=execution)
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                self.assertEqual(ownership_result_main(), 0)
            record = json.loads((root / "result.json").read_text())
            error = json.loads((root / "error.json").read_text())
            ownership = OwnershipResult.from_json((root / "ownership.json").read_text())

        self.assertEqual(
            ownership,
            make_ownership_result(
                llm_run_status="failed", codepath_owners=["@pytorch/baseline"]
            ),
        )
        self.assertEqual(record["ownership_result"], ownership.to_dict())
        self.assertIsNone(record["llm_result"])
        self.assertEqual(record["llm_metadata"]["status"], "failed")
        self.assertEqual(error["stage"], "ownership")

    def test_invalid_stage_input_emits_no_result(self) -> None:
        serialized = llm_input().to_dict()
        serialized["trusted_context"]["reviewer"] = "attacker"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(path=root / "intake.json", value=intake_result().to_dict())
            write_json(path=root / "llm_input.json", value=serialized)
            argv = ownership_result_argv(
                root=root, execution=root / "missing-execution.json"
            )
            with mock.patch.object(sys, "argv", argv), mock.patch("builtins.print"):
                self.assertEqual(ownership_result_main(), 1)
            error = json.loads((root / "error.json").read_text())

            self.assertEqual(error["stage"], "ownership")
            self.assertFalse((root / "ownership.json").exists())
            self.assertFalse((root / "result.json").exists())
            self.assertFalse((root / "github-step-summary").exists())


class ActionExecutionTest(unittest.TestCase):
    def test_action_execution_loads_final_structured_result(self) -> None:
        execution = [
            {"type": "assistant", "message": "intermediate"},
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "structured_output": llm_result(),
                "num_turns": 1,
                "total_cost_usd": 0.01,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.json"
            path.write_text(json.dumps(execution))
            result, metadata = load_action_execution(path)

        self.assertEqual(result, LLMResult.from_dict(llm_result()))
        self.assertEqual(result.additional_owner_concerns, ())
        self.assertEqual(metadata["num_turns"], 1)
        self.assertEqual(metadata["total_cost_usd"], 0.01)

    def test_action_execution_rejects_failed_result(self) -> None:
        execution = [
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": True,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.json"
            path.write_text(json.dumps(execution))
            with self.assertRaisesRegex(RuntimeError, "successful result"):
                load_action_execution(path)

    def test_log_output_exposes_reasoning_without_workflow_commands(self) -> None:
        record = {
            "ownership_result": make_ownership_result(
                additional_owners=["owner"]
            ).to_dict(),
            "processing_error": None,
            "llm_result": {
                "additional_owners": [
                    {
                        "owner_id": "owner",
                        "owned_concern": "A distinct changed contract.",
                        "rationale": [
                            "line one\n::error::not a command ##[add-mask]secret"
                        ],
                        "files": ["torch/file.py"],
                        "evidence": [
                            {
                                "file": "torch/file.py",
                                "diff_excerpt": "+::notice::not a command",
                                "relevance": "##[warning] This is still untrusted text.",
                            }
                        ],
                    }
                ],
                "uncovered_concerns": [
                    {
                        "description": "An uncovered serialization contract.",
                        "reason": "No configured owner describes serialization.",
                        "files": ["torch/file.py"],
                    }
                ],
                "confidence": "high",
            },
            "validation_errors": [],
        }

        with mock.patch("builtins.print") as output:
            log_analysis_record(record)

        lines = [call.args[0] for call in output.call_args_list]
        self.assertTrue(lines)
        self.assertTrue(
            all(line.startswith("Auto PR Triage result | ") for line in lines)
        )
        rendered = "\n".join(lines)
        self.assertIn('"owned_concern": "A distinct changed contract."', rendered)
        self.assertIn('"description": "An uncovered serialization contract."', rendered)
        self.assertIn(r"line one\n\u003a\u003aerror\u003a\u003anot a command", rendered)
        self.assertNotIn("::", rendered)
        self.assertNotIn("##[", rendered)
        self.assertIn(r"+\u003a\u003anotice\u003a\u003anot a command", rendered)
        self.assertIn(r"\u0023\u0023[warning] This is still untrusted text.", rendered)
        self.assertIn(r"\u0023\u0023[add-mask]secret", rendered)


class ValidationTest(unittest.TestCase):
    def test_valid_empty_additive_result(self) -> None:
        self.assertEqual(validate(prepared=llm_input(), result=llm_result()), [])

    def test_codepath_owners_are_immutable(self) -> None:
        prepared = llm_input(
            codepath_owners=["@baseline-user", "@pytorch/baseline"],
        )

        result = normalize(prepared=prepared, result=llm_result())

        self.assertEqual(
            result,
            make_ownership_result(
                codepath_owners=["@baseline-user", "@pytorch/baseline"],
            ),
        )

    def test_additional_owner_is_added_without_replacing_codepath_owners(self) -> None:
        prepared = llm_input(
            team_rosters={"owner": ["@owner"], "extra": ["@extra"]},
        )
        result = llm_result(additional_owners=["extra"])

        self.assertEqual(validate(prepared=prepared, result=result), [])
        normalized = normalize(prepared=prepared, result=result)

        self.assertEqual(
            normalized,
            make_ownership_result(
                codepath_owners=["@pytorch/baseline"],
                additional_owners=["extra"],
            ),
        )

    def test_duplicate_and_unknown_additional_owners_fail(self) -> None:
        duplicate = llm_result(additional_owners=["owner", "owner"])
        unknown = llm_result(additional_owners=["unknown"])
        apply_time_unavailable_input = llm_input(
            team_rosters={"owner": ["@owner"], "author_only": ["@author"]}
        )
        apply_time_unavailable = llm_result(additional_owners=["author_only"])

        self.assertTrue(
            any(
                "duplicate additional owners" in error
                for error in validate(prepared=llm_input(), result=duplicate)
            )
        )
        self.assertTrue(
            any(
                "unknown owners" in error
                for error in validate(prepared=llm_input(), result=unknown)
            )
        )
        self.assertEqual(
            validate(
                prepared=apply_time_unavailable_input, result=apply_time_unavailable
            ),
            [],
        )

    def test_suggestion_files_must_be_changed_paths(self) -> None:
        result = llm_result(additional_owners=["owner"])
        result["additional_owner_concerns"][0]["concern"]["files"] = ["not/changed.py"]

        errors = validate(prepared=llm_input(), result=result)

        self.assertIn("reported paths absent from PR: ['not/changed.py']", errors)
        self.assertIn(
            "evidence file for owner is not supporting: torch/file.py",
            errors,
        )

    def test_semantic_evidence_must_quote_a_changed_line_from_its_patch(self) -> None:
        cases = [
            (
                "unknown path",
                {"file": "not/changed.py"},
                "evidence path for owner is absent from PR: not/changed.py",
            ),
            (
                "unlisted file",
                {"file": "torch/file.py"},
                "evidence file for owner is not supporting: torch/file.py",
            ),
            (
                "non-verbatim excerpt",
                {"diff_excerpt": "+invented behavior"},
                "evidence excerpt for owner is not in patch: torch/file.py",
            ),
            (
                "context only",
                {"diff_excerpt": " unchanged context"},
                "evidence excerpt for owner has no changed line: torch/file.py",
            ),
        ]
        for name, evidence_update, expected in cases:
            result = llm_result(additional_owners=["owner"])
            evidence = result["additional_owner_concerns"][0]["concern"]["evidence"][0]
            evidence.update(evidence_update)
            if name == "unlisted file":
                result["additional_owner_concerns"][0]["concern"]["files"] = [
                    "other.py"
                ]
            with self.subTest(name=name):
                self.assertIn(expected, validate(prepared=llm_input(), result=result))

        prepared = llm_input(diff_truncated_or_unavailable=True)
        prepared = replace(
            prepared,
            untrusted_context=replace(
                prepared.untrusted_context, files=(changed_file(patch=None),)
            ),
        )
        self.assertIn(
            "evidence patch for owner is unavailable: torch/file.py",
            validate(prepared=prepared, result=llm_result(additional_owners=["owner"])),
        )

        prepared = llm_input()
        patch = f"{PATCH}\n+++counter"
        prepared = replace(
            prepared,
            untrusted_context=replace(
                prepared.untrusted_context, files=(changed_file(patch=patch),)
            ),
        )
        result = llm_result(additional_owners=["owner"])
        evidence = result["additional_owner_concerns"][0]["concern"]["evidence"][0]
        evidence["diff_excerpt"] = "+++counter"
        self.assertEqual(validate(prepared=prepared, result=result), [])

    def test_codepath_owner_concern_must_name_codepath_owners(self) -> None:
        result = llm_result(
            codepath_owner_concerns=[codepath_owner_concern(owners=["@someone-else"])]
        )

        self.assertEqual(
            validate(prepared=llm_input(), result=result),
            ["concern names owners that are not codepath owners: ['@someone-else']"],
        )

    def test_codepath_owner_concern_paths_and_evidence_are_verified(self) -> None:
        absent = llm_result(
            codepath_owner_concerns=[
                codepath_owner_concern(
                    owners=["@pytorch/baseline"], file="not/changed.py"
                )
            ]
        )
        invented = llm_result(
            codepath_owner_concerns=[
                codepath_owner_concern(owners=["@pytorch/baseline"])
            ]
        )
        evidence = invented["codepath_owner_concerns"][0]["concern"]["evidence"][0]
        evidence["diff_excerpt"] = "+invented line"

        self.assertEqual(
            validate(prepared=llm_input(), result=absent),
            [
                "codepath owner concern paths absent from PR: ['not/changed.py']",
                "evidence path for codepath owner concern is absent from PR: not/changed.py",
            ],
        )
        self.assertEqual(
            validate(prepared=llm_input(), result=invented),
            [
                "evidence excerpt for codepath owner concern is not in patch: torch/file.py"
            ],
        )

    def test_codepath_owner_concerns_are_recorded_without_changing_owners(
        self,
    ) -> None:
        prepared = llm_input()
        plain = llm_result()
        explained = llm_result(
            codepath_owner_concerns=[
                codepath_owner_concern(owners=["@pytorch/baseline"])
            ]
        )

        self.assertEqual(validate(prepared=prepared, result=explained), [])
        result = normalize(prepared=prepared, result=explained)
        self.assertEqual(
            result.codepath_owner_concerns,
            LLMResult.from_dict(explained).codepath_owner_concerns,
        )
        self.assertEqual(
            replace(result, codepath_owner_concerns=()),
            normalize(prepared=prepared, result=plain),
        )

    def test_uncovered_concern_files_must_be_changed_paths(self) -> None:
        result = llm_result(
            uncovered_concerns=[
                {
                    "description": "A material contract has no configured owner.",
                    "reason": "The available metadata does not describe this contract.",
                    "files": ["not/changed.py"],
                }
            ]
        )

        self.assertEqual(
            validate(prepared=llm_input(), result=result),
            [
                "uncovered concern paths absent from PR: ['not/changed.py']",
                "evidence path for uncovered concern is absent from PR: not/changed.py",
            ],
        )

    def test_uncovered_concern_keeps_analysis_completed(self) -> None:
        prepared = llm_input(
            team_rosters={"owner": ["@owner"], "extra": ["@extra"]},
        )
        result = llm_result(
            additional_owners=["extra"],
            uncovered_concerns=[
                {
                    "description": "A material contract has no configured owner.",
                    "reason": "The available metadata does not describe this contract.",
                    "files": ["torch/file.py"],
                }
            ],
        )

        self.assertEqual(validate(prepared=prepared, result=result), [])
        normalized = normalize(prepared=prepared, result=result)
        self.assertEqual(normalized.llm_run_status, "succeeded")
        self.assertEqual(normalized.additional_owners, ("extra",))
        self.assertTrue(normalized.has_uncovered_concerns)

    def test_eligible_result_without_owners_is_still_completed(self) -> None:
        prepared = llm_input(
            codepath_owners=[],
        )

        result = normalize(prepared=prepared, result=llm_result())

        self.assertEqual(
            result,
            make_ownership_result(),
        )

    def test_low_confidence_and_truncation_keep_analysis_completed(self) -> None:
        low = normalize(
            prepared=llm_input(),
            result=llm_result(additional_owners=["owner"], confidence="low"),
        )
        truncated = normalize(
            prepared=llm_input(files=(changed_file(truncated=True),)),
            result=llm_result(additional_owners=["owner"]),
        )

        for result in (low, truncated):
            self.assertEqual(result.llm_run_status, "succeeded")
            self.assertEqual(result.additional_owner_concerns, ())
            self.assertEqual(len(result.discarded_additional_owner_concerns), 1)

    def test_each_owner_is_accepted_on_its_own_confidence(self) -> None:
        prepared = llm_input(team_rosters={"alpha": ["@a"], "zeta": ["@z"]})
        result = normalize(
            prepared=prepared,
            result=llm_result(
                additional_owners=["alpha", "zeta"], confidences={"zeta": "low"}
            ),
        )

        self.assertEqual(result.additional_owners, ("alpha",))
        self.assertEqual(tuple(result.codepath_owners), ("@pytorch/baseline",))

    def test_truncated_patch_drops_only_owners_that_rely_on_it(self) -> None:
        files = (
            changed_file(path="torch/a.py"),
            changed_file(path="torch/z.py", truncated=True),
        )
        prepared = llm_input(
            team_rosters={"alpha": ["@a"], "zeta": ["@z"]},
            files=files,
            diff_truncated_or_unavailable=True,
        )
        result = llm_result(
            additional_owners=["alpha", "zeta"],
            owner_files={"alpha": "torch/a.py", "zeta": "torch/z.py"},
        )

        self.assertEqual(validate(prepared=prepared, result=result), [])
        normalized = normalize(prepared=prepared, result=result)
        self.assertEqual(normalized.additional_owners, ("alpha",))

    def test_accepted_bypass_claim_is_recorded(self) -> None:
        prepared = llm_input(
            team_rosters={"alpha": ["@a"], "zeta": ["@z"]},
            bypass={"alpha": "PRs that fix incorrect gradients."},
        )
        result = llm_result(additional_owners=["alpha", "zeta"], bypass=["alpha"])

        self.assertEqual(validate(prepared=prepared, result=result), [])
        normalized = normalize(prepared=prepared, result=result)
        self.assertEqual(normalized.additional_owners, ("alpha", "zeta"))
        self.assertEqual(normalized.bypass_intake_matches, ("alpha",))
        self.assertFalse(normalized.has_discarded_bypass_intake_match)

    def test_discarded_bypass_claim_is_recorded_without_the_owner(self) -> None:
        prepared = llm_input(bypass={"owner": "PRs that fix incorrect gradients."})
        result = llm_result(
            additional_owners=["owner"], bypass=["owner"], confidence="low"
        )

        normalized = normalize(prepared=prepared, result=result)
        self.assertEqual(normalized.additional_owners, ())
        self.assertEqual(normalized.bypass_intake_matches, ())
        self.assertTrue(normalized.has_discarded_bypass_intake_match)

    def test_bypass_claim_without_criteria_is_a_validation_error(self) -> None:
        result = llm_result(additional_owners=["owner"], bypass=["owner"])

        self.assertEqual(
            validate(prepared=llm_input(), result=result),
            ["bypass claimed without criteria: ['owner']"],
        )

    def test_bypass_criteria_quote_must_come_from_the_criteria(self) -> None:
        prepared = llm_input(bypass={"owner": BYPASS_CRITERIA})
        verbatim = llm_result(
            additional_owners=["owner"],
            bypass=["owner"],
            bypass_quote="  fix   incorrect\ngradients. ",
        )
        invented = llm_result(
            additional_owners=["owner"],
            bypass=["owner"],
            bypass_quote="PRs that refactor anything.",
        )

        self.assertEqual(validate(prepared=prepared, result=verbatim), [])
        self.assertEqual(
            validate(prepared=prepared, result=invented),
            ["bypass criteria quote for owner is not in its criteria"],
        )

    def test_bypass_evidence_must_quote_a_supporting_changed_hunk(self) -> None:
        other = changed_file(path="torch/other.py")
        prepared = llm_input(
            bypass={"owner": BYPASS_CRITERIA}, files=(changed_file(), other)
        )
        cases = {
            "evidence excerpt for owner bypass is not in patch: torch/file.py": (
                "diff_excerpt",
                "+invented line",
            ),
            "evidence file for owner bypass is not supporting: torch/other.py": (
                "file",
                "torch/other.py",
            ),
        }
        for expected, (key, value) in cases.items():
            with self.subTest(expected=expected):
                result = llm_result(
                    additional_owners=["owner"],
                    bypass=["owner"],
                )
                owner = result["additional_owner_concerns"][0]
                owner["bypass_intake_match"]["evidence"][0][key] = value

                self.assertEqual(validate(prepared=prepared, result=result), [expected])

    def test_uncovered_concern_evidence_must_quote_a_supporting_hunk(self) -> None:
        other = changed_file(path="torch/other.py")
        prepared = llm_input(files=(changed_file(), other))
        cases = {
            "evidence excerpt for uncovered concern is not in patch: torch/file.py": (
                "diff_excerpt",
                "+invented line",
            ),
            "evidence file for uncovered concern is not supporting: torch/other.py": (
                "file",
                "torch/other.py",
            ),
        }
        for expected, (key, value) in cases.items():
            with self.subTest(expected=expected):
                result = llm_result(
                    uncovered_concerns=[
                        {
                            "description": "A material contract has no owner.",
                            "reason": "The available metadata does not cover it.",
                            "files": ["torch/file.py"],
                        }
                    ],
                )
                result["uncovered_concerns"][0]["concern"]["evidence"][0][key] = value

                self.assertEqual(validate(prepared=prepared, result=result), [expected])

    def test_uncovered_concern_without_evidence_is_rejected_at_parse(self) -> None:
        result = llm_result(
            uncovered_concerns=[
                {
                    "description": "A material contract has no owner.",
                    "reason": "The available metadata does not cover it.",
                    "files": ["torch/file.py"],
                }
            ]
        )
        del result["uncovered_concerns"][0]["concern"]["evidence"]

        with self.assertRaisesRegex(RuntimeError, "missing \\['evidence'\\]"):
            LLMResult.from_dict(result)

    def test_execution_failure_preserves_only_codepath_owners(self) -> None:
        result = normalize(
            prepared=llm_input(),
            llm_failed=True,
        )

        self.assertEqual(
            result,
            make_ownership_result(
                llm_run_status="failed",
                codepath_owners=["@pytorch/baseline"],
            ),
        )

    def test_validation_failure_preserves_codepath_but_discards_additional(
        self,
    ) -> None:
        prepared = llm_input()
        result = llm_result(additional_owners=["owner"])
        errors = ["invalid LLM result"]

        normalized = normalize(
            prepared=prepared, result=result, llm_failed=bool(errors)
        )

        self.assertEqual(
            normalized,
            make_ownership_result(
                llm_run_status="failed",
                codepath_owners=["@pytorch/baseline"],
            ),
        )

    def test_validation_failure_without_codepath_owners_has_no_owners(self) -> None:
        result = normalize(
            prepared=llm_input(codepath_owners=[]),
            result=llm_result(),
            llm_failed=True,
        )

        self.assertEqual(result, make_ownership_result(llm_run_status="failed"))

    def test_normalized_result_keeps_only_accepted_owner_justification(self) -> None:
        normalized = normalize(
            prepared=llm_input(), result=llm_result(additional_owners=["owner"])
        )

        serialized = normalized.to_json()
        self.assertNotIn("free-form LLM text", serialized)
        self.assertIn("owner owns a distinct changed contract", serialized)
        self.assertIn("rationale", serialized)
        self.assertEqual(
            normalized,
            make_ownership_result(
                codepath_owners=["@pytorch/baseline"],
                additional_owners=["owner"],
            ),
        )


if __name__ == "__main__":
    unittest.main()
