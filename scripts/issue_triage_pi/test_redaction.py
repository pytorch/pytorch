"""The apply step edits download links, never model-supplied issue prose."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import apply_plan


NOTICE = (
    "[Link removed - external file downloads are not permitted for security reasons]"
)


class RedactionTest(unittest.TestCase):
    def test_only_download_and_storage_links_are_removed(self):
        downloads = [
            "https://example.org/repro.zip?token=abc#download",  # @lint-ignore
            "https://example.org/weights.PTH",  # @lint-ignore
            "https://example.org/model%2Ept",  # @lint-ignore
            "https://huggingface.co/org/model/resolve/main/model.safetensors",  # @lint-ignore
            "https://drive.google.com/file/d/123/view",  # @lint-ignore
            "https://www.dropbox.com/s/123/repro",  # @lint-ignore
            "https://1drv.ms/u/s!123",  # @lint-ignore
            "https://mega.nz/file/123#key",  # @lint-ignore
            "https://we.tl/t-123",  # @lint-ignore
        ]
        for url in downloads:
            with self.subTest(url=url):
                body = f"Before [repro]({url})\nAfter"
                self.assertEqual(
                    apply_plan.redact_external_links(body), f"Before {NOTICE}\nAfter"
                )
        kept = "\n".join(
            [
                "https://pytorch.org/docs/stable/index.html",  # @lint-ignore
                "https://github.com/pytorch/pytorch/issues/123",  # @lint-ignore
                "https://huggingface.co/org/model",  # @lint-ignore
                "https://drive.google.com.example.org/readme",  # @lint-ignore
                "https://example.org/docs/model.pt.html",  # @lint-ignore
                "path/to/local.pt",
            ]
        )
        self.assertEqual(apply_plan.redact_external_links(kept), kept)

    def test_markdown_autolinks_punctuation_and_retries(self):
        body = '![weights](https://example.org/weights.pt "title")\n<https://example.org/repro.zip>\nSee https://example.org/repro.pkl.'  # @lint-ignore
        expected = f"{NOTICE}\n{NOTICE}\nSee {NOTICE}."
        self.assertEqual(apply_plan.redact_external_links(body), expected)
        self.assertEqual(apply_plan.redact_external_links(expected), expected)

    def test_adjacent_links_and_quoted_downloads_preserve_surrounding_text(self):
        self.assertEqual(
            apply_plan.redact_external_links(
                "Inputs: [batch=32, [repro](https://example.org/repro.zip)]"  # @lint-ignore
            ),
            f"Inputs: [batch=32, {NOTICE}]",
        )
        docs = "[docs](https://pytorch.org/docs/stable/index.html)"  # @lint-ignore
        for url in (
            "https://drive.google.com/file/d/123/view",  # @lint-ignore
            "https://example.org/repro.zip",  # @lint-ignore
            "https://example.org/repro(1).zip",  # @lint-ignore
        ):
            with self.subTest(url=url):
                self.assertEqual(
                    apply_plan.redact_external_links(f"[repro]({url}),{docs}"),
                    f"{NOTICE},{docs}",
                )
                self.assertEqual(
                    apply_plan.redact_external_links(f"wget '{url}'"),
                    f"wget '{NOTICE}'",
                )

    def test_apply_redacts_current_body_not_planning_snapshot(self):
        original = "Original https://example.org/weights.pt"  # @lint-ignore
        current = original + "\nNew human explanation."
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp) / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "decision": "request_reproduction",
                        "labels": [],
                        "templates": ["request_self_contained_reproduction"],
                    }
                )
            )
            with (
                patch(
                    "sys.argv",
                    ["apply_plan", "pytorch/ciforge", "123", str(plan), "--apply"],
                ),
                patch(
                    "apply_plan.fetch_issue",
                    return_value=(
                        {"body": original, "labels": [], "state": "open"},
                        [],
                    ),
                ),
                patch(
                    "apply_plan.gh_api", return_value=json.dumps({"body": current})
                ) as api,
                patch("apply_plan.publish"),
            ):
                self.assertEqual(apply_plan.main(), 0)
        writes = [c for c in api.call_args_list if c.args[0][:2] == ["-X", "PATCH"]]
        self.assertEqual(len(writes), 1)
        self.assertEqual(
            writes[0].args[0],
            [
                "-X",
                "PATCH",
                "repos/pytorch/ciforge/issues/123",
                "-f",
                f"body=Original {NOTICE}\nNew human explanation.",
            ],
        )
        self.assertEqual(writes[0].kwargs, {"retry": False})
        comments = [
            c.args[0]
            for c in api.call_args_list
            if c.args[0][2:3] == ["repos/pytorch/ciforge/issues/123/comments"]
        ]
        self.assertEqual(len(comments), 1)

    def test_dry_run_does_not_write_or_fetch_for_redaction(self):
        with tempfile.TemporaryDirectory() as temp:
            plan = Path(temp) / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "decision": "request_reproduction",
                        "labels": [],
                        "templates": ["request_self_contained_reproduction"],
                    }
                )
            )
            with (
                patch("sys.argv", ["apply_plan", "pytorch/ciforge", "123", str(plan)]),
                patch(
                    "apply_plan.fetch_issue",
                    return_value=(
                        {
                            "body": "https://example.org/model.pt",  # @lint-ignore
                            "labels": [],
                            "state": "open",
                        },
                        [],
                    ),
                ),
                patch("apply_plan.gh_api") as api,
                patch("apply_plan.publish") as publish,
            ):
                self.assertEqual(apply_plan.main(), 0)
                api.assert_not_called()
                self.assertIn(
                    "redact external download links", publish.call_args.args[0]
                )

    def test_reproduction_request_enforces_comment_and_leaves_issue_open(self):
        effects = apply_plan.plan_effects(
            {"decision": "request_reproduction", "labels": ["triaged"]},
            set(),
            apply_plan.load_valid_labels(),
            apply_plan.load_templates(apply_plan.TRIAGE_SKILL),
            [],
        )
        self.assertTrue(effects.redact_body)
        self.assertNotIn("triaged", effects.add_labels)
        self.assertIn("without the external files", effects.comment)
        self.assertFalse(effects.close)

    def test_failed_body_patch_is_not_retried_with_stale_text(self):
        with (
            patch(
                "apply_plan.subprocess.run",
                return_value=subprocess.CompletedProcess([], 1, "", "failed"),
            ) as run,
            patch("apply_plan.time.sleep") as sleep,
            patch("builtins.print"),
        ):
            with self.assertRaises(RuntimeError):
                apply_plan.gh_api(
                    ["-X", "PATCH", "repos/pytorch/ciforge/issues/123"], retry=False
                )
            self.assertEqual(run.call_count, 1)
            sleep.assert_not_called()

    def test_no_body_write_when_no_matching_links_remain(self):
        with patch(
            "apply_plan.gh_api", return_value=json.dumps({"body": NOTICE})
        ) as api:
            apply_plan.apply_effects(
                "pytorch/ciforge", 123, apply_plan.Effects(redact_body=True)
            )
        self.assertEqual(len(api.call_args_list), 1)
        self.assertEqual(api.call_args.args[0], ["repos/pytorch/ciforge/issues/123"])


if __name__ == "__main__":
    unittest.main()
