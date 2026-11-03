"""Check that the cron does not finalize issues still needing review."""

import json
import os
import shlex
import subprocess
import unittest
from pathlib import Path

import yaml
from apply_distributed_plan import plan_effects
from labels import DISTRIBUTED_SKILL, load_labels, load_templates


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.safe_load(
    (ROOT / ".github/workflows/claude-distributed-triage-cron.yml").read_text()
)
PHASE1 = next(
    step["run"]
    for step in WORKFLOW["jobs"]["sweep"]["steps"]
    if step.get("name", "").startswith("Phase 1:")
)


class CronCompletionTest(unittest.TestCase):
    def test_pending_review_or_reproduction_is_not_auto_finalized(self):
        valid = load_labels(DISTRIBUTED_SKILL / "distributed-labels.json")
        existing = {"oncall: distributed", "bot-triaged"}
        effects = plan_effects(
            {
                "decision": "high_priority",
                "labels": ["oncall: distributed parallelisms", "module: fsdp"],
            },
            existing,
            valid,
            load_templates(DISTRIBUTED_SKILL),
            [],
        )
        classified = existing | {"oncall: distributed parallelisms", "module: fsdp"}
        cases = [
            existing | set(effects.add_labels),
            classified | {"needs reproduction"},
            classified,
            existing | {"oncall: distributed parallelisms"},
        ]
        issues = [
            {"number": i, "labels": [{"name": label} for label in sorted(labels)]}
            for i, labels in enumerate(cases, 1)
        ]
        modules = sorted(
            label
            for label in valid
            if label.startswith("module: ") and label != "module: performance"
        )
        queues = sorted(label for label in valid if label.startswith("oncall: "))
        # Run the actual classification loop; only its API fetch and mapfile setup
        # are replaced with fixtures, so the test also works with macOS Bash 3.
        setup = (
            f"MODULE_LABELS=({shlex.join(modules)})\n"
            f"SUB_ONCALL_LABELS=({shlex.join(queues)})\n"
        )
        script = PHASE1[PHASE1.index("PHASE1_COUNT=0") :].replace(
            "${{ github.repository }}", "pytorch/ciforge"
        )
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", setup + script],
            env={
                **os.environ,
                "DRY_RUN": "true",
                "ISSUES": "\n".join(json.dumps(issue) for issue in issues),
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        marked = [line for line in result.stdout.splitlines() if "Would add" in line]
        self.assertEqual(len(marked), 1, result.stdout)
        self.assertIn("to #3 ", marked[0])


if __name__ == "__main__":
    unittest.main()
