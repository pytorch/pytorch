"""Runs check_triage_outcome.py against trimmed real transcripts.

testdata/transcript_195701.json is the run where every get_issue call was
rejected and the agent asked for a Bash approval; transcript_196067.json is
the run where the body edit and comment succeeded after retries. Both are
the archived execution output with everything the checker ignores removed.
"""

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import main, mock, TestCase

import check_triage_outcome


TESTDATA = Path(__file__).parent / "testdata"


def fake_gh_api(labels, state="open"):
    calls = []

    def gh_api(args, **kwargs):
        calls.append(args)
        if args[0] == "-X":
            return "[]"
        return json.dumps({"state": state, "labels": [{"name": n} for n in labels]})

    return gh_api, calls


class TestCheckTriageOutcome(TestCase):
    def run_main(self, issue, transcript, labels, state="open"):
        gh_api, calls = fake_gh_api(labels, state)
        argv = ["check", "pytorch/pytorch", issue, str(TESTDATA / transcript)]
        out = io.StringIO()
        with (
            mock.patch.object(check_triage_outcome, "gh_api", gh_api),
            mock.patch.object(sys, "argv", argv),
            mock.patch.dict("os.environ", {"GITHUB_OUTPUT": ""}),
            redirect_stdout(out),
        ):
            rc = check_triage_outcome.main()
        return rc, out.getvalue(), calls

    def test_no_action_fails_with_diagnostics(self):
        rc, out, calls = self.run_main("195701", "transcript_195701.json", [])
        self.assertEqual(rc, 1)
        self.assertIn("Outcome: no_action", out)
        self.assertIn("::error::Triage run took no action on #195701", out)
        self.assertIn("parameter issue_number is not of type float64", out)
        self.assertIn("Permission denials: ['Bash', 'Bash', 'Bash']", out)
        self.assertIn("needs your approval", out)
        self.assertEqual(len(calls), 1)

    def test_mutation_without_marker_reapplies_bot_triaged(self):
        rc, out, calls = self.run_main("196067", "transcript_196067.json", [])
        self.assertEqual(rc, 0)
        self.assertIn("Outcome: waiting_on_reporter", out)
        self.assertEqual(calls[1][:2], ["-X", "POST"])
        self.assertIn("labels[]=bot-triaged", calls[1])

    def test_mutation_with_marker_leaves_issue_alone(self):
        rc, out, calls = self.run_main(
            "196067", "transcript_196067.json", ["bot-triaged"]
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(calls), 1)

    def test_missing_transcript_uses_live_labels(self):
        rc, out, _ = self.run_main("1", "missing.json", ["oncall: distributed"])
        self.assertEqual(rc, 0)
        self.assertIn("Outcome: routed", out)
        rc, out, _ = self.run_main("1", "missing.json", [])
        self.assertEqual(rc, 1)
        self.assertIn("No execution transcript was produced", out)

    def test_classify_label_only_outcomes(self):
        classify = check_triage_outcome.classify
        self.assertEqual(classify({"triaged", "module: mps"}, "open", False), "triaged")
        self.assertEqual(classify({"triage review"}, "open", False), "review")
        self.assertEqual(classify(set(), "closed", False), "closed")
        self.assertEqual(classify({"oncall: pt2", "triaged"}, "open", True), "routed")


if __name__ == "__main__":
    main()
