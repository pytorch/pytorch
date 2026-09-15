#!/bin/bash
# Shared validator entry point for the hardened PR review.
#
# Both hooks call this. The model CANNOT invoke it directly — Bash is refused in
# that job — so its on-demand check is the write itself: every Write to the
# findings file runs this and injects the result back. Rewriting is unlimited,
# so the loop is the same one a manual command would give, minus the shell.
#
# WHAT A PASS MEANS, EXACTLY: the file will publish without losing anything. It
# is NOT a claim that a finding is correct. A line number that is wrong but
# happens to fall inside the diff passes here and always will; only anchors that
# would be DISCARDED are catchable this way.
#
# Paths come from the environment so the workflow owns them and nothing here
# has to guess a workspace layout. Every value has a fallback matching what
# hardened-pr-review-run.yml sets, so a hand-run outside CI still works.
#
# All output goes to STDERR, which is what Claude Code surfaces back to the
# model. Exit 0 valid, 1 invalid.
set -uo pipefail

FINDINGS="${PR_REVIEW_FINDINGS_FILE:-/tmp/pr-review-findings.json}"
DIFF="${PR_REVIEW_DIFF_FILE:-/tmp/pr-diff.txt}"
SCRIPTS="${PR_REVIEW_SCRIPTS_DIR:-}"

if [[ -z "$SCRIPTS" ]]; then
  echo "CANNOT VALIDATE: PR_REVIEW_SCRIPTS_DIR is unset." >&2
  echo "This is a workflow problem, not a problem with your file." >&2
  exit 1
fi

# PYTHONPATH rather than a `cd`: validate_findings.py imports extract_verdict as
# a sibling module, and the model's cwd is the PR checkout, which must not
# become an import root — a PR carrying its own extract_verdict.py would
# otherwise be imported and get to define what "valid" means.
PYTHONPATH="$SCRIPTS" python3 "$SCRIPTS/validate_findings.py" \
  --findings-file "$FINDINGS" \
  --diff-file "$DIFF"
