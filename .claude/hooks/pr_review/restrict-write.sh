#!/bin/bash
# PreToolUse(Write) — the actual boundary on where the model may write.
#
# WHY THIS EXISTS RATHER THAN A PATH-QUALIFIED --allowedTools RULE. A rule like
# `Write(//path/to/file)` depends on how the CLI normalises and globs a path,
# and when it does not match, the tool call is simply refused with no reason
# recorded anywhere a workflow can read. Three separate runs were lost to that:
# the model attempted its write, the call was denied, and the only artefact was
# `permission_denials_count`. So the allowlist now carries a BARE `Write` and
# the restriction lives here, where it is an explicit string comparison whose
# outcome is logged either way.
#
# This is a real gate, not advice: a non-matching path is denied via
# `permissionDecision: "deny"`, which stops the call. It runs BEFORE the tool
# executes, which is what makes it a boundary rather than a report.
#
# It also LOGS every attempt to $PR_REVIEW_HOOK_LOG, which a later workflow step
# prints. Hook stderr is not reliably surfaced through the action, so a file is
# the only channel that certainly reaches the job log.
set -uo pipefail

FINDINGS="${PR_REVIEW_FINDINGS_FILE:-/tmp/pr-review-findings.json}"
LOG="${PR_REVIEW_HOOK_LOG:-/dev/null}"

input=$(cat)
target=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)
tool=$(printf '%s' "$input" | jq -r '.tool_name // "?"' 2>/dev/null || echo '?')

if [[ "$target" == "$FINDINGS" ]]; then
  printf 'ALLOW %s -> %s\n' "$tool" "$target" >> "$LOG" 2>/dev/null || true
  # Emit nothing: no opinion, so the ordinary permission path applies and the
  # bare `Write` grant lets it through.
  exit 0
fi

printf 'DENY  %s -> %s (expected %s)\n' "$tool" "$target" "$FINDINGS" >> "$LOG" 2>/dev/null || true
jq -n --arg t "$target" --arg f "$FINDINGS" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("This job may write exactly one file: " + $f +
      ". Refused a write to " + (if $t == "" then "(no path given)" else $t end) + ".")
  }
}'
exit 0
