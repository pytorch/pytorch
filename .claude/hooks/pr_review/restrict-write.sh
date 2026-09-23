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

# FAIL CLOSED, and note which way "closed" points here. Every refusal below is
# built by `jq`, while the ALLOW branch refuses by saying nothing -- an empty
# stdout is "no opinion", after which the bare `Write` grant lets the call
# through. So a jq that is missing or errors would convert every denial into a
# permission. Exit 2 is what PreToolUse reads as a block and it needs no jq to
# produce, so every failure path below takes it.
die_closed() {
  printf 'restrict-write.sh: %s\n' "$1" >&2
  printf 'BLOCK %s\n' "$1" >> "$LOG" 2>/dev/null || true
  exit 2
}

# The tool name and the write target below are MODEL-controlled, and this log is
# `cat`ed into the job log, which the Actions runner parses for workflow
# commands. Refusing the filesystem operation does not close that output
# channel, so the value is neutralized here. THREE things, and the runner's own
# parser (actions/runner, Runner.Common/ActionCommand.cs) is why each is needed:
#
#   * newline and every other non-printable byte -> `?`. TryParseV2 accepts
#     `::cmd::` only when the line STARTS with it after TrimStart, so keeping
#     the value on one line behind a `DENY `/`ALLOW ` prefix defeats that form.
#   * `::` and `##[` rewritten anyway. TryParse, the LEGACY form, uses
#     `IndexOf("##[")` and matches ANYWHERE in the line, so position is no
#     defence against it; `::` is rewritten too rather than relying on the
#     prefix alone holding for every future caller of this function.
#   * length bounded, so a multi-megabyte path cannot flood the job log.
#
# LC_ALL=C is exported for the WHOLE pipeline, not just printf: under a UTF-8
# locale `[:print:]` includes non-ASCII printables and `cut -c` counts
# characters, so the byte-level guarantee would not hold. Fails safe: if `tr` is
# missing the substitution is empty and the log loses the path, not its shape.
logsafe() {
  (
    export LC_ALL=C
    printf '%s' "$1" \
      | tr -c '[:print:]' '?' \
      | sed -e 's/::/;;/g' -e 's/##\[/#(/g' \
      | cut -c1-200
  )
}

command -v jq >/dev/null 2>&1 || die_closed "jq not found on PATH; refusing the write"

input=$(cat) || die_closed "could not read the hook payload from stdin"

# Read the path through a file rather than `$(...)`: command substitution
# strips trailing newlines, so `<findings>\n` would compare EQUAL to the
# allowed path and be let through as a different file. `-j` also suppresses
# the newline jq itself would append; the X sentinel preserves any the value
# genuinely carries.
TMP=$(mktemp) || die_closed "mktemp failed"
trap 'rm -f "$TMP"' EXIT
printf '%s' "$input" | jq -j '.tool_input.file_path // empty' > "$TMP" \
  || die_closed "jq failed reading .tool_input.file_path"

# `$(cat ...; printf X)` takes its exit status from the printf, so a `cat`
# that fails HALFWAY -- emitting the allowed prefix and then erroring -- would
# reach the comparison below as a truncated path that happens to match. Size
# the read against the file and refuse a short one.
#
# BOTH SIDES IN BYTES. `wc -c` counts bytes and `${#var}` counts CHARACTERS in
# a UTF-8 locale, so an explicitly configured non-ASCII findings path would
# fail this check as a phantom short read -- closed, but a refusal of a legal
# configuration. `LC_ALL=C` makes the shell count bytes too.
expected=$(wc -c < "$TMP") || die_closed "could not size the extracted path"
target=$(cat "$TMP"; printf X) || true
target=${target%X}
actual=$(LC_ALL=C bash -c 'printf %s "$1" | wc -c' _ "$target") \
  || die_closed "could not size the read-back path"
[ "$actual" -eq "$expected" ] || die_closed "short read of the extracted path ($actual of $expected bytes)"

tool=$(printf '%s' "$input" | jq -r '.tool_name // "?"' 2>/dev/null || echo '?')

if [[ "$target" == "$FINDINGS" ]]; then
  printf 'ALLOW %s -> %s\n' "$(logsafe "$tool")" "$(logsafe "$target")" >> "$LOG" 2>/dev/null || true
  # Emit nothing: no opinion, so the ordinary permission path applies and the
  # bare `Write` grant lets it through.
  exit 0
fi

printf 'DENY  %s -> %s (expected %s)\n' "$(logsafe "$tool")" "$(logsafe "$target")" "$FINDINGS" >> "$LOG" 2>/dev/null || true
# Build first, print second: a half-written object on stdout alongside a
# non-zero exit is ambiguous to the caller.
deny=$(jq -n --arg t "$target" --arg f "$FINDINGS" '{
  hookSpecificOutput: {
    hookEventName: "PreToolUse",
    permissionDecision: "deny",
    permissionDecisionReason: ("This job may write exactly one file: " + $f +
      ". Refused a write to " + (if $t == "" then "(no path given)" else $t end) + ".")
  }
}') || die_closed "jq failed building the denial"
printf '%s\n' "$deny"
exit 0
