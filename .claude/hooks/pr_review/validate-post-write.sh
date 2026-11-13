#!/bin/bash
# PostToolUse(Write) hook — advisory, and the reason the whole loop works.
#
# HOW FEEDBACK ACTUALLY REACHES THE MODEL. On exit 0 a hook's stderr goes to the
# transcript and NOWHERE the model can read, so a validator that merely printed
# errors and exited 0 would be inert — the model would rewrite nothing because
# it was never told. The two mechanisms that do reach it are exit 2 (stderr fed
# back as a tool error) and a `hookSpecificOutput.additionalContext` string on
# STDOUT, injected as a system message. This uses the second: the write already
# happened and is not an error, so an error is the wrong shape for it.
#
# Still always exits 0. PostToolUse fires after the file is on disk, so failing
# here could not undo the write; it would only turn a fixable state into a dead
# step. The Stop hook is what refuses to let the session END badly.
set -uo pipefail

FINDINGS="${PR_REVIEW_FINDINGS_FILE:-/tmp/pr-review-findings.json}"

emit_nothing() { exit 0; }

input=$(cat)
written=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)

# Only speak about the findings file. Write is scoped to that one path anyway,
# but a write elsewhere would otherwise be answered with a missing-file error
# about a file the model never claimed to be writing.
[[ "$written" == "$FINDINGS" ]] || emit_nothing

report=$("$(dirname "$0")/validate-findings.sh" 2>&1)
status=$?
printf 'POSTWRITE validate rc=%s\n' "$status" >> "${PR_REVIEW_HOOK_LOG:-/dev/null}" 2>/dev/null || true

if [[ $status -eq 0 ]]; then
  # Silence on success. Injecting "it is valid" after every write spends
  # context on the case that needs no action, and trains the model to skim
  # exactly the channel the failures arrive on.
  emit_nothing
fi

jq -n --arg r "$report" '{
  hookSpecificOutput: {
    hookEventName: "PostToolUse",
    additionalContext: ("Your findings file is not publishable yet:\n\n" + $r +
      "\n\nRewrite it with the Write tool. This check runs again on every write.")
  }
}'
exit 0
