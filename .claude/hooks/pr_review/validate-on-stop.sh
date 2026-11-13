#!/bin/bash
# Stop hook — blocking. The session may not end on a findings file that would
# lose findings at publication.
#
# Exit 2 is what Claude Code reads as "do not stop, here is why"; the stderr
# text becomes the model's next instruction.
#
# `stop_hook_active` guards the obvious loop: it is true when the model was
# already resumed by this hook, so a file the model cannot fix stops the session
# rather than cycling forever. That trades a lost finding for a terminating run,
# which is the right way round — `extract_verdict.py` still records exactly what
# was dropped and why, so the loss is visible in the row rather than silent.
set -uo pipefail

input=$(cat)
active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false' 2>/dev/null || echo false)
LOG="${PR_REVIEW_HOOK_LOG:-/dev/null}"

# A findings file written before the sub-agents reported is a draft, not the
# review. The model gets one chance to rewrite it; if the session still ends on
# the draft, the marker below makes the publish step record a failed run
# instead of publishing it.
# Fails closed: a transcript that is missing or unreadable counts as stale,
# since the check cannot show the file is the final verdict.
transcript=$(printf '%s' "$input" | jq -r '.transcript_path // empty' 2>/dev/null || true)
stale=1
if [[ -n "$transcript" ]] && python3 \
    "$(dirname "$0")/../../../scripts/pr_review/verdict_after_subagents.py" \
    "$transcript" "${PR_REVIEW_FINDINGS_FILE:-}"; then
  stale=0
fi

if [[ "$active" == "true" ]]; then
  if [[ "$stale" == "1" ]]; then
    printf 'STALE_VERDICT the session ended on a findings file older than the last sub-agent report\n' >> "$LOG" 2>/dev/null || true
  fi
  exit 0
fi

printf 'STOP hook fired (stop_hook_active=%s)\n' "$active" >> "$LOG" 2>/dev/null || true
if [[ "$stale" == "1" ]]; then
  echo "A sub-agent reported after your last write of the findings file, so that file is a draft. Rewrite it with your final verdict, then finish." >&2
  exit 2
fi
if ! "$(dirname "$0")/validate-findings.sh"; then
  echo "Rewrite the findings file so nothing is discarded, then finish." >&2
  exit 2
fi

exit 0
