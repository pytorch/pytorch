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
if [[ "$active" == "true" ]]; then
  exit 0
fi

printf 'STOP hook fired (stop_hook_active=%s)\n' "$active" >> "${PR_REVIEW_HOOK_LOG:-/dev/null}" 2>/dev/null || true
if ! "$(dirname "$0")/validate-findings.sh"; then
  echo "Rewrite the findings file so nothing is discarded, then finish." >&2
  exit 2
fi

exit 0
