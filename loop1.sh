#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./claude_snapshots.sh "your prompt"
#   echo "your prompt" | ./claude_snapshots.sh

CLAUDE_BIN="${CLAUDE_BIN:-claude}"
CLAUDE_FLAGS="${CLAUDE_FLAGS:-}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" 1>&2; exit 1; }; }
need_cmd git

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: run this inside a git repo" 1>&2
  exit 1
fi

read_prompt() {
  if [[ $# -gt 0 ]]; then
    printf "%s" "$*"
  else
    cat
  fi
}

run_claude() {
  # shellcheck disable=SC2086
  "${CLAUDE_BIN}" ${CLAUDE_FLAGS}
}

slop_cleanup_amend() {
  # Let Claude review the last commit's diff and fix style issues, then amend.
  # We ask it to use git show so it sees exactly what changed.
  cat <<'EOF' | run_claude >/dev/null
Run `git show` and remove all AI generated slop introduced

This includes:
- Extra comments that a human wouldn't add or is inconsistent with the rest of the file
- Extra defensive checks or try/catch blocks that are abnormal for that area of the codebase (especially if called by trusted / validated codepaths)
- Casts to any to get around type issues
- Variables that are only used a single time right after declaration, prefer inlining the rhs.
- Any other style that is inconsistent with the file

Make the minimal necessary edits to fix these issues.
When finished, output ONLY: DONE
EOF

  git add .
  if git diff --cached --quiet; then
    echo "No cleanup changes to amend."
    return 0
  fi

  git commit --amend --no-edit
}

PROMPT="$(read_prompt "$@")"
if [[ -z "${PROMPT// }" ]]; then
  echo "error: empty prompt" 1>&2
  exit 1
fi

i=0

# 1) Ask Claude for plan.md
cat <<EOF | run_claude > plan.md
Create a plan.md with a checklist of TODOs and for each TODO include a small test to verify it's done.
Output ONLY the contents of plan.md.
User request:
${PROMPT}
EOF

git add plan.md
git commit -m "snapshot${i}"
slop_cleanup_amend
i=$((i+1))

# 2+) Loop: implement one TODO at a time, snapshot after each
while grep -qE '^\- \[ \] ' plan.md; do
  todo="$(grep -m1 -E '^\- \[ \] ' plan.md | sed -E 's/^\- \[ \] (TODO: )?//')"

  cat <<EOF | run_claude >/dev/null
Implement ONLY the next TODO from plan.md: "${todo}".
Also implement the associated test for that TODO.
Do not work on any other TODOs.
When finished, output ONLY: DONE
EOF

  # mark first unchecked item as done
  perl -0777 -i -pe 's/^\- \[ \] /\- [x] /m' plan.md

  git add .
  if git diff --cached --quiet; then
    echo "No changes to commit for snapshot${i}, stopping."
    exit 0
  fi

  git commit -m "snapshot${i}"
  slop_cleanup_amend
  i=$((i+1))
done

echo "All TODOs complete."