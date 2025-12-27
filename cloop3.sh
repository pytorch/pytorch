#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./codex_snapshots.sh "your prompt"
#   echo "your prompt" | ./codex_snapshots.sh
#
# Env overrides:
#   CODEX_BIN=codex
#   CODEX_EXEC_FLAGS="..."   (for example approval/non-interactive flags)

CODEX_BIN="${CODEX_BIN:-codex}"
CODEX_EXEC_FLAGS="${CODEX_EXEC_FLAGS:-}"

need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1" 1>&2; exit 1; }; }
need_cmd git
need_cmd "${CODEX_BIN}"

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

ensure_file_nonempty() {
  local path="$1"
  if [[ ! -f "$path" ]] || [[ ! -s "$path" ]]; then
    echo "error: expected non-empty file: $path" 1>&2
    exit 1
  fi
}

codex_run() {
  local prompt="$1"
  local help
  help="$("${CODEX_BIN}" exec --help 2>/dev/null || true)"

  # Variant A: codex exec --prompt "..."
  if grep -q -- '--prompt' <<<"$help"; then
    # shellcheck disable=SC2086
    "${CODEX_BIN}" exec ${CODEX_EXEC_FLAGS} --prompt "$prompt"
    return 0
  fi

  # Variant B: codex exec "..."
  set +e
  # shellcheck disable=SC2086
  "${CODEX_BIN}" exec ${CODEX_EXEC_FLAGS} "$prompt"
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    return 0
  fi

  # Variant C: codex exec reads prompt from stdin
  # shellcheck disable=SC2086
  printf "%s" "$prompt" | "${CODEX_BIN}" exec ${CODEX_EXEC_FLAGS}
}

slop_cleanup_amend() {
  local prompt
  prompt=$(
    cat <<'EOF'
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
  )

  codex_run "$prompt"

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

# 1) Ask Codex to write plan.md directly
plan_prompt=$(
  cat <<EOF
Create a file named plan.md in the current directory.

The file must contain a checklist of TODOs, and for each TODO include a small test to verify it's done.

Do not print the contents of plan.md.
When finished, output ONLY: DONE

User request:
${PROMPT}
EOF
)

codex_run "$plan_prompt"
ensure_file_nonempty "plan.md"

echo "----- plan.md -----"
cat plan.md
echo "-------------------"

git add plan.md
git commit -m "snapshot${i}"
slop_cleanup_amend
i=$((i+1))

# 2+) Loop: implement one TODO at a time, snapshot after each
while grep -qE '^- \[ \] ' plan.md; do
  todo="$(grep -m1 -E '^- \[ \] ' plan.md | sed -E 's/^- \[ \] (TODO: )?//')"

  step_prompt=$(
    cat <<EOF
Implement ONLY the next TODO from plan.md: "${todo}".
Also implement the associated test for that TODO.
Do not work on any other TODOs.
When finished, output ONLY: DONE
EOF
  )

  codex_run "$step_prompt"

  # mark first unchecked item as done
  perl -0777 -i -pe 's/^- \[ \] /- [x] /m' plan.md

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