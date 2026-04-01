#!/usr/bin/env bash
# This script generates a release note.
# It prints information about breaking-changes first and the rest.
# The content is mostly based on `git log --oneline --since 202X-YY-ZZ`.

# Usage: the script takes one command-line argument that will be used on '-since' option of git command.
# As an example, you can run a script with a following command, and it will print commit titles between today and 2024-07-01.
# ```
# docs/scripts/release-note.sh 2024-07-01
# ```

# This script is supposed to work on all Windows based shell systems including WSL and git-bash.
# If you make any modifications, please test them, because CI doesn't test this script.

verbose=true
$verbose && echo "Reminder: PLEASE make sure your local repo is up-to-date before running the script." >&2

gh=""
for candidate in "$(which gh.exe)" "/mnt/c/Program Files/GitHub CLI/gh.exe" "/c/Program Files/GitHub CLI/gh.exe" "/cygdrive/c/Program Files/GitHub CLI/gh.exe"; do
  if [ -x "$candidate" ]; then
    gh="$candidate"
    break
  fi
done
if [ "x$gh" = "x" ] || ! [ -x "$gh" ]; then
  echo "File not found: gh.exe"
  echo "gh.exe can be downloaded from https://cli.github.com"
  exit 1
fi
$verbose && echo "gh.exe is found from: $gh" >&2

if [ "x$1" = "x" ]; then
  echo "This script requires 'since' information for git-log command."
  echo "Usage: $0 2024-07-30"
  exit 1
fi
since="$1"

commits="$(git log --oneline --since $since)"
commitsCount="$(echo "$commits" | wc -l)"

echo "=== Breaking changes ==="
breakingChanges=""
for i in $(seq $commitsCount); do
  line="$(echo "$commits" | head -$i | tail -1)"

  # Get PR number from the git commit title
  pr="$(echo "$line" | grep '#[1-9][0-9][0-9][0-9][0-9]*' | sed 's|.* (\#\([1-9][0-9][0-9][0-9][0-9]*\))|\1|')"
  [ "x$pr" = "x" ] && continue

  # Check if the PR is marked as a breaking change
  if "$gh" issue view $pr --json labels | grep -q 'pr: breaking change'; then
    breakingChanges+="$line"
  fi
done
if [ "x$breakingChanges" = "x" ]; then
  echo "No breaking changes"
else
  echo "$breakingChanges"
fi
echo ""

echo "=== All changes for this release ==="
for i in $(seq $commitsCount); do
  line="$(echo "$commits" | head -$i | tail -1)"

  result="$line"
  for dummy in 1; do
    # Get PR number from the git commit title
    pr="$(echo "$line" | grep '#[1-9][0-9][0-9][0-9][0-9]*' | sed 's|.* (\#\([1-9][0-9][0-9][0-9][0-9]*\))|\1|')"
    [ "x$pr" = "x" ] && break

    # Mark breaking changes with "[BREAKING]"
    if "$gh" issue view $pr --json labels | grep -q 'pr: breaking change'; then
      result="[BREAKING] $line"
    fi

    # Get the issue number for the PR
    body="$("$gh" issue view $pr --json body)"
    [ "x$body" = "x" ] && break
    issue="$(echo "$body" | grep '#[1-9][0-9][0-9][0-9][0-9]*' | sed 's|.*\#\([1-9][0-9][0-9][0-9][0-9]*\).*|\1|')"
    [ "x$issue" = "x" ] && break

    # Get the labels of the issue
    label="$("$gh" issue view $issue --json labels)"
    [ "x$label" = "x" ] && break

    # Get the goal type from the labels
    goal="$(echo "$label" | grep '"goal:' | sed 's|.*"goal:\([^"]*\)".*|\1|')"
    [ "x$goal" = "x" ] && break

    result+=" (#$issue:$goal)"
  done
  echo "$result"
done
