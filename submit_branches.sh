#!/bin/bash

# List of branches to submit
branches=(
  "refactor/builtin-builder-consolidation"
  "refactor/dicts-builder-consolidation"
  "refactor/higher-order-ops-builder-consolidation"
  "refactor/lists-builder-consolidation"
  "refactor/remaining-files-builder-consolidation"
  "refactor/symbolic-convert-builder-consolidation"
  "refactor/torch-builder-consolidation"
  "refactor/user-defined-builder-consolidation"
)

# Save current branch to return to later
original_branch=$(git branch --show-current)

for branch in "${branches[@]}"; do
  echo "=========================================="
  echo "Processing: $branch"
  echo "=========================================="

  git checkout "$branch"
  if [ $? -ne 0 ]; then
    echo "ERROR: Failed to checkout $branch"
    continue
  fi

  ghstack
  if [ $? -ne 0 ]; then
    echo "ERROR: ghstack failed for $branch"
  else
    echo "SUCCESS: $branch submitted"
  fi

  echo ""
done

# Return to original branch
echo "Returning to original branch: $original_branch"
git checkout "$original_branch"
