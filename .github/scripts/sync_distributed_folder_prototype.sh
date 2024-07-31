#!/bin/bash

set -eoux pipefail

SYNC_BRANCH=pytorch-stable-prototype

git config user.email "fake@example.com"
git config user.name  "PyTorch Stable Bot"

git fetch origin main
git fetch origin "$SYNC_BRANCH"
git checkout "$SYNC_BRANCH"

# Using a hardcoded SHA here is a massive speedup as we can skip the entire history of the pytorch GitHub repo.
# This specific SHA was chosen as it was before the "branch point" of the stable branch
for SHA in $(git log ba3b05fdf37ddbc3c301294d6a560a816335e717..origin/main --pretty="%h" -- torch/distributed torch/csrc/distributed test/distributed test/cpp/c10d benchmarks/distributed)
do
    # `git merge-base --is-ancestor` exits with code 0 if the given SHA is an ancestor, and non-0 otherwise
    if git merge-base --is-ancestor $SHA HEAD || [[ $(git log --grep="(cherry picked from commit $SHA") ]]
    then
        echo "Skipping $SHA"
        continue
    fi
    echo "Copying $SHA"
    git cherry-pick -x "$SHA" -X theirs
    git reset --soft HEAD~1
    git add torch/distributed torch/csrc/distributed test/distributed test/cpp/c10d benchmarks/distributed
    git checkout .
    git commit --reuse-message=HEAD@{1}
    git clean -f
done

if [[ "${WITH_PUSH}" == true ]]; then
  git push
fi
