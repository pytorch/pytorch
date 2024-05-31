#!/bin/bash

set -eoux pipefail

SYNC_BRANCH=fbcode/pytorch-stable-prototype

git config user.email "fake@example.com"
git config user.name  "PyTorch Stable Bot"

git fetch origin main
git fetch origin "$SYNC_BRANCH"
git checkout "$SYNC_BRANCH"

for SHA in $(git log 4333e122d4b74cdf84351ed2907045c6a767b4cd..origin/main --pretty="%h" --reverse -- torch/distributed torch/csrc/distributed test/distributed test/cpp/c10d benchmarks/distributed)
do
    # `git merge-base --is-ancestor` exits with code 0 if the given SHA is an ancestor, and non-0 otherwise
    if git merge-base --is-ancestor $SHA HEAD || [[ $(git log --grep="(cherry picked from commit $SHA") ]]
    then
        echo "Skipping $SHA"
        continue
    fi
    echo "Copying $SHA"
    git cherry-pick -x "$SHA"
done

if [[ "${WITH_PUSH}" == true ]]; then
  git push
fi
