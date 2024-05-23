#!/bin/bash

ORIG_BRANCH=$(git rev-parse --abbrev-ref HEAD)
git checkout fbcode/pytorch-stable-prototype
for SHA in $(git log 4333e122d4b74cdf84351ed2907045c6a767b4cd..main --pretty="%h" --reverse -- torch/distributed torch/csrc/distributed test/distributed test/cpp/c10d benchmarks/distributed)
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
git checkout "$ORIG_BRANCH"
