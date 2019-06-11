#!/bin/sh

# This script is run by a cronjob managed by @zdevito
# which mirrors the ATen-specific directories of PyTorch
# to zdevito/ATen, for ease of use of projects that wish
# to depend solely on ATen.
#
# See also .travis.aten.yml, which is the Travis configuration
# for the ATen project (and ensures ATen is separately
# buildable.)

if [[ -z "$EXTRACTED_REPO" ]]; then
  echo "Need to set envvar EXTRACTED_REPO"
  exit 1
fi
if [[ -z "$FULL_REPO" ]]; then
  echo "Need to set envvar FULL_REPO"
  exit 1
fi
rm -rf aten-export-repo
git clone $EXTRACTED_REPO aten-export-repo
cd aten-export-repo
git config user.name "Zach DeVito"
git config user.email "zdevito@fb.com"
git remote add fullrepo $FULL_REPO
git fetch fullrepo
git checkout -b temporary-split-branch fullrepo/master
# Cribbed from https://stackoverflow.com/questions/2982055/detach-many-subdirectories-into-a-new-separate-git-repository
# and https://stackoverflow.com/questions/42355621/git-filter-branch-moving-a-folder-with-index-filter-does-not-work
git filter-branch -f --index-filter 'git rm --cached -qr --ignore-unmatch -- . && git reset -q $GIT_COMMIT -- aten cmake third_party/tbb third_party/catch third_party/cpuinfo && (git ls-files -s | sed "s-.travis.aten.yml-.travis.yml-" | sed "s-.gitmodules.aten-.gitmodules-" | git update-index --index-info)'
git checkout master
git merge temporary-split-branch
git push
