#!/bin/sh
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
git filter-branch -f --index-filter 'git rm --cached -qr --ignore-unmatch -- . && git reset -q $GIT_COMMIT -- aten cmake' --prune-empty
git checkout master
git merge temporary-split-branch
git push
