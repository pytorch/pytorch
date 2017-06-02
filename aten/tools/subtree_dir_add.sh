SRC_BRANCH=$1
SRC_PATH=$2
DST_PATH=$3

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
set -x
git branch -q -D temporary-split-branch
git checkout $SRC_BRANCH
git subtree split -P $SRC_PATH -b temporary-split-branch
git checkout $CURRENT_BRANCH
git subtree add -P $DST_PATH temporary-split-branch ${@:5}
git branch -D temporary-split-branch

#./subtree_dir.sh add pytorch/master torch/lib/TH lib/TH
#./subtree_dir.sh add pytorch/master torch/lib/THC lib/THC
#./subtree_dir.sh add pytorch/master torch/lib/THNN lib/THNN
#./subtree_dir.sh add pytorch/master torch/lib/THCUNN lib/THCUNN
