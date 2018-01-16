set -e
set -x
DIR=$(dirname $0)
git fetch pytorch
$DIR/subtree_dir_merge.sh pytorch/master torch/lib/TH lib/TH
$DIR/subtree_dir_merge.sh pytorch/master torch/lib/THC lib/THC
$DIR/subtree_dir_merge.sh pytorch/master torch/lib/THNN lib/THNN
$DIR/subtree_dir_merge.sh pytorch/master torch/lib/THCUNN lib/THCUNN
$DIR/subtree_dir_merge.sh pytorch/master torch/lib/THS lib/THS
$DIR/subtree_dir_merge.sh pytorch/master torch/lib/THCS lib/THCS
