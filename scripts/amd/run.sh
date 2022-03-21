# set -e

clear

ROOT_DIR=$(pwd)
DEFAULT_LOG_DIR=$ROOT_DIR/$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# build pytorch
# bash scripts/amd/prep.sh |tee $LOG_DIR/prep.log
# bash scripts/amd/build.sh |tee $LOG_DIR/build.log
# bash scripts/amd/build_torchvision.sh |tee $LOG_DIR/build_torchvision.log
bash scripts/amd/test.sh $LOG_DIR 2>&1 | tee $LOG_DIR/test.log
