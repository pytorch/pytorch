# set -e

clear

ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# copy cur dir to tmp dir
TMP_DIR=/tmp/pytorch
bash scripts/amd/create_temp_dir.sh $TMP_DIR | tee $LOG_DIR/create_temp_dir.log

# fix rccl issue
# sudo ln -s /opt/rocm-5.0.0/lib/librccl.so.1.0.50000 /usr/lib/librccl.so

# build pytorch
pip uninstall torch -y
export PYTORCH_ROCM_ARCH="gfx908"
# export PYTORCH_ROCM_ARCH="gfx1030"
# export PYTORCH_ROCM_ARCH="gfx908;gfx1030"
cd $TMP_DIR
bash .jenkins/pytorch/build.sh | tee $LOG_DIR/build.log

# bash scripts/amd/build.sh | tee $LOG_DIR/build.log
# bash scripts/amd/build_torchvision.sh | tee $LOG_DIR/build_torchvision.log
bash scripts/amd/test.sh $LOG_DIR 2>&1 |tee $LOG_DIR/test.log
# bash scripts/amd/benchmark.sh 2>&1 | tee $LOG_DIR/benchmark.log
