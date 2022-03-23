# set -ex
# clear

ROOT_DIR=$(pwd)
DEFAULT_LOG_DIR=$ROOT_DIR/$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"
# rm -rf $LOG_DIR
# mkdir -p $LOG_DIR
# chmod -R 777 $LOG_DIR

# export HIP_VISIBLE_DEVICES=0
# export HIP_HIDDEN_FREE_MEM 500
# export HIP_TRACE_API=1
# export HIP_DB=api+mem+copy
# export HIP_API_BLOCKING=1
# export HIP_LAUNCH_BLOCKING_KERNELS kernel1,kernel2,...
# export HCC_DB 0x48a
# export HCC_SERIALIZE_KERNEL=3
# export HCC_SERIALIZE_COPY=3

# export AMD_OCL_WAIT_COMMAND=1
# export AMD_LOG_LEVEL=3
# export HIP_LAUNCH_BLOCKING=1

# bash scripts/amd/copy.sh

export PYTORCH_TEST_WITH_ROCM=1

# PYTORCH_DIR="/var/lib/jenkins/pytorch"
# PYTORCH_DIR="/tmp/pytorch"
PYTORCH_DIR="/dockerx/pytorch"
# PYTORCH_DIR=$(pwd)

cd $PYTORCH_DIR/test
# ls
pwd

# tests
# python test_spectral_ops.py --verbose | tee $LOG_DIR/test_spectral_ops.log
# python test_ops.py --verbose | tee $LOG_DIR/test_ops.log

# pip3 install pandas openpyxl
# python3 /dockerx/pytorch/scripts/amd/run_fft_test.py

python test_spectral_ops.py --verbose  TestFFTCUDA.test_empty_fft_fft_fft2_cuda_complex128 | tee $LOG_DIR/test_empty_fft_fft_fft2_cuda_complex128.log
