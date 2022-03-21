# set -e

clear

# build pytorch
bash scripts/amd/prep.sh |tee /dockerx/pytorch/prep.log
bash scripts/amd/build.sh |tee /dockerx/pytorch/build.log
# bash scripts/amd/build_torchvision.sh |tee /dockerx/pytorch/build_torchvision.log
bash scripts/amd/test.sh 2>&1 |tee /dockerx/pytorch/test.log