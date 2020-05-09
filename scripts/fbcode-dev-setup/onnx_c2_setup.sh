#!/bin/bash

# This script helps developers set up the ONNX Caffe2 and PyTorch develop environment on devgpu.
# It creates an virtualenv instance, and installs all the dependencies in this environment.
# The script will creates a folder called onnx-dev folder under the $HOME directory.
# onnx, pytorch and caffe2 are installed separately.
# Please source $HOME/onnx-dev/.onnx_env_init to initialize the development before starting developing.


# TODO: support python 3.

# Set script configuration
set -e
shopt -s expand_aliases

# Proxy setup
alias with_proxy="HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 FTP_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 ftp_proxy=http://fwdproxy:8080 http_no_proxy='*.facebook.com|*.tfbnw.net|*.fb.com'"

# Set the variables
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'
onnx_root="$HOME/local/onnx-dev"   # I think hardcoding the onnx root dir is fine, just like fbsource
onnx_root_link="$HOME/onnx-dev"
venv="$onnx_root/onnxvenv"
onnx_init_file="$onnx_root_link/.onnx_env_init"
ccache_root="$onnx_root/ccache"
ccache_script="$(pwd)/ccache_install.sh"
sanity_script="$onnx_root/sanity.sh"

# Check whether default CUDA exists
# TODO check the required header and lib files
default_cuda="/usr/local/cuda"
if [[ ! -e "$default_cuda" ]]; then
  echo "Default CUDA is not found at $default_cuda"
fi

# Checking to see if CuDNN is present, and install it if not exists
if [ -f /usr/local/cuda/include/cudnn.h ]; then
  echo "CuDNN header already exists!!"
else
  sudo cp -R /home/engshare/third-party2/cudnn/6.0.21/src/cuda/include/* /usr/local/cuda/include/
  sudo cp -R /home/engshare/third-party2/cudnn/6.0.21/src/cuda/lib64/* /usr/local/cuda/lib64/
fi

# TODO set the specific version for each package
# Install the dependencies for Caffe2
sudo yum install python-virtualenv freetype-devel libpng-devel glog gflags protobuf protobuf-devel protobuf-compiler -y
rpm -q protobuf  # check the version and if necessary update the value below
protoc --version  # check protoc
protoc_path=$(which protoc)
if [[ "$protoc_path" != "/bin/protoc" ]]; then
  echo "Warning: Non-default protoc is detected, the script may not work with non-default protobuf!!!"
  echo "Please try to remove the protoc at $protoc_path and rerun this script."
  exit 1
fi

# Upgrade Cmake to the right version (>3.0)
sudo yum remove cmake3 -y
sudo yum install cmake -y

# Install the dependencies for CCache
sudo yum install autoconf asciidoc -y

# Create the root folder
if [ -e "$onnx_root" ]; then
  timestamp=$(date "+%Y.%m.%d-%H.%M.%S")
  mv --backup=t -T "$onnx_root" "${onnx_root}.old.$timestamp"
fi
mkdir -p "$onnx_root"
if [ -e "$onnx_root_link"]; then
  timestamp=$(date "+%Y.%m.%d-%H.%M.%S")
  mv --backup=t -T "$onnx_root_link" "${onnx_root_link}.old.$timestamp"
fi
ln -s "$onnx_root" "$onnx_root_link"

# Set the name of virtualenv instance
with_proxy virtualenv "$venv"

# Creating a script that can be sourced in the future for the environmental variable
touch "$onnx_init_file"
{
  # shellcheck disable=SC2016
  echo 'if [ -z "$LD_LIBRARY_PATH" ]; then';
  echo '  export LD_LIBRARY_PATH=/usr/local/cuda/lib64';
  echo 'else'
  # shellcheck disable=SC2016
  echo '  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH';
  echo "fi"
  # shellcheck disable=SC2016
  echo 'export PATH='"$ccache_root"'/lib:/usr/local/cuda/bin:$PATH';
  echo "source $venv/bin/activate";
  echo 'alias with_proxy="HTTPS_PROXY=http://fwdproxy:8080 HTTP_PROXY=http://fwdproxy:8080 FTP_PROXY=http://fwdproxy:8080 https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 ftp_proxy=http://fwdproxy:8080 http_no_proxy='"'"'*.facebook.com|*.tfbnw.net|*.fb.com'"'"'"'
} >> "$onnx_init_file"
chmod u+x "$onnx_init_file"

# Installing CCache
cd "$onnx_root"
if [ ! -f "$ccache_script" ]; then
  ccache_script="$onnx_root/ccache_install.sh"
  with_proxy wget https://raw.githubusercontent.com/pytorch/pytorch/master/scripts/fbcode-dev-setup/ccache_setup.sh -O "$ccache_script"
fi
chmod u+x "$ccache_script"
"$ccache_script" --path "$ccache_root"

# Test nvcc with CCache
own_ccache=true
if [ -f "$CUDA_NVCC_EXECUTABLE" ] && [[ "$ccache_root/cuda/nvcc" != "$CUDA_NVCC_EXECUTABLE" ]] && \
  [[ "$CUDA_NVCC_EXECUTABLE" == *"ccache"* ]]; then  # Heuristic rule
  if $CUDA_NVCC_EXECUTABLE --version; then
    own_ccache=false
  fi
fi
if [ "$own_ccache" = true ]; then
  echo "export CUDA_NVCC_EXECUTABLE=$ccache_root/cuda/nvcc" >> "$onnx_init_file"
fi

# Loading env vars
# shellcheck disable=SC1090
source "$onnx_init_file"

"$CUDA_NVCC_EXECUTABLE" --version

# Create a virtualenv, activate it, upgrade pip
if [ -f "$HOME/.pip/pip.conf" ]; then
  echo "${RED}Warning: $HOME/.pip/pip.conf is detected, pip install may fail!${NC}"
fi
with_proxy python -m pip install -U pip setuptools
with_proxy python -m pip install future numpy "protobuf>3.2" pytest-runner pyyaml typing ipython

# Cloning repos
cd "$onnx_root"
with_proxy git clone https://github.com/onnx/onnx --recursive
with_proxy git clone https://github.com/pytorch/pytorch --recursive

# Build ONNX
cd "$onnx_root/onnx"
with_proxy python setup.py develop

# Build PyTorch and Caffe2
cd "$onnx_root/pytorch"
with_proxy pip install -r "requirements.txt"
with_proxy python setup.py develop

# Sanity checks and useful info
cd "$onnx_root"
with_proxy wget https://raw.githubusercontent.com/pytorch/pytorch/master/scripts/fbcode-dev-setup/onnx_c2_sanity_check.sh -O "$sanity_script"
chmod u+x "$sanity_script"
$sanity_script

echo "Congrats, you are ready to rock!!"
echo "################ Please run the following command before development ################"
echo -e "${CYAN}source $onnx_init_file${NC}"
echo "#####################################################################################"
