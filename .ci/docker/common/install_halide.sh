#!/bin/bash
set -ex

source "$(dirname "${BASH_SOURCE[0]}")/common_utils.sh"

COMMIT=$(get_pinned_commit halide)
test -n "$COMMIT"

# activate conda to populate CONDA_PREFIX
test -n "$ANACONDA_PYTHON_VERSION"
eval "$(conda shell.bash hook)"
conda activate py_$ANACONDA_PYTHON_VERSION

if [ -n "${UBUNTU_VERSION}" ];then
    apt update
    apt-get install -y lld liblld-15-dev libpng-dev libjpeg-dev libgl-dev \
                  libopenblas-dev libeigen3-dev libatlas-base-dev libzstd-dev
fi

conda_install numpy scipy imageio cmake ninja

git clone --depth 1 --branch release/16.x --recursive https://github.com/llvm/llvm-project.git
cmake -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_PROJECTS="clang" \
        -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
        -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_BUILD_32_BITS=OFF \
        -S llvm-project/llvm -B llvm-build -G Ninja
cmake --build llvm-build
cmake --install llvm-build --prefix llvm-install
export LLVM_ROOT=`pwd`/llvm-install
export LLVM_CONFIG=$LLVM_ROOT/bin/llvm-config

git clone https://github.com/halide/Halide.git
pushd Halide
git checkout ${COMMIT} && git submodule update --init --recursive
pip_install -r requirements.txt
# NOTE: pybind has a requirement for cmake > 3.5 so set the minimum cmake version here with a flag
#       Context: https://github.com/pytorch/pytorch/issues/150420
cmake -G Ninja -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -S . -B build
cmake --build build
test -e ${CONDA_PREFIX}/lib/python3 || ln -s python${ANACONDA_PYTHON_VERSION} ${CONDA_PREFIX}/lib/python3
cmake --install build --prefix ${CONDA_PREFIX}
chown -R jenkins ${CONDA_PREFIX}
popd
rm -rf Halide llvm-build llvm-project llvm-install

python -c "import halide"  # check for errors
