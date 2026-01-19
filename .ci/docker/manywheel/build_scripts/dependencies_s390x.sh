#!/bin/bash
# Helper dependencies build script for s390x
# Script used only in CD pipeline

# Stop at any error, show all commands
set -ex

# Function to retry functions that sometimes timeout or have flaky failures
retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}


# build onnxruntime 1.21.0 from sources.
# it is not possible to build it from sources using pip,
# so just build it from upstream repository.
# h5py is dependency of onnxruntime_training.
# h5py==3.11.0 builds with hdf5-devel 1.10.5 from repository.
# h5py 3.11.0 doesn't build with numpy >= 2.3.0.
# install newest flatbuffers version first:
# for some reason old version is getting pulled in otherwise.
# packaging package is required for onnxruntime wheel build.
retry pip3 install flatbuffers
retry pip3 install cython 'pkgconfig>=1.5.5' 'setuptools>=77' 'numpy<2.3.0'
retry pip3 install --no-build-isolation h5py==3.11.0
retry pip3 install packaging
retry git clone https://github.com/microsoft/onnxruntime

cd onnxruntime
git checkout v1.21.0
retry git submodule update --init --recursive
retry wget https://github.com/microsoft/onnxruntime/commit/f57db79743c4d1a3553aa05cf95bcd10966030e6.patch
patch -p1 < f57db79743c4d1a3553aa05cf95bcd10966030e6.patch
retry ./build.sh --config Release --parallel 0 --enable_pybind \
  --build_wheel --enable_training --enable_training_apis \
  --enable_training_ops --skip_tests --allow_running_as_root \
  --compile_no_warning_as_error

pip3 install ./build/Linux/Release/dist/onnxruntime_training-*.whl
cd ..
/bin/rm -rf ./onnxruntime
