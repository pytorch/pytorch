#!/bin/bash

echo "Testing custom script operators"
pushd test/custom_operator
# Build the custom operator library.
rm -rf build && mkdir build
pushd build
SITE_PACKAGES="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')"
CMAKE_PREFIX_PATH="$SITE_PACKAGES/torch" cmake ..
make VERBOSE=1
popd

# Run tests Python-side and export a script module.
python test_custom_ops.py -v
python model.py --export-script-module=model.pt
# Run tests C++-side and load the exported script module.
build/test_custom_ops ./model.pt
popd