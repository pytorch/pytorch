libtorch (C++-only)
===================

The core of pytorch does not depend on Python. A
CMake-based build system compiles the C++ source code into a shared
object, libtorch.so.

Building libtorch
-----------------

You can use a python script/module located in tools package to build libtorch
::
   cd <pytorch_root>
   # export some required environment variables
   python -m tools.build_libtorch


Alternatively, you can invoke a shell script in the same directory to achieve the same goal
::
   cd <pytorch_root>
   ONNX_NAMESPACE=onnx_torch bash tools/build_pytorch_libs.sh --use-nnpack caffe2
   ls torch/lib/tmp_install # output is produced here
   ls torch/lib/tmp_install/lib/libtorch.so # of particular interest

To produce libtorch.a rather than libtorch.so, set the environment variable `BUILD_SHARED_LIBS=OFF`.

To use ninja rather than make, set `CMAKE_GENERATOR="-GNinja" CMAKE_INSTALL="ninja install"`.

Note that we are working on eliminating tools/build_pytorch_libs.sh in favor of a unified cmake build.
