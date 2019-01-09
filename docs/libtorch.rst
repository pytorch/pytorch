libtorch (C++-only)
===================

The core of pytorch can be built and used without Python. A
CMake-based build system compiles the C++ source code into a shared
object, libtorch.so.

Building libtorch
-----------------

There is a script which wraps the CMake build. Invoke it with

::
   cd pytorch
   BUILD_TORCH=ON ONNX_NAMESPACE=onnx_torch bash tools/build_pytorch_libs.sh --use-nnpack caffe2
   ls torch/lib/tmp_install # output is produced here
   ls torch/lib/tmp_install/lib/libtorch.so # of particular interest

To produce libtorch.a rather than libtorch.so, set the environment variable `BUILD_SHARED_LIBS=OFF`.

To use ninja rather than make, set `CMAKE_GENERATOR="-GNinja" CMAKE_INSTALL="ninja install"`.

Future work will simplify this further.
