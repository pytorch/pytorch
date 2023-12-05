libtorch (C++-only)
===================

The core of pytorch does not depend on Python. A
CMake-based build system compiles the C++ source code into a shared
object, libtorch.so.

Building libtorch using Python
------------------------------

You can use a python script/module located in tools package to build libtorch
::
   cd <pytorch_root>

   # Make a new folder to build in to avoid polluting the source directories
   mkdir build_libtorch && cd build_libtorch

   # You might need to export some required environment variables here.
   Normally setup.py sets good default env variables, but you'll have to do
   that manually.
   python ../tools/build_libtorch.py


Alternatively, you can call setup.py normally and then copy the built cpp libraries. This method may have side effects to your active Python installation.
::
   cd <pytorch_root>
   python setup.py build

   ls torch/lib/tmp_install # output is produced here
   ls torch/lib/tmp_install/lib/libtorch.so # of particular interest

To produce libtorch.a rather than libtorch.so, set the environment variable `BUILD_SHARED_LIBS=OFF`.

To use ninja rather than make, set `CMAKE_GENERATOR="-GNinja" CMAKE_INSTALL="ninja install"`.

Note that we are working on eliminating tools/build_pytorch_libs.sh in favor of a unified cmake build.

Building libtorch using CMake
--------------------------------------

You can build C++ libtorch.so directly with cmake.  For example, to build a Release version from the main branch and install it in the directory specified by CMAKE_INSTALL_PREFIX below, you can use
::
   git clone -b main --recurse-submodule https://github.com/pytorch/pytorch.git
   mkdir pytorch-build
   cd pytorch-build
   cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch
   cmake --build . --target install

To use release branch v1.6.0, for example, replace ``master`` with ``v1.6.0``.  

Troubleshooting libtorch CMake builds
-------------------------------------

A warning about `ccache` can be resolved via `sudo apt-get install ccache`.

An error `No CMAKE_CUDA_COMPILER could be found.` may imply you need the nvidia-cuda-toolkit (not the same thing as cuda):

1. Install the toolkit with `sudo apt install nvidia-cuda-toolkit`.
2. Verify the nvidia-cuda-toolkit is installed with `nvcc --version`.

An error due to missing CMAKE_CUDA_ARCHITECTURES may be resolved by following these steps:

1. Navigate to `https://developer.nvidia.com/cuda-gpus`_.
2. Locate your GPU and find the corresponding "Compute Capability".
3. Pass the compute capability without the period symbol as a flag.

   - For example, RTX 4090 has compute capability 8.9. It is natural to pass: `-DCMAKE_CUDA_ARCHITECTURES=89`.
   - Unfortunately, nvidia-cuda-toolkit from `apt` sometimes does not support the latest compute capability architecture.
     - Roll back to earlier architectures in the list: `-DCMAKE_CUDA_ARCHITECTURES=86` worked with RTX 4090 on 2023-12-05.

A Python traceback during the build indicates you need to install PyTorch Python requirements:

1. Change directory into the PyTorch source repo cloned earlier `cd ../pytorch`.
2. Install necessary packages with `pip install -r requirements.txt`.
3. Return to the pytorch-build directory with `cd ../pytorch-build`.
4. Re-run the build command from before.

An error `nvcc fatal: Unsupported gpu architecture 'compute_89'` implies autodetection overwrites the architecture rollback:

1. Locate the file which regenerates `detect_cuda_compute_capabilities.cu` inside `pytorch-build`.
   - Ex: `../pytorch/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake`.
2. Edit the file to replace the contents with a direct call to `std::printf`.

::
   ################################################################################################
   # A function for automatic detection of GPUs installed (if autodetection is enabled)
   # Usage:
   #   CUDA_DETECT_INSTALLED_GPUS(OUT_VARIABLE)
   #
   function(CUDA_DETECT_INSTALLED_GPUS OUT_VARIABLE)
     if(NOT CUDA_GPU_DETECT_OUTPUT)
       if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
         set(file "${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cu")
       else()
         set(file "${PROJECT_BINARY_DIR}/detect_cuda_compute_capabilities.cpp")
       endif()
   # begin patch
       file(WRITE ${file} ""
         "#include <cuda_runtime.h>\n"
         "#include <cstdio>\n"
         "int main()\n"
         "{\n"
         "  std::printf(\"8.6\");\n"  # Hardcode the desired compute capability here
         "  return 0;\n"
         "}\n")
   # end patch (continuing as normal)
       if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
         try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file}
                 RUN_OUTPUT_VARIABLE compute_capabilities)
       else()
         try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file}
                 CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
                 LINK_LIBRARIES ${CUDA_LIBRARIES}
                 RUN_OUTPUT_VARIABLE compute_capabilities)
       endif()

       # Filter unrelated content out of the output.
       string(REGEX MATCHALL "[0-9]+\\.[0-9]+" compute_capabilities "${compute_capabilities}")

       if(run_result EQUAL 0)
         string(REPLACE "2.1" "2.1(2.0)" compute_capabilities "${compute_capabilities}")
         set(CUDA_GPU_DETECT_OUTPUT ${compute_capabilities}
           CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
       endif()
     endif()
   endif()

An error with `error: parameter packs not expanded with '...'` requires you to patch the `std_functions.h` of c++:

1. [Github issue with fix](https://github.com/NVIDIA/nccl/issues/650#issuecomment-1145173577).
Excerpt / Paraphrase of the fix instructions:   
Comment out these two lines beginning with noexcept in `/usr/include/c++/*/bits/std_function.h`:

Line 433+ (approximate):
::
   template<typename _Functor,
            typename _Constraints = _Requires<_Callable<_Functor>>>
     function(_Functor&& __f)
     //noexcept(_Handler<_Functor>::template _S_nothrow_init<_Functor>()) // CUDA BOTCHES THIS
     : _Function_base()

Line 529+ (approximate):
::
   template<typename _Functor>
     _Requires<_Callable<_Functor>, function&>
     operator=(_Functor&& __f)
     //noexcept(_Handler<_Functor>::template _S_nothrow_init<_Functor>()) // CUDA BOTCHES THIS
     {
       function(std::forward<_Functor>(__f)).swap(*this);
       return *this;
     }


Once you've edited the files, remember to continue where you left off in the "Building libtorch using CMake" section.

Also, please note: the final command will install libtorch under `../pytorch-install`.
Remember to use the install directory for your environment variables and not the build directory. 

For example:
::
   export LIBTORCH=/path/to/pytorch-install
   export LIBTORCH_LIB=/path/to/pytorch-install/lib
   export LIBTORCH_INCLUDE=/path/to/pytorch-install/include
   export LD_LIBRARY_PATH=$LIBTORCH/lib:$LIBTORCH:$LD_LIBRARY_PATH
