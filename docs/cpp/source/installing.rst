Installing C++ Distributions of PyTorch
=======================================

We provide binary distributions of all headers, libraries and CMake
configuration files required to depend on PyTorch. We call this distribution
*LibTorch*, and you can download ZIP archives containing the latest LibTorch
distribution on `our website <https://pytorch.org/get-started/locally/>`_. Below
is a small example of writing a minimal application that depends on LibTorch
and uses the ``torch::Tensor`` class which comes with the PyTorch C++ API.

Minimal Example
---------------

The first step is to download the LibTorch ZIP archive via the link above. For
example:

.. code-block:: sh

  wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
  unzip libtorch-shared-with-deps-latest.zip


Next, we can write a minimal CMake build configuration to develop a small
application that depends on LibTorch. CMake is not a hard requirement for using
LibTorch, but it is the recommended and blessed build system and will be well
supported into the future. A most basic `CMakeLists.txt` file could look like
this:

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
  project(example-app)

  find_package(Torch REQUIRED)

  add_executable(example-app example-app.cpp)
  target_link_libraries(example-app "${TORCH_LIBRARIES}")
  set_property(TARGET example-app PROPERTY CXX_STANDARD 11)

The implementation of our example will simply create a new `torch::Tensor` and
print it:

.. code-block:: cpp

  #include <torch/torch.h>
  #include <iostream>

  int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
  }

While there are more fine-grained headers you can include to access only parts
of the PyTorch C++ API, including `torch/torch.h` is the most sure-proof way of
including most of its functionality.

The last step is to build the application. For this, assume our example
directory is laid out like this:

.. code-block:: sh

  example-app/
    CMakeLists.txt
    example-app.cpp

We can now run the following commands to build the application from within the
``example-app/`` folder:

.. code-block:: sh

  mkdir build
  cd build
  cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
  make

where ``/absolute/path/to/libtorch`` should be the absolute (!) path to the unzipped LibTorch
distribution. If all goes well, it will look something like this:

.. code-block:: sh

  root@4b5a67132e81:/example-app# mkdir build
  root@4b5a67132e81:/example-app# cd build
  root@4b5a67132e81:/example-app/build# cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
  -- The C compiler identification is GNU 5.4.0
  -- The CXX compiler identification is GNU 5.4.0
  -- Check for working C compiler: /usr/bin/cc
  -- Check for working C compiler: /usr/bin/cc -- works
  -- Detecting C compiler ABI info
  -- Detecting C compiler ABI info - done
  -- Detecting C compile features
  -- Detecting C compile features - done
  -- Check for working CXX compiler: /usr/bin/c++
  -- Check for working CXX compiler: /usr/bin/c++ -- works
  -- Detecting CXX compiler ABI info
  -- Detecting CXX compiler ABI info - done
  -- Detecting CXX compile features
  -- Detecting CXX compile features - done
  -- Looking for pthread.h
  -- Looking for pthread.h - found
  -- Looking for pthread_create
  -- Looking for pthread_create - not found
  -- Looking for pthread_create in pthreads
  -- Looking for pthread_create in pthreads - not found
  -- Looking for pthread_create in pthread
  -- Looking for pthread_create in pthread - found
  -- Found Threads: TRUE
  -- Configuring done
  -- Generating done
  -- Build files have been written to: /example-app/build
  root@4b5a67132e81:/example-app/build# make
  Scanning dependencies of target example-app
  [ 50%] Building CXX object CMakeFiles/example-app.dir/example-app.cpp.o
  [100%] Linking CXX executable example-app
  [100%] Built target example-app

Executing the resulting ``example-app`` binary found in the ``build`` folder
should now merrily print the tensor (exact output subject to randomness):

.. code-block:: sh

  root@4b5a67132e81:/example-app/build# ./example-app model.pt
  0.2063  0.6593  0.0866
  0.0796  0.5841  0.1569
  [ Variable[CPUFloatType]{2,3} ]

Support
-------

If you run into any troubles with this installation and minimal usage guide,
please use our `forum <https://discuss.pytorch.org/>`_ or `GitHub issues
<https://github.com/pytorch/pytorch/issues>`_ to get in touch.
