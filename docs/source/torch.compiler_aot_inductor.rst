

AOTInductor: Ahead-Of-Time Compilation for Torch.Export-ed Models
=================================================================

.. warning::

    AOTInductor and its related features are in prototype status and are
    subject to backwards compatibility breaking changes.

AOTInductor is a specialized version of
`TorchInductor <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
, designed to process exported PyTorch models, optimize them, and produce shared libraries as well
as other relevant artifacts.
These compiled artifacts are specifically crafted for deployment in non-Python environments,
which are frequently employed for inference deployments. AOTInductor plays a pivotal role in the
`export <https://pytorch.org/docs/main/export.html>`__ path, offering a means to execute an exported
model independently of a Python runtime.

In this tutorial, you will gain insight into the process of taking a PyTorch model, exporting it,
compiling it into a shared library, and conducting model predictions using C++.


Model Compilation
---------------------------

Using AOTInductor, you can still author the model in Python. The following example demonstrates how to
invoke ``aot_compile`` to transform the model into a shared library.

.. note::

   To execute the following code, it's essential to have a CUDA-enabled device on your machine.
   If you do not possess a GPU, you can simply omit the ``.to(device="cuda")`` code within the snippet
   below. In such a case, the script will compile the model code into a shared library that is optmized
   for CPU execution.

.. code-block:: python

    import torch

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 16)
            self.relu = torch.nn.ReLU()
            self.fc2 = torch.nn.Linear(16, 1)
            self.sigmoid = torch.nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x

    with torch.no_grad():
        m = Model().to("cuda")
        example_inputs=(torch.randn(8, 10).to("cuda"),)
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        dynamic_shapes = {"x": {0: batch_dim}}
        so_path = torch._export.aot_compile(m, example_inputs, dynamic_shapes=dynamic_shapes)

    print(f"Compiled model into: {so_path}")
    with open("model_so_path.txt", "w") as file:
        file.write(so_path)

In this illustrative example, the ``Dim`` parameter is employed to designate the first dimension of
the input variable "x" as dynamic. Notably, the path and name of the compiled library remain unspecified,
resulting in the shared library being stored in a temporay directory.
To access this path from the C++ side, we save it to a file for later retrieval within the C++ code.
For comprehensive details on the 'torch._export.aot_compile' API,
you can refer to the code
`here <https://github.com/pytorch/pytorch/blob/92cc52ab0e48a27d77becd37f1683fd442992120/torch/_export/__init__.py#L891-L900C9>`__.


Inference in C++
---------------------------

Next, we use the following C++ file ``inference.cpp`` to load the shared library generated in the
previous step, enabling us to conduct model predictions directly within a C++ environment.

.. note::

    Once more, it's imperative to have a CUDA-enabled device to execute the subsequent code.
    In the absence of a GPU, it's necessary to make these adjustments in order to run it on a CPU:
    1. Modify ``aoti_model_runner_cuda.h`` to ``aoti_model_runner.h``.
    2. Change ``AOTIModelRunnerCuda`` to ``AOTIModelRunner``.
    3. Eliminate ``.to(at::kCUDA)`` from the initialization of input tensors.

.. code-block:: cpp

    #include <fstream>
    #include <iostream>
    #include <vector>
    #include <torch/torch.h>
    #include <torch/csrc/inductor/aoti_model_runner_cuda.h>

    int main() {
        torch::NoGradGuard no_grad;

        std::ifstream path_file("model_so_path.txt");
        if (!path_file.is_open()) {
            std::cerr << "Error: Unable to open model_so_path.txt." << std::endl;
            return 1;
        }
        std::string model_so;
        if (!std::getline(path_file, model_so)) {
            std::cerr << "Error: File is empty." << std::endl;
        }
        path_file.close();

        torch::inductor::AOTIModelRunnerCuda runner(model_so.c_str());
        std::vector<torch::Tensor> inputs = {torch::randn({8, 10}).to(at::kCUDA)};
        std::vector<torch::Tensor> outputs = runner.run(inputs);
        std::cout << "Result from first inference:"<< std::endl;
        std::cout << outputs[0] << std::endl;

        std::cout << "Result from second inference:"<< std::endl;
        std::cout << runner.run({torch::randn({2, 10}).to(at::kCUDA)})[0] << std::endl;
        return 0;
    }

For building the C++ file, you can make use of the provided ``CMakeLists.txt`` file, which
automates the process of invoking ``python model.py`` for AOT compilation of the model and compiling
``inference.cpp`` into an executable binary named ``aot_inductor_example``.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
    project(aot_inductor_example)

    find_package(Torch REQUIRED)

    add_executable(aot_inductor_example inference.cpp model_so_path.txt)

    add_custom_command(
        OUTPUT model_so_path.txt
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/model.py
        DEPENDS model.py
    )

    target_link_libraries(aot_inductor_example "${TORCH_LIBRARIES}")
    set_property(TARGET aot_inductor_example PROPERTY CXX_STANDARD 17)


Provided the directory structure resembles the following, you can execute the subsequent commands
to construct the binary. It is essential to note that the ``CMAKE_PREFIX_PATH`` variable
is crucial for CMake to locate the LibTorch library, and it should be set to an absolute path.
Please be mindful that your path may vary from the one illustrated in this example.

.. code-block:: shell

    aot_inductor_example/
        CMakeLists.txt
        inference.cpp
        model.py


.. code-block:: shell

    (nightly) [ ~/local/aot_inductor_example]$ mkdir build
    (nightly) [ ~/local/aot_inductor_example]$ cd build

    (nightly) [ ~/local/aot_inductor_example/build]$ CMAKE_PREFIX_PATH=/home/userid/local/miniconda3/envs/nightly/lib/python3.10/site-packages/torch/share/cmake cmake ..
    -- The C compiler identification is GNU 11.4.1
    -- The CXX compiler identification is GNU 11.4.1
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working C compiler: /home/userid/local/ccache/lib/cc - skipped
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Check for working CXX compiler: /home/userid/local/ccache/lib/c++ - skipped
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Found CUDA: /usr/local/cuda-12.1 (found version "12.1")
    -- The CUDA compiler identification is NVIDIA 12.1.105
    -- Detecting CUDA compiler ABI info
    -- Detecting CUDA compiler ABI info - done
    -- Check for working CUDA compiler: /usr/local/cuda-12.1/bin/nvcc - skipped
    -- Detecting CUDA compile features
    -- Detecting CUDA compile features - done
    -- Found CUDAToolkit: /usr/local/cuda-12.1/include (found version "12.1.105")
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Performing Test CMAKE_HAVE_LIBC_PTHREAD
    -- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Success
    -- Found Threads: TRUE
    -- Caffe2: CUDA detected: 12.1
    -- Caffe2: CUDA nvcc is: /home/userid/local/ccache/cuda/nvcc
    -- Caffe2: CUDA toolkit directory: /usr/local/cuda-12.1
    -- Caffe2: Header version is: 12.1
    -- /usr/local/cuda-12.1/lib64/libnvrtc.so shorthash is b51b459d
    -- USE_CUDNN is set to 0. Compiling without cuDNN support
    -- USE_CUSPARSELT is set to 0. Compiling without cuSPARSELt support
    -- Autodetected CUDA architecture(s):  8.0 8.0 8.0 8.0 8.0 8.0 8.0 8.0
    -- Added CUDA NVCC flags for: -gencode;arch=compute_80,code=sm_80
    -- Found Torch: /home/userid/local/miniconda3/envs/nightly/lib/python3.10/site-packages/torch/lib/libtorch.so
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /home/userid/local/aot_inductor_example/build

    (nightly) [ ~/local/aot_inductor_example/build]$ cmake --build . --config Release
    [ 33%] Generating model_so_path.txt
    Compiled model into: /tmp/torchinductor_userid/csnavcwn65mvhieu3jsqd2xkbzynhq2qif7rthy5l57qca3e4wwe/c64ucf56t5hvtulhodvu47apn2rcdhjre7ifghsuwovvniggmwd7.so
    [ 66%] Building CXX object CMakeFiles/aot_inductor_example.dir/inference.cpp.o
    [100%] Linking CXX executable aot_inductor_example
    [100%] Built target aot_inductor_example

After the ``aot_inductor_example`` binary has been generated in the ``build`` directory, executing it will
display results akin to the following:

.. code-block:: shell

    (nightly) [ ~/local/aot_inductor_example/build]$ ./aot_inductor_example
    Result from first inference:
    0.4866
    0.5184
    0.4462
    0.4611
    0.4744
    0.4811
    0.4938
    0.4193
    [ CUDAFloatType{8,1} ]
    Result from second inference:
    0.4883
    0.4703
    [ CUDAFloatType{2,1} ]
