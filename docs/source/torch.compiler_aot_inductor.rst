

AOTInductor: Ahead-Of-Time Compilation for Torch.Export-ed Models
=================================================================

.. warning::

    AOTInductor and its related features are in prototype status and are
    subject to backwards compatibility breaking changes.

AOTInductor is a specialized version of
`TorchInductor <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
which takes an exported PyTorch model, optimizes it, and generates a shared
library. These compiled artifacts can be deployed to non-Python environments,
which are commonly used for inference deployments. AOTInductor is a vital component along the
`export <https://pytorch.org/docs/main/export.html>`__ path as it provides a way
to run an exported model without a Python runtime.

In this tutorial, you will learn how to take a PyTorch model, export and compile into a shared library,
and run the model prediction in C++.


Model Compilation
---------------------------

With AOTInductor, the model is authored in Python. The following is an example model which shows how
to call ``aot_compile`` to compile it into a shared library.

.. note::

    To run the following script, you need to have at least one CUDA device on your machine.
    If you do not have a GPU, you can remove the ``.to(device="cuda")`` code
    in the snippet below and it will generate the model code into a shared library that runs on CPU.

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

In this example, ``Dim`` is used to specify the first dimension of input "x" as dynamic.
The compiled library path and name are not specified, so the compiled shared library will
be stored in a temporay directory. To use that path on the C++ side, we write that path into a file which will
be read in C++ later. The exact ``torch._export.aot_compile`` API can be found
`here <https://github.com/pytorch/pytorch/blob/92cc52ab0e48a27d77becd37f1683fd442992120/torch/_export/__init__.py#L891-L900C9>`__.


Inference in C++
---------------------------

Next, we write the following C++ file ``inference.cpp`` to load ``model.so`` generated from the last step,
and perform model prediction in C++.

.. note::

    Again, you need to have at least one CUDA device to run the following code. If you don't have a GPU,
    you need to make these changes to run it on CPU: change ``aoti_model_runner_cuda.h`` to ``aoti_model_runner.h``;
    change ``AOTIModelRunnerCuda`` to ``AOTIModelRunner``; remove ``.to(at::kCUDA)`` from input tensor initializations.


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


To build the cpp file, you can use the following CMakeLists.txt file, which takes care of invoking
``python model.py`` to AOT compile the model and compiling ``inference.cpp`` into a binary, ``aot_inductor_example``

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


.. code-block:: shell

    aot_inductor_example/
        CMakeLists.txt
        inference.cpp
        model.py


Assuming this is how the directory structure looks like, you can run the following commands
to build and run the binary. Note that ``CMAKE_PREFIX_PATH`` is required for ``cmake`` for find
``libtorch``, and it is required to use an absolute path. Your path may be different from the one used in this example.

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
