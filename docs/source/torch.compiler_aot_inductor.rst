

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
which are frequently employed for inference deployments on the server side.

In this tutorial, you will gain insight into the process of taking a PyTorch model, exporting it,
compiling it into a shared library, and conducting model predictions using C++.


Model Compilation
---------------------------

Using AOTInductor, you can still author the model in Python. The following
example demonstrates how to invoke ``aoti_compile_and_package`` to transform the model into a
shared library.

This API uses ``torch.export.export`` to capture the model into a computational graph,
and then uses TorchInductor to generate a .so which can be run in a non-Python
environment.  For comprehensive details on the
``torch._inductor.aoti_compile_and_package``
API, you can refer to the code
`here <https://github.com/pytorch/pytorch/blob/6ed237e5b528e3b01a7f1b6366b009dc6f30e6d6/torch/_inductor/__init__.py#L38-L105>`__.
For more details on ``torch.export.export``, you can refer to the :ref:`torch.export docs <torch.export>`.

.. note::

   If you have a CUDA-enabled device on your machine and you installed PyTorch with CUDA support,
   the following code will compile the model into a shared library for CUDA execution.
   Otherwise, the compiled artifact will run on CPU. For better performance during CPU inference,
   it is suggested to enable freezing by setting `export TORCHINDUCTOR_FREEZING=1`
   before running the Python script below.

.. code-block:: python

    import os
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
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Model().to(device=device)
        example_inputs=(torch.randn(8, 10, device=device),)
        batch_dim = torch.export.Dim("batch", min=1, max=1024)
        # [Optional] Specify the first dimension of the input x as dynamic.
        exported = torch.export.export(model, example_inputs, dynamic_shapes={"x": {0: batch_dim}})
        # [Note] In this example we directly feed the exported module to aoti_compile_and_package.
        # Depending on your use case, e.g. if your training platform and inference platform
        # are different, you may choose to save the exported model using torch.export.save and
        # then load it back using torch.export.load on your inference platform to run AOT compilation.
        output_path = torch._inductor.aoti_compile_and_package(
            exported,
            # [Optional] Specify the generated shared library path. If not specified,
            # the generated artifact is stored in your system temp directory.
            package_path=os.path.join(os.getcwd(), "model.pt2"),
        )


In this illustrative example, the ``Dim`` parameter is employed to designate the first dimension of
the input variable "x" as dynamic. Notably, the path and name of the compiled library remain unspecified,
resulting in the shared library being stored in a temporary directory.
To access this path from the C++ side, we save it to a file for later retrieval within the C++ code.


Inference in Python
---------------------------
There are multiple ways to deploy the compiled artifact for inference, and one of that is using Python.
We have provided a convinient utility API in Python ``torch._inductor.aoti_load_package`` for loading
and running the artifact, as shown in the following example:

.. code-block:: python

    import os
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch._inductor.aoti_load_package(os.path.join(os.getcwd(), "model.pt2"))
    print(model(torch.randn(8, 10, device=device)))


Inference in C++
---------------------------

Next, we use the following example C++ file ``inference.cpp`` to load the compiled artifact,
enabling us to conduct model predictions directly within a C++ environment.

.. code-block:: cpp

    #include <iostream>
    #include <vector>

    #include <torch/torch.h>
    #include <torch/csrc/inductor/aoti_package/model_package_loader.h>

    int main() {
        c10::InferenceMode mode;

        torch::inductor::AOTIModelPackageLoader loader("model.pt2");
        torch::inductor::AOTIModelContainerRunner* runner = loader.get_runner();
        // Assume running on CUDA
        std::vector<torch::Tensor> inputs = {torch::randn({8, 10}, at::kCUDA)};
        std::vector<torch::Tensor> outputs = runner->run(inputs);
        std::cout << "Result from the first inference:"<< std::endl;
        std::cout << outputs[0] << std::endl;

        // The second inference uses a different batch size and it works because we
        // specified that dimension as dynamic when compiling model.pt2.
        std::cout << "Result from the second inference:"<< std::endl;
        // Assume running on CUDA
        std::cout << runner->run({torch::randn({1, 10}, at::kCUDA)})[0] << std::endl;

        return 0;
    }

For building the C++ file, you can make use of the provided ``CMakeLists.txt`` file, which
automates the process of invoking ``python model.py`` for AOT compilation of the model and compiling
``inference.cpp`` into an executable binary named ``aoti_example``.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
    project(aoti_example)

    find_package(Torch REQUIRED)

    add_executable(aoti_example inference.cpp model.pt2)

    add_custom_command(
        OUTPUT model.pt2
        COMMAND python ${CMAKE_CURRENT_SOURCE_DIR}/model.py
        DEPENDS model.py
    )

    target_link_libraries(aoti_example "${TORCH_LIBRARIES}")
    set_property(TARGET aoti_example PROPERTY CXX_STANDARD 17)


Provided the directory structure resembles the following, you can execute the subsequent commands
to construct the binary. It is essential to note that the ``CMAKE_PREFIX_PATH`` variable
is crucial for CMake to locate the LibTorch library, and it should be set to an absolute path.
Please be mindful that your path may vary from the one illustrated in this example.

.. code-block:: shell

    aoti_example/
        CMakeLists.txt
        inference.cpp
        model.py


.. code-block:: shell

    $ mkdir build
    $ cd build
    $ CMAKE_PREFIX_PATH=/path/to/python/install/site-packages/torch/share/cmake cmake ..
    $ cmake --build . --config Release

After the ``aoti_example`` binary has been generated in the ``build`` directory, executing it will
display results akin to the following:

.. code-block:: shell

    $ ./aoti_example
    Result from the first inference:
    0.4866
    0.5184
    0.4462
    0.4611
    0.4744
    0.4811
    0.4938
    0.4193
    [ CUDAFloatType{8,1} ]
    Result from the second inference:
    0.4883
    0.4703
    [ CUDAFloatType{2,1} ]


Troubleshooting
---------------------------
Below are some useful tools for debugging AOT Inductor.

.. toctree::
   :caption: Debugging Tools
   :maxdepth: 1

   logging
   torch.compiler_aot_inductor_minifier
