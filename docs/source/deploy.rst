torch::deploy
=============

``torch::deploy`` is a system that allows you to run multiple embedded Python
interpreters in a C++ process without a shared global interpreter lock. For more
information on how ``torch::deploy`` works internally, please see the related
`arXiv paper <https://arxiv.org/pdf/2104.00254.pdf>`_.


.. warning::

    This is a prototype feature. Only Linux x86 is supported, and the API may
    change without warning.


Getting Started
---------------

Installing ``torch::deploy``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch::deploy`` is not yet built by default in our binary releases, so to get
a copy of libtorch with ``torch::deploy`` enabled, follow the instructions for
`building PyTorch from source <https://github.com/pytorch/pytorch/#from-source>`_.

When running ``setup.py``, you will need to specify ``USE_DEPLOY=1``, like:

.. code-block:: bash

    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
    export USE_DEPLOY=1
    python setup.py develop


Creating a model package in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``torch::deploy`` can load and run Python models that are packaged with
``torch.package``. You can learn more about ``torch.package`` in the
``torch.package`` `documentation <https://pytorch.org/docs/stable/package.html#tutorials>`_.

For now, let's create a simple model that we can load and run in ``torch::deploy``.

.. code-block:: py

    from torch.package import PackageExporter
    import torchvision

    # Instantiate some model
    model = torchvision.models.resnet.resnet18()

    # Package and export it.
    with PackageExporter("my_package.pt") as e:
        e.intern("torchvision.**")
        e.extern("numpy.**")
        e.extern("sys")
        e.extern("PIL.*")
        e.save_pickle("model", "model.pkl", model)

Note that since "numpy", "sys" and "PIL" were marked as "extern", `torch.package` will
look for these dependencies on the system that loads this package. They will not be packaged
with the model.

Now, there should be a file named ``my_package.pt`` in your working directory.


Loading and running the model in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set an environment variable (e.g. $PATH_TO_EXTERN_PYTHON_PACKAGES) to indicate to the interpreters
where the external Python dependencies can be found. In the example below, the path to the
site-packages of a conda environment is provided.

.. code-block:: bash

    export PATH_TO_EXTERN_PYTHON_PACKAGES= \
        "~/anaconda/envs/deploy-example-env/lib/python3.8/site-packages"


Let's create a minimal C++ program to that loads the model.

.. code-block:: cpp

    #include <torch/csrc/deploy/deploy.h>
    #include <torch/csrc/deploy/path_environment.h>
    #include <torch/script.h>
    #include <torch/torch.h>

    #include <iostream>
    #include <memory>

    int main(int argc, const char* argv[]) {
        if (argc != 2) {
            std::cerr << "usage: example-app <path-to-exported-script-module>\n";
            return -1;
        }

        // Start an interpreter manager governing 4 embedded interpreters.
        std::shared_ptr<torch::deploy::Environment> env =
            std::make_shared<torch::deploy::PathEnvironment>(
                std::getenv("PATH_TO_EXTERN_PYTHON_PACKAGES")
            );
        torch::deploy::InterpreterManager manager(4, env);

        try {
            // Load the model from the torch.package.
            torch::deploy::Package package = manager.loadPackage(argv[1]);
            torch::deploy::ReplicatedObj model = package.loadPickle("model", "model.pkl");
        } catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            std::cerr << e.msg();
            return -1;
        }

        std::cout << "ok\n";
    }

This small program introduces many of the core concepts of ``torch::deploy``.

An ``InterpreterManager`` abstracts over a collection of independent Python
interpreters, allowing you to load balance across them when running your code.

``PathEnvironment`` enables you to specify the location of Python
packages on your system which are external, but necessary, for your model.

Using the ``InterpreterManager::loadPackage`` method, you can load a
``torch.package`` from disk and make it available to all interpreters.

``Package::loadPickle`` allows you to retrieve specific Python objects
from the package, like the ResNet model we saved earlier.

Finally, the model itself is a ``ReplicatedObj``. This is an abstract handle to
an object that is replicated across multiple interpreters. When you interact
with a ``ReplicatedObj`` (for example, by calling ``forward``), it will select
an free interpreter to execute that interaction.


Building and running the application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Locate `libtorch_deployinterpreter.o` on your system. This should have been
built when PyTorch was built from source. In the same PyTorch directory, locate
the deploy source files. Set these locations to an environment variable for the build.
An example of where these can be found on a system is shown below.

.. code-block:: bash

    export DEPLOY_INTERPRETER_PATH="/pytorch/build/torch/csrc/deploy/"
    export DEPLOY_SRC_PATH="/pytorch/torch/csrc/deploy/"

As ``torch::deploy`` is in active development, these manual steps will be removed
soon.

Assuming the above C++ program was stored in a file called, `example-app.cpp`, a
minimal CMakeLists.txt file would look like:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
    project(deploy_tutorial)

    find_package(fmt REQUIRED)
    find_package(Torch REQUIRED)

    add_library(torch_deploy_internal STATIC
        ${DEPLOY_INTERPRETER_PATH}/libtorch_deployinterpreter.o
        ${DEPLOY_DIR}/deploy.cpp
        ${DEPLOY_DIR}/loader.cpp
        ${DEPLOY_DIR}/path_environment.cpp
        ${DEPLOY_DIR}/elf_file.cpp)

    # for python builtins
    target_link_libraries(torch_deploy_internal PRIVATE
        crypt pthread dl util m z ffi lzma readline nsl ncursesw panelw)
    target_link_libraries(torch_deploy_internal PUBLIC
        shm torch fmt::fmt-header-only)
    caffe2_interface_library(torch_deploy_internal torch_deploy)

    add_executable(example-app example.cpp)
    target_link_libraries(example-app PUBLIC
        "-Wl,--no-as-needed -rdynamic" dl torch_deploy "${TORCH_LIBRARIES}")

Currently, it is necessary to build ``torch::deploy`` as a static library.
In order to correctly link to a static library, the utility ``caffe2_interface_library``
is used to appropriately set and unset ``--whole-archive`` flag.

Furthermore, the ``-rdynamic`` flag is needed when linking to the executable
to ensure that symbols are exported to the dynamic table, making them accessible
to the deploy interpreters (which are dynamically loaded).

The last step is configuring and building the project. Assuming that our code
directory is laid out like this:

.. code-block:: none

    example-app/
        CMakeLists.txt
        example-app.cpp

We can now run the following commands to build the application from within the
``example-app/`` folder:

.. code-block:: bash

    mkdir build
    cd build
    # Point CMake at the built version of PyTorch we just installed.
    cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" .. \
        -DDEPLOY_INTERPRETER_PATH="$DEPLOY_INTERPRETER_PATH" \
        -DDEPLOY_DIR="$DEPLOY_DIR"
    cmake --build . --config Release

Now we can run our app:

.. code-block:: bash

        ./example-app /path/to/my_package.pt


Executing ``forward`` in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One you have your model loaded in C++, it is easy to execute it:

.. code-block:: cpp

    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({1, 3, 224, 224}));

    // Execute the model and turn its output into a tensor.
    at::Tensor output = model(inputs).toTensor();
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

Notably, the model's forward function is executing in Python, in an embedded
CPython interpreter. Note that the model is a ``ReplicatedObj``, which means
that you can call ``model()`` from multiple threads and the forward method will
be executed on multiple independent interpreters, with no global interpreter
lock.
