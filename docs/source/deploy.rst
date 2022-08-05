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

    from torch import nn
    from torch.package import PackageExporter

    # Instantiate some model
    model = nn.Sequential(
            nn.Conv2d(1,20,5),
            nn.ReLU(),
            nn.Conv2d(20,64,5),
            nn.ReLU()
        )

    # Package and export it.
    with PackageExporter("my_package.pt") as e:
        e.extern("numpy.**")
        e.save_pickle("model", "model.pkl", model)

Note that since "numpy" was marked as "extern", `torch.package` will
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

    #include <multipy/runtime/deploy.h>
    #include <multipy/runtime/path_environment.h>
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

            std::cout << "Loaded model \n";

            // Create a vector of inputs.
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(torch::ones({1, 1, 10, 10}));

            // Execute the model and turn its output into a tensor.
            at::Tensor output = model(inputs).toTensor();
            std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

        } catch (const c10::Error& e) {
            std::cerr << "error loading the model\n";
            std::cerr << e.msg();
            return -1;
        }
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

Follow the steps at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html to install
nvidia-docker on your system. Then run the following command.

.. code-block:: bash

    docker pull https://hub.docker.com/repository/docker/sahanp465/multipy/tags?page=1&ordering=last_updated

Create the following Dockerfile on your system:

.. code-block:: bash
    FROM sahanp465/multipy:latest as multipy-docker

    COPY . .

    ENV CMAKE_PREFIX_PATH "/opt/conda"
    ENV PATH_TO_EXTERN_PYTHON_PACKAGES "/opt/conda/lib/python3.8/site-packages"
    ENV MULTIPY_DIR_LOCATION "/opt/multipy"

    ENV LD_LIBRARY_PATH "$LD_LIBRARY_PATH:/opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib"
    ENV LIBRARY_PATH "$LIBRARY_PATH:/opt/conda/lib/python3.8/site-packages/torch/lib:/opt/conda/lib"


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

    find_package(Torch REQUIRED)

    include_directories($ENV{MULTIPY_DIR_LOCATION})

    add_library(torch_deploy_internal STATIC IMPORTED)

    set_target_properties(torch_deploy_internal
        PROPERTIES
        IMPORTED_LOCATION
        $ENV{MULTIPY_DIR_LOCATION}/multipy/runtime/lib/libtorch_deploy.a)

    caffe2_interface_library(torch_deploy_internal torch_deploy)

    # add headers from multipy
    add_executable(example-app example.cpp)
    target_link_libraries(example-app PUBLIC
        "-Wl,--no-as-needed -rdynamic"
        shm crypt pthread dl util m ffi lzma readline nsl ncursesw panelw z torch_deploy "${TORCH_LIBRARIES}")

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

    # Point CMake at the built version of PyTorch we just installed.
    cmake -S . -B build/ -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')"

Now we can run our app:

.. code-block:: bash

        ./example-app my_package.pt


Executing ``forward`` in C++
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notably, the model's forward function is executing in Python, in an embedded
CPython interpreter. Note that the model is a ``ReplicatedObj``, which means
that you can call ``model()`` from multiple threads and the forward method will
be executed on multiple independent interpreters, with no global interpreter
lock.
