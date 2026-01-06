:orphan:

C++
===================================
.. Note::
    If you are looking for the PyTorch C++ API docs, directly go `here <https://pytorch.org/cppdocs/>`__.

PyTorch provides several features for working with C++, and it’s best to choose from them based on your needs. At a high level, the following support is available:

Tensor and Autograd in C++
---------------------------
Most of the tensor and autograd operations in PyTorch Python API are also available in the C++ API. These include:

* ``torch::Tensor`` methods such as ``add`` / ``reshape`` / ``clone``. For the full list of methods available, please see: https://pytorch.org/cppdocs/api/classat_1_1_tensor.html
* C++ tensor indexing API that looks and behaves the same as the Python API. For details on its usage, please see: https://pytorch.org/cppdocs/notes/tensor_indexing.html
* The tensor autograd APIs and the ``torch::autograd`` package that are crucial for building dynamic neural networks in C++ frontend. For more details, please see: https://pytorch.org/tutorials/advanced/cpp_autograd.html

Authoring Models in C++
------------------------
We provide the full capability of authoring and training a neural net model purely in C++, with familiar components such as ``torch::nn`` / ``torch::nn::functional`` / ``torch::optim`` that closely resemble the Python API.

* For an overview of the PyTorch C++ model authoring and training API, please see: https://pytorch.org/cppdocs/frontend.html
* For a detailed tutorial on how to use the API, please see: https://pytorch.org/tutorials/advanced/cpp_frontend.html
* Docs for components such as ``torch::nn`` / ``torch::nn::functional`` / ``torch::optim`` can be found at: https://pytorch.org/cppdocs/api/library_root.html


Packaging for C++
------------------
For guidance on how to install and link with libtorch (the library that contains all of the above C++ APIs), please see: https://pytorch.org/cppdocs/installing.html. Note that on Linux there are two types of libtorch binaries provided: one compiled with GCC pre-cxx11 ABI and the other with GCC cxx11 ABI, and you should make the selection based on the GCC ABI your system is using.
