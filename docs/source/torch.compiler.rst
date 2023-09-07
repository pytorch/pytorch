.. _torch.compiler_overview:

torch.compiler
==============

``torch.compiler`` is a namespace through which some of the internal compiler
methods are surfaced for user consumption. The main function and the feature in
this namespace is ``torch.compile``.

``torch.compile`` is a PyTorch function introduced in PyTorch 2.x that aims to
solve the problem of accurate graph capturing in PyTorch and ultimately enable
software engineers to run their PyTorch programs faster. ``torch.compile`` is
written in Python and it marks the transition of PyTorch from C++ to Python.

``torch.compile`` leverages the following underlying technologies:

* **TorchDynamo (torch._dynamo)** is an internal API that uses a CPython
  feature called the Frame Evaluation API to safely capture PyTorch graphs.
  Methods that are available externally for PyTorch users are surfaced
  through the ``torch.compiler`` namespace.

* **TorchInductor** is the default ``torch.compile`` deep learning compiler
  that generates fast code for multiple accelerators and backends. You
  need to use a backend compiler to make speedups through ``torch.compile``
  possible. For NVIDIA and AMD GPUs, it leverages OpenAI Triton as the key
  building block.

* **AOT Autograd** captures not only the user-level code, but also backpropagation,
  which results in capturing the backwards pass "ahead-of-time". This enables
  acceleration of both forwards and backwards pass using TorchInductor.

.. note:: In some cases, the terms ``torch.compile``, TorchDynamo, ``torch.compiler``
   might be used interchangeably in this documentation.

As mentioned above, to run your workflows faster, ``torch.compile`` through
TorchDynamo requires a backend that converts the captured graphs into a fast
machine code. Different backends can result in various optimization gains.
The default backend is called TorchInductor, also known as *inductor*,
TorchDynamo has a list of supported backends developed by our partners,
which can be see by running ``torch.compile.list_backends()`` each of which
with its optional dependencies.

Some of the most commonly used backends include:

**Training & inference backends**

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Backend
     - Description
   * - ``torch.compile(m, backend="inductor")``
     - Uses the TorchInductor backend. `Read more <https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747>`__
   * - ``torch.compile(m, backend="cudagraphs")``
     - CUDA graphs with AOT Autograd. `Read more <https://github.com/pytorch/torchdynamo/pull/757>`__
   * - ``torch.compile(m, backend="ipex")``
     - Uses IPEX on CPU. `Read more <https://github.com/intel/intel-extension-for-pytorch>`__

**Inference-only backends**

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Backend
     - Description
   * - ``torch.compile(m, backend="onnxrt")``
     - Uses ONNXRT for inference on CPU/GPU. `Read more <https://onnxruntime.ai/>`__
   * - ``torch.compile(m, backend="tensorrt")``
     - Uses ONNXRT to run TensorRT for inference optimizations. `Read more <https://github.com/onnx/onnx-tensorrt>`__
   * - ``torch.compile(m, backend="tvm")``
     - Uses Apache TVM for inference optimizations. `Read more <https://tvm.apache.org/>`__

Read More
~~~~~~~~~

.. toctree::
   :caption: Getting Started for PyTorch Users
   :maxdepth: 1

   torch.compiler_get_started
   torch.compiler_api
   torch.compiler_performance_dashboard
   torch.compiler_fine_grain_apis
   torch.compiler_inductor_profiling
   torch.compiler_profiling_torch_compile
   torch.compiler_faq
   torch.compiler_troubleshooting

..
  _If you want to contribute a developer-level topic
   that provides in-depth overview of a torch._dynamo feature,
   add in the below toc.

.. toctree::
   :caption: Deep Dive for PyTorch Developers
   :maxdepth: 1

   torch.compiler_deepdive
   torch.compiler_guards_overview
   torch.compiler_dynamic_shapes
   torch.compiler_nn_module
   torch.compiler_best_practices_for_backends
   torch.compiler_cudagraph_trees
   torch.compiler_fake_tensor

.. toctree::
   :caption: HowTo for PyTorch Backend Vendors
   :maxdepth: 1

   torch.compiler_custom_backends
   torch.compiler_transformations
   torch.compiler_ir
