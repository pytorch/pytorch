.. PyTorch documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/pytorch/pytorch

PyTorch documentation
===================================

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

Features described in this documentation are classified by release status:

  *Stable:*  These features will be maintained long-term and there should generally
  be no major performance limitations or gaps in documentation.
  We also expect to maintain backwards compatibility (although
  breaking changes can happen and notice will be given one release ahead
  of time).

  *Beta:*  These features are tagged as Beta because the API may change based on
  user feedback, because the performance needs to improve, or because
  coverage across operators is not yet complete. For Beta features, we are
  committing to seeing the feature through to the Stable classification.
  We are not, however, committing to backwards compatibility.

  *Prototype:*  These features are typically not available as part of
  binary distributions like PyPI or Conda, except sometimes behind run-time
  flags, and are at an early stage for feedback and testing.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Community

   community/*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Developer Notes

   notes/*

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: torch.compile

   compile/index
   compile/get-started
   compile/troubleshooting
   compile/faq
   compile/technical-overview
   compile/guards-overview
   compile/custom-backends
   compile/fine_grained_apis
   compile/profiling_torch_compile
   compile/inductor_profiling
   compile/deep-dive
   compile/cudagraph_trees
   compile/performance-dashboard
   compile/torchfunc-and-torchcompile
   ir
   compile/dynamic-shapes
   compile/fake-tensor
   logging
   compile/transformations

.. toctree::
   :maxdepth: 1
   :caption: Language Bindings

   cpp_index
   Javadoc <https://pytorch.org/javadoc/>
   torch::deploy <deploy>

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Python API

   torch
   nn
   nn.functional
   tensors
   tensor_attributes
   tensor_view
   torch.amp <amp>
   torch.autograd <autograd>
   torch.library <library>
   cuda
   mps
   torch.backends <backends>
   export
   torch.distributed <distributed>
   torch.distributed.algorithms.join <distributed.algorithms.join>
   torch.distributed.elastic <distributed.elastic>
   torch.distributed.fsdp <fsdp>
   torch.distributed.optim <distributed.optim>
   torch.distributed.tensor.parallel <distributed.tensor.parallel>
   torch.distributed.checkpoint <distributed.checkpoint>
   torch.distributions <distributions>
   torch.compiler <compiler>
   torch.fft <fft>
   torch.func <func>
   futures
   fx
   torch.hub <hub>
   torch.jit <jit>
   torch.linalg <linalg>
   torch.monitor <monitor>
   torch.signal <signal>
   torch.special <special>
   torch.overrides
   torch.package <package>
   profiler
   nn.init
   onnx
   onnx_diagnostics
   optim
   complex_numbers
   ddp_comm_hooks
   pipeline
   quantization
   rpc
   torch.random <random>
   masked
   torch.nested <nested>
   sparse
   storage
   torch.testing <testing>
   torch.utils.benchmark <benchmark_utils>
   torch.utils.bottleneck <bottleneck>
   torch.utils.checkpoint <checkpoint>
   torch.utils.cpp_extension <cpp_extension>
   torch.utils.data <data>
   torch.utils.jit <jit_utils>
   torch.utils.dlpack <dlpack>
   torch.utils.mobile_optimizer <mobile_optimizer>
   torch.utils.model_zoo <model_zoo>
   torch.utils.tensorboard <tensorboard>
   type_info
   named_tensor
   name_inference
   torch.__config__ <config_mod>
   logging

.. toctree::
   :maxdepth: 1
   :caption: Libraries

   torchaudio <https://pytorch.org/audio/stable>
   TorchData <https://pytorch.org/data>
   TorchRec <https://pytorch.org/torchrec>
   TorchServe <https://pytorch.org/serve>
   torchtext <https://pytorch.org/text/stable>
   torchvision <https://pytorch.org/vision/stable>
   PyTorch on XLA Devices <https://pytorch.org/xla/>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
