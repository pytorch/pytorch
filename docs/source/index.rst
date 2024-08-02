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
   cpu
   cuda
   torch.cuda.memory <torch_cuda_memory>
   mps
   xpu
   mtia
   meta
   torch.backends <backends>
   torch.export <export>
   torch.distributed <distributed>
   torch.distributed.algorithms.join <distributed.algorithms.join>
   torch.distributed.elastic <distributed.elastic>
   torch.distributed.fsdp <fsdp>
   torch.distributed.optim <distributed.optim>
   torch.distributed.pipelining <distributed.pipelining>
   torch.distributed.tensor.parallel <distributed.tensor.parallel>
   torch.distributed.checkpoint <distributed.checkpoint>
   torch.distributions <distributions>
   torch.compiler <torch.compiler>
   torch.fft <fft>
   torch.func <func>
   futures
   fx
   fx.experimental
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
   nn.attention
   onnx
   optim
   complex_numbers
   ddp_comm_hooks
   quantization
   rpc
   torch.random <random>
   masked
   torch.nested <nested>
   size
   sparse
   storage
   torch.testing <testing>
   torch.utils <utils>
   torch.utils.benchmark <benchmark_utils>
   torch.utils.bottleneck <bottleneck>
   torch.utils.checkpoint <checkpoint>
   torch.utils.cpp_extension <cpp_extension>
   torch.utils.data <data>
   torch.utils.deterministic <deterministic>
   torch.utils.jit <jit_utils>
   torch.utils.dlpack <dlpack>
   torch.utils.mobile_optimizer <mobile_optimizer>
   torch.utils.model_zoo <model_zoo>
   torch.utils.tensorboard <tensorboard>
   torch.utils.module_tracker <module_tracker>
   type_info
   named_tensor
   name_inference
   torch.__config__ <config_mod>
   torch.__future__ <future_mod>
   logging
   torch_environment_variables

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
