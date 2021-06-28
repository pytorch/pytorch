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
   :caption: Notes

   notes/*

.. toctree::
   :maxdepth: 1
   :caption: Language Bindings

   cpp_index
   Javadoc <https://pytorch.org/javadoc/>

.. toctree::
   :maxdepth: 1
   :caption: Python API

   torch
   nn
   nn.functional
   tensors
   tensor_attributes
   tensor_view
   torch.autograd <autograd>
   cuda
   torch.cuda.amp <amp>
   torch.backends <backends>
   torch.distributed <distributed>
   torch.distributed.elastic <distributed.elastic>
   torch.distributed.optim <distributed.optim>
   torch.distributions <distributions>
   torch.fft <fft>
   futures
   fx
   torch.hub <hub>
   torch.jit <jit>
   torch.linalg <linalg>
   torch.special <special>
   torch.overrides
   torch.package <package>
   profiler
   nn.init
   onnx
   optim
   complex_numbers
   ddp_comm_hooks
   pipeline
   quantization
   rpc
   torch.random <random>
   sparse
   storage
   torch.testing <testing>
   torch.utils.benchmark <benchmark_utils>
   torch.utils.bottleneck <bottleneck>
   torch.utils.checkpoint <checkpoint>
   torch.utils.cpp_extension <cpp_extension>
   torch.utils.data <data>
   torch.utils.dlpack <dlpack>
   torch.utils.mobile_optimizer <mobile_optimizer>
   torch.utils.model_zoo <model_zoo>
   torch.utils.tensorboard <tensorboard>
   type_info
   named_tensor
   name_inference
   torch.__config__ <__config__>

.. toctree::
   :maxdepth: 1
   :caption: Libraries

   torchaudio <https://pytorch.org/audio/stable>
   torchtext <https://pytorch.org/text/stable>
   torchvision <https://pytorch.org/vision/stable>
   TorchServe <https://pytorch.org/serve>
   PyTorch on XLA Devices <http://pytorch.org/xla/>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Community

   community/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
