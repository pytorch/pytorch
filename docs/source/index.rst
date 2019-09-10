.. PyTorch documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/pytorch/pytorch

PyTorch documentation
===================================

PyTorch is an optimized tensor library for deep learning using GPUs and CPUs.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Notes

   notes/*

.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Community

  community/*

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   qat.modules
   quantized.modules
   torch.nn.qat
   quantization
   quantized
   torch
   nn.functional
   tensors
   tensor_attributes
   torch.autograd <autograd>
   cuda
   torch.distributed <distributed>
   torch.distributions <distributions>
   torch.hub <hub>
   torch.jit <jit>
   nn.init
   onnx
   optim
   torch.random <random>
   sparse
   storage
   torch.utils.bottleneck <bottleneck>
   torch.utils.checkpoint <checkpoint>
   torch.utils.cpp_extension <cpp_extension>
   torch.utils.data <data>
   torch.utils.dlpack <dlpack>
   torch.utils.model_zoo <model_zoo>
   torch.utils.tensorboard <tensorboard>
   type_info
   torch.__config__ <__config__>

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: torchvision Reference

   torchvision/index

* `torchaudio <https://pytorch.org/audio>`_

* `torchtext <https://pytorch.org/text>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
