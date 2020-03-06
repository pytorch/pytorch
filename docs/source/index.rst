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
   PyTorch on XLA Devices <http://pytorch.org/xla/>

.. toctree::
   :maxdepth: 1
   :caption: Language Bindings

   C++ API <https://pytorch.org/cppdocs/>
   packages

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
   torch.distributed <distributed>
   torch.distributions <distributions>
   torch.hub <hub>
   torch.jit <jit>
   nn.init
   onnx
   optim
   quantization
   rpc
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
   named_tensor
   name_inference
   torch.__config__ <__config__>

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: torchvision Reference

   torchvision/index

.. toctree::
   :maxdepth: 1
   :caption: torchaudio Reference

   torchaudio <https://pytorch.org/audio>

.. toctree::
   :maxdepth: 1
   :caption: torchtext Reference

   torchtext <https://pytorch.org/text>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Community

   community/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
