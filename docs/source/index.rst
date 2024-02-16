.. PyTorch documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/pytorch/pytorch

PyTorch documentation
===================================

PyTorch is a tensor library optimized for deep learning on GPUs and CPUs. 
Features in this documentation are classified by release status: 

            Stable, Beta, and Prototype. 

- **Stable**: These features are well-established and will be maintained long-term with comprehensive documentation and no major performance limitations.
 Backwards compatibility is generally maintained, with notice given for any breaking changes.

- **Beta**: Beta features are still in development, and their API may change based on user feedback.
 Performance improvements and broader operator coverage are often goals for Beta features.
  While there is a commitment to moving Beta features to Stable status, backwards compatibility is not guaranteed.

- **Prototype**: Prototype features are in early stages and may not be included in binary distributions like PyPI or Conda, except possibly behind runtime flags.
 They are available for feedback and testing to refine their functionality and usability.

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
   meta
   torch.backends <backends>
   torch.export <export>
   torch.distributed <distributed>
   torch.distributed.algorithms.join <distributed.algorithms.join>
   torch.distributed.elastic <distributed.elastic>
   torch.distributed.fsdp <fsdp>
   torch.distributed.optim <distributed.optim>
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
   pipeline
   quantization
   rpc
   torch.random <random>
   masked
   torch.nested <nested>
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
