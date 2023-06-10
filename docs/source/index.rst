.. PyTorch documentation master file, created by
   sphinx-quickstart on Fri Dec 23 13:31:47 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/pytorch/pytorch

PyTorch documentation
===================================

PyTorch is a powerful library used for deep learning on GPUs and CPUs. 
In this documentation, the features are divided into different release 
statuses to provide clarity on their stability and compatibility.

Stable: These features are well-established and will be supported in the long term. 
They have comprehensive documentation and are optimized for performance. 
While minor updates and bug fixes may occur, there are generally no major limitations. 
Backwards compatibility is maintained, but there may be rare instances where 
breaking changes are necessary. In such cases, advance notice will be 
given in the release documentation.

Beta: Features in the Beta stage have undergone initial development 
and are available for use. However, the API may still evolve based on user feedback 
or performance improvements. The coverage across different operators or functionalities 
might not be complete. Although these features are intended to become Stable in the future, 
they do not guarantee backwards compatibility. Users should be aware that changes 
may occur before the final Stable release.

Prototype: Prototype features are in the early stages of development and testing. 
They are not typically included in standard binary distributions like PyPI or Conda, 
except in some cases where they may be accessible through runtime flags. 
Prototypes are made available for feedback and experimentation, allowing users 
to provide input and report issues. These features are not considered production-ready 
and are subject to significant changes before they reach Stable status.

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
   torch.distributed <distributed>
   torch.distributed.algorithms.join <distributed.algorithms.join>
   torch.distributed.elastic <distributed.elastic>
   torch.distributed.fsdp <fsdp>
   torch.distributed.optim <distributed.optim>
   torch.distributed.tensor.parallel <distributed.tensor.parallel>
   torch.distributed.checkpoint <distributed.checkpoint>
   torch.distributions <distributions>
   torch._dynamo <_dynamo>
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
