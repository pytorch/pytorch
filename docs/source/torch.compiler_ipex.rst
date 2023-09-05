Intel® Extension for PyTorch* Backend
=====================================

To work better with `torch.compile`, Intel® Extension for PyTorch* implements a backend `ipex`. It targets to improve hardware resource usage efficiency on Intel platforms for better performance.

Currently the `ipex` backend is implemented with two compilation paths: `TorchScript` path, and `TorchInductor` path. The former path uses `torch.jit.trace` and `torch.jit.freeze` to compile models. The latter path uses `inductor` with further customizations designed in Intel® Extension for PyTorch* for the model compilation. While the `inductor` backend is getting mature, the `TorchScript` path is used as the default path for the `ipex` backend. An API, `ipex._set_compiler_backend("inductor")`, is exposed to turn on the experimental TorchInductor path.

Same as the functionality of `TorchScript` and `TorchInductor`, the `TorchScript` path optimizes for inference workloads only, while the `TorchInductor` path optimizes for both training and inference workloads.

Usage Example
~~~~~~~~~~~~~

Inference FP32
--------------

.. code:: python

   import torch
   import torchvision.models as models
   
   model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
   model.eval()
   data = torch.rand(1, 3, 224, 224)
   
   #################### code changes ####################
   import intel_extension_for_pytorch as ipex
   model = ipex.optimize(model, weights_prepack=False)
   
   # Invoke the following line to turn on the TorchInductor path
   ipex._set_compiler_backend("inductor")
   compile_model = torch.compile(model, backend="ipex")
   ######################################################
   
   with torch.no_grad():
       compile_model(data)


Inference BF16
--------------

.. code:: python

   import torch
   import torchvision.models as models
   
   model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
   model.eval()
   data = torch.rand(1, 3, 224, 224)
   
   #################### code changes ####################
   import intel_extension_for_pytorch as ipex
   model = ipex.optimize(model, dtype=torch.bfloat16, weights_prepack=False)
   
   # Invoke the following line to turn on the TorchInductor path
   ipex._set_compiler_backend("inductor")
   compile_model = torch.compile(model, backend="ipex")
   ######################################################
   
   with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16):
       compile_model(data)
