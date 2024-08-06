Pytorch 2.4: Getting Started on Intel GPU
=========================================

The support for Intel GPUs is released alongside PyTorch v2.4.

This release only supports build from source for Intel GPUs.

Hardware Prerequisites
----------------------

.. list-table::
   :header-rows: 1

   * - Supported Hardware
     - Intel® Data Center GPU Max Series
   * - Supported OS
     - Linux


PyTorch for Intel GPUs is compatible with Intel® Data Center GPU Max Series and only supports OS Linux with release 2.4.

Software Prerequisites
----------------------

As a prerequisite, install the driver and required packages by following the `PyTorch Installation Prerequisites for Intel GPUs <https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html>`_.

Set up Environment
------------------

Before you begin, you need to set up the environment. This can be done by sourcing the ``setvars.sh`` script provided by the ``intel-for-pytorch-gpu-dev`` and  ``intel-pti-dev`` packages.

.. code-block::

   source ${ONEAPI_ROOT}/setvars.sh

.. note::
   The ``ONEAPI_ROOT`` is the folder you installed your ``intel-for-pytorch-gpu-dev`` and  ``intel-pti-dev`` packages. Typically, it is located at ``/opt/intel/oneapi/`` or ``~/intel/oneapi/``.

Build from source
-----------------

Now we have all the required packages installed and environment acitvated. Use the following commands to install ``pytorch``, ``torchvision``, ``torchaudio`` by building from source. For more details, refer to official guides in `PyTorch from source <https://github.com/pytorch/pytorch?tab=readme-ov-file#intel-gpu-support>`_, `Vision from source <https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation>`_ and `Audio from source <https://pytorch.org/audio/main/build.linux.html>`_.

.. code-block::

   # Get PyTorch Source Code
   git clone --recursive https://github.com/pytorch/pytorch
   cd pytorch
   git checkout main # or checkout the specific release version >= v2.4
   git submodule sync
   git submodule update --init --recursive

   # Get required packages for compilation
   conda install cmake ninja
   pip install -r requirements.txt

   # Pytorch for Intel GPUs only support Linux platform for now.
   # Install the required packages for pytorch compilation.
   conda install intel::mkl-static intel::mkl-include

   # (optional) If using torch.compile with inductor/triton, install the matching version of triton
   # Run from the pytorch directory after cloning
   # For Intel GPU support, please explicitly `export USE_XPU=1` before running command.
   USE_XPU=1 make triton

   # If you would like to compile PyTorch with new C++ ABI enabled, then first run this command:
   export _GLIBCXX_USE_CXX11_ABI=1

   # pytorch build from source
   export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
   python setup.py develop
   cd ..

   # (optional) If using torchvison.
   # Get torchvision Code
   git clone https://github.com/pytorch/vision.git
   cd vision
   git checkout main # or specific version
   python setup.py develop
   cd ..

   # (optional) If using torchaudio.
   # Get torchaudio Code
   git clone https://github.com/pytorch/audio.git
   cd audio
   pip install -r requirements.txt
   git checkout main # or specific version
   git submodule sync
   git submodule update --init --recursive
   python setup.py develop
   cd ..

Check availability for Intel GPU
--------------------------------

.. note::
   Make sure the environment is properly set up by following `Environment Set up <#set-up-environment>`_ before running the code.

To check if your Intel GPU is available, you would typically use the following code:

.. code-block::

   import torch
   torch.xpu.is_available()  # torch.xpu is the API for Intel GPU support

If the output is ``False``, ensure that you have Intel GPU in your system and correctly follow the `PyTorch Installation Prerequisites for Intel GPUs <https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html>`_. Then, check that the PyTorch compilation is correctly finished.

Minimum Code Change
-------------------

If you are migrating code from ``cuda``, you would change references from ``cuda`` to ``xpu``. For example:

.. code-block::

   # CUDA CODE
   tensor = torch.tensor([1.0, 2.0]).to("cuda")

   # CODE for Intel GPU
   tensor = torch.tensor([1.0, 2.0]).to("xpu")

The following points outline the support and limitations for PyTorch with Intel GPU:

#. Both training and inference workflows are supported.
#. Both eager mode and ``torch.compile`` is supported.
#. Data types such as FP32, BF16, FP16, and Automatic Mixed Precision (AMP) are all supported.
#. Models that depend on third-party components, will not be supported until PyTorch v2.5 or later.

Examples
--------

This section contains usage examples for both inference and training workflows.

Inference Examples
^^^^^^^^^^^^^^^^^^

Here is a few inference workflow examples.


Inference with FP32
"""""""""""""""""""

.. code-block::

   import torch
   import torchvision.models as models

   model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   ######## code changes #######
   model = model.to("xpu")
   data = data.to("xpu")
   ######## code changes #######

   with torch.no_grad():
       model(data)

   print("Execution finished")

Inference with AMP
""""""""""""""""""

.. code-block::

   import torch
   import torchvision.models as models

   model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
   model.eval()
   data = torch.rand(1, 3, 224, 224)

   #################### code changes #################
   model = model.to("xpu")
   data = data.to("xpu")
   #################### code changes #################

   with torch.no_grad():
       d = torch.rand(1, 3, 224, 224)
       ############################# code changes #####################
       d = d.to("xpu")
       # set dtype=torch.bfloat16 for BF16
       with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=True):
       ############################# code changes #####################
           model(data)

   print("Execution finished")

Inference with ``torch.compile``
""""""""""""""""""""""""""""""""

.. code-block::

   import torch
   import torchvision.models as models

   model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
   model.eval()
   data = torch.rand(1, 3, 224, 224)
   ITERS = 10

   ######## code changes #######
   model = model.to("xpu")
   data = data.to("xpu")
   ######## code changes #######

   model = torch.compile(model)
   for i in range(ITERS):
       with torch.no_grad():
           model(data)

   print("Execution finished")

Training Examples
^^^^^^^^^^^^^^^^^

Here is a few training workflow examples.

Train with FP32
"""""""""""""""

.. code-block::

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = "datasets/cifar10/"

   transform = torchvision.transforms.Compose(
       [
           torchvision.transforms.Resize((224, 224)),
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
       ]
   )
   train_dataset = torchvision.datasets.CIFAR10(
       root=DATA,
       train=True,
       transform=transform,
       download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
   model.train()
   ######################## code changes #######################
   model = model.to("xpu")
   criterion = criterion.to("xpu")
   ######################## code changes #######################

   for batch_idx, (data, target) in enumerate(train_loader):
       ########## code changes ##########
       data = data.to("xpu")
       target = target.to("xpu")
       ########## code changes ##########
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       print(batch_idx)
   torch.save(
       {
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
       },
       "checkpoint.pth",
   )

   print("Execution finished")

Train with AMP
""""""""""""""

.. code-block::

   import torch
   import torchvision

   LR = 0.001
   DOWNLOAD = True
   DATA = "datasets/cifar10/"

   use_amp=True

   transform = torchvision.transforms.Compose(
       [
           torchvision.transforms.Resize((224, 224)),
           torchvision.transforms.ToTensor(),
           torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
       ]
   )
   train_dataset = torchvision.datasets.CIFAR10(
       root=DATA,
       train=True,
       transform=transform,
       download=DOWNLOAD,
   )
   train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128)

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
   scaler = torch.amp.GradScaler(enabled=use_amp)

   model.train()
   ######################## code changes #######################
   model = model.to("xpu")
   criterion = criterion.to("xpu")
   ######################## code changes #######################

   for batch_idx, (data, target) in enumerate(train_loader):
       ########## code changes ##########
       data = data.to("xpu")
       target = target.to("xpu")
       ########## code changes ##########
       # set dtype=torch.bfloat16 for BF16
       with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=use_amp):
           output = model(data)
           loss = criterion(output, target)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad()
       print(batch_idx)

   torch.save(
       {
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
       },
       "checkpoint.pth",
   )

   print("Execution finished")
