Getting Started on Intel GPU
============================

Hardware Prerequisite
---------------------

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Validated Hardware
     - Supported OS
   * - Intel® Data Center GPU Max Series
     - Linux
   * - Intel Client GPU
     - Windows/Linux

Intel GPUs support (Prototype) is ready in PyTorch* 2.5 for Intel® Data Center GPU Max Series and Intel® Client GPUs on both Linux and Windows, which brings Intel GPUs and the SYCL* software stack into the official PyTorch stack with consistent user experience to embrace more AI application scenarios.

Software Prerequisite
---------------------

Visit `PyTorch Installation Prerequisites for Intel GPUs <https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html>`_ for more detailed information regarding:

#. Intel GPU driver installation
#. Intel support package installation
#. Environment setup

Installation
------------

Binaries
^^^^^^^^

Platform Linux
""""""""""""""


Now we have all the required packages installed and environment activated. Use the following commands to install ``pytorch``, ``torchvision``, ``torchaudio`` on Linux.

For preview wheels

.. code-block::

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu

For nightly wheels

.. code-block::

    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

Platform Windows
""""""""""""""""

Now we have all the required packages installed and environment activated. Use the following commands to install ``pytorch`` on Windows, build from source for ``torchvision`` and ``torchaudio``.

For preview wheels

.. code-block::

    pip3 install torch --index-url https://download.pytorch.org/whl/test/xpu

For nightly wheels

.. code-block::

    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/xpu

From Source
^^^^^^^^^^^

Build from source for ``torch`` refer to `PyTorch Installation Build from source <https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source>`_.

Build from source for ``torchvision`` refer to `Torchvision Installation Build from source <https://github.com/pytorch/vision/blob/main/CONTRIBUTING.md#development-installation>`_.

Build from source for ``torchaudio`` refert to `Torchaudio Installation Build from source <https://github.com/pytorch/audio/blob/main/CONTRIBUTING.md#building-torchaudio-from-source>`_.

Check availability for Intel GPU
--------------------------------

To check if your Intel GPU is available, you would typically use the following code:

.. code-block::

   import torch
   torch.xpu.is_available()  # torch.xpu is the API for Intel GPU support

If the output is ``False``, double check following steps below.

#. Intel GPU driver installation
#. Intel support package installation
#. Environment setup

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

   model = model.to("xpu")
   data = data.to("xpu")

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

   model = model.to("xpu")
   data = data.to("xpu")

   with torch.no_grad():
       d = torch.rand(1, 3, 224, 224)
       d = d.to("xpu")
       # set dtype=torch.bfloat16 for BF16
       with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=True):
           model(data)

   print("Execution finished")

Inference with ``torch.compile``
""""""""""""""""""""""""""""""""

.. code-block::

   import torch
   import torchvision.models as models
   import time

   model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
   model.eval()
   data = torch.rand(1, 3, 224, 224)
   ITERS = 10

   model = model.to("xpu")
   data = data.to("xpu")

    for i in range(ITERS):
        start = time.time()
        with torch.no_grad():
            model(data)
            torch.xpu.synchronize()
        end = time.time()
        print(f"Inference time before torch.compile for iteration {i}: {(end-start)*1000} ms")

    model = torch.compile(model)
    for i in range(ITERS):
        start = time.time()
        with torch.no_grad():
            model(data)
            torch.xpu.synchronize()
        end = time.time()
        print(f"Inference time after torch.compile for iteration {i}: {(end-start)*1000} ms")

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
   train_len = len(train_loader)

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
   model.train()
   model = model.to("xpu")
   criterion = criterion.to("xpu")

   print(f"Initiating training")
   for batch_idx, (data, target) in enumerate(train_loader):
       data = data.to("xpu")
       target = target.to("xpu")
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       if (batch_idx + 1) % 10 == 0:
            iteration_loss = loss.item()
            print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")
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
   train_len = len(train_loader)

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
   scaler = torch.amp.GradScaler(enabled=use_amp)

   model.train()
   model = model.to("xpu")
   criterion = criterion.to("xpu")

   print(f"Initiating training")
   for batch_idx, (data, target) in enumerate(train_loader):
       data = data.to("xpu")
       target = target.to("xpu")
       # set dtype=torch.bfloat16 for BF16
       with torch.autocast(device_type="xpu", dtype=torch.float16, enabled=use_amp):
           output = model(data)
           loss = criterion(output, target)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
       optimizer.zero_grad()
       if (batch_idx + 1) % 10 == 0:
            iteration_loss = loss.item()
            print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")

   torch.save(
       {
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
       },
       "checkpoint.pth",
   )

   print("Execution finished")

Train with ``torch.compile``
""""""""""""""""""""""""""""

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
   train_len = len(train_loader)

   model = torchvision.models.resnet50()
   criterion = torch.nn.CrossEntropyLoss()
   optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
   model.train()
   model = model.to("xpu")
   criterion = criterion.to("xpu")
   model = torch.compile(model)

   print(f"Initiating training with torch compile")
   for batch_idx, (data, target) in enumerate(train_loader):
       data = data.to("xpu")
       target = target.to("xpu")
       optimizer.zero_grad()
       output = model(data)
       loss = criterion(output, target)
       loss.backward()
       optimizer.step()
       if (batch_idx + 1) % 10 == 0:
            iteration_loss = loss.item()
            print(f"Iteration [{batch_idx+1}/{train_len}], Loss: {iteration_loss:.4f}")
   torch.save(
       {
           "model_state_dict": model.state_dict(),
           "optimizer_state_dict": optimizer.state_dict(),
       },
       "checkpoint.pth",
   )

   print("Execution finished")
