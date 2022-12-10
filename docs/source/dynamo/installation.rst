Installing TorchDynamo
======================

This section describes how to install TorchDynamo.
TorchDynamo is included in the nightly binaries of PyTorch. For
more information, see `Getting Started <https://pytorch.org/get-started/locally/>`__.

Requirements
------------

You must have the following prerequisites to use TorchDynamo:

* A Linux or macOS environment
* Python 3.8 (recommended). Python 3.7 through 3.10 are supported and
  tested. Make sure to have a development version of Python installed
  locally as well.

GPU/CUDA Requirements
~~~~~~~~~~~~~~~~~~~~~

To use GPU back ends, and in particular Triton, make sure that
the CUDA that you have installed locally matches the PyTorch version you
are running.

The following command installs GPU PyTorch + TorchDynamo along with GPU
TorchDynamo dependencies (for CUDA 11.7):

.. code-block:: shell

   pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117

CPU requirements
~~~~~~~~~~~~~~~~

There are no additional requirements for CPU TorchDynamo. CPU
TorchDynamo is included in the nightly versions of PyTorch.
To install, run the following command:

.. code-block:: shell

   pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu


Install from Local Source
~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can build PyTorch from `source
<https://github.com/pytorch/pytorch#from-source>`__, which has TorchDynamo
included.

To install GPU TorchDynamo dependencies, run ``make triton`` in the
PyTorch repo root directory.

Verify Installation
~~~~~~~~~~~~~~~~~~~

If you built PyTorch from source, then you can run the following
commands (from the PyTorch repo root directory)
to check that TorchDynamo is installed correctly:

.. code-block:: shell

   cd tools/dynamo
   python verify_dynamo.py

If you do not have the PyTorch source locally, you can alternatively
copy the script (``tools/dynamo/verify_dynamo.py``) from the PyTorch
repository and run it locally.

Docker Installation
-------------------

We also provide all the required dependencies in the PyTorch nightly
binaries which you can download with the following command:

.. code-block::

   docker pull ghcr.io/pytorch/pytorch-nightly

And for ad hoc experiments just make sure that your container has access
to all your GPUs:

.. code-block:: bash

   docker run --gpus all -it ghcr.io/pytorch/pytorch-nightly:latest /bin/bash
