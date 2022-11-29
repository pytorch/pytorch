Installing TorchDynamo
======================

This section describes how to install TorchDynamo.

Requirements and Setup
----------------------

Python 3.8 is recommended. Python 3.7 through 3.10 are supported and
tested. Make sure to have a development version of Python installed
locally as well.

TorchDynamo is included in the nightly binaries of PyTorch. You can
find more information `here <https://pytorch.org/get-started/locally/>`__

Install GPU/CUDA version requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To use GPU back ends (and in particular Triton), please make sure that
the CUDA that you have installed locally matches the PyTorch version you
are running.

The following command installs GPU PyTorch+TorchDynamo along with GPU
TorchDynamo dependencies (for CUDA 11.7):

.. code-block:: python

   pip3 install numpy --pre torch[dynamo] --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cu117

CPU requirements
~~~~~~~~~~~~~~~~

There are no additional requirements for CPU TorchDynamo. CPU
TorchDynamo is included in the nightly versions of PyTorch, which, for
reference, can be installed with the following command:

.. code-block:: shell

   pip3 install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu


Install from local source
~~~~~~~~~~~~~~~~~~~~~~~~~

Build PyTorch from source:
https://github.com/pytorch/pytorch#from-source, which has TorchDynamo
included.

To install GPU TorchDynamo dependencies, run ``make triton`` in the
PyTorch repo root directory.

Verify Installation
~~~~~~~~~~~~~~~~~~~

If you built PyTorch from source, then you can run the following
commands (from the PyTorch repo root directory) that run minimal
examples to check that TorchDynamo is installed correctly:

.. code:: shell

   cd tools/dynamo
   python verify_dynamo.py

If you do not have the PyTorch source locally, you can alternatively
copy the script (``tools/dynamo/verify_dynamo.py``) from the PyTorch
repo and run it locally.

Docker installation
-------------------

We also provide all the required dependencies in the PyTorch nightly
binaries which you can download with

.. code-block::

   docker pull ghcr.io/pytorch/pytorch-nightly

And for ad hoc experiments just make sure that your container has access
to all your GPUs

.. code-block:: bash

   docker run --gpus all -it ghcr.io/pytorch/pytorch-nightly:latest /bin/bash
