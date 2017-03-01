PyTorch for former Torchies
---------------------------

In this tutorial, you will learn the following:

1. Using torch Tensors, and important difference against (Lua)Torch
2. Using the autograd package
3. Building neural networks

  -  Building a ConvNet
  -  Building a Recurrent Net
4. Use multiple GPUs

.. ##########################  tensor ########################## 
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tensors behave almost exactly the same way in PyTorch as they do in Torch. ">

.. only:: html

    .. figure:: /_static/img/tensor_illustration.png

        :ref:`sphx_glr_tutorial_tutorial_torchies_tensor.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorial/tutorial_torchies_tensor


.. ##########################  autograd ########################## 
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Autograd is now a core torch package for automatic differentiation.">

.. only:: html

    .. figure:: /_static/img/Variable.png

        :ref:`sphx_glr_tutorial_tutorial_torchies_autograd.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorial/tutorial_torchies_autograd


.. ##########################  nn ########################## 
.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We’ve redesigned the nn package, so that it’s fully integrated with autograd.">

.. only:: html

    .. figure:: /_static/img/torch-nn-vs-pytorch-nn.png

        :ref:`sphx_glr_tutorial_tutorial_torchies_nn.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorial/tutorial_torchies_nn

.. ########################## parallelism ########################## 

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Data Parallelism is when we split the mini-batch of samples into multiple smaller mini-batches ...">

.. only:: html

    .. figure:: /tutorial/images/thumb/sphx_glr_tutorial_torchies_parallelism_thumb.png

        :ref:`sphx_glr_tutorial_tutorial_torchies_parallelism.py`

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /tutorial/tutorial_torchies_parallelism

.. raw:: html

    <div style='clear:both'></div>
