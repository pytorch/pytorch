Javadoc
=========

Main part of the java api includes 3 classes:
::
    org.pytorch.Tensor
    org.pytorch.IValue
    org.pytorch.Module

If the reader is familiar with pytorch python api, we can think that
``org.pytorch.Tensor`` represents ``torch.tensor``, ``org.pytorch.Module`` represents ``torch.Module``,
while ``org.pytorch.IValue`` represents value of TorchScript variable, supporting all
its `types <https://pytorch.org/docs/stable/jit.html#types>`_.

Additionally, there is the DType class which contains code representing tensor data types and 
TensorImageUtils which contains utility functions for \ :java:ref:`org.pytorch.Tensor`\ created from \ :java:ref:`android.graphics.Bitmap`\  or \ :java:ref:`android.media.Image`\  source.

You can find details to each of these classes linked below:

.. java:package:: org.pytorch

.. toctree::
   :maxdepth: 1
    
   org/pytorch/DType 
   org/pytorch/Tensor
   org/pytorch/IValue
   org/pytorch/Module
   org/pytorch/TensorImageUtils
