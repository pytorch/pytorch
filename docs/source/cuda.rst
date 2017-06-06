torch.cuda
===================================

.. currentmodule:: torch.cuda

.. automodule:: torch.cuda
   :members:

Communication collectives
-------------------------

.. autofunction:: torch.cuda.comm.broadcast

.. autofunction:: torch.cuda.comm.reduce_add

.. autofunction:: torch.cuda.comm.scatter

.. autofunction:: torch.cuda.comm.gather

Streams and events
------------------

.. autoclass:: Stream
   :members:

.. autoclass:: Event
   :members:

NVIDIA Tools Extension (NVTX)
-----------------------------

.. autofunction:: torch.cuda.nvtx.mark
.. autofunction:: torch.cuda.nvtx.range_push
.. autofunction:: torch.cuda.nvtx.range_pop
