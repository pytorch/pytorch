.. _torch_cuda_memory:

Understanding CUDA Memory Usage
===============================
To debug CUDA memory use, PyTorch provides a way to generate memory snapshots that record the state of allocated CUDA memory
at any point in time, and optionally record the history of allocation events that led up to that snapshot.

The generated snapshots can then be drag and dropped onto the interactiver viewer hosted at `pytorch.org/memory_viz <https://pytorch.org/memory_viz>`_ which
can be used to explore the snapshot.

Generating a Snapshot
=====================
The common pattern for recording a snapshot is to enable memory history, run the code to be observed, and then save a file with a pickled snapshot:

.. code-block:: python

   # enable memory history, which will
   # add tracebacks and event history to snapshots
   torch.cuda.memory._record_memory_history()

   run_your_code()
   torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

Using the visualizer
====================

Open `pytorch.org/memory_viz <https://pytorch.org/memory_viz>`_ and drag/drop the pickled snapshot file into the visualizer.
The visualizer is a javascript application that runs locally on your computer. It does not upload any snapshot data.


Active Memory Timeline
----------------------

The Active Memory Timeline shows all the live tensors over time in the snapshot on a particular GPU. Pan/Zoom over the plot to look at smaller allocations.
Mouse over allocated blocks to see a stack trace for when that block was allocated, and details like its address. The detail slider can be adjusted to
render fewer allocations and improve performance when there is a lot of data.

.. image:: _static/img/torch_cuda_memory/active_memory_timeline.png


Allocator State History
-----------------------

The Allocator State History shows individual allocator events in a timeline on the left. Select an event in the timeline to see a visual summary of the
allocator state at that event. This summary shows each individual segment returned from cudaMalloc and how it is split up into blocks of individual allocations
or free space. Mouse over segments and blocks to see the stack trace when the memory was allocated. Mouse over events to see the stack trace when the event occurred,
such as when a tensor was freed. Out of memory errors are reported as OOM events. Looking at the state of memory during an OOM may provide insight into why
an allocation failed even though reserved memory still exists.

.. image:: _static/img/torch_cuda_memory/allocator_state_history.png

The stack trace information also reports the address at which an allocation occurred.
The address b7f064c000000_0 refers to the (b)lock at address 7f064c000000 which is the "_0"th time this address was allocated.
This unique string can be looked up in the Active Memory Timeline and searched
in the Active State History to examine the memory state when a tensor was allocated or freed.

Snapshot API Reference
======================

.. currentmodule:: torch.cuda.memory
.. autofunction:: _record_memory_history
.. autofunction:: _snapshot
.. autofunction:: _dump_snapshot
