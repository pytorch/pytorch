.. _local_tensor_tutorial:

LocalTensor Tutorial: Single-Process SPMD Debugging
====================================================

This tutorial introduces ``LocalTensor``, a powerful debugging tool for developing
and testing distributed tensor operations without requiring multiple processes or GPUs.

.. contents:: Table of Contents
   :local:
   :depth: 2

What is LocalTensor?
--------------------

``LocalTensor`` is a ``torch.Tensor`` subclass that simulates distributed SPMD
(Single Program, Multiple Data) computations on a single process. It internally
maintains a mapping from rank IDs to their corresponding local tensor shards,
allowing you to debug and test distributed code without infrastructure overhead.

Key Benefits
~~~~~~~~~~~~

1. **No Multi-Process Setup Required**: Test distributed algorithms on a single CPU/GPU
2. **Faster Debugging Cycles**: Iterate quickly without launching multiple processes
3. **Full Visibility**: Inspect each rank's tensor state directly
4. **CI-Friendly**: Run distributed tests in single-process CI pipelines
5. **DTensor Integration**: Seamlessly test DTensor code locally

.. note::

   ``LocalTensor`` is intended for **debugging and testing** only, not production use.
   The overhead of simulating multiple ranks locally is significant.

Installation and Setup
----------------------

``LocalTensor`` is part of PyTorch's distributed package. No additional installation
is required beyond PyTorch itself.

Usage Examples
--------------

The following examples demonstrate core patterns for using ``LocalTensor``. Each
example's code is included directly from source files that are also tested to
ensure correctness. The tests directly invoke these same functions.

Example 1: Basic LocalTensor Creation and Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Creating a LocalTensor from per-rank tensors:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py
   :language: python
   :start-after: # [core_create_local_tensor]
   :end-before: # [end_core_create_local_tensor]
   :dedent: 0

**Arithmetic operations (applied per-rank):**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py
   :language: python
   :start-after: # [core_arithmetic_operations]
   :end-before: # [end_core_arithmetic_operations]
   :dedent: 0

**Extracting a tensor when all shards are identical:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py
   :language: python
   :start-after: # [core_reconcile]
   :end-before: # [end_core_reconcile]
   :dedent: 0

**Using LocalTensorMode for automatic LocalTensor creation:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py
   :language: python
   :start-after: # [core_local_tensor_mode]
   :end-before: # [end_core_local_tensor_mode]
   :dedent: 0

**Full source:** `example_01_basic_operations.py <https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_01_basic_operations.py>`_

Example 2: Simulating Collective Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Test collective operations like ``all_reduce``, ``broadcast``, and ``all_gather``
without multiple processes.

**All-reduce with SUM:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_02_collective_operations.py
   :language: python
   :start-after: # [core_all_reduce]
   :end-before: # [end_core_all_reduce]
   :dedent: 0

**Broadcast from a source rank:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_02_collective_operations.py
   :language: python
   :start-after: # [core_broadcast]
   :end-before: # [end_core_broadcast]
   :dedent: 0

**All-gather to collect tensors from all ranks:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_02_collective_operations.py
   :language: python
   :start-after: # [core_all_gather]
   :end-before: # [end_core_all_gather]
   :dedent: 0

**Full source:** `example_02_collective_operations.py <https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_02_collective_operations.py>`_

Example 3: Working with DTensor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``LocalTensor`` integrates with DTensor for testing distributed tensor parallelism.

**Distribute a tensor and verify reconstruction:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_03_dtensor_integration.py
   :language: python
   :start-after: # [core_dtensor_distribute]
   :end-before: # [end_core_dtensor_distribute]
   :dedent: 0

**Distributed matrix multiplication:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_03_dtensor_integration.py
   :language: python
   :start-after: # [core_dtensor_matmul]
   :end-before: # [end_core_dtensor_matmul]
   :dedent: 0

**Simulating a distributed linear layer:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_03_dtensor_integration.py
   :language: python
   :start-after: # [core_dtensor_nn_layer]
   :end-before: # [end_core_dtensor_nn_layer]
   :dedent: 0

**Full source:** `example_03_dtensor_integration.py <https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_03_dtensor_integration.py>`_

Example 4: Handling Uneven Sharding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real-world distributed systems often have uneven data distribution across ranks.
``LocalTensor`` handles this using ``LocalIntNode``.

**Creating LocalTensor with different sizes per rank:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_04_uneven_sharding.py
   :language: python
   :start-after: # [core_uneven_shards]
   :end-before: # [end_core_uneven_shards]
   :dedent: 0

**LocalIntNode arithmetic operations:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_04_uneven_sharding.py
   :language: python
   :start-after: # [core_local_int_node]
   :end-before: # [end_core_local_int_node]
   :dedent: 0

**DTensor with unevenly divisible dimensions:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_04_uneven_sharding.py
   :language: python
   :start-after: # [core_dtensor_uneven]
   :end-before: # [end_core_dtensor_uneven]
   :dedent: 0

**Full source:** `example_04_uneven_sharding.py <https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_04_uneven_sharding.py>`_

Example 5: Rank-Specific Computations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you need to perform different operations on different ranks.

**Using rank_map() to create per-rank values:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_05_rank_specific.py
   :language: python
   :start-after: # [core_rank_map]
   :end-before: # [end_core_rank_map]
   :dedent: 0

**Using tensor_map() to transform shards per-rank:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_05_rank_specific.py
   :language: python
   :start-after: # [core_tensor_map]
   :end-before: # [end_core_tensor_map]
   :dedent: 0

**Temporarily exiting LocalTensorMode:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_05_rank_specific.py
   :language: python
   :start-after: # [core_disable_mode]
   :end-before: # [end_core_disable_mode]
   :dedent: 0

**Full source:** `example_05_rank_specific.py <https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_05_rank_specific.py>`_

Example 6: Multi-Dimensional Meshes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use 2D/3D device meshes for hybrid parallelism (e.g., data parallel + tensor parallel).

**Creating a 2D mesh:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_06_multidim_mesh.py
   :language: python
   :start-after: # [core_2d_mesh]
   :end-before: # [end_core_2d_mesh]
   :dedent: 0

**Hybrid parallelism (DP + TP):**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_06_multidim_mesh.py
   :language: python
   :start-after: # [core_hybrid_parallel]
   :end-before: # [end_core_hybrid_parallel]
   :dedent: 0

**3D mesh for DP + TP + PP:**

.. literalinclude:: ../../../test/distributed/local_tensor_tutorial_examples/example_06_multidim_mesh.py
   :language: python
   :start-after: # [core_3d_mesh]
   :end-before: # [end_core_3d_mesh]
   :dedent: 0

**Full source:** `example_06_multidim_mesh.py <https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/example_06_multidim_mesh.py>`_

Testing Tutorial Examples
-------------------------

All examples in this tutorial are tested to ensure correctness. The test suite
directly invokes the same functions included above:

.. code-block:: python

   # From test_local_tensor_tutorial_examples.py
   from example_01_basic_operations import create_local_tensor

   def test_create_local_tensor(self):
       lt = create_local_tensor()
       self.assertIsInstance(lt, LocalTensor)
       self.assertEqual(lt.shape, torch.Size([2, 2]))

**Test suite:** `test_local_tensor_tutorial_examples.py <https://github.com/pytorch/pytorch/blob/main/test/distributed/local_tensor_tutorial_examples/test_local_tensor_tutorial_examples.py>`_

API Reference
-------------

Core Classes
~~~~~~~~~~~~

.. autoclass:: torch.distributed._local_tensor.LocalTensor
   :members: reconcile, is_contiguous, contiguous, tolist, numpy

.. autoclass:: torch.distributed._local_tensor.LocalTensorMode
   :members: disable, rank_map, tensor_map

.. autoclass:: torch.distributed._local_tensor.LocalIntNode

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: torch.distributed._local_tensor.local_tensor_mode
.. autofunction:: torch.distributed._local_tensor.enabled_local_tensor_mode
.. autofunction:: torch.distributed._local_tensor.maybe_run_for_local_tensor
.. autofunction:: torch.distributed._local_tensor.maybe_disable_local_tensor_mode

Best Practices
--------------

1. **Use for Testing Only**: LocalTensor has significant overhead and should not
   be used in production code.

2. **Initialize Process Groups**: Even for local testing, you need to initialize
   a process group (use the "fake" backend).

3. **Avoid requires_grad on Inner Tensors**: LocalTensor expects inner tensors
   to not have ``requires_grad=True``. Set gradients on the LocalTensor wrapper instead.

4. **Reconcile for Assertions**: Use ``reconcile()`` to extract a single tensor
   when all ranks should have identical values (e.g., after an all-reduce).

5. **Debug with Direct Access**: Access individual shards via ``tensor._local_tensors[rank]``
   for debugging.

Common Pitfalls
---------------

1. **Forgetting the Context Manager**: Operations on LocalTensor outside
   ``LocalTensorMode`` still work but won't create new LocalTensors from factories.

2. **Mismatched Ranks**: Ensure all LocalTensors in an operation have compatible ranks.

3. **Inner Tensor Gradients**: Creating LocalTensor from tensors with ``requires_grad=True``
   will raise an error.
