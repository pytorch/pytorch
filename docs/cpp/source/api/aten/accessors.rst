Tensor Accessors
================

For element-wise operations in custom kernels, use *accessors* to avoid
dynamic dispatch overhead.

CPU Accessors
-------------

.. code-block:: cpp

   torch::Tensor foo = torch::rand({12, 12});

   // Create accessor - validates type and dimensions once
   auto foo_a = foo.accessor<float, 2>();

   float trace = 0;
   for (int i = 0; i < foo_a.size(0); i++) {
       trace += foo_a[i][i];
   }

CUDA Packed Accessors
---------------------

For CUDA kernels, use *packed accessors* which copy metadata instead of
pointing to it:

.. code-block:: cpp

   __global__ void kernel(torch::PackedTensorAccessor64<float, 2> foo, float* trace) {
       int i = threadIdx.x;
       gpuAtomicAdd(trace, foo[i][i]);
   }

   torch::Tensor foo = torch::rand({12, 12}).cuda();
   auto foo_a = foo.packed_accessor64<float, 2>();

   float trace = 0;
   kernel<<<1, 12>>>(foo_a, &trace);

.. tip::

   Use ``PackedTensorAccessor32`` and ``packed_accessor32`` for 32-bit indexing,
   which is faster on CUDA but may overflow for large tensors.
