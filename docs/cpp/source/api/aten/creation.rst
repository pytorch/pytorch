Tensor Creation
===============

Factory functions create new tensors with different initialization patterns.
All factory functions follow a general schema:

.. code-block:: cpp

   torch::<function-name>(<function-specific-options>, <sizes>, <tensor-options>)

Available Factory Functions
---------------------------

- ``torch::zeros`` - Tensor filled with zeros
- ``torch::ones`` - Tensor filled with ones
- ``torch::empty`` - Uninitialized tensor
- ``torch::full`` - Tensor filled with a single value
- ``torch::rand`` - Uniform random values on [0, 1)
- ``torch::randn`` - Standard normal distribution
- ``torch::randint`` - Random integers in a range
- ``torch::arange`` - Sequence of integers
- ``torch::linspace`` - Linearly spaced values
- ``torch::logspace`` - Logarithmically spaced values
- ``torch::eye`` - Identity matrix
- ``torch::randperm`` - Random permutation of integers

**Examples:**

.. code-block:: cpp

   // Create tensors with various shapes
   torch::Tensor vector = torch::ones(5);
   torch::Tensor matrix = torch::randn({3, 4});
   torch::Tensor tensor3d = torch::zeros({2, 3, 4});

   // With TensorOptions for dtype, device, etc.
   auto options = torch::dtype(torch::kFloat32).device(torch::kCUDA, 0);
   torch::Tensor gpu_tensor = torch::randn({3, 4}, options);

   // Shorthand syntax
   torch::Tensor t = torch::ones(10, torch::kFloat32);
   torch::Tensor t2 = torch::randn({3, 4}, torch::dtype(torch::kFloat64).device(torch::kCUDA));

Using Externally Created Data
-----------------------------

If you already have tensor data allocated in memory, use ``from_blob``:

.. code-block:: cpp

   float data[] = {1, 2, 3, 4, 5, 6};
   torch::Tensor tensor = torch::from_blob(data, {2, 3});

.. note::

   Tensors created with ``from_blob`` cannot be resized because ATen does not
   own the memory.

Tensor Conversion
-----------------

Use ``to()`` to convert tensors between dtypes and devices:

.. code-block:: cpp

   torch::Tensor source = torch::randn({2, 3}, torch::kInt64);

   // Convert dtype
   torch::Tensor float_tensor = source.to(torch::kFloat32);

   // Move to GPU
   torch::Tensor gpu_tensor = float_tensor.to(torch::kCUDA);

   // Specific GPU device
   torch::Tensor gpu1_tensor = float_tensor.to(torch::Device(torch::kCUDA, 1));

   // Async copy
   torch::Tensor async_tensor = gpu_tensor.to(torch::kCPU, /*non_blocking=*/true);

Scalars and Zero-Dimensional Tensors
------------------------------------

``Scalar`` represents a single dynamically-typed number:

.. code-block:: cpp

   // Scalars can be implicitly constructed from C++ number types
   torch::Tensor result = torch::addmm(1.0, a, 0.5, b, c);

Zero-dimensional tensors hold a single value and can reference elements in
larger tensors:

.. code-block:: cpp

   torch::Tensor matrix = torch::rand({10, 20});
   matrix[1][2] = 4;  // matrix[1][2] is a zero-dimensional tensor
