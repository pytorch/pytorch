Tensor Class
============

The ``at::Tensor`` class is the primary tensor class in ATen, representing
a multi-dimensional array with a specific data type and device.

Tensor
------

.. cpp:class:: at::Tensor

   The primary tensor class in ATen. Represents a multi-dimensional array
   with a specific data type and device.

   .. cpp:function:: Tensor()

      Default constructor. Creates an undefined tensor.

   .. cpp:function:: int64_t dim() const

      Returns the number of dimensions of the tensor.

   .. cpp:function:: int64_t size(int64_t dim) const

      Returns the size of the tensor at the given dimension.

   .. cpp:function:: IntArrayRef sizes() const

      Returns the sizes of all dimensions.

   .. cpp:function:: IntArrayRef strides() const

      Returns the strides of all dimensions.

   .. cpp:function:: ScalarType scalar_type() const

      Returns the data type of the tensor.

   .. cpp:function:: Device device() const

      Returns the device where the tensor is stored.

   .. cpp:function:: bool is_cuda() const

      Returns true if the tensor is on a CUDA device.

   .. cpp:function:: bool is_cpu() const

      Returns true if the tensor is on CPU.

   .. cpp:function:: bool requires_grad() const

      Returns true if gradients need to be computed for this tensor.

   .. cpp:function:: Tensor& requires_grad_(bool requires_grad = true)

      Sets whether gradients should be computed for this tensor.

   .. cpp:function:: Tensor to(Device device) const

      Returns a tensor on the specified device.

   .. cpp:function:: Tensor to(ScalarType dtype) const

      Returns a tensor with the specified data type.

   .. cpp:function:: Tensor contiguous() const

      Returns a contiguous tensor with the same data.

   .. cpp:function:: void* data_ptr() const

      Returns a pointer to the underlying data.

   **Example:**

   .. code-block:: cpp

      #include <ATen/ATen.h>

      at::Tensor a = at::ones({2, 2}, at::kInt);
      at::Tensor b = at::randn({2, 2});
      auto c = a + b.to(at::kInt);

TensorOptions
-------------

.. cpp:class:: at::TensorOptions

   A class to specify options for tensor creation, including dtype, device,
   layout, and requires_grad.

   .. cpp:function:: TensorOptions()

      Default constructor.

   .. cpp:function:: TensorOptions dtype(ScalarType dtype) const

      Returns options with the specified data type.

   .. cpp:function:: TensorOptions device(Device device) const

      Returns options with the specified device.

   .. cpp:function:: TensorOptions layout(Layout layout) const

      Returns options with the specified layout.

   .. cpp:function:: TensorOptions requires_grad(bool requires_grad) const

      Returns options with the specified requires_grad setting.

   **Example:**

   .. code-block:: cpp

      auto options = at::TensorOptions()
          .dtype(at::kFloat)
          .device(at::kCUDA, 0)
          .requires_grad(true);

      at::Tensor tensor = at::zeros({3, 4}, options);

Scalar
------

.. cpp:class:: at::Scalar

   Represents a scalar value that can be converted to various numeric types.

   .. cpp:function:: Scalar(int64_t v)

      Construct from an integer.

   .. cpp:function:: Scalar(double v)

      Construct from a double.

   .. cpp:function:: template<typename T> T to() const

      Convert to the specified type.

   .. cpp:function:: bool isIntegral(bool includeBool = false) const

      Returns true if the scalar is an integral type.

   .. cpp:function:: bool isFloatingPoint() const

      Returns true if the scalar is a floating point type.

ScalarType
----------

.. cpp:enum-class:: at::ScalarType

   Enumeration of data types supported by tensors.

   .. cpp:enumerator:: Byte

      8-bit unsigned integer (uint8_t)

   .. cpp:enumerator:: Char

      8-bit signed integer (int8_t)

   .. cpp:enumerator:: Short

      16-bit signed integer (int16_t)

   .. cpp:enumerator:: Int

      32-bit signed integer (int32_t)

   .. cpp:enumerator:: Long

      64-bit signed integer (int64_t)

   .. cpp:enumerator:: Half

      16-bit floating point (float16)

   .. cpp:enumerator:: Float

      32-bit floating point (float)

   .. cpp:enumerator:: Double

      64-bit floating point (double)

   .. cpp:enumerator:: Bool

      Boolean type

   .. cpp:enumerator:: BFloat16

      Brain floating point (bfloat16)

Convenience constants:

- ``at::kByte``, ``at::kChar``, ``at::kShort``, ``at::kInt``, ``at::kLong``
- ``at::kHalf``, ``at::kFloat``, ``at::kDouble``, ``at::kBFloat16``
- ``at::kBool``

DeviceGuard
-----------

.. cpp:class:: at::DeviceGuard

   RAII guard that sets the current device and restores the previous device
   on destruction.

   .. cpp:function:: explicit DeviceGuard(Device device)

      Sets the current device to the specified device.

   **Example:**

   .. code-block:: cpp

      {
          at::DeviceGuard guard(at::Device(at::kCUDA, 1));
          // Operations here run on CUDA device 1
          auto tensor = at::zeros({2, 2});
      }
      // Previous device is restored

Layout
------

.. cpp:enum-class:: at::Layout

   Enumeration of tensor memory layouts.

   .. cpp:enumerator:: Strided

      Dense tensor with strides.

   .. cpp:enumerator:: Sparse

      Sparse tensor (COO format).

   .. cpp:enumerator:: SparseCsr

      Sparse tensor in CSR format.

   .. cpp:enumerator:: SparseCsc

      Sparse tensor in CSC format.
