Stable Operators
================

The stable API provides tensor operations that maintain binary compatibility
across PyTorch versions.

Tensor Class
------------

.. doxygenclass:: torch::stable::Tensor
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   torch::stable::Tensor tensor = torch::stable::empty({3, 4}, ...);
   float* data = tensor.data_ptr<float>();
   auto shape = tensor.sizes();

Device Class
------------

.. doxygenclass:: torch::stable::Device
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   torch::stable::Device cpu_device(torch::headeronly::DeviceType::CPU);
   torch::stable::Device cuda_device(torch::headeronly::DeviceType::CUDA, 0);

Tensor Creation
---------------

.. cpp:function:: torch::stable::Tensor torch::stable::empty(IntArrayRef size, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<MemoryFormat> memory_format)

   Create an uninitialized tensor with the given shape.

.. cpp:function:: torch::stable::Tensor torch::stable::empty_like(const Tensor& self, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory, std::optional<MemoryFormat> memory_format)

   Create an uninitialized tensor with the same shape as the input.

.. cpp:function:: torch::stable::Tensor torch::stable::new_empty(const Tensor& self, IntArrayRef size, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory)

   Create a new empty tensor with properties from self.

.. cpp:function:: torch::stable::Tensor torch::stable::new_zeros(const Tensor& self, IntArrayRef size, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory)

   Create a new zero-filled tensor with properties from self.

.. cpp:function:: torch::stable::Tensor torch::stable::full(IntArrayRef size, const Scalar& fill_value, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory)

   Create a tensor filled with a scalar value.

.. cpp:function:: torch::stable::Tensor torch::stable::from_blob(void* data, IntArrayRef sizes, IntArrayRef strides, const std::function<void(void*)>& deleter, std::optional<ScalarType> dtype, std::optional<Device> device)

   Create a tensor from an existing data pointer.

**Example:**

.. code-block:: cpp

   auto tensor = torch::stable::empty(
       {3, 4},
       torch::headeronly::ScalarType::Float,
       torch::headeronly::Layout::Strided,
       torch::stable::Device(torch::headeronly::DeviceType::CUDA, 0),
       false,
       torch::headeronly::MemoryFormat::Contiguous);

Tensor Manipulation
-------------------

.. cpp:function:: torch::stable::Tensor torch::stable::clone(const Tensor& self, std::optional<MemoryFormat> memory_format)

   Create a copy of the tensor.

.. cpp:function:: torch::stable::Tensor torch::stable::contiguous(const Tensor& self, MemoryFormat memory_format)

   Return a contiguous tensor.

.. cpp:function:: torch::stable::Tensor torch::stable::reshape(const Tensor& self, IntArrayRef shape)

   Reshape the tensor to the given shape.

.. cpp:function:: torch::stable::Tensor torch::stable::view(const Tensor& self, IntArrayRef size)

   Create a view with a different shape.

.. cpp:function:: torch::stable::Tensor torch::stable::flatten(const Tensor& self, int64_t start_dim, int64_t end_dim)

   Flatten dimensions of the tensor.

.. cpp:function:: torch::stable::Tensor torch::stable::squeeze(const Tensor& self)

   Remove dimensions of size 1.

.. cpp:function:: torch::stable::Tensor torch::stable::unsqueeze(const Tensor& self, int64_t dim)

   Add a dimension of size 1.

.. cpp:function:: torch::stable::Tensor torch::stable::transpose(const Tensor& self, int64_t dim0, int64_t dim1)

   Transpose two dimensions.

.. cpp:function:: torch::stable::Tensor torch::stable::select(const Tensor& self, int64_t dim, int64_t index)

   Select a slice along a dimension.

.. cpp:function:: torch::stable::Tensor torch::stable::narrow(const Tensor& self, int64_t dim, int64_t start, int64_t length)

   Narrow the tensor along a dimension.

.. cpp:function:: torch::stable::Tensor torch::stable::pad(const Tensor& self, IntArrayRef pad, const std::string& mode, std::optional<double> value)

   Pad the tensor.

Device and Type Conversion
--------------------------

.. cpp:function:: torch::stable::Tensor torch::stable::to(const Tensor& self, std::optional<ScalarType> dtype, std::optional<Layout> layout, std::optional<Device> device, std::optional<bool> pin_memory, bool non_blocking, bool copy, std::optional<MemoryFormat> memory_format)

   Convert tensor to specified dtype, layout, device.

.. cpp:function:: torch::stable::Tensor torch::stable::to(const Tensor& self, Device device, bool non_blocking, bool copy)

   Move tensor to the specified device.

In-place Operations
-------------------

.. cpp:function:: torch::stable::Tensor& torch::stable::fill_(Tensor& self, const Scalar& value)

   Fill tensor with a scalar value in-place.

.. cpp:function:: torch::stable::Tensor& torch::stable::zero_(Tensor& self)

   Fill tensor with zeros in-place.

.. cpp:function:: torch::stable::Tensor& torch::stable::copy_(Tensor& self, const Tensor& src, bool non_blocking)

   Copy data from src to self in-place.

Mathematical Operations
-----------------------

.. cpp:function:: torch::stable::Tensor torch::stable::matmul(const Tensor& self, const Tensor& other)

   Matrix multiplication.

.. cpp:function:: torch::stable::Tensor torch::stable::amax(const Tensor& self, int64_t dim, bool keepdim)

   Maximum value along a dimension.

.. cpp:function:: torch::stable::Tensor torch::stable::amax(const Tensor& self, IntArrayRef dims, bool keepdim)

   Maximum value along multiple dimensions.

.. cpp:function:: torch::stable::Tensor torch::stable::sum(const Tensor& self, std::optional<ScalarType> dtype)

   Sum of all elements.

.. cpp:function:: torch::stable::Tensor& torch::stable::sum_out(Tensor& out, const Tensor& self, IntArrayRef dims, bool keepdim, std::optional<ScalarType> dtype)

   Sum with output tensor.

.. cpp:function:: torch::stable::Tensor torch::stable::subtract(const Tensor& self, const Tensor& other, const Scalar& alpha)

   Element-wise subtraction.
