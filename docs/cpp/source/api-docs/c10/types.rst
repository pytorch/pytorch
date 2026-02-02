Core Types
==========

C10 provides fundamental types used throughout PyTorch.

ArrayRef
--------

.. doxygenclass:: c10::ArrayRef
   :members:
   :undoc-members:

**Example:**

.. code-block:: cpp

   std::vector<int64_t> sizes = {3, 4, 5};
   c10::ArrayRef<int64_t> sizes_ref(sizes);

   // Can also use initializer list
   auto tensor = at::zeros({3, 4, 5});  // implicitly converts

Optional
--------

.. cpp:class:: c10::optional

   A wrapper type that may or may not contain a value.
   Similar to ``std::optional``.

   .. cpp:function:: bool has_value() const

      Returns true if a value is present.

   .. cpp:function:: T& value()

      Returns the contained value. Throws if empty.

   .. cpp:function:: T value_or(T default_value) const

      Returns the value if present, otherwise returns the default.

**Example:**

.. code-block:: cpp

   c10::optional<int64_t> maybe_dim = c10::nullopt;

   if (maybe_dim.has_value()) {
       std::cout << "Dim: " << maybe_dim.value() << std::endl;
   }

   int64_t dim = maybe_dim.value_or(-1);  // Returns -1 if empty

Half
----

.. cpp:class:: c10::Half

   16-bit floating point type (IEEE 754 half-precision).

   .. cpp:function:: Half(float value)

      Construct from a float.

   .. cpp:function:: operator float() const

      Convert to float.

**Example:**

.. code-block:: cpp

   c10::Half h = 3.14f;
   float f = static_cast<float>(h);

IValue
------

``c10::IValue`` (Interpreter Value) is a type-erased container used extensively
in TorchScript for storing values of different types. It can hold tensors,
scalars, lists, dictionaries, and other types.

.. note::
   The full API documentation for IValue is complex due to its many type
   conversion methods. See the header file ``ATen/core/ivalue.h`` for complete
   details.

**Common methods:**

- ``isTensor()`` / ``toTensor()`` - Check if tensor / convert to tensor
- ``isInt()`` / ``toInt()`` - Check if int / convert to int
- ``isDouble()`` / ``toDouble()`` - Check if double / convert to double
- ``isBool()`` / ``toBool()`` - Check if bool / convert to bool
- ``isString()`` / ``toString()`` - Check if string / convert to string
- ``isList()`` / ``toList()`` - Check if list / convert to list
- ``isGenericDict()`` / ``toGenericDict()`` - Check if dict / convert to dict
- ``isTuple()`` / ``toTuple()`` - Check if tuple / convert to tuple
- ``isNone()`` - Check if None/null

**Example:**

.. code-block:: cpp

   c10::IValue val = at::ones({2, 2});

   if (val.isTensor()) {
       at::Tensor t = val.toTensor();
   }
