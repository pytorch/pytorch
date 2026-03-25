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

OptionalArrayRef
----------------

.. doxygenclass:: c10::OptionalArrayRef
   :members:
   :no-link:

**Example:**

.. code-block:: cpp

   void my_function(c10::OptionalArrayRef<int64_t> sizes = c10::nullopt) {
       if (sizes.has_value()) {
           for (auto s : sizes.value()) {
               // process sizes
           }
       }
   }

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

Containers
----------

C10 provides container types that store ``IValue`` elements internally. These
are pointer types: copies share the same underlying storage.

Dict
^^^^

An ordered hash map from ``Key`` to ``Value``. Valid key types are ``int64_t``,
``double``, ``bool``, ``std::string``, and ``at::Tensor``.

.. cpp:class:: template<class Key, class Value> c10::Dict

   An ordered dictionary mapping keys to values.

   .. cpp:function:: Dict()

      Construct an empty dictionary.

   .. cpp:function:: iterator begin() const

      Returns an iterator to the first element.

   .. cpp:function:: iterator end() const

      Returns an iterator past the last element.

   .. cpp:function:: size_type size() const

      Returns the number of elements.

   .. cpp:function:: bool empty() const

      Returns true if the dictionary is empty.

   .. cpp:function:: bool contains(const Key& key) const

      Returns true if ``key`` is present.

   .. cpp:function:: Value at(const Key& key) const

      Returns the value for ``key``. Throws if not found.

   .. cpp:function:: iterator insert(Key key, Value value)

      Inserts a key-value pair (or updates if key exists).

   .. cpp:function:: iterator insert_or_assign(Key key, Value value)

      Inserts or overwrites the value for ``key``.

   .. cpp:function:: void erase(const Key& key)

      Removes the element with the given key.

   .. cpp:function:: iterator find(const Key& key) const

      Returns an iterator to the element with the given key, or ``end()``.

   .. cpp:function:: void reserve(size_type n)

      Reserves space for at least ``n`` elements.

**Example:**

.. code-block:: cpp

   #include <ATen/core/Dict.h>

   c10::Dict<std::string, at::Tensor> named_tensors;
   named_tensors.insert("weight", torch::randn({3, 3}));
   named_tensors.insert("bias", torch::zeros({3}));

   if (named_tensors.contains("weight")) {
       at::Tensor w = named_tensors.at("weight");
   }

   for (const auto& entry : named_tensors) {
       std::cout << entry.key() << ": " << entry.value().sizes() << std::endl;
   }

List
^^^^

A type-safe list container backed by ``IValue`` elements.

.. cpp:class:: template<class T> c10::List

   A reference-counted list of values.

   .. cpp:function:: List()

      Construct an empty list.

   .. cpp:function:: T get(size_type pos) const

      Returns the element at position ``pos``.

   .. cpp:function:: void set(size_type pos, const T& value)

      Sets the element at position ``pos``.

   .. cpp:function:: void push_back(T value)

      Appends an element to the end.

   .. cpp:function:: T extract(size_type pos)

      Removes and returns the element at position ``pos``.

   .. cpp:function:: iterator begin() const

      Returns an iterator to the first element.

   .. cpp:function:: iterator end() const

      Returns an iterator past the last element.

   .. cpp:function:: size_type size() const

      Returns the number of elements.

   .. cpp:function:: bool empty() const

      Returns true if the list is empty.

   .. cpp:function:: void reserve(size_type n)

      Reserves space for at least ``n`` elements.

   .. cpp:function:: void clear()

      Removes all elements.

   .. cpp:function:: void copy_(const List& rhs)

      Copies elements from another list (element-wise, not pointer copy).

**Example:**

.. code-block:: cpp

   #include <ATen/core/List.h>

   c10::List<at::Tensor> tensor_list;
   tensor_list.push_back(torch::randn({2, 3}));
   tensor_list.push_back(torch::zeros({2, 3}));

   at::Tensor first = tensor_list.get(0);
   std::cout << "List size: " << tensor_list.size() << std::endl;

   c10::List<int64_t> int_list;
   int_list.push_back(1);
   int_list.push_back(2);
   int_list.push_back(3);

IListRef
^^^^^^^^

``c10::IListRef<T>`` is a lightweight reference type that provides a unified
interface over different list-like types (``List<T>``, ``ArrayRef<T>``,
``std::vector<T>``). It avoids copying when passing list arguments to operators.

.. cpp:class:: template<class T> c10::IListRef

   A type-erased reference to a list. Does not own the underlying data.

   .. cpp:function:: size_type size() const

      Returns the number of elements.

   .. cpp:function:: bool empty() const

      Returns true if the list is empty.

   .. cpp:function:: iterator begin() const

      Returns an iterator to the first element.

   .. cpp:function:: iterator end() const

      Returns an iterator past the last element.

**Example:**

.. code-block:: cpp

   #include <ATen/core/IListRef.h>

   // IListRef can wrap different underlying types
   std::vector<at::Tensor> vec = {torch::randn({2}), torch::randn({3})};
   c10::IListRef<at::Tensor> ref(vec);

   for (const auto& t : ref) {
       std::cout << t.sizes() << std::endl;
   }

IValue
------

``c10::IValue`` (Interpreter Value) is a type-erased container used extensively
for storing values of different types. It can hold tensors,
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
