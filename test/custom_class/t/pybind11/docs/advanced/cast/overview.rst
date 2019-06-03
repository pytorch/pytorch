Overview
########

.. rubric:: 1. Native type in C++, wrapper in Python

Exposing a custom C++ type using :class:`py::class_` was covered in detail
in the :doc:`/classes` section. There, the underlying data structure is
always the original C++ class while the :class:`py::class_` wrapper provides
a Python interface. Internally, when an object like this is sent from C++ to
Python, pybind11 will just add the outer wrapper layer over the native C++
object. Getting it back from Python is just a matter of peeling off the
wrapper.

.. rubric:: 2. Wrapper in C++, native type in Python

This is the exact opposite situation. Now, we have a type which is native to
Python, like a ``tuple`` or a ``list``. One way to get this data into C++ is
with the :class:`py::object` family of wrappers. These are explained in more
detail in the :doc:`/advanced/pycpp/object` section. We'll just give a quick
example here:

.. code-block:: cpp

    void print_list(py::list my_list) {
        for (auto item : my_list)
            std::cout << item << " ";
    }

.. code-block:: pycon

    >>> print_list([1, 2, 3])
    1 2 3

The Python ``list`` is not converted in any way -- it's just wrapped in a C++
:class:`py::list` class. At its core it's still a Python object. Copying a
:class:`py::list` will do the usual reference-counting like in Python.
Returning the object to Python will just remove the thin wrapper.

.. rubric:: 3. Converting between native C++ and Python types

In the previous two cases we had a native type in one language and a wrapper in
the other. Now, we have native types on both sides and we convert between them.

.. code-block:: cpp

    void print_vector(const std::vector<int> &v) {
        for (auto item : v)
            std::cout << item << "\n";
    }

.. code-block:: pycon

    >>> print_vector([1, 2, 3])
    1 2 3

In this case, pybind11 will construct a new ``std::vector<int>`` and copy each
element from the Python ``list``. The newly constructed object will be passed
to ``print_vector``. The same thing happens in the other direction: a new
``list`` is made to match the value returned from C++.

Lots of these conversions are supported out of the box, as shown in the table
below. They are very convenient, but keep in mind that these conversions are
fundamentally based on copying data. This is perfectly fine for small immutable
types but it may become quite expensive for large data structures. This can be
avoided by overriding the automatic conversion with a custom wrapper (i.e. the
above-mentioned approach 1). This requires some manual effort and more details
are available in the :ref:`opaque` section.

.. _conversion_table:

List of all builtin conversions
-------------------------------

The following basic data types are supported out of the box (some may require
an additional extension header to be included). To pass other data structures
as arguments and return values, refer to the section on binding :ref:`classes`.

+------------------------------------+---------------------------+-------------------------------+
|  Data type                         |  Description              | Header file                   |
+====================================+===========================+===============================+
| ``int8_t``, ``uint8_t``            | 8-bit integers            | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``int16_t``, ``uint16_t``          | 16-bit integers           | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``int32_t``, ``uint32_t``          | 32-bit integers           | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``int64_t``, ``uint64_t``          | 64-bit integers           | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``ssize_t``, ``size_t``            | Platform-dependent size   | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``float``, ``double``              | Floating point types      | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``bool``                           | Two-state Boolean type    | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``char``                           | Character literal         | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``char16_t``                       | UTF-16 character literal  | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``char32_t``                       | UTF-32 character literal  | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``wchar_t``                        | Wide character literal    | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``const char *``                   | UTF-8 string literal      | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``const char16_t *``               | UTF-16 string literal     | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``const char32_t *``               | UTF-32 string literal     | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``const wchar_t *``                | Wide string literal       | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``std::string``                    | STL dynamic UTF-8 string  | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``std::u16string``                 | STL dynamic UTF-16 string | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``std::u32string``                 | STL dynamic UTF-32 string | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``std::wstring``                   | STL dynamic wide string   | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``std::string_view``,              | STL C++17 string views    | :file:`pybind11/pybind11.h`   |
| ``std::u16string_view``, etc.      |                           |                               |
+------------------------------------+---------------------------+-------------------------------+
| ``std::pair<T1, T2>``              | Pair of two custom types  | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``std::tuple<...>``                | Arbitrary tuple of types  | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``std::reference_wrapper<...>``    | Reference type wrapper    | :file:`pybind11/pybind11.h`   |
+------------------------------------+---------------------------+-------------------------------+
| ``std::complex<T>``                | Complex numbers           | :file:`pybind11/complex.h`    |
+------------------------------------+---------------------------+-------------------------------+
| ``std::array<T, Size>``            | STL static array          | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::vector<T>``                 | STL dynamic array         | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::deque<T>``                  | STL double-ended queue    | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::valarray<T>``               | STL value array           | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::list<T>``                   | STL linked list           | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::map<T1, T2>``               | STL ordered map           | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::unordered_map<T1, T2>``     | STL unordered map         | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::set<T>``                    | STL ordered set           | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::unordered_set<T>``          | STL unordered set         | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::optional<T>``               | STL optional type (C++17) | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::experimental::optional<T>`` | STL optional type (exp.)  | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::variant<...>``              | Type-safe union (C++17)   | :file:`pybind11/stl.h`        |
+------------------------------------+---------------------------+-------------------------------+
| ``std::function<...>``             | STL polymorphic function  | :file:`pybind11/functional.h` |
+------------------------------------+---------------------------+-------------------------------+
| ``std::chrono::duration<...>``     | STL time duration         | :file:`pybind11/chrono.h`     |
+------------------------------------+---------------------------+-------------------------------+
| ``std::chrono::time_point<...>``   | STL date/time             | :file:`pybind11/chrono.h`     |
+------------------------------------+---------------------------+-------------------------------+
| ``Eigen::Matrix<...>``             | Eigen: dense matrix       | :file:`pybind11/eigen.h`      |
+------------------------------------+---------------------------+-------------------------------+
| ``Eigen::Map<...>``                | Eigen: mapped memory      | :file:`pybind11/eigen.h`      |
+------------------------------------+---------------------------+-------------------------------+
| ``Eigen::SparseMatrix<...>``       | Eigen: sparse matrix      | :file:`pybind11/eigen.h`      |
+------------------------------------+---------------------------+-------------------------------+
