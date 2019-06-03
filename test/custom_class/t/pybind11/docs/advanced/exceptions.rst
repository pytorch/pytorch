Exceptions
##########

Built-in exception translation
==============================

When C++ code invoked from Python throws an ``std::exception``, it is
automatically converted into a Python ``Exception``. pybind11 defines multiple
special exception classes that will map to different types of Python
exceptions:

.. tabularcolumns:: |p{0.5\textwidth}|p{0.45\textwidth}|

+--------------------------------------+--------------------------------------+
|  C++ exception type                  |  Python exception type               |
+======================================+======================================+
| :class:`std::exception`              | ``RuntimeError``                     |
+--------------------------------------+--------------------------------------+
| :class:`std::bad_alloc`              | ``MemoryError``                      |
+--------------------------------------+--------------------------------------+
| :class:`std::domain_error`           | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::invalid_argument`       | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::length_error`           | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::out_of_range`           | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::range_error`            | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::stop_iteration`    | ``StopIteration`` (used to implement |
|                                      | custom iterators)                    |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::index_error`       | ``IndexError`` (used to indicate out |
|                                      | of bounds access in ``__getitem__``, |
|                                      | ``__setitem__``, etc.)               |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::value_error`       | ``ValueError`` (used to indicate     |
|                                      | wrong value passed in                |
|                                      | ``container.remove(...)``)           |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::key_error`         | ``KeyError`` (used to indicate out   |
|                                      | of bounds access in ``__getitem__``, |
|                                      | ``__setitem__`` in dict-like         |
|                                      | objects, etc.)                       |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::error_already_set` | Indicates that the Python exception  |
|                                      | flag has already been set via Python |
|                                      | API calls from C++ code; this C++    |
|                                      | exception is used to propagate such  |
|                                      | a Python exception back to Python.   |
+--------------------------------------+--------------------------------------+

When a Python function invoked from C++ throws an exception, it is converted
into a C++ exception of type :class:`error_already_set` whose string payload
contains a textual summary.

There is also a special exception :class:`cast_error` that is thrown by
:func:`handle::call` when the input arguments cannot be converted to Python
objects.

Registering custom translators
==============================

If the default exception conversion policy described above is insufficient,
pybind11 also provides support for registering custom exception translators.
To register a simple exception conversion that translates a C++ exception into
a new Python exception using the C++ exception's ``what()`` method, a helper
function is available:

.. code-block:: cpp

    py::register_exception<CppExp>(module, "PyExp");

This call creates a Python exception class with the name ``PyExp`` in the given
module and automatically converts any encountered exceptions of type ``CppExp``
into Python exceptions of type ``PyExp``.

When more advanced exception translation is needed, the function
``py::register_exception_translator(translator)`` can be used to register
functions that can translate arbitrary exception types (and which may include
additional logic to do so).  The function takes a stateless callable (e.g.  a
function pointer or a lambda function without captured variables) with the call
signature ``void(std::exception_ptr)``.

When a C++ exception is thrown, the registered exception translators are tried
in reverse order of registration (i.e. the last registered translator gets the
first shot at handling the exception).

Inside the translator, ``std::rethrow_exception`` should be used within
a try block to re-throw the exception.  One or more catch clauses to catch
the appropriate exceptions should then be used with each clause using
``PyErr_SetString`` to set a Python exception or ``ex(string)`` to set
the python exception to a custom exception type (see below).

To declare a custom Python exception type, declare a ``py::exception`` variable
and use this in the associated exception translator (note: it is often useful
to make this a static declaration when using it inside a lambda expression
without requiring capturing).


The following example demonstrates this for a hypothetical exception classes
``MyCustomException`` and ``OtherException``: the first is translated to a
custom python exception ``MyCustomError``, while the second is translated to a
standard python RuntimeError:

.. code-block:: cpp

    static py::exception<MyCustomException> exc(m, "MyCustomError");
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyCustomException &e) {
            exc(e.what());
        } catch (const OtherException &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

Multiple exceptions can be handled by a single translator, as shown in the
example above. If the exception is not caught by the current translator, the
previously registered one gets a chance.

If none of the registered exception translators is able to handle the
exception, it is handled by the default converter as described in the previous
section.

.. seealso::

    The file :file:`tests/test_exceptions.cpp` contains examples
    of various custom exception translators and custom exception types.

.. note::

    You must call either ``PyErr_SetString`` or a custom exception's call
    operator (``exc(string)``) for every exception caught in a custom exception
    translator.  Failure to do so will cause Python to crash with ``SystemError:
    error return without exception set``.

    Exceptions that you do not plan to handle should simply not be caught, or
    may be explicitly (re-)thrown to delegate it to the other,
    previously-declared existing exception translators.
