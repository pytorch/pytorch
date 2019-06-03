Python types
############

Available wrappers
==================

All major Python types are available as thin C++ wrapper classes. These
can also be used as function parameters -- see :ref:`python_objects_as_args`.

Available types include :class:`handle`, :class:`object`, :class:`bool_`,
:class:`int_`, :class:`float_`, :class:`str`, :class:`bytes`, :class:`tuple`,
:class:`list`, :class:`dict`, :class:`slice`, :class:`none`, :class:`capsule`,
:class:`iterable`, :class:`iterator`, :class:`function`, :class:`buffer`,
:class:`array`, and :class:`array_t`.

Casting back and forth
======================

In this kind of mixed code, it is often necessary to convert arbitrary C++
types to Python, which can be done using :func:`py::cast`:

.. code-block:: cpp

    MyClass *cls = ..;
    py::object obj = py::cast(cls);

The reverse direction uses the following syntax:

.. code-block:: cpp

    py::object obj = ...;
    MyClass *cls = obj.cast<MyClass *>();

When conversion fails, both directions throw the exception :class:`cast_error`.

.. _python_libs:

Accessing Python libraries from C++
===================================

It is also possible to import objects defined in the Python standard
library or available in the current Python environment (``sys.path``) and work
with these in C++.

This example obtains a reference to the Python ``Decimal`` class.

.. code-block:: cpp

    // Equivalent to "from decimal import Decimal"
    py::object Decimal = py::module::import("decimal").attr("Decimal");

.. code-block:: cpp

    // Try to import scipy
    py::object scipy = py::module::import("scipy");
    return scipy.attr("__version__");

.. _calling_python_functions:

Calling Python functions
========================

It is also possible to call Python classes, functions and methods 
via ``operator()``.

.. code-block:: cpp

    // Construct a Python object of class Decimal
    py::object pi = Decimal("3.14159");

.. code-block:: cpp

    // Use Python to make our directories
    py::object os = py::module::import("os");
    py::object makedirs = os.attr("makedirs");
    makedirs("/tmp/path/to/somewhere");

One can convert the result obtained from Python to a pure C++ version 
if a ``py::class_`` or type conversion is defined.

.. code-block:: cpp

    py::function f = <...>;
    py::object result_py = f(1234, "hello", some_instance);
    MyClass &result = result_py.cast<MyClass>();

.. _calling_python_methods:

Calling Python methods
========================

To call an object's method, one can again use ``.attr`` to obtain access to the
Python method.

.. code-block:: cpp

    // Calculate e^π in decimal
    py::object exp_pi = pi.attr("exp")();
    py::print(py::str(exp_pi));

In the example above ``pi.attr("exp")`` is a *bound method*: it will always call
the method for that same instance of the class. Alternately one can create an 
*unbound method* via the Python class (instead of instance) and pass the ``self`` 
object explicitly, followed by other arguments.

.. code-block:: cpp

    py::object decimal_exp = Decimal.attr("exp");

    // Compute the e^n for n=0..4
    for (int n = 0; n < 5; n++) {
        py::print(decimal_exp(Decimal(n));
    }

Keyword arguments
=================

Keyword arguments are also supported. In Python, there is the usual call syntax:

.. code-block:: python

    def f(number, say, to):
        ...  # function code

    f(1234, say="hello", to=some_instance)  # keyword call in Python

In C++, the same call can be made using:

.. code-block:: cpp

    using namespace pybind11::literals; // to bring in the `_a` literal
    f(1234, "say"_a="hello", "to"_a=some_instance); // keyword call in C++

Unpacking arguments
===================

Unpacking of ``*args`` and ``**kwargs`` is also possible and can be mixed with
other arguments:

.. code-block:: cpp

    // * unpacking
    py::tuple args = py::make_tuple(1234, "hello", some_instance);
    f(*args);

    // ** unpacking
    py::dict kwargs = py::dict("number"_a=1234, "say"_a="hello", "to"_a=some_instance);
    f(**kwargs);

    // mixed keywords, * and ** unpacking
    py::tuple args = py::make_tuple(1234);
    py::dict kwargs = py::dict("to"_a=some_instance);
    f(*args, "say"_a="hello", **kwargs);

Generalized unpacking according to PEP448_ is also supported:

.. code-block:: cpp

    py::dict kwargs1 = py::dict("number"_a=1234);
    py::dict kwargs2 = py::dict("to"_a=some_instance);
    f(**kwargs1, "say"_a="hello", **kwargs2);

.. seealso::

    The file :file:`tests/test_pytypes.cpp` contains a complete
    example that demonstrates passing native Python types in more detail. The
    file :file:`tests/test_callbacks.cpp` presents a few examples of calling
    Python functions from C++, including keywords arguments and unpacking.

.. _PEP448: https://www.python.org/dev/peps/pep-0448/
