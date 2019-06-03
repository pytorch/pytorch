Functional
##########

The following features must be enabled by including :file:`pybind11/functional.h`.


Callbacks and passing anonymous functions
=========================================

The C++11 standard brought lambda functions and the generic polymorphic
function wrapper ``std::function<>`` to the C++ programming language, which
enable powerful new ways of working with functions. Lambda functions come in
two flavors: stateless lambda function resemble classic function pointers that
link to an anonymous piece of code, while stateful lambda functions
additionally depend on captured variables that are stored in an anonymous
*lambda closure object*.

Here is a simple example of a C++ function that takes an arbitrary function
(stateful or stateless) with signature ``int -> int`` as an argument and runs
it with the value 10.

.. code-block:: cpp

    int func_arg(const std::function<int(int)> &f) {
        return f(10);
    }

The example below is more involved: it takes a function of signature ``int -> int``
and returns another function of the same kind. The return value is a stateful
lambda function, which stores the value ``f`` in the capture object and adds 1 to
its return value upon execution.

.. code-block:: cpp

    std::function<int(int)> func_ret(const std::function<int(int)> &f) {
        return [f](int i) {
            return f(i) + 1;
        };
    }

This example demonstrates using python named parameters in C++ callbacks which
requires using ``py::cpp_function`` as a wrapper. Usage is similar to defining
methods of classes:

.. code-block:: cpp

    py::cpp_function func_cpp() {
        return py::cpp_function([](int i) { return i+1; },
           py::arg("number"));
    }

After including the extra header file :file:`pybind11/functional.h`, it is almost
trivial to generate binding code for all of these functions.

.. code-block:: cpp

    #include <pybind11/functional.h>

    PYBIND11_MODULE(example, m) {
        m.def("func_arg", &func_arg);
        m.def("func_ret", &func_ret);
        m.def("func_cpp", &func_cpp);
    }

The following interactive session shows how to call them from Python.

.. code-block:: pycon

    $ python
    >>> import example
    >>> def square(i):
    ...     return i * i
    ...
    >>> example.func_arg(square)
    100L
    >>> square_plus_1 = example.func_ret(square)
    >>> square_plus_1(4)
    17L
    >>> plus_1 = func_cpp()
    >>> plus_1(number=43)
    44L

.. warning::

    Keep in mind that passing a function from C++ to Python (or vice versa)
    will instantiate a piece of wrapper code that translates function
    invocations between the two languages. Naturally, this translation
    increases the computational cost of each function call somewhat. A
    problematic situation can arise when a function is copied back and forth
    between Python and C++ many times in a row, in which case the underlying
    wrappers will accumulate correspondingly. The resulting long sequence of
    C++ -> Python -> C++ -> ... roundtrips can significantly decrease
    performance.

    There is one exception: pybind11 detects case where a stateless function
    (i.e. a function pointer or a lambda function without captured variables)
    is passed as an argument to another C++ function exposed in Python. In this
    case, there is no overhead. Pybind11 will extract the underlying C++
    function pointer from the wrapped function to sidestep a potential C++ ->
    Python -> C++ roundtrip. This is demonstrated in :file:`tests/test_callbacks.cpp`.

.. note::

    This functionality is very useful when generating bindings for callbacks in
    C++ libraries (e.g. GUI libraries, asynchronous networking libraries, etc.).

    The file :file:`tests/test_callbacks.cpp` contains a complete example
    that demonstrates how to work with callbacks and anonymous functions in
    more detail.
