.. _embedding:

Embedding the interpreter
#########################

While pybind11 is mainly focused on extending Python using C++, it's also
possible to do the reverse: embed the Python interpreter into a C++ program.
All of the other documentation pages still apply here, so refer to them for
general pybind11 usage. This section will cover a few extra things required
for embedding.

Getting started
===============

A basic executable with an embedded interpreter can be created with just a few
lines of CMake and the ``pybind11::embed`` target, as shown below. For more
information, see :doc:`/compiling`.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.0)
    project(example)

    find_package(pybind11 REQUIRED)  # or `add_subdirectory(pybind11)`

    add_executable(example main.cpp)
    target_link_libraries(example PRIVATE pybind11::embed)

The essential structure of the ``main.cpp`` file looks like this:

.. code-block:: cpp

    #include <pybind11/embed.h> // everything needed for embedding
    namespace py = pybind11;

    int main() {
        py::scoped_interpreter guard{}; // start the interpreter and keep it alive

        py::print("Hello, World!"); // use the Python API
    }

The interpreter must be initialized before using any Python API, which includes
all the functions and classes in pybind11. The RAII guard class `scoped_interpreter`
takes care of the interpreter lifetime. After the guard is destroyed, the interpreter
shuts down and clears its memory. No Python functions can be called after this.

Executing Python code
=====================

There are a few different ways to run Python code. One option is to use `eval`,
`exec` or `eval_file`, as explained in :ref:`eval`. Here is a quick example in
the context of an executable with an embedded interpreter:

.. code-block:: cpp

    #include <pybind11/embed.h>
    namespace py = pybind11;

    int main() {
        py::scoped_interpreter guard{};

        py::exec(R"(
            kwargs = dict(name="World", number=42)
            message = "Hello, {name}! The answer is {number}".format(**kwargs)
            print(message)
        )");
    }

Alternatively, similar results can be achieved using pybind11's API (see
:doc:`/advanced/pycpp/index` for more details).

.. code-block:: cpp

    #include <pybind11/embed.h>
    namespace py = pybind11;
    using namespace py::literals;

    int main() {
        py::scoped_interpreter guard{};

        auto kwargs = py::dict("name"_a="World", "number"_a=42);
        auto message = "Hello, {name}! The answer is {number}"_s.format(**kwargs);
        py::print(message);
    }

The two approaches can also be combined:

.. code-block:: cpp

    #include <pybind11/embed.h>
    #include <iostream>

    namespace py = pybind11;
    using namespace py::literals;

    int main() {
        py::scoped_interpreter guard{};

        auto locals = py::dict("name"_a="World", "number"_a=42);
        py::exec(R"(
            message = "Hello, {name}! The answer is {number}".format(**locals())
        )", py::globals(), locals);

        auto message = locals["message"].cast<std::string>();
        std::cout << message;
    }

Importing modules
=================

Python modules can be imported using `module::import()`:

.. code-block:: cpp

    py::module sys = py::module::import("sys");
    py::print(sys.attr("path"));

For convenience, the current working directory is included in ``sys.path`` when
embedding the interpreter. This makes it easy to import local Python files:

.. code-block:: python

    """calc.py located in the working directory"""

    def add(i, j):
        return i + j


.. code-block:: cpp

    py::module calc = py::module::import("calc");
    py::object result = calc.attr("add")(1, 2);
    int n = result.cast<int>();
    assert(n == 3);

Modules can be reloaded using `module::reload()` if the source is modified e.g.
by an external process. This can be useful in scenarios where the application
imports a user defined data processing script which needs to be updated after
changes by the user. Note that this function does not reload modules recursively.

.. _embedding_modules:

Adding embedded modules
=======================

Embedded binary modules can be added using the `PYBIND11_EMBEDDED_MODULE` macro.
Note that the definition must be placed at global scope. They can be imported
like any other module.

.. code-block:: cpp

    #include <pybind11/embed.h>
    namespace py = pybind11;

    PYBIND11_EMBEDDED_MODULE(fast_calc, m) {
        // `m` is a `py::module` which is used to bind functions and classes
        m.def("add", [](int i, int j) {
            return i + j;
        });
    }

    int main() {
        py::scoped_interpreter guard{};

        auto fast_calc = py::module::import("fast_calc");
        auto result = fast_calc.attr("add")(1, 2).cast<int>();
        assert(result == 3);
    }

Unlike extension modules where only a single binary module can be created, on
the embedded side an unlimited number of modules can be added using multiple
`PYBIND11_EMBEDDED_MODULE` definitions (as long as they have unique names).

These modules are added to Python's list of builtins, so they can also be
imported in pure Python files loaded by the interpreter. Everything interacts
naturally:

.. code-block:: python

    """py_module.py located in the working directory"""
    import cpp_module

    a = cpp_module.a
    b = a + 1


.. code-block:: cpp

    #include <pybind11/embed.h>
    namespace py = pybind11;

    PYBIND11_EMBEDDED_MODULE(cpp_module, m) {
        m.attr("a") = 1;
    }

    int main() {
        py::scoped_interpreter guard{};

        auto py_module = py::module::import("py_module");

        auto locals = py::dict("fmt"_a="{} + {} = {}", **py_module.attr("__dict__"));
        assert(locals["a"].cast<int>() == 1);
        assert(locals["b"].cast<int>() == 2);

        py::exec(R"(
            c = a + b
            message = fmt.format(a, b, c)
        )", py::globals(), locals);

        assert(locals["c"].cast<int>() == 3);
        assert(locals["message"].cast<std::string>() == "1 + 2 = 3");
    }


Interpreter lifetime
====================

The Python interpreter shuts down when `scoped_interpreter` is destroyed. After
this, creating a new instance will restart the interpreter. Alternatively, the
`initialize_interpreter` / `finalize_interpreter` pair of functions can be used
to directly set the state at any time.

Modules created with pybind11 can be safely re-initialized after the interpreter
has been restarted. However, this may not apply to third-party extension modules.
The issue is that Python itself cannot completely unload extension modules and
there are several caveats with regard to interpreter restarting. In short, not
all memory may be freed, either due to Python reference cycles or user-created
global data. All the details can be found in the CPython documentation.

.. warning::

    Creating two concurrent `scoped_interpreter` guards is a fatal error. So is
    calling `initialize_interpreter` for a second time after the interpreter
    has already been initialized.

    Do not use the raw CPython API functions ``Py_Initialize`` and
    ``Py_Finalize`` as these do not properly handle the lifetime of
    pybind11's internal data.


Sub-interpreter support
=======================

Creating multiple copies of `scoped_interpreter` is not possible because it
represents the main Python interpreter. Sub-interpreters are something different
and they do permit the existence of multiple interpreters. This is an advanced
feature of the CPython API and should be handled with care. pybind11 does not
currently offer a C++ interface for sub-interpreters, so refer to the CPython
documentation for all the details regarding this feature.

We'll just mention a couple of caveats the sub-interpreters support in pybind11:

 1. Sub-interpreters will not receive independent copies of embedded modules.
    Instead, these are shared and modifications in one interpreter may be
    reflected in another.

 2. Managing multiple threads, multiple interpreters and the GIL can be
    challenging and there are several caveats here, even within the pure
    CPython API (please refer to the Python docs for details). As for
    pybind11, keep in mind that `gil_scoped_release` and `gil_scoped_acquire`
    do not take sub-interpreters into account.
