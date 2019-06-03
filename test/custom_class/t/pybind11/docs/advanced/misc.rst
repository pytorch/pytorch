Miscellaneous
#############

.. _macro_notes:

General notes regarding convenience macros
==========================================

pybind11 provides a few convenience macros such as
:func:`PYBIND11_DECLARE_HOLDER_TYPE` and ``PYBIND11_OVERLOAD_*``. Since these
are "just" macros that are evaluated in the preprocessor (which has no concept
of types), they *will* get confused by commas in a template argument; for
example, consider:

.. code-block:: cpp

    PYBIND11_OVERLOAD(MyReturnType<T1, T2>, Class<T3, T4>, func)

The limitation of the C preprocessor interprets this as five arguments (with new
arguments beginning after each comma) rather than three.  To get around this,
there are two alternatives: you can use a type alias, or you can wrap the type
using the ``PYBIND11_TYPE`` macro:

.. code-block:: cpp

    // Version 1: using a type alias
    using ReturnType = MyReturnType<T1, T2>;
    using ClassType = Class<T3, T4>;
    PYBIND11_OVERLOAD(ReturnType, ClassType, func);

    // Version 2: using the PYBIND11_TYPE macro:
    PYBIND11_OVERLOAD(PYBIND11_TYPE(MyReturnType<T1, T2>),
                      PYBIND11_TYPE(Class<T3, T4>), func)

The ``PYBIND11_MAKE_OPAQUE`` macro does *not* require the above workarounds.

.. _gil:

Global Interpreter Lock (GIL)
=============================

When calling a C++ function from Python, the GIL is always held.
The classes :class:`gil_scoped_release` and :class:`gil_scoped_acquire` can be
used to acquire and release the global interpreter lock in the body of a C++
function call. In this way, long-running C++ code can be parallelized using
multiple Python threads. Taking :ref:`overriding_virtuals` as an example, this
could be realized as follows (important changes highlighted):

.. code-block:: cpp
    :emphasize-lines: 8,9,31,32

    class PyAnimal : public Animal {
    public:
        /* Inherit the constructors */
        using Animal::Animal;

        /* Trampoline (need one for each virtual function) */
        std::string go(int n_times) {
            /* Acquire GIL before calling Python code */
            py::gil_scoped_acquire acquire;

            PYBIND11_OVERLOAD_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function */
                n_times      /* Argument(s) */
            );
        }
    };

    PYBIND11_MODULE(example, m) {
        py::class_<Animal, PyAnimal> animal(m, "Animal");
        animal
            .def(py::init<>())
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", [](Animal *animal) -> std::string {
            /* Release GIL before calling into (potentially long-running) C++ code */
            py::gil_scoped_release release;
            return call_go(animal);
        });
    }

The ``call_go`` wrapper can also be simplified using the `call_guard` policy
(see :ref:`call_policies`) which yields the same result:

.. code-block:: cpp

    m.def("call_go", &call_go, py::call_guard<py::gil_scoped_release>());


Binding sequence data types, iterators, the slicing protocol, etc.
==================================================================

Please refer to the supplemental example for details.

.. seealso::

    The file :file:`tests/test_sequences_and_iterators.cpp` contains a
    complete example that shows how to bind a sequence data type, including
    length queries (``__len__``), iterators (``__iter__``), the slicing
    protocol and other kinds of useful operations.


Partitioning code over multiple extension modules
=================================================

It's straightforward to split binding code over multiple extension modules,
while referencing types that are declared elsewhere. Everything "just" works
without any special precautions. One exception to this rule occurs when
extending a type declared in another extension module. Recall the basic example
from Section :ref:`inheritance`.

.. code-block:: cpp

    py::class_<Pet> pet(m, "Pet");
    pet.def(py::init<const std::string &>())
       .def_readwrite("name", &Pet::name);

    py::class_<Dog>(m, "Dog", pet /* <- specify parent */)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Suppose now that ``Pet`` bindings are defined in a module named ``basic``,
whereas the ``Dog`` bindings are defined somewhere else. The challenge is of
course that the variable ``pet`` is not available anymore though it is needed
to indicate the inheritance relationship to the constructor of ``class_<Dog>``.
However, it can be acquired as follows:

.. code-block:: cpp

    py::object pet = (py::object) py::module::import("basic").attr("Pet");

    py::class_<Dog>(m, "Dog", pet)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Alternatively, you can specify the base class as a template parameter option to
``class_``, which performs an automated lookup of the corresponding Python
type. Like the above code, however, this also requires invoking the ``import``
function once to ensure that the pybind11 binding code of the module ``basic``
has been executed:

.. code-block:: cpp

    py::module::import("basic");

    py::class_<Dog, Pet>(m, "Dog")
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Naturally, both methods will fail when there are cyclic dependencies.

Note that pybind11 code compiled with hidden-by-default symbol visibility (e.g.
via the command line flag ``-fvisibility=hidden`` on GCC/Clang), which is
required for proper pybind11 functionality, can interfere with the ability to
access types defined in another extension module.  Working around this requires
manually exporting types that are accessed by multiple extension modules;
pybind11 provides a macro to do just this:

.. code-block:: cpp

    class PYBIND11_EXPORT Dog : public Animal {
        ...
    };

Note also that it is possible (although would rarely be required) to share arbitrary
C++ objects between extension modules at runtime. Internal library data is shared
between modules using capsule machinery [#f6]_ which can be also utilized for
storing, modifying and accessing user-defined data. Note that an extension module
will "see" other extensions' data if and only if they were built with the same
pybind11 version. Consider the following example:

.. code-block:: cpp

    auto data = (MyData *) py::get_shared_data("mydata");
    if (!data)
        data = (MyData *) py::set_shared_data("mydata", new MyData(42));

If the above snippet was used in several separately compiled extension modules,
the first one to be imported would create a ``MyData`` instance and associate
a ``"mydata"`` key with a pointer to it. Extensions that are imported later
would be then able to access the data behind the same pointer.

.. [#f6] https://docs.python.org/3/extending/extending.html#using-capsules

Module Destructors
==================

pybind11 does not provide an explicit mechanism to invoke cleanup code at
module destruction time. In rare cases where such functionality is required, it
is possible to emulate it using Python capsules or weak references with a
destruction callback.

.. code-block:: cpp

    auto cleanup_callback = []() {
        // perform cleanup here -- this function is called with the GIL held
    };

    m.add_object("_cleanup", py::capsule(cleanup_callback));

This approach has the potential downside that instances of classes exposed
within the module may still be alive when the cleanup callback is invoked
(whether this is acceptable will generally depend on the application).

Alternatively, the capsule may also be stashed within a type object, which
ensures that it not called before all instances of that type have been
collected:

.. code-block:: cpp

    auto cleanup_callback = []() { /* ... */ };
    m.attr("BaseClass").attr("_cleanup") = py::capsule(cleanup_callback);

Both approaches also expose a potentially dangerous ``_cleanup`` attribute in
Python, which may be undesirable from an API standpoint (a premature explicit
call from Python might lead to undefined behavior). Yet another approach that 
avoids this issue involves weak reference with a cleanup callback:

.. code-block:: cpp

    // Register a callback function that is invoked when the BaseClass object is colelcted
    py::cpp_function cleanup_callback(
        [](py::handle weakref) {
            // perform cleanup here -- this function is called with the GIL held

            weakref.dec_ref(); // release weak reference
        }
    );

    // Create a weak reference with a cleanup callback and initially leak it
    (void) py::weakref(m.attr("BaseClass"), cleanup_callback).release();

.. note::

    PyPy (at least version 5.9) does not garbage collect objects when the
    interpreter exits. An alternative approach (which also works on CPython) is to use
    the :py:mod:`atexit` module [#f7]_, for example:

    .. code-block:: cpp

        auto atexit = py::module::import("atexit");
        atexit.attr("register")(py::cpp_function([]() {
            // perform cleanup here -- this function is called with the GIL held
        }));

    .. [#f7] https://docs.python.org/3/library/atexit.html


Generating documentation using Sphinx
=====================================

Sphinx [#f4]_ has the ability to inspect the signatures and documentation
strings in pybind11-based extension modules to automatically generate beautiful
documentation in a variety formats. The python_example repository [#f5]_ contains a
simple example repository which uses this approach.

There are two potential gotchas when using this approach: first, make sure that
the resulting strings do not contain any :kbd:`TAB` characters, which break the
docstring parsing routines. You may want to use C++11 raw string literals,
which are convenient for multi-line comments. Conveniently, any excess
indentation will be automatically be removed by Sphinx. However, for this to
work, it is important that all lines are indented consistently, i.e.:

.. code-block:: cpp

    // ok
    m.def("foo", &foo, R"mydelimiter(
        The foo function

        Parameters
        ----------
    )mydelimiter");

    // *not ok*
    m.def("foo", &foo, R"mydelimiter(The foo function

        Parameters
        ----------
    )mydelimiter");

By default, pybind11 automatically generates and prepends a signature to the docstring of a function 
registered with ``module::def()`` and ``class_::def()``. Sometimes this
behavior is not desirable, because you want to provide your own signature or remove 
the docstring completely to exclude the function from the Sphinx documentation.
The class ``options`` allows you to selectively suppress auto-generated signatures:

.. code-block:: cpp

    PYBIND11_MODULE(example, m) {
        py::options options;
        options.disable_function_signatures();

        m.def("add", [](int a, int b) { return a + b; }, "A function which adds two numbers");
    }

Note that changes to the settings affect only function bindings created during the 
lifetime of the ``options`` instance. When it goes out of scope at the end of the module's init function, 
the default settings are restored to prevent unwanted side effects.

.. [#f4] http://www.sphinx-doc.org
.. [#f5] http://github.com/pybind/python_example
