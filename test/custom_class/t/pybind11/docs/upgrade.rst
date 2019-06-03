Upgrade guide
#############

This is a companion guide to the :doc:`changelog`. While the changelog briefly
lists all of the new features, improvements and bug fixes, this upgrade guide
focuses only the subset which directly impacts your experience when upgrading
to a new version. But it goes into more detail. This includes things like
deprecated APIs and their replacements, build system changes, general code
modernization and other useful information.


v2.2
====

Deprecation of the ``PYBIND11_PLUGIN`` macro
--------------------------------------------

``PYBIND11_MODULE`` is now the preferred way to create module entry points.
The old macro emits a compile-time deprecation warning.

.. code-block:: cpp

    // old
    PYBIND11_PLUGIN(example) {
        py::module m("example", "documentation string");

        m.def("add", [](int a, int b) { return a + b; });

        return m.ptr();
    }

    // new
    PYBIND11_MODULE(example, m) {
        m.doc() = "documentation string"; // optional

        m.def("add", [](int a, int b) { return a + b; });
    }


New API for defining custom constructors and pickling functions
---------------------------------------------------------------

The old placement-new custom constructors have been deprecated. The new approach
uses ``py::init()`` and factory functions to greatly improve type safety.

Placement-new can be called accidentally with an incompatible type (without any
compiler errors or warnings), or it can initialize the same object multiple times
if not careful with the Python-side ``__init__`` calls. The new-style custom
constructors prevent such mistakes. See :ref:`custom_constructors` for details.

.. code-block:: cpp

    // old -- deprecated (runtime warning shown only in debug mode)
    py::class<Foo>(m, "Foo")
        .def("__init__", [](Foo &self, ...) {
            new (&self) Foo(...); // uses placement-new
        });

    // new
    py::class<Foo>(m, "Foo")
        .def(py::init([](...) { // Note: no `self` argument
            return new Foo(...); // return by raw pointer
            // or: return std::make_unique<Foo>(...); // return by holder
            // or: return Foo(...); // return by value (move constructor)
        }));

Mirroring the custom constructor changes, ``py::pickle()`` is now the preferred
way to get and set object state. See :ref:`pickling` for details.

.. code-block:: cpp

    // old -- deprecated (runtime warning shown only in debug mode)
    py::class<Foo>(m, "Foo")
        ...
        .def("__getstate__", [](const Foo &self) {
            return py::make_tuple(self.value1(), self.value2(), ...);
        })
        .def("__setstate__", [](Foo &self, py::tuple t) {
            new (&self) Foo(t[0].cast<std::string>(), ...);
        });

    // new
    py::class<Foo>(m, "Foo")
        ...
        .def(py::pickle(
            [](const Foo &self) { // __getstate__
                return py::make_tuple(f.value1(), f.value2(), ...); // unchanged
            },
            [](py::tuple t) { // __setstate__, note: no `self` argument
                return new Foo(t[0].cast<std::string>(), ...);
                // or: return std::make_unique<Foo>(...); // return by holder
                // or: return Foo(...); // return by value (move constructor)
            }
        ));

For both the constructors and pickling, warnings are shown at module
initialization time (on import, not when the functions are called).
They're only visible when compiled in debug mode. Sample warning:

.. code-block:: none

    pybind11-bound class 'mymodule.Foo' is using an old-style placement-new '__init__'
    which has been deprecated. See the upgrade guide in pybind11's docs.


Stricter enforcement of hidden symbol visibility for pybind11 modules
---------------------------------------------------------------------

pybind11 now tries to actively enforce hidden symbol visibility for modules.
If you're using either one of pybind11's :doc:`CMake or Python build systems
<compiling>` (the two example repositories) and you haven't been exporting any
symbols, there's nothing to be concerned about. All the changes have been done
transparently in the background. If you were building manually or relied on
specific default visibility, read on.

Setting default symbol visibility to *hidden* has always been recommended for
pybind11 (see :ref:`faq:symhidden`). On Linux and macOS, hidden symbol
visibility (in conjunction with the ``strip`` utility) yields much smaller
module binaries. `CPython's extension docs`_ also recommend hiding symbols
by default, with the goal of avoiding symbol name clashes between modules.
Starting with v2.2, pybind11 enforces this more strictly: (1) by declaring
all symbols inside the ``pybind11`` namespace as hidden and (2) by including
the ``-fvisibility=hidden`` flag on Linux and macOS (only for extension
modules, not for embedding the interpreter).

.. _CPython's extension docs: https://docs.python.org/3/extending/extending.html#providing-a-c-api-for-an-extension-module

The namespace-scope hidden visibility is done automatically in pybind11's
headers and it's generally transparent to users. It ensures that:

* Modules compiled with different pybind11 versions don't clash with each other.

* Some new features, like ``py::module_local`` bindings, can work as intended.

The ``-fvisibility=hidden`` flag applies the same visibility to user bindings
outside of the ``pybind11`` namespace. It's now set automatic by pybind11's
CMake and Python build systems, but this needs to be done manually by users
of other build systems. Adding this flag:

* Minimizes the chances of symbol conflicts between modules. E.g. if two
  unrelated modules were statically linked to different (ABI-incompatible)
  versions of the same third-party library, a symbol clash would be likely
  (and would end with unpredictable results).

* Produces smaller binaries on Linux and macOS, as pointed out previously.

Within pybind11's CMake build system, ``pybind11_add_module`` has always been
setting the ``-fvisibility=hidden`` flag in release mode. From now on, it's
being applied unconditionally, even in debug mode and it can no longer be opted
out of with the ``NO_EXTRAS`` option. The ``pybind11::module`` target now also
adds this flag to it's interface. The ``pybind11::embed`` target is unchanged.

The most significant change here is for the ``pybind11::module`` target. If you
were previously relying on default visibility, i.e. if your Python module was
doubling as a shared library with dependents, you'll need to either export
symbols manually (recommended for cross-platform libraries) or factor out the
shared library (and have the Python module link to it like the other
dependents). As a temporary workaround, you can also restore default visibility
using the CMake code below, but this is not recommended in the long run:

.. code-block:: cmake

    target_link_libraries(mymodule PRIVATE pybind11::module)

    add_library(restore_default_visibility INTERFACE)
    target_compile_options(restore_default_visibility INTERFACE -fvisibility=default)
    target_link_libraries(mymodule PRIVATE restore_default_visibility)


Local STL container bindings
----------------------------

Previous pybind11 versions could only bind types globally -- all pybind11
modules, even unrelated ones, would have access to the same exported types.
However, this would also result in a conflict if two modules exported the
same C++ type, which is especially problematic for very common types, e.g.
``std::vector<int>``. :ref:`module_local` were added to resolve this (see
that section for a complete usage guide).

``py::class_`` still defaults to global bindings (because these types are
usually unique across modules), however in order to avoid clashes of opaque
types, ``py::bind_vector`` and ``py::bind_map`` will now bind STL containers
as ``py::module_local`` if their elements are: builtins (``int``, ``float``,
etc.), not bound using ``py::class_``, or bound as ``py::module_local``. For
example, this change allows multiple modules to bind ``std::vector<int>``
without causing conflicts. See :ref:`stl_bind` for more details.

When upgrading to this version, if you have multiple modules which depend on
a single global binding of an STL container, note that all modules can still
accept foreign  ``py::module_local`` types in the direction of Python-to-C++.
The locality only affects the C++-to-Python direction. If this is needed in
multiple modules, you'll need to either:

* Add a copy of the same STL binding to all of the modules which need it.

* Restore the global status of that single binding by marking it
  ``py::module_local(false)``.

The latter is an easy workaround, but in the long run it would be best to
localize all common type bindings in order to avoid conflicts with
third-party modules.


Negative strides for Python buffer objects and numpy arrays
-----------------------------------------------------------

Support for negative strides required changing the integer type from unsigned
to signed in the interfaces of ``py::buffer_info`` and ``py::array``. If you
have compiler warnings enabled, you may notice some new conversion warnings
after upgrading. These can be resolved using ``static_cast``.


Deprecation of some ``py::object`` APIs
---------------------------------------

To compare ``py::object`` instances by pointer, you should now use
``obj1.is(obj2)`` which is equivalent to ``obj1 is obj2`` in Python.
Previously, pybind11 used ``operator==`` for this (``obj1 == obj2``), but
that could be confusing and is now deprecated (so that it can eventually
be replaced with proper rich object comparison in a future release).

For classes which inherit from ``py::object``, ``borrowed`` and ``stolen``
were previously available as protected constructor tags. Now the types
should be used directly instead: ``borrowed_t{}`` and ``stolen_t{}``
(`#771 <https://github.com/pybind/pybind11/pull/771>`_).


Stricter compile-time error checking
------------------------------------

Some error checks have been moved from run time to compile time. Notably,
automatic conversion of ``std::shared_ptr<T>`` is not possible when ``T`` is
not directly registered with ``py::class_<T>`` (e.g. ``std::shared_ptr<int>``
or ``std::shared_ptr<std::vector<T>>`` are not automatically convertible).
Attempting to bind a function with such arguments now results in a compile-time
error instead of waiting to fail at run time.

``py::init<...>()`` constructor definitions are also stricter and now prevent
bindings which could cause unexpected behavior:

.. code-block:: cpp

    struct Example {
        Example(int &);
    };

    py::class_<Example>(m, "Example")
        .def(py::init<int &>()); // OK, exact match
        // .def(py::init<int>()); // compile-time error, mismatch

A non-``const`` lvalue reference is not allowed to bind to an rvalue. However,
note that a constructor taking ``const T &`` can still be registered using
``py::init<T>()`` because a ``const`` lvalue reference can bind to an rvalue.

v2.1
====

Minimum compiler versions are enforced at compile time
------------------------------------------------------

The minimums also apply to v2.0 but the check is now explicit and a compile-time
error is raised if the compiler does not meet the requirements:

* GCC >= 4.8
* clang >= 3.3 (appleclang >= 5.0)
* MSVC >= 2015u3
* Intel C++ >= 15.0


The ``py::metaclass`` attribute is not required for static properties
---------------------------------------------------------------------

Binding classes with static properties is now possible by default. The
zero-parameter version of ``py::metaclass()`` is deprecated. However, a new
one-parameter ``py::metaclass(python_type)`` version was added for rare
cases when a custom metaclass is needed to override pybind11's default.

.. code-block:: cpp

    // old -- emits a deprecation warning
    py::class_<Foo>(m, "Foo", py::metaclass())
        .def_property_readonly_static("foo", ...);

    // new -- static properties work without the attribute
    py::class_<Foo>(m, "Foo")
        .def_property_readonly_static("foo", ...);

    // new -- advanced feature, override pybind11's default metaclass
    py::class_<Bar>(m, "Bar", py::metaclass(custom_python_type))
        ...


v2.0
====

Breaking changes in ``py::class_``
----------------------------------

These changes were necessary to make type definitions in pybind11
future-proof, to support PyPy via its ``cpyext`` mechanism (`#527
<https://github.com/pybind/pybind11/pull/527>`_), and to improve efficiency
(`rev. 86d825 <https://github.com/pybind/pybind11/commit/86d825>`_).

1. Declarations of types that provide access via the buffer protocol must
   now include the ``py::buffer_protocol()`` annotation as an argument to
   the ``py::class_`` constructor.

   .. code-block:: cpp

       py::class_<Matrix>("Matrix", py::buffer_protocol())
           .def(py::init<...>())
           .def_buffer(...);

2. Classes which include static properties (e.g. ``def_readwrite_static()``)
   must now include the ``py::metaclass()`` attribute. Note: this requirement
   has since been removed in v2.1. If you're upgrading from 1.x, it's
   recommended to skip directly to v2.1 or newer.

3. This version of pybind11 uses a redesigned mechanism for instantiating
   trampoline classes that are used to override virtual methods from within
   Python. This led to the following user-visible syntax change:

   .. code-block:: cpp

       // old v1.x syntax
       py::class_<TrampolineClass>("MyClass")
           .alias<MyClass>()
           ...

       // new v2.x syntax
       py::class_<MyClass, TrampolineClass>("MyClass")
           ...

   Importantly, both the original and the trampoline class are now specified
   as arguments to the ``py::class_`` template, and the ``alias<..>()`` call
   is gone. The new scheme has zero overhead in cases when Python doesn't
   override any functions of the underlying C++ class.
   `rev. 86d825 <https://github.com/pybind/pybind11/commit/86d825>`_.

   The class type must be the first template argument given to ``py::class_``
   while the trampoline can be mixed in arbitrary order with other arguments
   (see the following section).


Deprecation of the ``py::base<T>()`` attribute
----------------------------------------------

``py::base<T>()`` was deprecated in favor of specifying ``T`` as a template
argument to ``py::class_``. This new syntax also supports multiple inheritance.
Note that, while the type being exported must be the first argument in the
``py::class_<Class, ...>`` template, the order of the following types (bases,
holder and/or trampoline) is not important.

.. code-block:: cpp

    // old v1.x
    py::class_<Derived>("Derived", py::base<Base>());

    // new v2.x
    py::class_<Derived, Base>("Derived");

    // new -- multiple inheritance
    py::class_<Derived, Base1, Base2>("Derived");

    // new -- apart from `Derived` the argument order can be arbitrary
    py::class_<Derived, Base1, Holder, Base2, Trampoline>("Derived");


Out-of-the-box support for ``std::shared_ptr``
----------------------------------------------

The relevant type caster is now built in, so it's no longer necessary to
include a declaration of the form:

.. code-block:: cpp

    PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)

Continuing to do so won’t cause an error or even a deprecation warning,
but it's completely redundant.


Deprecation of a few ``py::object`` APIs
----------------------------------------

All of the old-style calls emit deprecation warnings.

+---------------------------------------+---------------------------------------------+
|  Old syntax                           |  New syntax                                 |
+=======================================+=============================================+
| ``obj.call(args...)``                 | ``obj(args...)``                            |
+---------------------------------------+---------------------------------------------+
| ``obj.str()``                         | ``py::str(obj)``                            |
+---------------------------------------+---------------------------------------------+
| ``auto l = py::list(obj); l.check()`` | ``py::isinstance<py::list>(obj)``           |
+---------------------------------------+---------------------------------------------+
| ``py::object(ptr, true)``             | ``py::reinterpret_borrow<py::object>(ptr)`` |
+---------------------------------------+---------------------------------------------+
| ``py::object(ptr, false)``            | ``py::reinterpret_steal<py::object>(ptr)``  |
+---------------------------------------+---------------------------------------------+
| ``if (obj.attr("foo"))``              | ``if (py::hasattr(obj, "foo"))``            |
+---------------------------------------+---------------------------------------------+
| ``if (obj["bar"])``                   | ``if (obj.contains("bar"))``                |
+---------------------------------------+---------------------------------------------+
