.. _changelog:

Changelog
#########

Starting with version 1.8.0, pybind11 releases use a `semantic versioning
<http://semver.org>`_ policy.

v2.3.0 (Not yet released)
-----------------------------------------------------

* Significantly reduced module binary size (10-20%) when compiled in C++11 mode
  with GCC/Clang, or in any mode with MSVC. Function signatures are now always
  precomputed at compile time (this was previously only available in C++14 mode
  for non-MSVC compilers).
  `#934 <https://github.com/pybind/pybind11/pull/934>`_.

* Add basic support for tag-based static polymorphism, where classes
  provide a method to returns the desired type of an instance.
  `#1326 <https://github.com/pybind/pybind11/pull/1326>`_.

* Added support for write only properties.
  `#1144 <https://github.com/pybind/pybind11/pull/1144>`_.

* Python type wrappers (``py::handle``, ``py::object``, etc.)
  now support map Python's number protocol onto C++ arithmetic
  operators such as ``operator+``, ``operator/=``, etc.
  `#1511 <https://github.com/pybind/pybind11/pull/1511>`_.

* A number of improvements related to enumerations:

   1. The ``enum_`` implementation was rewritten from scratch to reduce
      code bloat. Rather than instantiating a full implementation for each
      enumeration, most code is now contained in a generic base class.
      `#1511 <https://github.com/pybind/pybind11/pull/1511>`_.

   2. The ``value()``  method of ``py::enum_`` now accepts an optional
      docstring that will be shown in the documentation of the associated
      enumeration. `#1160 <https://github.com/pybind/pybind11/pull/1160>`_.

   3. check for already existing enum value and throw an error if present.
      `#1453 <https://github.com/pybind/pybind11/pull/1453>`_.

* added ``py::ellipsis()`` method for slicing of multidimensional NumPy arrays
  `#1502 <https://github.com/pybind/pybind11/pull/1502>`_.

* ``pybind11_add_module()``: allow including Python as a ``SYSTEM`` include path.
  `#1416 <https://github.com/pybind/pybind11/pull/1416>`_.

* ``pybind11/stl.h`` does not convert strings to ``vector<string>`` anymore.
  `#1258 <https://github.com/pybind/pybind11/issues/1258>`_.

v2.2.4 (September 11, 2018)
-----------------------------------------------------

* Use new Python 3.7 Thread Specific Storage (TSS) implementation if available.
  `#1454 <https://github.com/pybind/pybind11/pull/1454>`_,
  `#1517 <https://github.com/pybind/pybind11/pull/1517>`_.

* Fixes for newer MSVC versions and C++17 mode.
  `#1347 <https://github.com/pybind/pybind11/pull/1347>`_,
  `#1462 <https://github.com/pybind/pybind11/pull/1462>`_.

* Propagate return value policies to type-specific casters
  when casting STL containers.
  `#1455 <https://github.com/pybind/pybind11/pull/1455>`_.

* Allow ostream-redirection of more than 1024 characters.
  `#1479 <https://github.com/pybind/pybind11/pull/1479>`_.

* Set ``Py_DEBUG`` define when compiling against a debug Python build.
  `#1438 <https://github.com/pybind/pybind11/pull/1438>`_.

* Untangle integer logic in number type caster to work for custom
  types that may only be castable to a restricted set of builtin types.
  `#1442 <https://github.com/pybind/pybind11/pull/1442>`_.

* CMake build system: Remember Python version in cache file.
  `#1434 <https://github.com/pybind/pybind11/pull/1434>`_.

* Fix for custom smart pointers: use ``std::addressof`` to obtain holder
  address instead of ``operator&``.
  `#1435 <https://github.com/pybind/pybind11/pull/1435>`_.

* Properly report exceptions thrown during module initialization.
  `#1362 <https://github.com/pybind/pybind11/pull/1362>`_.

* Fixed a segmentation fault when creating empty-shaped NumPy array.
  `#1371 <https://github.com/pybind/pybind11/pull/1371>`_.

* The version of Intel C++ compiler must be >= 2017, and this is now checked by
  the header files. `#1363 <https://github.com/pybind/pybind11/pull/1363>`_.

* A few minor typo fixes and improvements to the test suite, and
  patches that silence compiler warnings.

v2.2.3 (April 29, 2018)
-----------------------------------------------------

* The pybind11 header location detection was replaced by a new implementation
  that no longer depends on ``pip`` internals (the recently released ``pip``
  10 has restricted access to this API).
  `#1190 <https://github.com/pybind/pybind11/pull/1190>`_.

* Small adjustment to an implementation detail to work around a compiler segmentation fault in Clang 3.3/3.4.
  `#1350 <https://github.com/pybind/pybind11/pull/1350>`_.

* The minimal supported version of the Intel compiler was >= 17.0 since
  pybind11 v2.1. This check is now explicit, and a compile-time error is raised
  if the compiler meet the requirement.
  `#1363 <https://github.com/pybind/pybind11/pull/1363>`_.

* Fixed an endianness-related fault in the test suite.
  `#1287 <https://github.com/pybind/pybind11/pull/1287>`_.

v2.2.2 (February 7, 2018)
-----------------------------------------------------

* Fixed a segfault when combining embedded interpreter
  shutdown/reinitialization with external loaded pybind11 modules.
  `#1092 <https://github.com/pybind/pybind11/pull/1092>`_.

* Eigen support: fixed a bug where Nx1/1xN numpy inputs couldn't be passed as
  arguments to Eigen vectors (which for Eigen are simply compile-time fixed
  Nx1/1xN matrices).
  `#1106 <https://github.com/pybind/pybind11/pull/1106>`_.

* Clarified to license by moving the licensing of contributions from
  ``LICENSE`` into ``CONTRIBUTING.md``: the licensing of contributions is not
  actually part of the software license as distributed.  This isn't meant to be
  a substantial change in the licensing of the project, but addresses concerns
  that the clause made the license non-standard.
  `#1109 <https://github.com/pybind/pybind11/issues/1109>`_.

* Fixed a regression introduced in 2.1 that broke binding functions with lvalue
  character literal arguments.
  `#1128 <https://github.com/pybind/pybind11/pull/1128>`_.

* MSVC: fix for compilation failures under /permissive-, and added the flag to
  the appveyor test suite.
  `#1155 <https://github.com/pybind/pybind11/pull/1155>`_.

* Fixed ``__qualname__`` generation, and in turn, fixes how class names
  (especially nested class names) are shown in generated docstrings.
  `#1171 <https://github.com/pybind/pybind11/pull/1171>`_.

* Updated the FAQ with a suggested project citation reference.
  `#1189 <https://github.com/pybind/pybind11/pull/1189>`_.

* Added fixes for deprecation warnings when compiled under C++17 with
  ``-Wdeprecated`` turned on, and add ``-Wdeprecated`` to the test suite
  compilation flags.
  `#1191 <https://github.com/pybind/pybind11/pull/1191>`_.

* Fixed outdated PyPI URLs in ``setup.py``.
  `#1213 <https://github.com/pybind/pybind11/pull/1213>`_.

* Fixed a refcount leak for arguments that end up in a ``py::args`` argument
  for functions with both fixed positional and ``py::args`` arguments.
  `#1216 <https://github.com/pybind/pybind11/pull/1216>`_.

* Fixed a potential segfault resulting from possible premature destruction of
  ``py::args``/``py::kwargs`` arguments with overloaded functions.
  `#1223 <https://github.com/pybind/pybind11/pull/1223>`_.

* Fixed ``del map[item]`` for a ``stl_bind.h`` bound stl map.
  `#1229 <https://github.com/pybind/pybind11/pull/1229>`_.

* Fixed a regression from v2.1.x where the aggregate initialization could
  unintentionally end up at a constructor taking a templated
  ``std::initializer_list<T>`` argument.
  `#1249 <https://github.com/pybind/pybind11/pull/1249>`_.

* Fixed an issue where calling a function with a keep_alive policy on the same
  nurse/patient pair would cause the internal patient storage to needlessly
  grow (unboundedly, if the nurse is long-lived).
  `#1251 <https://github.com/pybind/pybind11/issues/1251>`_.

* Various other minor fixes.

v2.2.1 (September 14, 2017)
-----------------------------------------------------

* Added ``py::module::reload()`` member function for reloading a module.
  `#1040 <https://github.com/pybind/pybind11/pull/1040>`_.

* Fixed a reference leak in the number converter.
  `#1078 <https://github.com/pybind/pybind11/pull/1078>`_.

* Fixed compilation with Clang on host GCC < 5 (old libstdc++ which isn't fully
  C++11 compliant). `#1062 <https://github.com/pybind/pybind11/pull/1062>`_.

* Fixed a regression where the automatic ``std::vector<bool>`` caster would
  fail to compile. The same fix also applies to any container which returns
  element proxies instead of references.
  `#1053 <https://github.com/pybind/pybind11/pull/1053>`_.

* Fixed a regression where the ``py::keep_alive`` policy could not be applied
  to constructors. `#1065 <https://github.com/pybind/pybind11/pull/1065>`_.

* Fixed a nullptr dereference when loading a ``py::module_local`` type
  that's only registered in an external module.
  `#1058 <https://github.com/pybind/pybind11/pull/1058>`_.

* Fixed implicit conversion of accessors to types derived from ``py::object``.
  `#1076 <https://github.com/pybind/pybind11/pull/1076>`_.

* The ``name`` in ``PYBIND11_MODULE(name, variable)`` can now be a macro.
  `#1082 <https://github.com/pybind/pybind11/pull/1082>`_.

* Relaxed overly strict ``py::pickle()`` check for matching get and set types.
  `#1064 <https://github.com/pybind/pybind11/pull/1064>`_.

* Conversion errors now try to be more informative when it's likely that
  a missing header is the cause (e.g. forgetting ``<pybind11/stl.h>``).
  `#1077 <https://github.com/pybind/pybind11/pull/1077>`_.

v2.2.0 (August 31, 2017)
-----------------------------------------------------

* Support for embedding the Python interpreter. See the
  :doc:`documentation page </advanced/embedding>` for a
  full overview of the new features.
  `#774 <https://github.com/pybind/pybind11/pull/774>`_,
  `#889 <https://github.com/pybind/pybind11/pull/889>`_,
  `#892 <https://github.com/pybind/pybind11/pull/892>`_,
  `#920 <https://github.com/pybind/pybind11/pull/920>`_.

  .. code-block:: cpp

      #include <pybind11/embed.h>
      namespace py = pybind11;

      int main() {
          py::scoped_interpreter guard{}; // start the interpreter and keep it alive

          py::print("Hello, World!"); // use the Python API
      }

* Support for inheriting from multiple C++ bases in Python.
  `#693 <https://github.com/pybind/pybind11/pull/693>`_.

  .. code-block:: python

      from cpp_module import CppBase1, CppBase2

      class PyDerived(CppBase1, CppBase2):
          def __init__(self):
              CppBase1.__init__(self)  # C++ bases must be initialized explicitly
              CppBase2.__init__(self)

* ``PYBIND11_MODULE`` is now the preferred way to create module entry points.
  ``PYBIND11_PLUGIN`` is deprecated. See :ref:`macros` for details.
  `#879 <https://github.com/pybind/pybind11/pull/879>`_.

  .. code-block:: cpp

      // new
      PYBIND11_MODULE(example, m) {
          m.def("add", [](int a, int b) { return a + b; });
      }

      // old
      PYBIND11_PLUGIN(example) {
          py::module m("example");
          m.def("add", [](int a, int b) { return a + b; });
          return m.ptr();
      }

* pybind11's headers and build system now more strictly enforce hidden symbol
  visibility for extension modules. This should be seamless for most users,
  but see the :doc:`upgrade` if you use a custom build system.
  `#995 <https://github.com/pybind/pybind11/pull/995>`_.

* Support for ``py::module_local`` types which allow multiple modules to
  export the same C++ types without conflicts. This is useful for opaque
  types like ``std::vector<int>``. ``py::bind_vector`` and ``py::bind_map``
  now default to ``py::module_local`` if their elements are builtins or
  local types. See :ref:`module_local` for details.
  `#949 <https://github.com/pybind/pybind11/pull/949>`_,
  `#981 <https://github.com/pybind/pybind11/pull/981>`_,
  `#995 <https://github.com/pybind/pybind11/pull/995>`_,
  `#997 <https://github.com/pybind/pybind11/pull/997>`_.

* Custom constructors can now be added very easily using lambdas or factory
  functions which return a class instance by value, pointer or holder. This
  supersedes the old placement-new ``__init__`` technique.
  See :ref:`custom_constructors` for details.
  `#805 <https://github.com/pybind/pybind11/pull/805>`_,
  `#1014 <https://github.com/pybind/pybind11/pull/1014>`_.

  .. code-block:: cpp

      struct Example {
          Example(std::string);
      };

      py::class_<Example>(m, "Example")
          .def(py::init<std::string>()) // existing constructor
          .def(py::init([](int n) { // custom constructor
              return std::make_unique<Example>(std::to_string(n));
          }));

* Similarly to custom constructors, pickling support functions are now bound
  using the ``py::pickle()`` adaptor which improves type safety. See the
  :doc:`upgrade` and :ref:`pickling` for details.
  `#1038 <https://github.com/pybind/pybind11/pull/1038>`_.

* Builtin support for converting C++17 standard library types and general
  conversion improvements:

  1. C++17 ``std::variant`` is supported right out of the box. C++11/14
     equivalents (e.g. ``boost::variant``) can also be added with a simple
     user-defined specialization. See :ref:`cpp17_container_casters` for details.
     `#811 <https://github.com/pybind/pybind11/pull/811>`_,
     `#845 <https://github.com/pybind/pybind11/pull/845>`_,
     `#989 <https://github.com/pybind/pybind11/pull/989>`_.

  2. Out-of-the-box support for C++17 ``std::string_view``.
     `#906 <https://github.com/pybind/pybind11/pull/906>`_.

  3. Improved compatibility of the builtin ``optional`` converter.
     `#874 <https://github.com/pybind/pybind11/pull/874>`_.

  4. The ``bool`` converter now accepts ``numpy.bool_`` and types which
     define ``__bool__`` (Python 3.x) or ``__nonzero__`` (Python 2.7).
     `#925 <https://github.com/pybind/pybind11/pull/925>`_.

  5. C++-to-Python casters are now more efficient and move elements out
     of rvalue containers whenever possible.
     `#851 <https://github.com/pybind/pybind11/pull/851>`_,
     `#936 <https://github.com/pybind/pybind11/pull/936>`_,
     `#938 <https://github.com/pybind/pybind11/pull/938>`_.

  6. Fixed ``bytes`` to ``std::string/char*`` conversion on Python 3.
     `#817 <https://github.com/pybind/pybind11/pull/817>`_.

  7. Fixed lifetime of temporary C++ objects created in Python-to-C++ conversions.
     `#924 <https://github.com/pybind/pybind11/pull/924>`_.

* Scope guard call policy for RAII types, e.g. ``py::call_guard<py::gil_scoped_release>()``,
  ``py::call_guard<py::scoped_ostream_redirect>()``. See :ref:`call_policies` for details.
  `#740 <https://github.com/pybind/pybind11/pull/740>`_.

* Utility for redirecting C++ streams to Python (e.g. ``std::cout`` ->
  ``sys.stdout``). Scope guard ``py::scoped_ostream_redirect`` in C++ and
  a context manager in Python. See :ref:`ostream_redirect`.
  `#1009 <https://github.com/pybind/pybind11/pull/1009>`_.

* Improved handling of types and exceptions across module boundaries.
  `#915 <https://github.com/pybind/pybind11/pull/915>`_,
  `#951 <https://github.com/pybind/pybind11/pull/951>`_,
  `#995 <https://github.com/pybind/pybind11/pull/995>`_.

* Fixed destruction order of ``py::keep_alive`` nurse/patient objects
  in reference cycles.
  `#856 <https://github.com/pybind/pybind11/pull/856>`_.

* Numpy and buffer protocol related improvements:

  1. Support for negative strides in Python buffer objects/numpy arrays. This
     required changing integers from unsigned to signed for the related C++ APIs.
     Note: If you have compiler warnings enabled, you may notice some new conversion
     warnings after upgrading. These can be resolved with ``static_cast``.
     `#782 <https://github.com/pybind/pybind11/pull/782>`_.

  2. Support ``std::complex`` and arrays inside ``PYBIND11_NUMPY_DTYPE``.
     `#831 <https://github.com/pybind/pybind11/pull/831>`_,
     `#832 <https://github.com/pybind/pybind11/pull/832>`_.

  3. Support for constructing ``py::buffer_info`` and ``py::arrays`` using
     arbitrary containers or iterators instead of requiring a ``std::vector``.
     `#788 <https://github.com/pybind/pybind11/pull/788>`_,
     `#822 <https://github.com/pybind/pybind11/pull/822>`_,
     `#860 <https://github.com/pybind/pybind11/pull/860>`_.

  4. Explicitly check numpy version and require >= 1.7.0.
     `#819 <https://github.com/pybind/pybind11/pull/819>`_.

* Support for allowing/prohibiting ``None`` for specific arguments and improved
  ``None`` overload resolution order. See :ref:`none_arguments` for details.
  `#843 <https://github.com/pybind/pybind11/pull/843>`_.
  `#859 <https://github.com/pybind/pybind11/pull/859>`_.

* Added ``py::exec()`` as a shortcut for ``py::eval<py::eval_statements>()``
  and support for C++11 raw string literals as input. See :ref:`eval`.
  `#766 <https://github.com/pybind/pybind11/pull/766>`_,
  `#827 <https://github.com/pybind/pybind11/pull/827>`_.

* ``py::vectorize()`` ignores non-vectorizable arguments and supports
  member functions.
  `#762 <https://github.com/pybind/pybind11/pull/762>`_.

* Support for bound methods as callbacks (``pybind11/functional.h``).
  `#815 <https://github.com/pybind/pybind11/pull/815>`_.

* Allow aliasing pybind11 methods: ``cls.attr("foo") = cls.attr("bar")``.
  `#802 <https://github.com/pybind/pybind11/pull/802>`_.

* Don't allow mixed static/non-static overloads.
  `#804 <https://github.com/pybind/pybind11/pull/804>`_.

* Fixed overriding static properties in derived classes.
  `#784 <https://github.com/pybind/pybind11/pull/784>`_.

* Improved deduction of member functions of a derived class when its bases
  aren't registered with pybind11.
  `#855 <https://github.com/pybind/pybind11/pull/855>`_.

  .. code-block:: cpp

      struct Base {
          int foo() { return 42; }
      }

      struct Derived : Base {}

      // Now works, but previously required also binding `Base`
      py::class_<Derived>(m, "Derived")
          .def("foo", &Derived::foo); // function is actually from `Base`

* The implementation of ``py::init<>`` now uses C++11 brace initialization
  syntax to construct instances, which permits binding implicit constructors of
  aggregate types. `#1015 <https://github.com/pybind/pybind11/pull/1015>`_.

    .. code-block:: cpp

        struct Aggregate {
            int a;
            std::string b;
        };

        py::class_<Aggregate>(m, "Aggregate")
            .def(py::init<int, const std::string &>());

* Fixed issues with multiple inheritance with offset base/derived pointers.
  `#812 <https://github.com/pybind/pybind11/pull/812>`_,
  `#866 <https://github.com/pybind/pybind11/pull/866>`_,
  `#960 <https://github.com/pybind/pybind11/pull/960>`_.

* Fixed reference leak of type objects.
  `#1030 <https://github.com/pybind/pybind11/pull/1030>`_.

* Improved support for the ``/std:c++14`` and ``/std:c++latest`` modes
  on MSVC 2017.
  `#841 <https://github.com/pybind/pybind11/pull/841>`_,
  `#999 <https://github.com/pybind/pybind11/pull/999>`_.

* Fixed detection of private operator new on MSVC.
  `#893 <https://github.com/pybind/pybind11/pull/893>`_,
  `#918 <https://github.com/pybind/pybind11/pull/918>`_.

* Intel C++ compiler compatibility fixes.
  `#937 <https://github.com/pybind/pybind11/pull/937>`_.

* Fixed implicit conversion of `py::enum_` to integer types on Python 2.7.
  `#821 <https://github.com/pybind/pybind11/pull/821>`_.

* Added ``py::hash`` to fetch the hash value of Python objects, and
  ``.def(hash(py::self))`` to provide the C++ ``std::hash`` as the Python
  ``__hash__`` method.
  `#1034 <https://github.com/pybind/pybind11/pull/1034>`_.

* Fixed ``__truediv__`` on Python 2 and ``__itruediv__`` on Python 3.
  `#867 <https://github.com/pybind/pybind11/pull/867>`_.

* ``py::capsule`` objects now support the ``name`` attribute. This is useful
  for interfacing with ``scipy.LowLevelCallable``.
  `#902 <https://github.com/pybind/pybind11/pull/902>`_.

* Fixed ``py::make_iterator``'s ``__next__()`` for past-the-end calls.
  `#897 <https://github.com/pybind/pybind11/pull/897>`_.

* Added ``error_already_set::matches()`` for checking Python exceptions.
  `#772 <https://github.com/pybind/pybind11/pull/772>`_.

* Deprecated ``py::error_already_set::clear()``. It's no longer needed
  following a simplification of the ``py::error_already_set`` class.
  `#954 <https://github.com/pybind/pybind11/pull/954>`_.

* Deprecated ``py::handle::operator==()`` in favor of ``py::handle::is()``
  `#825 <https://github.com/pybind/pybind11/pull/825>`_.

* Deprecated ``py::object::borrowed``/``py::object::stolen``.
  Use ``py::object::borrowed_t{}``/``py::object::stolen_t{}`` instead.
  `#771 <https://github.com/pybind/pybind11/pull/771>`_.

* Changed internal data structure versioning to avoid conflicts between
  modules compiled with different revisions of pybind11.
  `#1012 <https://github.com/pybind/pybind11/pull/1012>`_.

* Additional compile-time and run-time error checking and more informative messages.
  `#786 <https://github.com/pybind/pybind11/pull/786>`_,
  `#794 <https://github.com/pybind/pybind11/pull/794>`_,
  `#803 <https://github.com/pybind/pybind11/pull/803>`_.

* Various minor improvements and fixes.
  `#764 <https://github.com/pybind/pybind11/pull/764>`_,
  `#791 <https://github.com/pybind/pybind11/pull/791>`_,
  `#795 <https://github.com/pybind/pybind11/pull/795>`_,
  `#840 <https://github.com/pybind/pybind11/pull/840>`_,
  `#844 <https://github.com/pybind/pybind11/pull/844>`_,
  `#846 <https://github.com/pybind/pybind11/pull/846>`_,
  `#849 <https://github.com/pybind/pybind11/pull/849>`_,
  `#858 <https://github.com/pybind/pybind11/pull/858>`_,
  `#862 <https://github.com/pybind/pybind11/pull/862>`_,
  `#871 <https://github.com/pybind/pybind11/pull/871>`_,
  `#872 <https://github.com/pybind/pybind11/pull/872>`_,
  `#881 <https://github.com/pybind/pybind11/pull/881>`_,
  `#888 <https://github.com/pybind/pybind11/pull/888>`_,
  `#899 <https://github.com/pybind/pybind11/pull/899>`_,
  `#928 <https://github.com/pybind/pybind11/pull/928>`_,
  `#931 <https://github.com/pybind/pybind11/pull/931>`_,
  `#944 <https://github.com/pybind/pybind11/pull/944>`_,
  `#950 <https://github.com/pybind/pybind11/pull/950>`_,
  `#952 <https://github.com/pybind/pybind11/pull/952>`_,
  `#962 <https://github.com/pybind/pybind11/pull/962>`_,
  `#965 <https://github.com/pybind/pybind11/pull/965>`_,
  `#970 <https://github.com/pybind/pybind11/pull/970>`_,
  `#978 <https://github.com/pybind/pybind11/pull/978>`_,
  `#979 <https://github.com/pybind/pybind11/pull/979>`_,
  `#986 <https://github.com/pybind/pybind11/pull/986>`_,
  `#1020 <https://github.com/pybind/pybind11/pull/1020>`_,
  `#1027 <https://github.com/pybind/pybind11/pull/1027>`_,
  `#1037 <https://github.com/pybind/pybind11/pull/1037>`_.

* Testing improvements.
  `#798 <https://github.com/pybind/pybind11/pull/798>`_,
  `#882 <https://github.com/pybind/pybind11/pull/882>`_,
  `#898 <https://github.com/pybind/pybind11/pull/898>`_,
  `#900 <https://github.com/pybind/pybind11/pull/900>`_,
  `#921 <https://github.com/pybind/pybind11/pull/921>`_,
  `#923 <https://github.com/pybind/pybind11/pull/923>`_,
  `#963 <https://github.com/pybind/pybind11/pull/963>`_.

v2.1.1 (April 7, 2017)
-----------------------------------------------------

* Fixed minimum version requirement for MSVC 2015u3
  `#773 <https://github.com/pybind/pybind11/pull/773>`_.

v2.1.0 (March 22, 2017)
-----------------------------------------------------

* pybind11 now performs function overload resolution in two phases. The first
  phase only considers exact type matches, while the second allows for implicit
  conversions to take place. A special ``noconvert()`` syntax can be used to
  completely disable implicit conversions for specific arguments.
  `#643 <https://github.com/pybind/pybind11/pull/643>`_,
  `#634 <https://github.com/pybind/pybind11/pull/634>`_,
  `#650 <https://github.com/pybind/pybind11/pull/650>`_.

* Fixed a regression where static properties no longer worked with classes
  using multiple inheritance. The ``py::metaclass`` attribute is no longer
  necessary (and deprecated as of this release) when binding classes with
  static properties.
  `#679 <https://github.com/pybind/pybind11/pull/679>`_,

* Classes bound using ``pybind11`` can now use custom metaclasses.
  `#679 <https://github.com/pybind/pybind11/pull/679>`_,

* ``py::args`` and ``py::kwargs`` can now be mixed with other positional
  arguments when binding functions using pybind11.
  `#611 <https://github.com/pybind/pybind11/pull/611>`_.

* Improved support for C++11 unicode string and character types; added
  extensive documentation regarding pybind11's string conversion behavior.
  `#624 <https://github.com/pybind/pybind11/pull/624>`_,
  `#636 <https://github.com/pybind/pybind11/pull/636>`_,
  `#715 <https://github.com/pybind/pybind11/pull/715>`_.

* pybind11 can now avoid expensive copies when converting Eigen arrays to NumPy
  arrays (and vice versa). `#610 <https://github.com/pybind/pybind11/pull/610>`_.

* The "fast path" in ``py::vectorize`` now works for any full-size group of C or
  F-contiguous arrays. The non-fast path is also faster since it no longer performs
  copies of the input arguments (except when type conversions are necessary).
  `#610 <https://github.com/pybind/pybind11/pull/610>`_.

* Added fast, unchecked access to NumPy arrays via a proxy object.
  `#746 <https://github.com/pybind/pybind11/pull/746>`_.

* Transparent support for class-specific ``operator new`` and
  ``operator delete`` implementations.
  `#755 <https://github.com/pybind/pybind11/pull/755>`_.

* Slimmer and more efficient STL-compatible iterator interface for sequence types.
  `#662 <https://github.com/pybind/pybind11/pull/662>`_.

* Improved custom holder type support.
  `#607 <https://github.com/pybind/pybind11/pull/607>`_.

* ``nullptr`` to ``None`` conversion fixed in various builtin type casters.
  `#732 <https://github.com/pybind/pybind11/pull/732>`_.

* ``enum_`` now exposes its members via a special ``__members__`` attribute.
  `#666 <https://github.com/pybind/pybind11/pull/666>`_.

* ``std::vector`` bindings created using ``stl_bind.h`` can now optionally
  implement the buffer protocol. `#488 <https://github.com/pybind/pybind11/pull/488>`_.

* Automated C++ reference documentation using doxygen and breathe.
  `#598 <https://github.com/pybind/pybind11/pull/598>`_.

* Added minimum compiler version assertions.
  `#727 <https://github.com/pybind/pybind11/pull/727>`_.

* Improved compatibility with C++1z.
  `#677 <https://github.com/pybind/pybind11/pull/677>`_.

* Improved ``py::capsule`` API. Can be used to implement cleanup
  callbacks that are involved at module destruction time.
  `#752 <https://github.com/pybind/pybind11/pull/752>`_.

* Various minor improvements and fixes.
  `#595 <https://github.com/pybind/pybind11/pull/595>`_,
  `#588 <https://github.com/pybind/pybind11/pull/588>`_,
  `#589 <https://github.com/pybind/pybind11/pull/589>`_,
  `#603 <https://github.com/pybind/pybind11/pull/603>`_,
  `#619 <https://github.com/pybind/pybind11/pull/619>`_,
  `#648 <https://github.com/pybind/pybind11/pull/648>`_,
  `#695 <https://github.com/pybind/pybind11/pull/695>`_,
  `#720 <https://github.com/pybind/pybind11/pull/720>`_,
  `#723 <https://github.com/pybind/pybind11/pull/723>`_,
  `#729 <https://github.com/pybind/pybind11/pull/729>`_,
  `#724 <https://github.com/pybind/pybind11/pull/724>`_,
  `#742 <https://github.com/pybind/pybind11/pull/742>`_,
  `#753 <https://github.com/pybind/pybind11/pull/753>`_.

v2.0.1 (Jan 4, 2017)
-----------------------------------------------------

* Fix pointer to reference error in type_caster on MSVC
  `#583 <https://github.com/pybind/pybind11/pull/583>`_.

* Fixed a segmentation in the test suite due to a typo
  `cd7eac <https://github.com/pybind/pybind11/commit/cd7eac>`_.

v2.0.0 (Jan 1, 2017)
-----------------------------------------------------

* Fixed a reference counting regression affecting types with custom metaclasses
  (introduced in v2.0.0-rc1).
  `#571 <https://github.com/pybind/pybind11/pull/571>`_.

* Quenched a CMake policy warning.
  `#570 <https://github.com/pybind/pybind11/pull/570>`_.

v2.0.0-rc1 (Dec 23, 2016)
-----------------------------------------------------

The pybind11 developers are excited to issue a release candidate of pybind11
with a subsequent v2.0.0 release planned in early January next year.

An incredible amount of effort by went into pybind11 over the last ~5 months,
leading to a release that is jam-packed with exciting new features and numerous
usability improvements. The following list links PRs or individual commits
whenever applicable.

Happy Christmas!

* Support for binding C++ class hierarchies that make use of multiple
  inheritance. `#410 <https://github.com/pybind/pybind11/pull/410>`_.

* PyPy support: pybind11 now supports nightly builds of PyPy and will
  interoperate with the future 5.7 release. No code changes are necessary,
  everything "just" works as usual. Note that we only target the Python 2.7
  branch for now; support for 3.x will be added once its ``cpyext`` extension
  support catches up. A few minor features remain unsupported for the time
  being (notably dynamic attributes in custom types).
  `#527 <https://github.com/pybind/pybind11/pull/527>`_.

* Significant work on the documentation -- in particular, the monolithic
  ``advanced.rst`` file was restructured into a easier to read hierarchical
  organization. `#448 <https://github.com/pybind/pybind11/pull/448>`_.

* Many NumPy-related improvements:

  1. Object-oriented API to access and modify NumPy ``ndarray`` instances,
     replicating much of the corresponding NumPy C API functionality.
     `#402 <https://github.com/pybind/pybind11/pull/402>`_.

  2. NumPy array ``dtype`` array descriptors are now first-class citizens and
     are exposed via a new class ``py::dtype``.

  3. Structured dtypes can be registered using the ``PYBIND11_NUMPY_DTYPE()``
     macro. Special ``array`` constructors accepting dtype objects were also
     added.

     One potential caveat involving this change: format descriptor strings
     should now be accessed via ``format_descriptor::format()`` (however, for
     compatibility purposes, the old syntax ``format_descriptor::value`` will
     still work for non-structured data types). `#308
     <https://github.com/pybind/pybind11/pull/308>`_.

  4. Further improvements to support structured dtypes throughout the system.
     `#472 <https://github.com/pybind/pybind11/pull/472>`_,
     `#474 <https://github.com/pybind/pybind11/pull/474>`_,
     `#459 <https://github.com/pybind/pybind11/pull/459>`_,
     `#453 <https://github.com/pybind/pybind11/pull/453>`_,
     `#452 <https://github.com/pybind/pybind11/pull/452>`_, and
     `#505 <https://github.com/pybind/pybind11/pull/505>`_.

  5. Fast access operators. `#497 <https://github.com/pybind/pybind11/pull/497>`_.

  6. Constructors for arrays whose storage is owned by another object.
     `#440 <https://github.com/pybind/pybind11/pull/440>`_.

  7. Added constructors for ``array`` and ``array_t`` explicitly accepting shape
     and strides; if strides are not provided, they are deduced assuming
     C-contiguity. Also added simplified constructors for 1-dimensional case.

  8. Added buffer/NumPy support for ``char[N]`` and ``std::array<char, N>`` types.

  9. Added ``memoryview`` wrapper type which is constructible from ``buffer_info``.

* Eigen: many additional conversions and support for non-contiguous
  arrays/slices.
  `#427 <https://github.com/pybind/pybind11/pull/427>`_,
  `#315 <https://github.com/pybind/pybind11/pull/315>`_,
  `#316 <https://github.com/pybind/pybind11/pull/316>`_,
  `#312 <https://github.com/pybind/pybind11/pull/312>`_, and
  `#267 <https://github.com/pybind/pybind11/pull/267>`_

* Incompatible changes in ``class_<...>::class_()``:

    1. Declarations of types that provide access via the buffer protocol must
       now include the ``py::buffer_protocol()`` annotation as an argument to
       the ``class_`` constructor.

    2. Declarations of types that require a custom metaclass (i.e. all classes
       which include static properties via commands such as
       ``def_readwrite_static()``) must now include the ``py::metaclass()``
       annotation as an argument to the ``class_`` constructor.

       These two changes were necessary to make type definitions in pybind11
       future-proof, and to support PyPy via its cpyext mechanism. `#527
       <https://github.com/pybind/pybind11/pull/527>`_.


    3. This version of pybind11 uses a redesigned mechanism for instantiating
       trampoline classes that are used to override virtual methods from within
       Python. This led to the following user-visible syntax change: instead of

       .. code-block:: cpp

           py::class_<TrampolineClass>("MyClass")
             .alias<MyClass>()
             ....

       write

       .. code-block:: cpp

           py::class_<MyClass, TrampolineClass>("MyClass")
             ....

       Importantly, both the original and the trampoline class are now
       specified as an arguments (in arbitrary order) to the ``py::class_``
       template, and the ``alias<..>()`` call is gone. The new scheme has zero
       overhead in cases when Python doesn't override any functions of the
       underlying C++ class. `rev. 86d825
       <https://github.com/pybind/pybind11/commit/86d825>`_.

* Added ``eval`` and ``eval_file`` functions for evaluating expressions and
  statements from a string or file. `rev. 0d3fc3
  <https://github.com/pybind/pybind11/commit/0d3fc3>`_.

* pybind11 can now create types with a modifiable dictionary.
  `#437 <https://github.com/pybind/pybind11/pull/437>`_ and
  `#444 <https://github.com/pybind/pybind11/pull/444>`_.

* Support for translation of arbitrary C++ exceptions to Python counterparts.
  `#296 <https://github.com/pybind/pybind11/pull/296>`_ and
  `#273 <https://github.com/pybind/pybind11/pull/273>`_.

* Report full backtraces through mixed C++/Python code, better reporting for
  import errors, fixed GIL management in exception processing.
  `#537 <https://github.com/pybind/pybind11/pull/537>`_,
  `#494 <https://github.com/pybind/pybind11/pull/494>`_,
  `rev. e72d95 <https://github.com/pybind/pybind11/commit/e72d95>`_, and
  `rev. 099d6e <https://github.com/pybind/pybind11/commit/099d6e>`_.

* Support for bit-level operations, comparisons, and serialization of C++
  enumerations. `#503 <https://github.com/pybind/pybind11/pull/503>`_,
  `#508 <https://github.com/pybind/pybind11/pull/508>`_,
  `#380 <https://github.com/pybind/pybind11/pull/380>`_,
  `#309 <https://github.com/pybind/pybind11/pull/309>`_.
  `#311 <https://github.com/pybind/pybind11/pull/311>`_.

* The ``class_`` constructor now accepts its template arguments in any order.
  `#385 <https://github.com/pybind/pybind11/pull/385>`_.

* Attribute and item accessors now have a more complete interface which makes
  it possible to chain attributes as in
  ``obj.attr("a")[key].attr("b").attr("method")(1, 2, 3)``. `#425
  <https://github.com/pybind/pybind11/pull/425>`_.

* Major redesign of the default and conversion constructors in ``pytypes.h``.
  `#464 <https://github.com/pybind/pybind11/pull/464>`_.

* Added built-in support for ``std::shared_ptr`` holder type. It is no longer
  necessary to to include a declaration of the form
  ``PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)`` (though continuing to
  do so won't cause an error).
  `#454 <https://github.com/pybind/pybind11/pull/454>`_.

* New ``py::overload_cast`` casting operator to select among multiple possible
  overloads of a function. An example:

    .. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def("set", py::overload_cast<int>(&Pet::set), "Set the pet's age")
            .def("set", py::overload_cast<const std::string &>(&Pet::set), "Set the pet's name");

  This feature only works on C++14-capable compilers.
  `#541 <https://github.com/pybind/pybind11/pull/541>`_.

* C++ types are automatically cast to Python types, e.g. when assigning
  them as an attribute. For instance, the following is now legal:

    .. code-block:: cpp

        py::module m = /* ... */
        m.attr("constant") = 123;

  (Previously, a ``py::cast`` call was necessary to avoid a compilation error.)
  `#551 <https://github.com/pybind/pybind11/pull/551>`_.

* Redesigned ``pytest``-based test suite. `#321 <https://github.com/pybind/pybind11/pull/321>`_.

* Instance tracking to detect reference leaks in test suite. `#324 <https://github.com/pybind/pybind11/pull/324>`_

* pybind11 can now distinguish between multiple different instances that are
  located at the same memory address, but which have different types.
  `#329 <https://github.com/pybind/pybind11/pull/329>`_.

* Improved logic in ``move`` return value policy.
  `#510 <https://github.com/pybind/pybind11/pull/510>`_,
  `#297 <https://github.com/pybind/pybind11/pull/297>`_.

* Generalized unpacking API to permit calling Python functions from C++ using
  notation such as ``foo(a1, a2, *args, "ka"_a=1, "kb"_a=2, **kwargs)``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* ``py::print()`` function whose behavior matches that of the native Python
  ``print()`` function. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::dict`` keyword constructor:``auto d = dict("number"_a=42,
  "name"_a="World");``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::str::format()`` method and ``_s`` literal: ``py::str s = "1 + 2
  = {}"_s.format(3);``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::repr()`` function which is equivalent to Python's builtin
  ``repr()``. `#333 <https://github.com/pybind/pybind11/pull/333>`_.

* Improved construction and destruction logic for holder types. It is now
  possible to reference instances with smart pointer holder types without
  constructing the holder if desired. The ``PYBIND11_DECLARE_HOLDER_TYPE``
  macro now accepts an optional second parameter to indicate whether the holder
  type uses intrusive reference counting.
  `#533 <https://github.com/pybind/pybind11/pull/533>`_ and
  `#561 <https://github.com/pybind/pybind11/pull/561>`_.

* Mapping a stateless C++ function to Python and back is now "for free" (i.e.
  no extra indirections or argument conversion overheads). `rev. 954b79
  <https://github.com/pybind/pybind11/commit/954b79>`_.

* Bindings for ``std::valarray<T>``.
  `#545 <https://github.com/pybind/pybind11/pull/545>`_.

* Improved support for C++17 capable compilers.
  `#562 <https://github.com/pybind/pybind11/pull/562>`_.

* Bindings for ``std::optional<t>``.
  `#475 <https://github.com/pybind/pybind11/pull/475>`_,
  `#476 <https://github.com/pybind/pybind11/pull/476>`_,
  `#479 <https://github.com/pybind/pybind11/pull/479>`_,
  `#499 <https://github.com/pybind/pybind11/pull/499>`_, and
  `#501 <https://github.com/pybind/pybind11/pull/501>`_.

* ``stl_bind.h``: general improvements and support for ``std::map`` and
  ``std::unordered_map``.
  `#490 <https://github.com/pybind/pybind11/pull/490>`_,
  `#282 <https://github.com/pybind/pybind11/pull/282>`_,
  `#235 <https://github.com/pybind/pybind11/pull/235>`_.

* The ``std::tuple``, ``std::pair``, ``std::list``, and ``std::vector`` type
  casters now accept any Python sequence type as input. `rev. 107285
  <https://github.com/pybind/pybind11/commit/107285>`_.

* Improved CMake Python detection on multi-architecture Linux.
  `#532 <https://github.com/pybind/pybind11/pull/532>`_.

* Infrastructure to selectively disable or enable parts of the automatically
  generated docstrings. `#486 <https://github.com/pybind/pybind11/pull/486>`_.

* ``reference`` and ``reference_internal`` are now the default return value
  properties for static and non-static properties, respectively. `#473
  <https://github.com/pybind/pybind11/pull/473>`_. (the previous defaults
  were ``automatic``). `#473 <https://github.com/pybind/pybind11/pull/473>`_.

* Support for ``std::unique_ptr`` with non-default deleters or no deleter at
  all (``py::nodelete``). `#384 <https://github.com/pybind/pybind11/pull/384>`_.

* Deprecated ``handle::call()`` method. The new syntax to call Python
  functions is simply ``handle()``. It can also be invoked explicitly via
  ``handle::operator<X>()``, where ``X`` is an optional return value policy.

* Print more informative error messages when ``make_tuple()`` or ``cast()``
  fail. `#262 <https://github.com/pybind/pybind11/pull/262>`_.

* Creation of holder types for classes deriving from
  ``std::enable_shared_from_this<>`` now also works for ``const`` values.
  `#260 <https://github.com/pybind/pybind11/pull/260>`_.

* ``make_iterator()`` improvements for better compatibility with various
  types (now uses prefix increment operator); it now also accepts iterators
  with different begin/end types as long as they are equality comparable.
  `#247 <https://github.com/pybind/pybind11/pull/247>`_.

* ``arg()`` now accepts a wider range of argument types for default values.
  `#244 <https://github.com/pybind/pybind11/pull/244>`_.

* Support ``keep_alive`` where the nurse object may be ``None``. `#341
  <https://github.com/pybind/pybind11/pull/341>`_.

* Added constructors for ``str`` and ``bytes`` from zero-terminated char
  pointers, and from char pointers and length. Added constructors for ``str``
  from ``bytes`` and for ``bytes`` from ``str``, which will perform UTF-8
  decoding/encoding as required.

* Many other improvements of library internals without user-visible changes


1.8.1 (July 12, 2016)
----------------------
* Fixed a rare but potentially very severe issue when the garbage collector ran
  during pybind11 type creation.

1.8.0 (June 14, 2016)
----------------------
* Redesigned CMake build system which exports a convenient
  ``pybind11_add_module`` function to parent projects.
* ``std::vector<>`` type bindings analogous to Boost.Python's ``indexing_suite``
* Transparent conversion of sparse and dense Eigen matrices and vectors (``eigen.h``)
* Added an ``ExtraFlags`` template argument to the NumPy ``array_t<>`` wrapper
  to disable an enforced cast that may lose precision, e.g. to create overloads
  for different precisions and complex vs real-valued matrices.
* Prevent implicit conversion of floating point values to integral types in
  function arguments
* Fixed incorrect default return value policy for functions returning a shared
  pointer
* Don't allow registering a type via ``class_`` twice
* Don't allow casting a ``None`` value into a C++ lvalue reference
* Fixed a crash in ``enum_::operator==`` that was triggered by the ``help()`` command
* Improved detection of whether or not custom C++ types can be copy/move-constructed
* Extended ``str`` type to also work with ``bytes`` instances
* Added a ``"name"_a`` user defined string literal that is equivalent to ``py::arg("name")``.
* When specifying function arguments via ``py::arg``, the test that verifies
  the number of arguments now runs at compile time.
* Added ``[[noreturn]]`` attribute to ``pybind11_fail()`` to quench some
  compiler warnings
* List function arguments in exception text when the dispatch code cannot find
  a matching overload
* Added ``PYBIND11_OVERLOAD_NAME`` and ``PYBIND11_OVERLOAD_PURE_NAME`` macros which
  can be used to override virtual methods whose name differs in C++ and Python
  (e.g. ``__call__`` and ``operator()``)
* Various minor ``iterator`` and ``make_iterator()`` improvements
* Transparently support ``__bool__`` on Python 2.x and Python 3.x
* Fixed issue with destructor of unpickled object not being called
* Minor CMake build system improvements on Windows
* New ``pybind11::args`` and ``pybind11::kwargs`` types to create functions which
  take an arbitrary number of arguments and keyword arguments
* New syntax to call a Python function from C++ using ``*args`` and ``*kwargs``
* The functions ``def_property_*`` now correctly process docstring arguments (these
  formerly caused a segmentation fault)
* Many ``mkdoc.py`` improvements (enumerations, template arguments, ``DOC()``
  macro accepts more arguments)
* Cygwin support
* Documentation improvements (pickling support, ``keep_alive``, macro usage)

1.7 (April 30, 2016)
----------------------
* Added a new ``move`` return value policy that triggers C++11 move semantics.
  The automatic return value policy falls back to this case whenever a rvalue
  reference is encountered
* Significantly more general GIL state routines that are used instead of
  Python's troublesome ``PyGILState_Ensure`` and ``PyGILState_Release`` API
* Redesign of opaque types that drastically simplifies their usage
* Extended ability to pass values of type ``[const] void *``
* ``keep_alive`` fix: don't fail when there is no patient
* ``functional.h``: acquire the GIL before calling a Python function
* Added Python RAII type wrappers ``none`` and ``iterable``
* Added ``*args`` and ``*kwargs`` pass-through parameters to
  ``pybind11.get_include()`` function
* Iterator improvements and fixes
* Documentation on return value policies and opaque types improved

1.6 (April 30, 2016)
----------------------
* Skipped due to upload to PyPI gone wrong and inability to recover
  (https://github.com/pypa/packaging-problems/issues/74)

1.5 (April 21, 2016)
----------------------
* For polymorphic types, use RTTI to try to return the closest type registered with pybind11
* Pickling support for serializing and unserializing C++ instances to a byte stream in Python
* Added a convenience routine ``make_iterator()`` which turns a range indicated
  by a pair of C++ iterators into a iterable Python object
* Added ``len()`` and a variadic ``make_tuple()`` function
* Addressed a rare issue that could confuse the current virtual function
  dispatcher and another that could lead to crashes in multi-threaded
  applications
* Added a ``get_include()`` function to the Python module that returns the path
  of the directory containing the installed pybind11 header files
* Documentation improvements: import issues, symbol visibility, pickling, limitations
* Added casting support for ``std::reference_wrapper<>``

1.4 (April 7, 2016)
--------------------------
* Transparent type conversion for ``std::wstring`` and ``wchar_t``
* Allow passing ``nullptr``-valued strings
* Transparent passing of ``void *`` pointers using capsules
* Transparent support for returning values wrapped in ``std::unique_ptr<>``
* Improved docstring generation for compatibility with Sphinx
* Nicer debug error message when default parameter construction fails
* Support for "opaque" types that bypass the transparent conversion layer for STL containers
* Redesigned type casting interface to avoid ambiguities that could occasionally cause compiler errors
* Redesigned property implementation; fixes crashes due to an unfortunate default return value policy
* Anaconda package generation support

1.3 (March 8, 2016)
--------------------------

* Added support for the Intel C++ compiler (v15+)
* Added support for the STL unordered set/map data structures
* Added support for the STL linked list data structure
* NumPy-style broadcasting support in ``pybind11::vectorize``
* pybind11 now displays more verbose error messages when ``arg::operator=()`` fails
* pybind11 internal data structures now live in a version-dependent namespace to avoid ABI issues
* Many, many bugfixes involving corner cases and advanced usage

1.2 (February 7, 2016)
--------------------------

* Optional: efficient generation of function signatures at compile time using C++14
* Switched to a simpler and more general way of dealing with function default
  arguments. Unused keyword arguments in function calls are now detected and
  cause errors as expected
* New ``keep_alive`` call policy analogous to Boost.Python's ``with_custodian_and_ward``
* New ``pybind11::base<>`` attribute to indicate a subclass relationship
* Improved interface for RAII type wrappers in ``pytypes.h``
* Use RAII type wrappers consistently within pybind11 itself. This
  fixes various potential refcount leaks when exceptions occur
* Added new ``bytes`` RAII type wrapper (maps to ``string`` in Python 2.7)
* Made handle and related RAII classes const correct, using them more
  consistently everywhere now
* Got rid of the ugly ``__pybind11__`` attributes on the Python side---they are
  now stored in a C++ hash table that is not visible in Python
* Fixed refcount leaks involving NumPy arrays and bound functions
* Vastly improved handling of shared/smart pointers
* Removed an unnecessary copy operation in ``pybind11::vectorize``
* Fixed naming clashes when both pybind11 and NumPy headers are included
* Added conversions for additional exception types
* Documentation improvements (using multiple extension modules, smart pointers,
  other minor clarifications)
* unified infrastructure for parsing variadic arguments in ``class_`` and cpp_function
* Fixed license text (was: ZLIB, should have been: 3-clause BSD)
* Python 3.2 compatibility
* Fixed remaining issues when accessing types in another plugin module
* Added enum comparison and casting methods
* Improved SFINAE-based detection of whether types are copy-constructible
* Eliminated many warnings about unused variables and the use of ``offsetof()``
* Support for ``std::array<>`` conversions

1.1 (December 7, 2015)
--------------------------

* Documentation improvements (GIL, wrapping functions, casting, fixed many typos)
* Generalized conversion of integer types
* Improved support for casting function objects
* Improved support for ``std::shared_ptr<>`` conversions
* Initial support for ``std::set<>`` conversions
* Fixed type resolution issue for types defined in a separate plugin module
* Cmake build system improvements
* Factored out generic functionality to non-templated code (smaller code size)
* Added a code size / compile time benchmark vs Boost.Python
* Added an appveyor CI script

1.0 (October 15, 2015)
------------------------
* Initial release
