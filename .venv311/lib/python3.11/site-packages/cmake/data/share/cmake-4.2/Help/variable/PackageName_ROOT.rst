<PackageName>_ROOT
------------------

.. versionadded:: 3.12

Calls to :command:`find_package(<PackageName>)` will search in prefixes
specified by the ``<PackageName>_ROOT`` CMake variable, where
``<PackageName>`` is the (case-preserved) name given to the
:command:`find_package` call and ``_ROOT`` is literal.
For example, ``find_package(Foo)`` will search prefixes specified in the
``Foo_ROOT`` CMake variable (if set).  See policy :policy:`CMP0074`.

This variable may hold a single prefix or a
:ref:`semicolon-separated list <CMake Language Lists>` of multiple prefixes.

See also the :envvar:`<PackageName>_ROOT` environment variable.

.. variable:: <PACKAGENAME>_ROOT

  .. versionadded:: 3.27

  Calls to :command:`find_package(<PackageName>)` will also search in
  prefixes specified by the upper-case ``<PACKAGENAME>_ROOT`` CMake
  variable.  See policy :policy:`CMP0144`.
