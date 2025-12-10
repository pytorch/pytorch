<PackageName>_ROOT
------------------

.. versionadded:: 3.12

.. include:: include/ENV_VAR.rst

Calls to :command:`find_package(<PackageName>)` will search in prefixes
specified by the ``<PackageName>_ROOT`` environment variable, where
``<PackageName>`` is the (case-preserved) name given to the
:command:`find_package` call and ``_ROOT`` is literal.
For example, ``find_package(Foo)`` will search prefixes specified in the
``Foo_ROOT`` environment variable (if set).  See policy :policy:`CMP0074`.

This variable may hold a single prefix or a list of prefixes separated
by ``:`` on UNIX or ``;`` on Windows (the same as the ``PATH`` environment
variable convention on those platforms).

See also the :variable:`<PackageName>_ROOT` CMake variable.

.. envvar:: <PACKAGENAME>_ROOT

  .. versionadded:: 3.27

  Calls to :command:`find_package(<PackageName>)` will also search in
  prefixes specified by the upper-case ``<PACKAGENAME>_ROOT`` environment
  variable.  See policy :policy:`CMP0144`.

.. note::

  Note that the ``<PackageName>_ROOT`` and ``<PACKAGENAME>_ROOT``
  environment variables are distinct only on platforms that have
  case-sensitive environments.
