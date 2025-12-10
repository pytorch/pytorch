include
-------

Load and run CMake code from a file or module.

.. code-block:: cmake

  include(<file|module> [OPTIONAL] [RESULT_VARIABLE <var>]
                        [NO_POLICY_SCOPE])

Loads and runs CMake code from the file given.  Variable reads and
writes access the scope of the caller (dynamic scoping).  If ``OPTIONAL``
is present, then no error is raised if the file does not exist.  If
``RESULT_VARIABLE`` is given the variable ``<var>`` will be set to the
full filename which has been included or ``NOTFOUND`` if it failed.

If a module is specified instead of a file, the file with name
``<modulename>.cmake`` is searched first in :variable:`CMAKE_MODULE_PATH`,
then in the CMake module directory.  There is one exception to this: if
the file which calls ``include()`` is located itself in the CMake builtin
module directory, then first the CMake builtin module directory is searched and
:variable:`CMAKE_MODULE_PATH` afterwards.  See also policy :policy:`CMP0017`.

See the :command:`cmake_policy` command documentation for discussion of the
``NO_POLICY_SCOPE`` option.
