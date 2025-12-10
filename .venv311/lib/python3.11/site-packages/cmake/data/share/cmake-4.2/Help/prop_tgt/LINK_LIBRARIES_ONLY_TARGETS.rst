LINK_LIBRARIES_ONLY_TARGETS
---------------------------

.. versionadded:: 3.23

Enforce that link items that can be target names are actually existing targets.

Set this property to a true value to enable additional checks on the contents
of the :prop_tgt:`LINK_LIBRARIES` and :prop_tgt:`INTERFACE_LINK_LIBRARIES`
target properties, typically populated by :command:`target_link_libraries`.
Checks are also applied to libraries added to a target through the
:prop_tgt:`INTERFACE_LINK_LIBRARIES_DIRECT` properties of its dependencies.
CMake will verify that link items that might be target names actually name
existing targets.  An item is considered a possible target name if:

* it does not contain a ``/`` or ``\``, and
* it does not start in ``-``, and
* (for historical reasons) it does not start in ``$`` or `````.

This property is initialized by the value of the
:variable:`CMAKE_LINK_LIBRARIES_ONLY_TARGETS` variable when a non-imported
target is created.  The property may be explicitly enabled on an imported
target to check its link interface.

In the following example, CMake will halt with an error at configure time
because ``miLib`` is not a target:

.. code-block:: cmake

  set(CMAKE_LINK_LIBRARIES_ONLY_TARGETS ON)
  add_library(myLib STATIC myLib.c)
  add_executable(myExe myExe.c)
  target_link_libraries(myExe PRIVATE miLib) # typo for myLib

In order to link toolchain-provided libraries by name while still
enforcing ``LINK_LIBRARIES_ONLY_TARGETS``, use an
:ref:`imported <Imported Targets>`
:ref:`Interface Library <Interface Libraries>` with the
:prop_tgt:`IMPORTED_LIBNAME` target property:

.. code-block:: cmake

  add_library(toolchain::m INTERFACE IMPORTED)
  set_property(TARGET toolchain::m PROPERTY IMPORTED_LIBNAME "m")
  target_link_libraries(myExe PRIVATE toolchain::m)

See also policy :policy:`CMP0028`.

.. note::

  If :prop_tgt:`INTERFACE_LINK_LIBRARIES` contains generator expressions,
  its actual list of link items may depend on the type and properties of
  the consuming target.  In such cases CMake may not always detect names
  of missing targets that only appear for specific consumers.
  A future version of CMake with improved heuristics may start triggering
  errors on projects accepted by previous versions of CMake.
