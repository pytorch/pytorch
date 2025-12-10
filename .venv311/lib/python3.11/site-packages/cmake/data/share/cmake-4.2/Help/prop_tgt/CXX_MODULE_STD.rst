CXX_MODULE_STD
--------------

.. versionadded:: 3.30

``CXX_MODULE_STD`` is a boolean specifying whether the target may use
``import std;`` its C++ sources or not.

.. note::

   This setting is meaningful only when experimental support for ``import
   std;`` has been enabled by the ``CMAKE_EXPERIMENTAL_CXX_IMPORT_STD`` gate.

When this property is explicitly set to ``ON``, CMake will add a dependency to
a target which provides the C++ standard library's modules for the C++
standard applied to the target. This target is only applicable within the
current build and will not appear in the exported interfaces of the targets.
When consumed, these targets will be reapplied as necessary.

.. note::

   Similar to the introduction of :prop_tgt:`CXX_SCAN_FOR_MODULES`, this
   property defaults to **not** adding ``import std`` support to targets using
   ``cxx_std_23`` without an explicit request in order to preserve existing
   behavior for projects using C++23 without ``import std``. A future policy
   to change the default behavior is expected once the feature sees wider
   usage.

This property's value is not relevant for targets which disable scanning (see
:prop_tgt:`CXX_SCAN_FOR_MODULES`). Additionally, this property only applies to
targets utilizing C++23 (``cxx_std_23``) or newer.

The property supports
:manual:`generator expressions <cmake-generator-expressions(7)>`, however
expressions that depend upon the configuration, the consuming target, or the
linker language are not allowed. Whether a target uses ``import std`` should
not depend upon such things as it is a static property of the target's source
code.

Targets which are exported with C++ module sources will have this property's
resolved value exported.
