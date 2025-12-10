target_precompile_headers
-------------------------

.. versionadded:: 3.16

Add a list of header files to precompile.

Precompiling header files can speed up compilation by creating a partially
processed version of some header files, and then using that version during
compilations rather than repeatedly parsing the original headers.

Main Form
^^^^^^^^^

.. code-block:: cmake

  target_precompile_headers(<target>
    <INTERFACE|PUBLIC|PRIVATE> [header1...]
    [<INTERFACE|PUBLIC|PRIVATE> [header2...] ...])

The command adds header files to the :prop_tgt:`PRECOMPILE_HEADERS` and/or
:prop_tgt:`INTERFACE_PRECOMPILE_HEADERS` target properties of ``<target>``.
The named ``<target>`` must have been created by a command such as
:command:`add_executable` or :command:`add_library` and must not be an
:ref:`ALIAS target <Alias Targets>`.

The ``INTERFACE``, ``PUBLIC`` and ``PRIVATE`` keywords are required to
specify the :ref:`scope <Target Command Scope>` of the following arguments.
``PRIVATE`` and ``PUBLIC`` items will populate the :prop_tgt:`PRECOMPILE_HEADERS`
property of ``<target>``.  ``PUBLIC`` and ``INTERFACE`` items will populate the
:prop_tgt:`INTERFACE_PRECOMPILE_HEADERS` property of ``<target>``
(:ref:`IMPORTED targets <Imported Targets>` only support ``INTERFACE`` items).
Repeated calls for the same ``<target>`` will append items in the order called.

Projects should generally avoid using ``PUBLIC`` or ``INTERFACE`` for targets
that will be :command:`exported <install(EXPORT)>`, or they should at least use
the :genex:`$<BUILD_INTERFACE:...>` generator expression to prevent precompile
headers from appearing in an installed exported target.  Consumers of a target
should typically be in control of what precompile headers they use, not have
precompile headers forced on them by the targets being consumed (since
precompile headers are not typically usage requirements).  A notable exception
to this is where an :ref:`interface library <Interface Libraries>` is created
to define a commonly used set of precompile headers in one place and then other
targets link to that interface library privately.  In this case, the interface
library exists specifically to propagate the precompile headers to its
consumers and the consumer is effectively still in control, since it decides
whether to link to the interface library or not.

The list of header files is used to generate a header file named
``cmake_pch.h|xx`` which is used to generate the precompiled header file
(``.pch``, ``.gch``, ``.pchi``) artifact.  The ``cmake_pch.h|xx`` header
file will be force included (``-include`` for GCC, ``/FI`` for MSVC) to
all source files, so sources do not need to have ``#include "pch.h"``.

Header file names specified with angle brackets (e.g. ``<unordered_map>``) or
explicit double quotes (escaped for the :manual:`cmake-language(7)`,
e.g. ``[["other_header.h"]]``) will be treated as is, and include directories
must be available for the compiler to find them.  Other header file names
(e.g. ``project_header.h``) are interpreted as being relative to the current
source directory (e.g. :variable:`CMAKE_CURRENT_SOURCE_DIR`) and will be
included by absolute path.  For example:

.. code-block:: cmake

  target_precompile_headers(myTarget
    PUBLIC
      project_header.h
    PRIVATE
      [["other_header.h"]]
      <unordered_map>
  )

.. |command_name| replace:: ``target_precompile_headers``
.. |more_see_also| replace:: The :genex:`$<COMPILE_LANGUAGE:...>
   <COMPILE_LANGUAGE:languages>` generator
   expression is particularly useful for specifying a language-specific header
   to precompile for only one language (e.g. ``CXX`` and not ``C``).  In this
   case, header file names that are not explicitly in double quotes or angle
   brackets must be specified by absolute path.  Also, when specifying angle
   brackets inside a generator expression, be sure to encode the closing
   ``>`` as :genex:`$<ANGLE-R>`.  For example:
.. include:: include/GENEX_NOTE.rst
   :start-line: 2

.. code-block:: cmake

  target_precompile_headers(mylib PRIVATE
    "$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/cxx_only.h>"
    "$<$<COMPILE_LANGUAGE:C>:<stddef.h$<ANGLE-R>>"
    "$<$<COMPILE_LANGUAGE:CXX>:<cstddef$<ANGLE-R>>"
  )


Reusing Precompile Headers
^^^^^^^^^^^^^^^^^^^^^^^^^^

The command also supports a second signature which can be used to specify that
one target reuses a precompiled header file artifact from another target
instead of generating its own:

.. code-block:: cmake

  target_precompile_headers(<target> REUSE_FROM <other_target>)

This form sets the :prop_tgt:`PRECOMPILE_HEADERS_REUSE_FROM` property to
``<other_target>`` and adds a dependency such that ``<target>`` will depend
on ``<other_target>``.  CMake will halt with an error if the
:prop_tgt:`PRECOMPILE_HEADERS` property of ``<target>`` is already set when
the ``REUSE_FROM`` form is used.

.. note::

  The ``REUSE_FROM`` form requires the same set of compiler options,
  compiler flags and compiler definitions for both ``<target>`` and
  ``<other_target>``.  Some compilers (e.g. GCC) may issue a warning if the
  precompiled header file cannot be used (``-Winvalid-pch``).

See Also
^^^^^^^^

* To disable precompile headers for specific targets, see the
  :prop_tgt:`DISABLE_PRECOMPILE_HEADERS` target property.

* To prevent precompile headers from being used when compiling a specific
  source file, see the :prop_sf:`SKIP_PRECOMPILE_HEADERS` source file property.

* :command:`target_compile_definitions`
* :command:`target_compile_features`
* :command:`target_compile_options`
* :command:`target_include_directories`
* :command:`target_link_libraries`
* :command:`target_link_directories`
* :command:`target_link_options`
* :command:`target_sources`
