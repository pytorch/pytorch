VERIFY_INTERFACE_HEADER_SETS
----------------------------

.. versionadded:: 3.24

Used to verify that all headers in a target's ``PUBLIC`` and ``INTERFACE``
header sets can be included on their own.

When this property is set to true, and the target is an object library, static
library, shared library, interface library, or executable with exports enabled,
and the target has one or more ``PUBLIC`` or ``INTERFACE`` header sets, an
object library target named ``<target_name>_verify_interface_header_sets`` is
created. This verification target has one source file per header in the
``PUBLIC`` and ``INTERFACE`` header sets. Each source file only includes its
associated header file. The verification target links against the original
target to get all of its usage requirements. The verification target has its
:prop_tgt:`EXCLUDE_FROM_ALL` and :prop_tgt:`DISABLE_PRECOMPILE_HEADERS`
properties set to true, and its :prop_tgt:`AUTOMOC`, :prop_tgt:`AUTORCC`,
:prop_tgt:`AUTOUIC`, and :prop_tgt:`UNITY_BUILD` properties set to false.

If the header's :prop_sf:`LANGUAGE` property is set, the value of that property
is used to determine the language with which to compile the header file.
Otherwise, if the target has any C++ sources, the header is compiled as C++.
Otherwise, if the target has any C sources, the header is compiled as C.
Otherwise, if C++ is enabled globally, the header is compiled as C++.
Otherwise, if C is enabled globally, the header is compiled as C. Otherwise,
the header file is not compiled.

If the header's :prop_sf:`SKIP_LINTING` property is set to true, the file is
not compiled.

If any verification targets are created, a top-level target called
``all_verify_interface_header_sets`` is created which depends on all
verification targets.

This property is initialized by the value of the
:variable:`CMAKE_VERIFY_INTERFACE_HEADER_SETS` variable if it is set when
a target is created.

If the project wishes to control which header sets are verified by this
property, it can set :prop_tgt:`INTERFACE_HEADER_SETS_TO_VERIFY`.
