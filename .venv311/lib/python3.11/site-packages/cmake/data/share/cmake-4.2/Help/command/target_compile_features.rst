target_compile_features
-----------------------

.. versionadded:: 3.1

Add expected compiler features to a target.

.. code-block:: cmake

  target_compile_features(<target> <PRIVATE|PUBLIC|INTERFACE> <feature> [...])

Specifies compiler features required when compiling a given target.  If the
feature is not listed in the :variable:`CMAKE_C_COMPILE_FEATURES`,
:variable:`CMAKE_CUDA_COMPILE_FEATURES`, or :variable:`CMAKE_CXX_COMPILE_FEATURES`
variables, then an error will be reported by CMake.  If the use of the feature requires
an additional compiler flag, such as ``-std=gnu++11``, the flag will be added
automatically.

The ``INTERFACE``, ``PUBLIC`` and ``PRIVATE`` keywords are required to
specify the scope of the features.  ``PRIVATE`` and ``PUBLIC`` items will
populate the :prop_tgt:`COMPILE_FEATURES` property of ``<target>``.
``PUBLIC`` and ``INTERFACE`` items will populate the
:prop_tgt:`INTERFACE_COMPILE_FEATURES` property of ``<target>``.
Repeated calls for the same ``<target>`` append items.

.. versionadded:: 3.11
  Allow setting ``INTERFACE`` items on :ref:`IMPORTED targets <Imported Targets>`.

The named ``<target>`` must have been created by a command such as
:command:`add_executable` or :command:`add_library` and must not be an
:ref:`ALIAS target <Alias Targets>`.

.. |command_name| replace:: ``target_compile_features``
.. |more_see_also| replace:: See the :manual:`cmake-compile-features(7)`
   manual for information on compile features and a list of supported compilers.
.. include:: include/GENEX_NOTE.rst
   :start-line: 2

See Also
^^^^^^^^

* :command:`target_compile_definitions`
* :command:`target_compile_options`
* :command:`target_include_directories`
* :command:`target_link_libraries`
* :command:`target_link_directories`
* :command:`target_link_options`
* :command:`target_precompile_headers`
* :command:`target_sources`
