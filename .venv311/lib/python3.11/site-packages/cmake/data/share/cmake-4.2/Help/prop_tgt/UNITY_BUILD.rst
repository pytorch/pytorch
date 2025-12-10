UNITY_BUILD
-----------

.. versionadded:: 3.16

When this property is set to true, the target source files will be combined
into batches for faster compilation.  This is done by creating a (set of)
unity sources which ``#include`` the original sources, then compiling these
unity sources instead of the originals.  This is known as a *Unity* or *Jumbo*
build.

CMake provides different algorithms for selecting which sources are grouped
together into a *bucket*. Algorithm selection is decided by the
:prop_tgt:`UNITY_BUILD_MODE` target property, which has the following acceptable
values:

* ``BATCH``
  When in this mode CMake determines which files are grouped together.
  The :prop_tgt:`UNITY_BUILD_BATCH_SIZE` property controls the upper limit on
  how many sources can be combined per unity source file.

* ``GROUP``
  When in this mode each target explicitly specifies how to group
  source files. Each source file that has the same
  :prop_sf:`UNITY_GROUP` value will be grouped together. Any sources
  that don't have this property will be compiled individually. The
  :prop_tgt:`UNITY_BUILD_BATCH_SIZE` property is ignored when using
  this mode.

If no explicit :prop_tgt:`UNITY_BUILD_MODE` has been specified, CMake will
default to ``BATCH``.

Unity builds are supported for the following languages:

``C``
  .. versionadded:: 3.16

``CXX``
  .. versionadded:: 3.16

``CUDA``
  .. versionadded:: 3.31

``OBJC``
  .. versionadded:: 3.29

``OBJCXX``
  .. versionadded:: 3.29

For targets that mix source files from more than one language, CMake
separates the languages such that each generated unity source file only
contains sources for a single language.

This property is initialized by the value of the :variable:`CMAKE_UNITY_BUILD`
variable when a target is created.

.. note::

  Projects should not directly set the ``UNITY_BUILD`` property or its
  associated :variable:`CMAKE_UNITY_BUILD` variable to true.  Depending
  on the capabilities of the build machine and compiler used, it might or
  might not be appropriate to enable unity builds.  Therefore, this feature
  should be under developer control, which would normally be through the
  developer choosing whether or not to set the :variable:`CMAKE_UNITY_BUILD`
  variable on the :manual:`cmake(1)` command line or some other equivalent
  method.  However, it IS recommended to set the ``UNITY_BUILD`` target
  property to false if it is known that enabling unity builds for the
  target can lead to problems.

ODR (One definition rule) errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When multiple source files are included into one source file, as is done
for unity builds, it can potentially lead to ODR errors.  CMake provides
a number of measures to help address such problems:

* Any source file that has a non-empty :prop_sf:`COMPILE_OPTIONS`,
  :prop_sf:`COMPILE_DEFINITIONS`, :prop_sf:`COMPILE_FLAGS`, or
  :prop_sf:`INCLUDE_DIRECTORIES` source property will not be combined
  into a unity source.

* Any source file which is scanned for C++ module sources via
  :prop_tgt:`CXX_SCAN_FOR_MODULES`, :prop_sf:`CXX_SCAN_FOR_MODULES`, or
  membership of a ``CXX_MODULES`` file set will not be combined into a unity
  source.  See :manual:`cmake-cxxmodules(7)` for details.

* Projects can prevent an individual source file from being combined into
  a unity source by setting its :prop_sf:`SKIP_UNITY_BUILD_INCLUSION`
  source property to true.  This can be a more effective way to prevent
  problems with specific files than disabling unity builds for an entire
  target.

* Projects can set :prop_tgt:`UNITY_BUILD_UNIQUE_ID` to cause a valid
  C-identifier to be generated which is unique per file in a unity
  build.  This can be used to avoid problems with anonymous namespaces
  in unity builds.

* The :prop_tgt:`UNITY_BUILD_CODE_BEFORE_INCLUDE` and
  :prop_tgt:`UNITY_BUILD_CODE_AFTER_INCLUDE` target properties can be used
  to inject code into the unity source files before and after every
  ``#include`` statement.

* The order of source files added to the target via commands like
  :command:`add_library`, :command:`add_executable` or
  :command:`target_sources` will be preserved in the generated unity source
  files.  This can be used to manually enforce a specific grouping based on
  the :prop_tgt:`UNITY_BUILD_BATCH_SIZE` target property.
