UNITY_BUILD_FILENAME_PREFIX
---------------------------

.. versionadded:: 4.2

By default, the unity file generated when :prop_tgt:`UNITY_BUILD` is enabled
is of the form ``unity_<index>_<suffix>``, where ``<suffix>`` is language-specific.

If several targets are using unity builds, the build output may give no
indication which target a unity file belongs to. This property allows
customizing the prefix of the generated unity file name. If unset,
the default prefix ``unity_`` is used.

Example usage:

.. code-block:: cmake

  add_library(example_library
              source1.cxx
              source2.cxx
              source3.cxx)

  set_target_properties(example_library PROPERTIES
                        UNITY_BUILD True
                        UNITY_BUILD_FILENAME_PREFIX "example_")
