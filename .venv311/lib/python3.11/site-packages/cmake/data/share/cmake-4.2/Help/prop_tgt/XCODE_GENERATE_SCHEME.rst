XCODE_GENERATE_SCHEME
---------------------

.. versionadded:: 3.15

If enabled, the :generator:`Xcode` generator will generate schema files.  These
are useful to invoke analyze, archive, build-for-testing and test
actions from the command line.

This property is initialized by the value of the variable
:variable:`CMAKE_XCODE_GENERATE_SCHEME` if it is set when a target
is created.

The following target properties overwrite the default of the
corresponding settings on the "Diagnostic" tab for each schema file.
Each of those is initialized by the respective ``CMAKE_`` variable
at target creation time.

- :prop_tgt:`XCODE_SCHEME_ADDRESS_SANITIZER`
- :prop_tgt:`XCODE_SCHEME_ADDRESS_SANITIZER_USE_AFTER_RETURN`
- :prop_tgt:`XCODE_SCHEME_DISABLE_MAIN_THREAD_CHECKER`
- :prop_tgt:`XCODE_SCHEME_DYNAMIC_LIBRARY_LOADS`
- :prop_tgt:`XCODE_SCHEME_DYNAMIC_LINKER_API_USAGE`
- :prop_tgt:`XCODE_SCHEME_GUARD_MALLOC`
- :prop_tgt:`XCODE_SCHEME_MAIN_THREAD_CHECKER_STOP`
- :prop_tgt:`XCODE_SCHEME_MALLOC_GUARD_EDGES`
- :prop_tgt:`XCODE_SCHEME_MALLOC_SCRIBBLE`
- :prop_tgt:`XCODE_SCHEME_MALLOC_STACK`
- :prop_tgt:`XCODE_SCHEME_THREAD_SANITIZER`
- :prop_tgt:`XCODE_SCHEME_THREAD_SANITIZER_STOP`
- :prop_tgt:`XCODE_SCHEME_UNDEFINED_BEHAVIOUR_SANITIZER`
- :prop_tgt:`XCODE_SCHEME_UNDEFINED_BEHAVIOUR_SANITIZER_STOP`
- :prop_tgt:`XCODE_SCHEME_ENABLE_GPU_API_VALIDATION`
- :prop_tgt:`XCODE_SCHEME_ENABLE_GPU_SHADER_VALIDATION`
- :prop_tgt:`XCODE_SCHEME_ZOMBIE_OBJECTS`

The following target properties will be applied on the
"Info", "Arguments", and "Options" tab:

- :prop_tgt:`XCODE_SCHEME_ARGUMENTS`
- :prop_tgt:`XCODE_SCHEME_DEBUG_AS_ROOT`
- :prop_tgt:`XCODE_SCHEME_DEBUG_DOCUMENT_VERSIONING`
- :prop_tgt:`XCODE_SCHEME_ENABLE_GPU_FRAME_CAPTURE_MODE`
- :prop_tgt:`XCODE_SCHEME_ENVIRONMENT`
- :prop_tgt:`XCODE_SCHEME_EXECUTABLE`
- :prop_tgt:`XCODE_SCHEME_LAUNCH_CONFIGURATION`
- :prop_tgt:`XCODE_SCHEME_LAUNCH_MODE`
- :prop_tgt:`XCODE_SCHEME_TEST_CONFIGURATION`
- :prop_tgt:`XCODE_SCHEME_LLDB_INIT_FILE`
- :prop_tgt:`XCODE_SCHEME_WORKING_DIRECTORY`
