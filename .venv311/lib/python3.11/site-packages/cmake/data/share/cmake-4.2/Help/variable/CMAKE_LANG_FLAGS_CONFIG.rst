CMAKE_<LANG>_FLAGS_<CONFIG>
---------------------------

Language-wide flags for language ``<LANG>`` used when building for
the ``<CONFIG>`` configuration.  These flags will be passed to all
invocations of the compiler in the corresponding configuration.
This includes invocations that drive compiling and those that drive
linking.

The flags in this variable will be passed after those in the
:variable:`CMAKE_<LANG>_FLAGS` variable.  On invocations driving compiling,
flags from both variables will be passed before flags added by commands
such as :command:`add_compile_options` and :command:`target_compile_options`.
On invocations driving linking, they will be passed before flags added by
commands such as :command:`add_link_options` and
:command:`target_link_options`.
