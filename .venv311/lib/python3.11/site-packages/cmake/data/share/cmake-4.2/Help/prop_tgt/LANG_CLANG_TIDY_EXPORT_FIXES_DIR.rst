<LANG>_CLANG_TIDY_EXPORT_FIXES_DIR
----------------------------------

.. versionadded:: 3.26

This property is implemented only when ``<LANG>`` is ``C``, ``CXX``, ``OBJC``
or ``OBJCXX``, and only has an effect when :prop_tgt:`<LANG>_CLANG_TIDY` is
set.

Specify a directory for the ``clang-tidy`` tool to put ``.yaml`` files
containing its suggested changes in. This can be used for automated mass
refactoring by ``clang-tidy``. Each object file that gets compiled will have a
corresponding ``.yaml`` file in this directory. After the build is completed,
you can run ``clang-apply-replacements`` on this directory to simultaneously
apply all suggested changes to the code base. If this property is not an
absolute directory, it is assumed to be relative to the target's binary
directory. This property should be preferred over adding an ``--export-fixes``
or ``--fix`` argument directly to the :prop_tgt:`<LANG>_CLANG_TIDY` property.

When this property is set, CMake takes ownership of the specified directory,
and may create, modify, or delete files and directories within the directory
at any time during configure or build time. Users should use a dedicated
directory for exporting clang-tidy fixes to avoid having files deleted or
overwritten by CMake. Users should not create, modify, or delete files in this
directory.

This property is initialized by the value of
the :variable:`CMAKE_<LANG>_CLANG_TIDY_EXPORT_FIXES_DIR` variable if it is set
when a target is created.
