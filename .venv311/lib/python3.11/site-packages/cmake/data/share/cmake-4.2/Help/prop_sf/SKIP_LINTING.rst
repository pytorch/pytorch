SKIP_LINTING
------------

.. versionadded:: 3.27

This property allows you to exclude a specific source file
from the linting process. The linting process involves running
tools such as :prop_tgt:`<LANG>_CPPLINT`, :prop_tgt:`<LANG>_CLANG_TIDY`,
:prop_tgt:`<LANG>_CPPCHECK`, :prop_tgt:`<LANG>_ICSTAT` and
:prop_tgt:`<LANG>_INCLUDE_WHAT_YOU_USE` on the source files, as well
as compiling header files as part of :prop_tgt:`VERIFY_INTERFACE_HEADER_SETS`.
By setting ``SKIP_LINTING`` on a source file, the mentioned linting tools
will not be executed for that particular file.

Example
^^^^^^^

Consider a C++ project that includes multiple source files,
such as ``main.cpp``, ``things.cpp``, and ``generatedBindings.cpp``.
In this example, you want to exclude the ``generatedBindings.cpp``
file from the linting process. To achieve this, you can utilize
the ``SKIP_LINTING`` property with the :command:`set_source_files_properties`
command as shown below:

.. code-block:: cmake

  add_executable(MyApp main.cpp things.cpp generatedBindings.cpp)

  set_source_files_properties(generatedBindings.cpp PROPERTIES
      SKIP_LINTING ON
  )

In the provided code snippet, the ``SKIP_LINTING`` property is set to true
for the ``generatedBindings.cpp`` source file. As a result, when the linting
tools specified by :prop_tgt:`<LANG>_CPPLINT`, :prop_tgt:`<LANG>_CLANG_TIDY`,
:prop_tgt:`<LANG>_CPPCHECK`, :prop_tgt:`<LANG>_ICSTAT` or
:prop_tgt:`<LANG>_INCLUDE_WHAT_YOU_USE` are executed, they will skip analyzing
the ``generatedBindings.cpp`` file.

By using the ``SKIP_LINTING`` property, you can selectively exclude specific
source files from the linting process. This allows you to focus the
linting tools on the relevant parts of your project, enhancing the efficiency
and effectiveness of the linting workflow.

See Also
^^^^^^^^

* :prop_tgt:`SKIP_LINTING` target property
