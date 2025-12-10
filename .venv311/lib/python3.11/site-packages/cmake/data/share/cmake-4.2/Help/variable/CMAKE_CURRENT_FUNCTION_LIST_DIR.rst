CMAKE_CURRENT_FUNCTION_LIST_DIR
-------------------------------

.. versionadded:: 3.17

When executing code inside a :command:`function`, this variable
contains the full directory of the listfile that defined the current function.

It is quite common practice in CMake for modules to use some additional files,
such as templates to be copied in after substituting CMake variables.
In such cases, a function needs to know where to locate those files in a way
that doesn't depend on where the function is called.  Without
``CMAKE_CURRENT_FUNCTION_LIST_DIR``, the code to do that would typically use
the following pattern:

.. code-block:: cmake

  set(_THIS_MODULE_BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")

  function(foo)
    configure_file(
      "${_THIS_MODULE_BASE_DIR}/some.template.in"
      some.output
    )
  endfunction()

Using ``CMAKE_CURRENT_FUNCTION_LIST_DIR`` inside the function instead
eliminates the need for the extra variable which would otherwise be visible
outside the function's scope.
The above example can be written in the more concise and more robust form:

.. code-block:: cmake

  function(foo)
    configure_file(
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/some.template.in"
      some.output
    )
  endfunction()

See also :variable:`CMAKE_CURRENT_FUNCTION`,
:variable:`CMAKE_CURRENT_FUNCTION_LIST_FILE`,
:variable:`CMAKE_CURRENT_FUNCTION_LIST_LINE` and
:variable:`CMAKE_CURRENT_LIST_DIR`.
