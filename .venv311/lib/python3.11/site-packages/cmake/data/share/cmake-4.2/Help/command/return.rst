return
------

Return from a file, directory or function.

.. code-block:: cmake

  return([PROPAGATE <var-name>...])

When this command is encountered in an included file (via :command:`include` or
:command:`find_package`), it causes processing of the current file to stop
and control is returned to the including file.  If it is encountered in a
file which is not included by another file, e.g. a ``CMakeLists.txt``,
deferred calls scheduled by :command:`cmake_language(DEFER)` are invoked and
control is returned to the parent directory if there is one.

If ``return()`` is called in a function, control is returned to the caller
of that function.  Note that a :command:`macro`, unlike a :command:`function`,
is expanded in place and therefore cannot handle ``return()``.

Policy :policy:`CMP0140` controls the behavior regarding the arguments of the
command.  All arguments are ignored unless that policy is set to ``NEW``.

``PROPAGATE``
  .. versionadded:: 3.25

  This option sets or unsets the specified variables in the parent directory or
  function caller scope. This is equivalent to :command:`set(PARENT_SCOPE)` or
  :command:`unset(PARENT_SCOPE)` commands, except for the way it interacts
  with the :command:`block` command, as described below.

  The ``PROPAGATE`` option can be very useful in conjunction with the
  :command:`block` command.  A ``return`` will propagate the
  specified variables through any enclosing block scopes created by the
  :command:`block` commands.  Inside a function, this ensures the variables
  are propagated to the function's caller, regardless of any blocks within
  the function.  If not inside a function, it ensures the variables are
  propagated to the parent file or directory scope. For example:

  .. code-block:: cmake
    :caption: CMakeLists.txt

    cmake_minimum_required(VERSION 3.25)
    project(example)

    set(var1 "top-value")

    block(SCOPE_FOR VARIABLES)
      add_subdirectory(subDir)
      # var1 has the value "block-nested"
    endblock()

    # var1 has the value "top-value"

  .. code-block:: cmake
    :caption: subDir/CMakeLists.txt

    function(multi_scopes result_var1 result_var2)
      block(SCOPE_FOR VARIABLES)
        # This would only propagate out of the immediate block, not to
        # the caller of the function.
        #set(${result_var1} "new-value" PARENT_SCOPE)
        #unset(${result_var2} PARENT_SCOPE)

        # This propagates the variables through the enclosing block and
        # out to the caller of the function.
        set(${result_var1} "new-value")
        unset(${result_var2})
        return(PROPAGATE ${result_var1} ${result_var2})
      endblock()
    endfunction()

    set(var1 "some-value")
    set(var2 "another-value")

    multi_scopes(var1 var2)
    # Now var1 will hold "new-value" and var2 will be unset

    block(SCOPE_FOR VARIABLES)
      # This return() will set var1 in the directory scope that included us
      # via add_subdirectory(). The surrounding block() here does not limit
      # propagation to the current file, but the block() in the parent
      # directory scope does prevent propagation going any further.
      set(var1 "block-nested")
      return(PROPAGATE var1)
    endblock()

See Also
^^^^^^^^

* :command:`block`
* :command:`function`
