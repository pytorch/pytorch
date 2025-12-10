block
-----

.. versionadded:: 3.25

Evaluate a group of commands with a dedicated variable and/or policy scope.

.. code-block:: cmake

  block([SCOPE_FOR [POLICIES] [VARIABLES]] [PROPAGATE <var-name>...])
    <commands>
  endblock()

All commands between ``block()`` and the matching :command:`endblock` are
recorded without being invoked.  Once the :command:`endblock` is evaluated, the
recorded list of commands is invoked inside the requested scopes, then the
scopes created by the ``block()`` command are removed.

``SCOPE_FOR``
  Specify which scopes must be created.

  ``POLICIES``
    Create a new policy scope. This is equivalent to
    :command:`cmake_policy(PUSH)` with an automatic
    :command:`cmake_policy(POP)` when leaving the block scope.

  ``VARIABLES``
    Create a new variable scope.

  If ``SCOPE_FOR`` is not specified, this is equivalent to:

  .. code-block:: cmake

    block(SCOPE_FOR VARIABLES POLICIES)

``PROPAGATE``
  When a variable scope is created by the :command:`block` command, this
  option sets or unsets the specified variables in the parent scope. This is
  equivalent to :command:`set(PARENT_SCOPE)` or :command:`unset(PARENT_SCOPE)`
  commands.

  .. code-block:: cmake

    set(var1 "INIT1")
    set(var2 "INIT2")
    set(var3 "INIT3")

    block(PROPAGATE var1 var2)
      set(var1 "VALUE1")
      unset(var2)
      set(var3 "VALUE3")
    endblock()

    # Now var1 holds VALUE1, var2 is unset, and var3 holds the initial value INIT3

  This option is only allowed when a variable scope is created. An error will
  be raised in the other cases.

When the ``block()`` is inside a :command:`foreach` or :command:`while`
command, the :command:`break` and :command:`continue` commands can be used
inside the block.

.. code-block:: cmake

  while(TRUE)
    block()
       ...
       # the break() command will terminate the while() command
       break()
    endblock()
  endwhile()


See Also
^^^^^^^^

* :command:`endblock`
* :command:`return`
* :command:`cmake_policy`
