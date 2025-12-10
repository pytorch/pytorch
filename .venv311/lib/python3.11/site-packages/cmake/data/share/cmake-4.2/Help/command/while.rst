while
-----

Evaluate a group of commands while a condition is true

.. code-block:: cmake

  while(<condition>)
    <commands>
  endwhile()

All commands between while and the matching :command:`endwhile` are recorded
without being invoked.  Once the :command:`endwhile` is evaluated, the
recorded list of commands is invoked as long as the ``<condition>`` is true.

The ``<condition>`` has the same syntax and is evaluated using the same logic
as described at length for the :command:`if` command.

The commands :command:`break` and :command:`continue` provide means to
escape from the normal control flow.

Per legacy, the :command:`endwhile` command admits
an optional ``<condition>`` argument.
If used, it must be a verbatim repeat of the argument of the opening
``while`` command.

See Also
^^^^^^^^

* :command:`break`
* :command:`continue`
* :command:`foreach`
* :command:`endwhile`
