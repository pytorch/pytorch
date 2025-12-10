endwhile
--------

Ends a list of commands in a while block.

.. code-block:: cmake

  endwhile([<condition>])

See the :command:`while` command.

The optional ``<condition>`` argument is supported for backward compatibility
only. If used it must be a verbatim repeat of the argument of the opening
``while`` clause.
