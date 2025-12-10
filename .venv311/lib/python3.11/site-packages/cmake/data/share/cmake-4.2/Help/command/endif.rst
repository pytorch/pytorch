endif
-----

Ends a list of commands in an if block.

.. code-block:: cmake

  endif([<condition>])

See the :command:`if` command.

The optional ``<condition>`` argument is supported for backward compatibility
only. If used it must be a verbatim repeat of the argument of the opening
``if`` clause.
