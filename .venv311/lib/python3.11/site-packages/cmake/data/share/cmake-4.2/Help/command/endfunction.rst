endfunction
-----------

Ends a list of commands in a function block.

.. code-block:: cmake

  endfunction([<name>])

See the :command:`function` command.

The optional ``<name>`` argument is supported for backward compatibility
only. If used it must be a verbatim repeat of the ``<name>`` argument
of the opening ``function`` command.
