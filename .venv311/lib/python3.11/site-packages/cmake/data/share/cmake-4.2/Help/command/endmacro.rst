endmacro
--------

Ends a list of commands in a macro block.

.. code-block:: cmake

  endmacro([<name>])

See the :command:`macro` command.

The optional ``<name>`` argument is supported for backward compatibility
only. If used it must be a verbatim repeat of the ``<name>`` argument
of the opening ``macro`` command.
