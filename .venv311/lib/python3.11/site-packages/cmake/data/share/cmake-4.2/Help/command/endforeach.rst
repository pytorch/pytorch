endforeach
----------

Ends a list of commands in a foreach block.

.. code-block:: cmake

  endforeach([<loop_var>])

See the :command:`foreach` command.

The optional ``<loop_var>`` argument is supported for backward compatibility
only. If used it must be a verbatim repeat of the ``<loop_var>`` argument of
the opening ``foreach`` clause.
