site_name
---------

Set the given variable to the name of the computer.

.. code-block:: cmake

  site_name(variable)

On UNIX-like platforms, if the variable ``HOSTNAME`` is set, its value
will be executed as a command expected to print out the host name,
much like the ``hostname`` command-line tool.
