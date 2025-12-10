ENV
---

Operator to read environment variables.

Use the syntax ``$ENV{VAR}`` to read environment variable ``VAR``.

To test whether an environment variable is defined, use the signature
``if(DEFINED ENV{<name>})`` of the :command:`if` command.

.. note::

  Environment variable names containing special characters like parentheses
  may need to be escaped.  (Policy :policy:`CMP0053` must also be enabled.)
  For example, to get the value of the Windows environment variable
  ``ProgramFiles(x86)``, use:

  .. code-block:: cmake

      set(ProgramFiles_x86 "$ENV{ProgramFiles\(x86\)}")

For general information on environment variables, see the
:ref:`Environment Variables <CMake Language Environment Variables>`
section in the :manual:`cmake-language(7)` manual.
