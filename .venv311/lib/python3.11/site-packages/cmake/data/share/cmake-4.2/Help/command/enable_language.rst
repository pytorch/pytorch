enable_language
---------------

Enable languages (CXX/C/OBJC/OBJCXX/Fortran/etc)

.. code-block:: cmake

  enable_language(<lang>... [OPTIONAL])

Enables support for the named languages in CMake.  This is the same as
the :command:`project` command but does not create any of the extra
variables that are created by the :command:`project` command.

.. include:: include/SUPPORTED_LANGUAGES.rst

The following restrictions apply to where ``enable_language()`` may be called:

* It must be called in file scope, not in a function call.
* It must not be called before the first call to :command:`project`.
  See policy :policy:`CMP0165`.
* It must be called in the highest directory common to all targets
  using the named language directly for compiling sources or
  indirectly through link dependencies.  It is simplest to enable all
  needed languages in the top-level directory of a project.

The ``OPTIONAL`` keyword is a placeholder for future implementation and
does not currently work. Instead you can use the :module:`CheckLanguage`
module to verify support before enabling.
