CMAKE_Swift_COMPILATION_MODE
----------------------------

.. versionadded:: 3.29

Specify how Swift compiles a target. This variable is used to initialize the
:prop_tgt:`Swift_COMPILATION_MODE` property on targets as they are created.

The allowed values are:

.. include:: ../prop_tgt/include/Swift_COMPILATION_MODE-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to support
per-configuration specification. For example, the code:

.. code-block:: cmake

   set(CMAKE_Swift_COMPILATION_MODE
     "$<IF:$<CONFIG:Release>,wholemodule,incremental>")

sets the default Swift compilation mode to wholemodule mode when building a
release configuration and to incremental mode in other configurations.

If this variable is not set then the :prop_tgt:`Swift_COMPILATION_MODE` target
property will not be set automatically. If that property is unset then CMake
uses the default value ``incremental`` to build the Swift source files.

.. note::

   This property only has effect when policy :policy:`CMP0157` is set to ``NEW``
   prior to the first :command:`project` or :command:`enable_language` command
   that enables the Swift language.
