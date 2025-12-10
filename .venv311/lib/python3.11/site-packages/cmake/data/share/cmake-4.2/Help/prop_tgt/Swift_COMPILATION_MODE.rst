Swift_COMPILATION_MODE
----------------------

.. versionadded:: 3.29

Specify how Swift compiles a target.

The allowed values are:

.. include:: include/Swift_COMPILATION_MODE-VALUES.rst

Use :manual:`generator expressions <cmake-generator-expressions(7)>` to support
per-configuration specification. For example, the code:

.. code-block:: cmake

  add_library(foo foo.swift)
  set_property(TARGET foo PROPERTY
    Swift_COMPILATION_MODE "$<IF:$<CONFIG:Release>,wholemodule,incremental>")

sets the Swift compilation mode to wholemodule mode in the release configuration
and sets the property to incremental mode in other configurations.

The property is initialized from the value of the
:variable:`CMAKE_Swift_COMPILATION_MODE` variable, if it is set. If the property
is not set or is empty, then CMake uses the default value ``incremental`` to
specify the swift compilation mode.

.. note::

   This property only has effect when policy :policy:`CMP0157` is set to ``NEW``
   prior to the first :command:`project` or :command:`enable_language` command
   that enables the Swift language.
