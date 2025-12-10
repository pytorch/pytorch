CMP0157
-------

.. versionadded:: 3.29

Swift compilation mode is selected by an abstraction.

The Swift compiler can compile modules in different modes. The desired build
mode depends whether the developer is iterating and wants to incrementally make
changes, or if they are building a release for distribution and want more
optimizations applied to the resulting binary.

CMake versions 3.26 through 3.28 build Swift binaries with whole-module
optimizations enabled when configured in a non-debug build type.
For CMake versions earlier than 3.26, the developer needs to specify
the necessary flag manually for the :ref:`Ninja Generators`, and cannot
not specify whole-module optimizations to the :generator:`Xcode` generator.

CMake versions 3.29 and above prefer to set the compilation mode using
the :prop_tgt:`Swift_COMPILATION_MODE` target property, which can be
initialized by the :variable:`CMAKE_Swift_COMPILATION_MODE` variable.

This policy provides compatibility for projects that have not been updated.
The policy setting takes effect as of the first :command:`project` or
:command:`enable_language` command that enables the ``Swift`` language.

.. note::

  Once the policy has taken effect at the top of a project, that choice
  must be used throughout the tree. In projects that have nested projects
  in subdirectories, be sure to convert everything together.

The ``OLD`` behavior for this policy builds all Swift targets in
``wholemodule`` mode for non-debug configurations.  :ref:`Ninja Generators`
prepend the ``-wmo`` flag to the default set of Swift flags.
The :generator:`Xcode` generator sets the ``SWIFT_COMPILATION_MODE``
attribute to ``wholemodule`` in the generated Xcode project file.

The ``NEW`` behavior for this policy is to apply the compilation mode specified
in the :prop_tgt:`Swift_COMPILATION_MODE` target property, initialized as each
target is created by the :variable:`CMAKE_Swift_COMPILATION_MODE` variable.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.29
.. |WARNS_OR_DOES_NOT_WARN| replace:: does *not* warn
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
