CMAKE_OSX_ARCHITECTURES
-----------------------

Target specific architectures for macOS and iOS.

This variable is used to initialize the :prop_tgt:`OSX_ARCHITECTURES`
property on each target as it is created.  See that target property
for additional information.

If ``CMAKE_OSX_ARCHITECTURES`` is not set, the compiler's default target
architecture is used.  For compilers provided by Xcode, this is the host
machine's architecture.

.. include:: include/CMAKE_OSX_VARIABLE.rst
