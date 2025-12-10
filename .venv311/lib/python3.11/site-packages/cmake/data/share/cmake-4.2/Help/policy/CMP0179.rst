CMP0179
-------

.. versionadded:: 3.31

De-duplication of static libraries on link lines keeps first occurrence.
This policy is only relevant when policy :policy:`CMP0156` is set to ``NEW``.

Based on the linker capabilities, the static libraries can
be de-duplicated. See policy :policy:`CMP0156` for more information.

CMake 3.30 and below may choose to keep, on some platforms, the last occurrence
of the static libraries rather than the fist occurrence when they are
de-duplicated.

CMake 3.31 and above prefer to keep, on all platforms, the first occurrence of
the static libraries when they are de-duplicated.

The ``OLD`` behavior for this policy is to keep, on some platforms, the last
occurrence of the static libraries when they are de-duplicated.  The ``NEW``
behavior for this policy is to keep the first occurrence of the static
libraries when they are de-duplicated, regardless of the platform.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 3.31
.. |WARNS_OR_DOES_NOT_WARN| replace:: does *not* warn
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
