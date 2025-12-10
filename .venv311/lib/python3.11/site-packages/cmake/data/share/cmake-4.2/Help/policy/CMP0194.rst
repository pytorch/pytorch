CMP0194
-------

.. versionadded:: 4.1

MSVC is not an assembler for language ASM.

When enabling the ``ASM`` language, CMake considers C compiler drivers
as assembler candidates.  CMake 4.0 and below accidentally selected
MSVC's ``cl`` compiler as the ``CMAKE_ASM_COMPILER``, allowing the ``ASM``
language to be enabled on Windows even though ``cl`` does not support
assembler sources.  CMake 4.1 and above prefer to reject ``cl`` as an
assembler candidate, but some existing projects unconditionally enable
``ASM`` on Windows even though they add no assembler sources.  This
policy provides compatibility for such projects to allow them to
configure as before.

The ``OLD`` behavior for this policy is to successfully enable ``ASM``
even if ``cl`` is the only available candidate.  The ``NEW`` behavior
for this policy is to not consider ``cl`` as a candidate assembler
for the ``ASM`` language.

.. |INTRODUCED_IN_CMAKE_VERSION| replace:: 4.1
.. |WARNS_OR_DOES_NOT_WARN| replace:: warns
.. include:: include/STANDARD_ADVICE.rst

.. include:: include/DEPRECATED.rst
