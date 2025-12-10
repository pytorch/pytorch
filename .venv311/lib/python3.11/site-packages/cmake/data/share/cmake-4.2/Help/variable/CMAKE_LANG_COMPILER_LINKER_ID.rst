CMAKE_<LANG>_COMPILER_LINKER_ID
-------------------------------

.. versionadded:: 3.29

Linker identification string.

A short string unique to the linker vendor.  Possible values
include:

=============================== ===============================================
Value                           Name
=============================== ===============================================
``AppleClang``                  Apple Clang
``LLD``                         `LLVM LLD`_
``GNU``                         `GNU Binutils - ld linker`_ (also known as
                                ``bfd``)
``GNUgold``                     `GNU Binutils - gold linker`_
``MSVC``                        `Microsoft Visual Studio`_
``MOLD``                        `mold: A Modern Linker`_, or on Apple the
                                `sold`_ linker
``AIX``                         AIX system linker
``Solaris``                     SunOS system linker
=============================== ===============================================

This variable is not guaranteed to be defined for all linkers or languages.

.. _LLVM LLD: https://lld.llvm.org
.. _GNU Binutils - ld linker: https://sourceware.org/binutils
.. _GNU Binutils - gold linker: https://sourceware.org/binutils
.. _Microsoft Visual Studio: https://visualstudio.microsoft.com
.. _mold\: A Modern Linker: https://github.com/rui314/mold
.. _sold: https://github.com/bluewhalesystems/sold
