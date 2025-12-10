
Supported languages are ``C``, ``CXX`` (i.e.  C++), ``CSharp`` (i.e.  C#), ``CUDA``,
``OBJC`` (i.e. Objective-C), ``OBJCXX`` (i.e. Objective-C++), ``Fortran``, ``HIP``,
``ISPC``, ``Swift``, ``ASM``, ``ASM_NASM``, ``ASM_MARMASM``, ``ASM_MASM``, and ``ASM-ATT``.

  .. versionadded:: 3.8
    Added ``CSharp`` and ``CUDA`` support.

  .. versionadded:: 3.15
    Added ``Swift`` support.

  .. versionadded:: 3.16
    Added ``OBJC`` and ``OBJCXX`` support.

  .. versionadded:: 3.18
    Added ``ISPC`` support.

  .. versionadded:: 3.21
    Added ``HIP`` support.

  .. versionadded:: 3.26
    Added ``ASM_MARMASM`` support.

If enabling ``ASM``, list it last so that CMake can check whether
compilers for other languages like ``C`` work for assembly too.
