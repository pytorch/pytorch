.. versionadded:: 3.8
  For each toolset that comes with this version of Visual Studio, there are
  variants that are themselves compiled for 32-bit (``x86``) and
  64-bit (``x64``) hosts (independent of the architecture they target).
  |VS_TOOLSET_HOST_ARCH_DEFAULT|
  One may explicitly request use of either the 32-bit or 64-bit host tools
  by adding either ``host=x86`` or ``host=x64`` to the toolset specification.
  See the :variable:`CMAKE_GENERATOR_TOOLSET` variable for details.

.. versionadded:: 3.14
  Added support for ``host=x86`` option.
