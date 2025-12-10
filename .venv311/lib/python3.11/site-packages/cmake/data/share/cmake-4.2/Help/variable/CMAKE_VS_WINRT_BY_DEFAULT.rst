CMAKE_VS_WINRT_BY_DEFAULT
-------------------------

.. versionadded:: 3.13

Inform :ref:`Visual Studio Generators` for VS 2010 and above that the
target platform enables WinRT compilation by default and it needs to
be explicitly disabled if ``/ZW`` or :prop_tgt:`VS_WINRT_COMPONENT` is
omitted (as opposed to enabling it when either of those options is
present)

This makes cmake configuration consistent in terms of WinRT among
platforms - if you did not enable the WinRT compilation explicitly, it
will be disabled (by either not enabling it or explicitly disabling it)

Note: WinRT compilation is always explicitly disabled for C language
source files, even if it is expliclty enabled for a project

This variable is meant to be set by a
:variable:`toolchain file <CMAKE_TOOLCHAIN_FILE>` for such platforms.
