CMAKE_<LANG>_COMPILER_ARCHITECTURE_ID
-------------------------------------

.. versionadded:: 3.10

:ref:`List <CMake Language Lists>` of identifiers indicating the
target architecture(s) of the compiler for language ``<LANG>``.

Typically the list has one entry unless :variable:`CMAKE_OSX_ARCHITECTURES`
lists multiple architectures.

Possible values for each platform are documented in the following sections.

.. Sync with:
     Modules/CMakeCompilerABI.h
     Modules/CMakeFortranCompilerABI.F
     Modules/CMakeFortranCompilerABI.F90
     Modules/Internal/CMakeParseCompilerArchitectureId.cmake

Apple Platforms
^^^^^^^^^^^^^^^

.. versionadded:: 4.1

These identifiers are used when the :variable:`CMAKE_<LANG>_COMPILER`
targets an Apple platform (``__APPLE__`` is defined).

``arm64``
  ARM 64-bit

``arm64e``
  ARM 64-bit with Pointer Authentication (PACs)

``arm64_32``
  ARM 64-bit with 32-bit pointers (watchOS)

``armv5``, ``armv6``, ``armv7``, ``armv7k``, ``armv7s``
  ARM 32-bit

``i386``, ``i486``, ``i586``, ``i686``
  Intel 32-bit

``ppc``
  PowerPC 32-bit

``x86_64``
  Intel 64-bit

UNIX Platforms
^^^^^^^^^^^^^^

.. versionadded:: 4.1

These identifiers are used when the :variable:`CMAKE_<LANG>_COMPILER`
targets a UNIX platform.

``aarch64``
  ARM 64-bit

``alpha``
  DEC Alpha

``armv5``, ``armv6``, ``armv7``
  ARM 32-bit

``i386``, ``i486``, ``i586``, ``i686``
  Intel 32-bit

``ia64``
  Itanium 64-bit

``loongarch32``
  LoongArch 32-bit

``loongarch64``
  LoongArch 64-bit

``m68k``
  Motorola 68000

``mips``
  MIPS 32-bit big-endian

``mipsel``
  MIPS 32-bit little-endian

``mips64``
  MIPS 64-bit big-endian

``mips64el``
  MIPS 64-bit little-endian

``parisc``
  PA-RISC 32-bit

``parisc64``
  PA-RISC 64-bit

``ppc``
  PowerPC 32-bit big-endian

``ppcle``
  PowerPC 32-bit little-endian

``ppc64``
  PowerPC 64-bit big-endian

``ppc64le``
  PowerPC 64-bit little-endian

``riscv32``
  RISC-V 32-bit

``riscv64``
  RISC-V 64-bit

``s390``, ``s390x``
  IBM Z

``sparc``
  SPARC 32-bit

``sparcv9``
  SPARC 64-bit

``sw_64``
  Sunway

``x86_64``
  Intel 64-bit

Windows Platforms with GNU ABI (MinGW)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 4.1

These identifiers are used when the :variable:`CMAKE_<LANG>_COMPILER`
targets Windows with a GNU ABI (``_WIN32`` and ``__MINGW32__`` are defined).

``aarch64``
  ARM 64-bit

``armv7``
  ARM 32-bit

``i386``, ``i486``, ``i586``, ``i686``
  Intel 32-bit

``x86_64``
  Intel 64-bit

Windows Platforms with MSVC ABI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.10

These identifiers are used when the :variable:`CMAKE_<LANG>_COMPILER`
targets Windows with a MSVC ABI (``_WIN32`` and ``_MSC_VER`` are defined).

``ARM64``
  ARM 64-bit

``ARM64EC``
  ARM 64-bit Emulation-Compatible

``ARMV4I``, ``ARMV5I``, ``ARMV7``
  ARM 32-bit

``IA64``
  Itanium 64-bit

``MIPS``
  MIPS

``SHx``, ``SH3``, ``SH3DSP``, ``SH4``, ``SH5``
  SuperH

``x64``
  Intel 64-bit

``X86``
  Intel 32-bit

Windows Platforms with Watcom ABI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.10

These identifiers are used when :variable:`CMAKE_<LANG>_COMPILER_ID` is
``OpenWatcom`` or ``Watcom``.

``I86``
  Intel 16-bit

``X86``
  Intel 32-bit

Green Hills MULTI Platforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.14

These identifiers are used when :variable:`CMAKE_<LANG>_COMPILER_ID` is
``GHS``.

``ARM``
  ARM 32-bit

``PPC``
  PowerPC 32-bit

``PPC64``
  PowerPC 64-bit

``x64``
  Intel 64-bit

``X86``
  Intel 32-bit

IAR Platforms
^^^^^^^^^^^^^

.. versionadded:: 3.10

These identifiers are used when :variable:`CMAKE_<LANG>_COMPILER_ID` is
``IAR``.

``8051``
  Intel 8051-compatible 8-bit

``ARM``
  ARM 32-/64-bit

``AVR``
  Microchip AVR 8-bit

``MSP430``
  Texas Instruments MSP430 16-bit

``RH850``
  Renesas Electronics RH850 32-bit

``RISCV``
  RISC-V 32-/64-bit

``RL78``
  Renesas Electronics RL78 16-bit

``RX``
  Renesas Electronics RX 32-bit

``STM8``
  STMicroelectronics STM8 8-bit

``V850``
  Renesas Electronics V850 32-bit

Renesas Compiler Platforms
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 4.1

These identifiers are used when :variable:`CMAKE_<LANG>_COMPILER_ID` is
``Renesas``.

``RH850``
  Renesas Electronics RH850

``RL78``
  Renesas Electronics RL78

``RX``
  Renesas Electronics RX

TASKING Platforms
^^^^^^^^^^^^^^^^^

.. versionadded:: 3.25

These identifiers are used when :variable:`CMAKE_<LANG>_COMPILER_ID` is
``Tasking``.

``8051``
  ..

``ARC``
  ..

``ARM``
  ..

``MCS``
  ..

``PCP``
  ..

``TriCore``
  ..

Texas Instruments Platforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.19

These identifiers are used when :variable:`CMAKE_<LANG>_COMPILER_ID` is
``TI``.

``ARM``
  ..

``Blackfin``
  ..

``MSP430``
  ..

``SHARC``
  ..

``TMS320C28x``
  ..

``TMS320C6x``
  ..
