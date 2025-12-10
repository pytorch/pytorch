CMAKE_SYSTEM_NAME
-----------------

The name of the operating system for which CMake is to build.
See the :variable:`CMAKE_SYSTEM_VERSION` variable for the OS version.

Note that ``CMAKE_SYSTEM_NAME`` is not set to anything by default when running
in script mode, since it's not building anything.

System Name for Host Builds
^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CMAKE_SYSTEM_NAME`` is by default set to the same value as the
:variable:`CMAKE_HOST_SYSTEM_NAME` variable so that the build
targets the host system.

System Name for Cross Compiling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``CMAKE_SYSTEM_NAME`` may be set explicitly when first configuring a new build
tree in order to enable :ref:`cross compiling <Cross Compiling Toolchain>`.
In this case the :variable:`CMAKE_SYSTEM_VERSION` variable must also be
set explicitly.

System Names Known to CMake
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is a list of possible values, each associated with corresponding
operating systems or environments.

========================= ======================================================
Value                     Name
========================= ======================================================
``ADSP``                  Analog Devices Audio Digital Signal Processing
``AIX``                   IBM Unix operating system
``Android``               Android operating system
``ARTOS``                 Operating system for microcontrollers
``BeOS``                  Operating system for personal computers (discontinued)
``BlueGeneL``             Blue Gene/L static environment
``BlueGeneP-dynamic``     Blue Gene/P dynamic environment
``BlueGeneP-static``      Blue Gene/P static environment
``BlueGeneQ-dynamic``     Blue Gene/Q dynamic environment
``BlueGeneQ-static``      Blue Gene/Q static environment
``BSDOS``                 BSD operating system (discontinued)
``Catamount``             Operating system for Cray XT series
``CrayLinuxEnvironment``  Cray Linux Environment
``CYGWIN``                Cygwin environment for Windows
``Darwin``                Apple stationary operating systems (macOS, OS X, etc.)
``DOS``                   MS-DOS or compatible
``DragonFly``             BSD-derived operating system
``eCos``                  Real-time embedded operating system
``Emscripten``            Compiler toolchain to WebAssembly
``Euros``                 Real-time operating system for embedded devices
``FreeBSD``               FreeBSD operating system
``Fuchsia``               Operating system by Google based on the Zircon kernel
``Generic-ADSP``          Generic ADSP (Audio DSP) environment
``Generic-ELF``           Generic ELF (Executable and Linkable Format) environment
``Generic``               Some platforms, e.g. bare metal embedded devices
``GHS-MULTI``             Green Hills Software MULTI environment
``GNU``                   GNU/Hurd-based operating system
``Haiku``                 Unix operating system inspired by BeOS
``HP-UX``                 Hewlett Packard Unix
``iOS``                   Apple mobile phone operating system
``Linux``                 All Linux-based distributions
``Midipix``               POSIX-compatible layer for Windows
``MirBSD``                MirOS BSD operating system
``MP-RAS``                MP-RAS UNIX operating system
``MSYS``                  MSYS environment (MSYSTEM=MSYS)
``NetBSD``                NetBSD operating systems
``OpenBSD``               OpenBSD operating systems
``OpenVMS``               OpenVMS operating system by HP
``OS2``                   OS/2 operating system
``OSF1``                  Compaq Tru64 UNIX (formerly DEC OSF/1, Digital Unix) (discontinued)
``QNX``                   Unix-like operating system by BlackBerry
``RISCos``                RISC OS operating system
``SCO_SV``                SCO OpenServer 5
``SerenityOS``            Unix-like operating system
``SINIX``                 SINIX operating system
``SunOS``                 Oracle Solaris and all illumos operating systems
``syllable``              Syllable operating system
``Tru64``                 Compaq Tru64 UNIX (formerly DEC OSF/1) operating system
``tvOS``                  Apple TV operating system
``ULTRIX``                Unix operating system (discontinued)
``UNIX_SV``               SCO UnixWare (pre release 7)
``UnixWare``              SCO UnixWare 7
``visionOS``              Apple mixed reality operating system
``WASI``                  WebAssembly System Interface
``watchOS``               Apple watch operating system
``Windows``               Windows stationary operating systems
``WindowsCE``             Windows Embedded Compact
``WindowsPhone``          Windows mobile phone operating system
``WindowsStore``          Universal Windows Platform applications
``Xenix``                 SCO Xenix Unix operating system (discontinued)
========================= ======================================================

Platform-specific notes:

* MSYS2's ``msys/cmake`` package (``/usr/bin/cmake``) works only under
  ``MSYSTEM=MSYS`` environments, with system name ``MSYS``.  Under other
  environments like ``MSYSTEM=MINGW64``, use another package such
  as ``mingw64/mingw-w64-x86_64-cmake`` (``/mingw64/bin/cmake``),
  which targets ``MSYSTEM=MINGW64`` with system name ``Windows``.

* Cygwin's ``cmake`` package (``/usr/bin/cmake``) uses system name ``CYGWIN``.
  A non-cygwin CMake on Windows (e.g. ``$PROGRAMFILES/CMake/bin/cmake``)
  uses system name ``Windows`` even when it runs under a Cygwin environment.

Removed Platforms
^^^^^^^^^^^^^^^^^

The following platforms were once supported by CMake and got removed either due
to platform's EOL, or other incompatibilities:

========================= ===================================== ================
Value                     Name                                  Removed in CMake
========================= ===================================== ================
``kFreeBSD``              FreeBSD kernel with a GNU userland    4.1
========================= ===================================== ================
