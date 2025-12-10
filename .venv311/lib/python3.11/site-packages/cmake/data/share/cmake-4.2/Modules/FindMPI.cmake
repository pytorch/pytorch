# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindMPI
-------

Finds a Message Passing Interface (MPI) implementation:

.. code-block:: cmake

  find_package(MPI [<version>] [COMPONENTS <langs>...] [...])

The Message Passing Interface (MPI) is a library used to write
high-performance distributed-memory parallel applications, and is
typically deployed on a cluster.  MPI is a standard interface (defined
by the MPI forum) for which many implementations are available.

.. versionadded:: 3.10
  Major overhaul of the module: many new variables, per-language components,
  and support for a wider variety of runtimes.

Components
^^^^^^^^^^

This module supports optional components that can be specified with the
:command:`find_package` command to control which MPI languages to search
for:

.. code-block:: cmake

  find_package(MPI [COMPONENTS <langs>...])

Supported components include:

``C``
  .. versionadded:: 3.10

  Finds MPI C API.

``CXX``
  .. versionadded:: 3.10

  Finds the MPI C API that is usable from C++.

``MPICXX``
  .. versionadded:: 3.10

  Finds the MPI-2 C++ API that was removed in MPI-3.

``Fortran``
  .. versionadded:: 3.10

  Finds the MPI Fortran API.

If no components are specified, module searches for the ``C``, ``CXX``, and
``Fortran`` components automatically, depending on which languages are
enabled in the project.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``MPI::MPI_<lang>``
  .. versionadded:: 3.9

  Target encapsulating usage requirements for using MPI from language
  ``<lang>``, available if MPI is found.  The ``<lang>`` is a specified
  component name as listed above.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``MPI_FOUND``
  Boolean indicating that MPI settings for all requested components
  (languages) were found.  If no components are specified, this variable is
  set to boolean true if MPI settings for all enabled languages were
  detected.  Note that the ``MPICXX`` component does not affect this
  variable.

``MPI_VERSION``
  Minimal version of MPI detected among the requested languages, or all
  enabled languages if no components were specified.

This module will set the following variables per language in CMake project,
where ``<lang>`` is one of C, CXX, or Fortran:

``MPI_<lang>_FOUND``
  Boolean indicating the MPI settings for ``<lang>`` were found and that
  simple MPI test programs compile with the provided settings.

``MPI_<lang>_COMPILER``
  MPI compiler for ``<lang>`` if such a program exists.

``MPI_<lang>_COMPILE_OPTIONS``
  Compilation options for MPI programs in ``<lang>``, given as a
  :ref:`semicolon-separated list <CMake Language Lists>`.

``MPI_<lang>_COMPILE_DEFINITIONS``
  Compilation definitions for MPI programs in ``<lang>``, given as a
  :ref:`semicolon-separated list <CMake Language Lists>`.

``MPI_<lang>_INCLUDE_DIRS``
  Include path(s) for MPI header.

``MPI_<lang>_LINK_FLAGS``
  Linker flags for MPI programs.

``MPI_<lang>_LIBRARIES``
  All libraries to link MPI programs against.

The following variables indicating which bindings are present will be
defined:

``MPI_MPICXX_FOUND``
  Boolean indicating whether the MPI-2 C++ bindings are present (introduced
  in MPI-2, removed with MPI-3).

``MPI_Fortran_HAVE_F77_HEADER``
  True if the Fortran 77 header ``<mpif.h>`` is available.

``MPI_Fortran_HAVE_F90_MODULE``
  True if the Fortran 90 module ``mpi`` can be used for accessing MPI (MPI-2
  and higher only).

``MPI_Fortran_HAVE_F08_MODULE``
  True if the Fortran 2008 ``mpi_f08`` is available to MPI programs (MPI-3
  and higher only).

If possible, the MPI version will be determined by this module.  The
facilities to detect the MPI version were introduced with MPI-1.2, and
therefore cannot be found for older MPI versions.

``MPI_<lang>_VERSION``
  MPI version implemented for ``<lang>`` by the MPI distribution.

``MPI_<lang>_VERSION_MAJOR``
  Major version of MPI implemented for ``<lang>`` by the MPI distribution.

``MPI_<lang>_VERSION_MINOR``
  Minor version of MPI implemented for ``<lang>`` by the MPI distribution.

Note that there's no variable for the C bindings being accessible through
``<mpi.h>``, since the MPI standards always have required this binding to
work in both C and C++ code.

For running MPI programs, the module sets the following variables:

``MPIEXEC_EXECUTABLE``
  Executable for running MPI programs, if such exists.

``MPIEXEC_NUMPROC_FLAG``
  Flag to pass to ``mpiexec`` before giving it the number of processors to
  run on.

``MPIEXEC_MAX_NUMPROCS``
  Number of MPI processors to utilize.  Defaults to the number of
  processors detected on the host system.

``MPIEXEC_PREFLAGS``
  Flags to pass to ``mpiexec`` directly before the executable to run.

``MPIEXEC_POSTFLAGS``
  Flags to pass to ``mpiexec`` after other flags.

Variables for Locating MPI
^^^^^^^^^^^^^^^^^^^^^^^^^^

This module performs a four-step search for an MPI implementation:

1. Searches for ``MPIEXEC_EXECUTABLE`` and, if found, uses its base
   directory.
2. Checks if the compiler has MPI support built-in.  This is the case if
   the user passed a compiler wrapper as :variable:`CMAKE_<LANG>_COMPILER`
   or if they use Cray system compiler wrappers.
3. Attempts to find an MPI compiler wrapper and determines the compiler
   information from it.
4. Tries to find an MPI implementation that does not ship such a wrapper by
   guessing settings.  Currently, only Microsoft MPI and MPICH2 on Windows
   are supported.

For controlling the ``MPIEXEC_EXECUTABLE`` step, the following variables
may be set:

``MPIEXEC_EXECUTABLE``
  Manually specify the location of ``mpiexec``.

``MPI_HOME``
  Specify the base directory of the MPI installation.

``ENV{MPI_HOME}``
  Environment variable to specify the base directory of the MPI installation.

``ENV{I_MPI_ROOT}``
  Environment variable to specify the base directory of the MPI installation.

For controlling the compiler wrapper step, the following variables may be
set:

``MPI_<lang>_COMPILER``
  Search for the specified compiler wrapper and use it.

``MPI_<lang>_COMPILER_FLAGS``
  Flags to pass to the MPI compiler wrapper during interrogation.  Some
  compiler wrappers support linking debug or tracing libraries if a specific
  flag is passed and this variable may be used to obtain them.

``MPI_COMPILER_FLAGS``
  Used to initialize ``MPI_<lang>_COMPILER_FLAGS`` if no language specific
  flag has been given.  Empty by default.

``MPI_EXECUTABLE_SUFFIX``
  A suffix which is appended to all names that are being looked for.  For
  instance, it may be set to ``.mpich`` or ``.openmpi`` to prefer the one
  or the other on Debian and its derivatives.

In order to control the guessing step, the following variable may be set:

``MPI_GUESS_LIBRARY_NAME``
  Valid values are ``MSMPI`` and ``MPICH2``.  If set, only the given library
  will be searched for.  By default, ``MSMPI`` will be preferred over
  ``MPICH2`` if both are available.  This also sets
  ``MPI_SKIP_COMPILER_WRAPPER`` variable to ``true``, which may be
  overridden.

Each of the search steps may be skipped with the following control variables:

``MPI_ASSUME_NO_BUILTIN_MPI``
  If true, the module assumes that the compiler itself does not provide an
  MPI implementation and skips to step 2.

``MPI_SKIP_COMPILER_WRAPPER``
  If true, no compiler wrapper will be searched for.

``MPI_SKIP_GUESSING``
  If true, the guessing step will be skipped.

Additionally, the following control variable is available to change search
behavior:

``MPI_CXX_SKIP_MPICXX``
  Add some definitions that will disable the MPI-2 C++ bindings.
  Currently supported are MPICH, Open MPI, Platform MPI and derivatives
  thereof, for example, MVAPICH or Intel MPI.

If the find procedure fails for the module's internal variable
``MPI_<lang>_WORKS``, then the settings detected by or passed to the module
did not work and even a simple MPI test program failed to compile.

If all of these parameters were not sufficient to find the right MPI
implementation, a user may disable the entire autodetection process by
specifying both a list of libraries in ``MPI_<lang>_LIBRARIES`` and a list
of include directories in ``MPI_<lang>_ADDITIONAL_INCLUDE_DIRS``.  Any other
variable may be set in addition to these two.  The module will then validate
the MPI settings and store the settings in the cache.

Cache Variables
^^^^^^^^^^^^^^^

The variable ``MPI_<lang>_INCLUDE_DIRS`` will be assembled from the
following variables.

For C and CXX:

``MPI_<lang>_HEADER_DIR``
  Location of the ``<mpi.h>`` header on disk.

For Fortran:

``MPI_Fortran_F77_HEADER_DIR``
  Location of the Fortran 77 header ``<mpif.h>``, if it exists.

``MPI_Fortran_MODULE_DIR``
  Location of the ``mpi`` or ``mpi_f08`` modules, if available.

For all languages the following variables are additionally considered:

``MPI_<lang>_ADDITIONAL_INCLUDE_DIRS``
  A :ref:`semicolon-separated list <CMake Language Lists>` of paths needed
  in addition to the normal include directories.

``MPI_<include-name>_INCLUDE_DIR``
  Path variables for include folders referred to by ``<include-name>``.

``MPI_<lang>_ADDITIONAL_INCLUDE_VARS``
  A :ref:`semicolon-separated list <CMake Language Lists>` of
  ``<include-name>`` that will be added to the include locations of
  ``<lang>``.

The variable ``MPI_<lang>_LIBRARIES`` will be assembled from the following
variables:

``MPI_<lib-name>_LIBRARY``
  The location of a library called ``<lib-name>`` for use with MPI.

``MPI_<lang>_LIB_NAMES``
  A :ref:`semicolon-separated list <CMake Language Lists>` of ``<lib-name>``
  that will be added to the include locations of ``<lang>``.

Advanced Variables for Using MPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module can perform some advanced feature detections upon explicit
request.

.. note::

  The following checks cannot be performed without *executing* an MPI test
  program.  Consider the special considerations for the behavior of
  :command:`try_run` during cross compilation.  Moreover, running an MPI
  program can cause additional issues, like a firewall notification on some
  systems.  These detections should be only enabled if information is
  absolutely needed.

If the following variables are set to true, the respective search will be
performed:

``MPI_DETERMINE_Fortran_CAPABILITIES``
  Determine for all available Fortran bindings what the values of
  ``MPI_SUBARRAYS_SUPPORTED`` and ``MPI_ASYNC_PROTECTS_NONBLOCKING`` are
  and make their values available as ``MPI_Fortran_<binding>_SUBARRAYS``
  and ``MPI_Fortran_<binding>_ASYNCPROT``, where ``<binding>`` is one of
  ``F77_HEADER``, ``F90_MODULE`` and ``F08_MODULE``.

``MPI_DETERMINE_LIBRARY_VERSION``
  For each language, find the output of ``MPI_Get_library_version`` and
  make it available as ``MPI_<lang>_LIBRARY_VERSION_STRING``.  This
  information is usually tied to the runtime component of an MPI
  implementation and might differ depending on ``<lang>``.
  Note that the return value is entirely implementation defined.  This
  information might be used to identify the MPI vendor and for example pick
  the correct one of multiple third party binaries that matches the MPI
  vendor.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``MPI_COMPILER``
  .. deprecated:: 2.8.5
    Use the ``MPI_<lang>_COMPILER`` instead.

``MPI_LIBRARY``
  .. deprecated:: 2.8.5
    Use the ``MPI_<lang>_LIBRARIES`` instead.

``MPI_EXTRA_LIBRARY``
  .. deprecated:: 2.8.5
    Use the ``MPI_<lang>_LIBRARIES`` instead.

``MPI_COMPILE_FLAGS``
  .. deprecated:: 2.8.5
    Use ``MPI_<lang>_COMPILE_OPTIONS`` and ``MPI_<lang>_COMPILE_DEFINITIONS``
    instead.

``MPI_INCLUDE_PATH``
  .. deprecated:: 2.8.5
    Use the ``MPI_<lang>_INCLUDE_DIRS`` instead.

``MPI_LINK_FLAGS``
  .. deprecated:: 2.8.5
    Use the ``MPI_<lang>_LINK_FLAGS`` instead.

``MPI_LIBRARIES``
  .. deprecated:: 2.8.5
    Use the ``MPI_<lang>_LIBRARIES`` instead.

``MPI_<lang>_COMPILE_FLAGS``
  .. deprecated:: 3.10
    Use the ``MPI_<lang>_COMPILE_OPTIONS`` and
    ``MPI_<lang>_COMPILE_DEFINITIONS`` instead.

``MPI_<lang>_INCLUDE_PATH``
  .. deprecated:: 3.10
    For consumption use ``MPI_<lang>_INCLUDE_DIRS`` and for specifying
    folders use ``MPI_<lang>_ADDITIONAL_INCLUDE_DIRS`` instead.

``MPIEXEC``
  .. deprecated:: 3.10
    Use ``MPIEXEC_EXECUTABLE`` instead.

Examples
^^^^^^^^

Example: Basic Usage
""""""""""""""""""""

Finding MPI and linking imported target to project target:

.. code-block:: cmake

  find_package(MPI)
  target_link_libraries(example PRIVATE MPI::MPI_C)

Example: Usage of mpiexec
"""""""""""""""""""""""""

When using ``MPIEXEC_EXECUTABLE`` to execute MPI applications, typically
all of the ``MPIEXEC_EXECUTABLE`` flags should be used as follows.

In the following example, the command is executed in a process.
``<executable>`` should be replaced with the MPI program, and ``<args>``
with the arguments to pass to the MPI program.

.. code-block:: cmake

  find_package(MPI)

  if(MPI_FOUND)
    execute_process(
      COMMAND
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG}
        ${MPIEXEC_MAX_NUMPROCS}
        ${MPIEXEC_PREFLAGS}
        <executable>
        ${MPIEXEC_POSTFLAGS}
        <args>
    )
  endif()
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

include(FindPackageHandleStandardArgs)
find_package(PkgConfig QUIET)

# Generic compiler names
set(_MPI_C_GENERIC_COMPILER_NAMES          mpicc    mpcc      mpicc_r mpcc_r)
set(_MPI_CXX_GENERIC_COMPILER_NAMES        mpicxx   mpiCC     mpcxx   mpCC    mpic++   mpc++
                                           mpicxx_r mpiCC_r   mpcxx_r mpCC_r  mpic++_r mpc++_r)
set(_MPI_Fortran_GENERIC_COMPILER_NAMES    mpif95   mpif95_r  mpf95   mpf95_r
                                           mpif90   mpif90_r  mpf90   mpf90_r
                                           mpif77   mpif77_r  mpf77   mpf77_r
                                           mpifc)

#Fujitsu cross/own compiler names
set(_MPI_Fujitsu_C_COMPILER_NAMES        mpifccpx mpifcc)
set(_MPI_Fujitsu_CXX_COMPILER_NAMES      mpiFCCpx mpiFCC)
set(_MPI_Fujitsu_Fortran_COMPILER_NAMES  mpifrtpx mpifrt)
set(_MPI_FujitsuClang_C_COMPILER_NAMES            mpifccpx mpifcc)
set(_MPI_FujitsuClang_CXX_COMPILER_NAMES          mpiFCCpx mpiFCC)
set(_MPI_FujitsuClang_Fortran_COMPILER_NAMES      mpifrtpx mpifrt)

# GNU compiler names
set(_MPI_GNU_C_COMPILER_NAMES              mpigcc mpgcc mpigcc_r mpgcc_r)
set(_MPI_GNU_CXX_COMPILER_NAMES            mpig++ mpg++ mpig++_r mpg++_r mpigxx)
set(_MPI_GNU_Fortran_COMPILER_NAMES        mpigfortran mpgfortran mpigfortran_r mpgfortran_r
                                           mpig77 mpig77_r mpg77 mpg77_r)

# Intel MPI compiler names on Windows
if(WIN32)
  list(APPEND _MPI_C_GENERIC_COMPILER_NAMES       mpicc.bat)
  list(APPEND _MPI_CXX_GENERIC_COMPILER_NAMES     mpicxx.bat)
  list(APPEND _MPI_Fortran_GENERIC_COMPILER_NAMES mpifc.bat)

  # Intel MPI compiler names
  set(_MPI_Intel_C_COMPILER_NAMES            mpiicc.bat)
  set(_MPI_Intel_CXX_COMPILER_NAMES          mpiicpc.bat)
  set(_MPI_Intel_Fortran_COMPILER_NAMES      mpiifort.bat mpif77.bat mpif90.bat)

  # Intel MPI compiler names
  set(_MPI_IntelLLVM_C_COMPILER_NAMES            mpiicx.bat mpiicc.bat)
  set(_MPI_IntelLLVM_CXX_COMPILER_NAMES          mpiicx.bat mpiicpc.bat) # Not GNU-like mpiicpx.bat
  set(_MPI_IntelLLVM_Fortran_COMPILER_NAMES      mpiifx.bat mpiifort.bat mpif77.bat mpif90.bat)

  # Intel MPI compiler names for MSMPI
  set(_MPI_MSVC_C_COMPILER_NAMES             mpicl.bat)
  set(_MPI_MSVC_CXX_COMPILER_NAMES           mpicl.bat)
else()
  # Intel compiler names
  set(_MPI_Intel_C_COMPILER_NAMES            mpiicc)
  set(_MPI_Intel_CXX_COMPILER_NAMES          mpiicpc  mpiicxx mpiic++)
  set(_MPI_Intel_Fortran_COMPILER_NAMES      mpiifort mpiif95 mpiif90 mpiif77)

  # Intel compiler names
  set(_MPI_IntelLLVM_C_COMPILER_NAMES            mpiicx mpiicc)
  set(_MPI_IntelLLVM_CXX_COMPILER_NAMES          mpiicpx mpiicpc mpiicxx mpiic++)
  set(_MPI_IntelLLVM_Fortran_COMPILER_NAMES      mpiifx mpiifort mpiif95 mpiif90 mpiif77)
endif()

# PGI compiler names
set(_MPI_PGI_C_COMPILER_NAMES              mpipgicc mpipgcc mppgcc)
set(_MPI_PGI_CXX_COMPILER_NAMES            mpipgic++ mpipgCC mppgCC)
set(_MPI_PGI_Fortran_COMPILER_NAMES        mpipgifort mpipgf95 mpipgf90 mppgf95 mppgf90 mpipgf77 mppgf77)

# XLC MPI Compiler names
set(_MPI_XL_C_COMPILER_NAMES               mpxlc      mpxlc_r    mpixlc     mpixlc_r)
set(_MPI_XL_CXX_COMPILER_NAMES             mpixlcxx   mpixlC     mpixlc++   mpxlcxx   mpxlc++   mpixlc++   mpxlCC
                                           mpixlcxx_r mpixlC_r   mpixlc++_r mpxlcxx_r mpxlc++_r mpixlc++_r mpxlCC_r)
set(_MPI_XL_Fortran_COMPILER_NAMES         mpixlf95   mpixlf95_r mpxlf95 mpxlf95_r
                                           mpixlf90   mpixlf90_r mpxlf90 mpxlf90_r
                                           mpixlf77   mpixlf77_r mpxlf77 mpxlf77_r
                                           mpixlf     mpixlf_r   mpxlf   mpxlf_r)

# Cray Compiler names
set(_MPI_Cray_C_COMPILER_NAMES             cc)
set(_MPI_Cray_CXX_COMPILER_NAMES           CC)
set(_MPI_Cray_Fortran_COMPILER_NAMES       ftn)

# Prepend vendor-specific compiler wrappers to the list. If we don't know the compiler,
# attempt all of them.
# By attempting vendor-specific compiler names first, we should avoid situations where the compiler wrapper
# stems from a proprietary MPI and won't know which compiler it's being used for. For instance, Intel MPI
# controls its settings via the I_MPI_CC environment variables if the generic name is being used.
# If we know which compiler we're working with, we can use the most specialized wrapper there is in order to
# pick up the right settings for it.
foreach (LANG IN ITEMS C CXX Fortran)
  set(_MPI_${LANG}_COMPILER_NAMES "")
  foreach (id IN ITEMS Fujitsu FujitsuClang GNU Intel IntelLLVM MSVC PGI XL)
    if (NOT CMAKE_${LANG}_COMPILER_ID OR CMAKE_${LANG}_COMPILER_ID STREQUAL id)
      foreach(_COMPILER_NAME IN LISTS _MPI_${id}_${LANG}_COMPILER_NAMES)
        list(APPEND _MPI_${LANG}_COMPILER_NAMES ${_COMPILER_NAME}${MPI_EXECUTABLE_SUFFIX})
      endforeach()
    endif()
    unset(_MPI_${id}_${LANG}_COMPILER_NAMES)
  endforeach()
  foreach(_COMPILER_NAME IN LISTS _MPI_${LANG}_GENERIC_COMPILER_NAMES)
    list(APPEND _MPI_${LANG}_COMPILER_NAMES ${_COMPILER_NAME}${MPI_EXECUTABLE_SUFFIX})
  endforeach()
  unset(_MPI_${LANG}_GENERIC_COMPILER_NAMES)
endforeach()

# Names to try for mpiexec
# Only mpiexec commands are guaranteed to behave as described in the standard,
# mpirun commands are not covered by the standard in any way whatsoever.
# lamexec is the executable for LAM/MPI, srun is for SLURM or Open MPI with SLURM support.
# srun -n X <executable> is however a valid command, so it behaves 'like' mpiexec.
set(_MPIEXEC_NAMES_BASE                   mpiexec mpiexec.hydra mpiexec.mpd mpirun lamexec srun)

unset(_MPIEXEC_NAMES)
foreach(_MPIEXEC_NAME IN LISTS _MPIEXEC_NAMES_BASE)
  list(APPEND _MPIEXEC_NAMES "${_MPIEXEC_NAME}${MPI_EXECUTABLE_SUFFIX}")
endforeach()
unset(_MPIEXEC_NAMES_BASE)

function (_MPI_check_compiler LANG QUERY_FLAG OUTPUT_VARIABLE RESULT_VARIABLE)
  if(DEFINED MPI_${LANG}_COMPILER_FLAGS)
    separate_arguments(_MPI_COMPILER_WRAPPER_OPTIONS NATIVE_COMMAND "${MPI_${LANG}_COMPILER_FLAGS}")
  else()
    separate_arguments(_MPI_COMPILER_WRAPPER_OPTIONS NATIVE_COMMAND "${MPI_COMPILER_FLAGS}")
  endif()
  execute_process(
    COMMAND ${MPI_${LANG}_COMPILER} ${_MPI_COMPILER_WRAPPER_OPTIONS} ${QUERY_FLAG}
    OUTPUT_VARIABLE  WRAPPER_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_VARIABLE   WRAPPER_OUTPUT ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE  WRAPPER_RETURN)
  # Some compiler wrappers will yield spurious zero return values, for example
  # Intel MPI tolerates unknown arguments and if the MPI wrappers loads a shared
  # library that has invalid or missing version information there would be warning
  # messages emitted by ld.so in the compiler output. In either case, we'll treat
  # the output as invalid.
  if(WRAPPER_OUTPUT MATCHES "undefined reference|unrecognized|need to set|no version information available|command not found")
    set(WRAPPER_RETURN 255)
  endif()
  # Ensure that no error output might be passed upwards.
  if(NOT WRAPPER_RETURN EQUAL "0")
    unset(WRAPPER_OUTPUT)
  else()
    # Strip leading whitespace
    string(REGEX REPLACE "^ +" "" WRAPPER_OUTPUT "${WRAPPER_OUTPUT}")
  endif()
  set(${OUTPUT_VARIABLE} "${WRAPPER_OUTPUT}" PARENT_SCOPE)
  set(${RESULT_VARIABLE} "${WRAPPER_RETURN}" PARENT_SCOPE)
endfunction()

macro(_MPI_env_set_ifnot VAR VALUE)
  if(NOT DEFINED ENV{${VAR}})
    set(_MPI_${VAR}_WAS_SET FALSE)
    set(ENV{${VAR}} ${${VALUE}})
  else()
    set(_MPI_${VAR}_WAS_SET TRUE)
  endif()
endmacro()

macro(_MPI_env_unset_ifnot VAR)
  if(NOT _MPI_${VAR}_WAS_SET)
    unset(ENV{${VAR}})
  endif()
endmacro()

function (_MPI_interrogate_compiler LANG)
  unset(MPI_COMPILE_CMDLINE)
  unset(MPI_LINK_CMDLINE)

  unset(MPI_COMPILE_OPTIONS_WORK)
  unset(MPI_COMPILE_DEFINITIONS_WORK)
  unset(MPI_INCLUDE_DIRS_WORK)
  unset(MPI_LINK_FLAGS_WORK)
  unset(MPI_LIB_NAMES_WORK)
  unset(MPI_LIB_FULLPATHS_WORK)

  # Define the MPICH and Intel MPI compiler variables to the compilers set in CMake.
  # It's possible to have a per-compiler configuration in these MPI implementations and
  # a particular MPICH derivate might check compiler interoperability.
  # Intel MPI in particular does this with I_MPI_CHECK_COMPILER.
  file(TO_NATIVE_PATH "${CMAKE_${LANG}_COMPILER}" _MPI_UNDERLYING_COMPILER)
  # On Windows, the Intel MPI batch scripts can only work with filenames - Full paths will break them.
  # Due to the lack of other MPICH-based wrappers for Visual C++, we may treat this as default.
  if(MSVC)
    get_filename_component(_MPI_UNDERLYING_COMPILER "${_MPI_UNDERLYING_COMPILER}" NAME)
  endif()
  if(LANG STREQUAL "C")
    _MPI_env_set_ifnot(I_MPI_CC _MPI_UNDERLYING_COMPILER)
    _MPI_env_set_ifnot(MPICH_CC _MPI_UNDERLYING_COMPILER)
  elseif(LANG STREQUAL "CXX")
    _MPI_env_set_ifnot(I_MPI_CXX _MPI_UNDERLYING_COMPILER)
    _MPI_env_set_ifnot(MPICH_CXX _MPI_UNDERLYING_COMPILER)
  elseif(LANG STREQUAL "Fortran")
    _MPI_env_set_ifnot(I_MPI_FC _MPI_UNDERLYING_COMPILER)
    _MPI_env_set_ifnot(MPICH_FC _MPI_UNDERLYING_COMPILER)
    _MPI_env_set_ifnot(I_MPI_F77 _MPI_UNDERLYING_COMPILER)
    _MPI_env_set_ifnot(MPICH_F77 _MPI_UNDERLYING_COMPILER)
    _MPI_env_set_ifnot(I_MPI_F90 _MPI_UNDERLYING_COMPILER)
    _MPI_env_set_ifnot(MPICH_F90 _MPI_UNDERLYING_COMPILER)
  endif()

  # Set these two variables for Intel MPI:
  #   - I_MPI_DEBUG_INFO_STRIP: It adds 'objcopy' lines to the compiler output. We support stripping them
  #     (see below), but if we can avoid them in the first place, we should.
  #   - I_MPI_FORT_BIND: By default Intel MPI makes the C/C++ compiler wrappers link Fortran bindings.
  #     This is so that mixed-language code doesn't require additional libraries when linking with mpicc.
  #     For our purposes, this makes little sense, since correct MPI usage from CMake already circumvenes this.
  set(_MPI_ENV_VALUE "disable")
  _MPI_env_set_ifnot(I_MPI_DEBUG_INFO_STRIP _MPI_ENV_VALUE)
  _MPI_env_set_ifnot(I_MPI_FORT_BIND _MPI_ENV_VALUE)

  # Check whether the -showme:compile option works. This indicates that we have either Open MPI
  # or a newer version of LAM/MPI, and implies that -showme:link will also work.
  # Open MPI also supports -show, but separates linker and compiler information
  _MPI_check_compiler(${LANG} "-showme:compile" MPI_COMPILE_CMDLINE MPI_COMPILER_RETURN)
  if (MPI_COMPILER_RETURN EQUAL "0")
    _MPI_check_compiler(${LANG} "-showme:link" MPI_LINK_CMDLINE MPI_COMPILER_RETURN)

    if (NOT MPI_COMPILER_RETURN EQUAL "0")
      unset(MPI_COMPILE_CMDLINE)
    endif()
  endif()

  # MPICH and MVAPICH offer -compile-info and -link-info.
  # For modern versions, both do the same as -show. However, for old versions, they do differ
  # when called for mpicxx and mpif90 and it's necessary to use them over -show in order to find the
  # removed MPI C++ bindings.
  if (NOT MPI_COMPILER_RETURN EQUAL "0")
    _MPI_check_compiler(${LANG} "-compile-info" MPI_COMPILE_CMDLINE MPI_COMPILER_RETURN)

    if (MPI_COMPILER_RETURN EQUAL "0")
      _MPI_check_compiler(${LANG} "-link-info" MPI_LINK_CMDLINE MPI_COMPILER_RETURN)

      if (NOT MPI_COMPILER_RETURN EQUAL "0")
        unset(MPI_COMPILE_CMDLINE)
      endif()
    endif()
  endif()

  # Cray compiler wrappers come usually without a separate mpicc/c++/ftn, but offer
  # --cray-print-opts=...
  if (NOT MPI_COMPILER_RETURN EQUAL "0")
    _MPI_check_compiler(${LANG} "--cray-print-opts=cflags"
                        MPI_COMPILE_CMDLINE MPI_COMPILER_RETURN)

    if (MPI_COMPILER_RETURN EQUAL "0")
      # Pass --no-as-needed so the mpi library is always linked. Otherwise, the
      # Cray compiler wrapper puts an --as-needed flag around the mpi library,
      # and it is not linked unless code directly refers to it.
      _MPI_check_compiler(${LANG} "--no-as-needed;--cray-print-opts=libs"
                          MPI_LINK_CMDLINE MPI_COMPILER_RETURN)

      if (NOT MPI_COMPILER_RETURN EQUAL "0")
        unset(MPI_COMPILE_CMDLINE)
        unset(MPI_LINK_CMDLINE)
      endif()
    endif()
  endif()

  # MPICH, MVAPICH2 and Intel MPI just use "-show". Open MPI also offers this, but the
  # -showme commands are more specialized.
  if (NOT MPI_COMPILER_RETURN EQUAL "0")
    _MPI_check_compiler(${LANG} "-show" MPI_COMPILE_CMDLINE MPI_COMPILER_RETURN)
  endif()

  # Older versions of LAM/MPI have "-showme". Open MPI also supports this.
  # Unknown to MPICH, MVAPICH and Intel MPI.
  if (NOT MPI_COMPILER_RETURN EQUAL "0")
    _MPI_check_compiler(${LANG} "-showme" MPI_COMPILE_CMDLINE MPI_COMPILER_RETURN)
  endif()

  if (MPI_COMPILER_RETURN EQUAL "0" AND DEFINED MPI_COMPILE_CMDLINE)
    # Intel MPI can be run with -compchk or I_MPI_CHECK_COMPILER set to 1.
    # In this case, -show will be prepended with a line to the compiler checker. This is a script that performs
    # compatibility checks and returns a non-zero exit code together with an error if something fails.
    # It has to be called as "compchk.sh <arch> <compiler>". Here, <arch> is one out of 32 (i686), 64 (ia64) or 32e (x86_64).
    # The compiler is identified by filename, and can be either the MPI compiler or the underlying compiler.
    # NOTE: It is vital to run this script while the environment variables are set up, otherwise it can check the wrong compiler.
    if(MPI_COMPILE_CMDLINE MATCHES "^([^\"\n ]+/compchk.sh|\"[^\"]+/compchk.sh\") +([^ ]+)")
      # Now CMAKE_MATCH_1 contains the path to the compchk.sh file and CMAKE_MATCH_2 the architecture flag.
      unset(COMPILER_CHECKER_OUTPUT)
      execute_process(
      COMMAND ${CMAKE_MATCH_1} ${CMAKE_MATCH_2} ${MPI_${LANG}_COMPILER}
      OUTPUT_VARIABLE  COMPILER_CHECKER_OUTPUT OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_VARIABLE   COMPILER_CHECKER_OUTPUT ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE  MPI_COMPILER_RETURN)
      # If it returned a non-zero value, the check below will fail and cause the interrogation to be aborted.
      if(NOT MPI_COMPILER_RETURN EQUAL "0")
        if(NOT MPI_FIND_QUIETLY)
          message(STATUS "Intel MPI compiler check failed: ${COMPILER_CHECKER_OUTPUT}")
        endif()
      else()
        # Since the check passed, we can remove the compchk.sh script.
        string(REGEX REPLACE "^([^\"\n ]+|\"[^\"]+\")/compchk.sh.*\n" "" MPI_COMPILE_CMDLINE "${MPI_COMPILE_CMDLINE}")
      endif()
    endif()
  endif()

  # Revert changes to the environment made previously
  if(LANG STREQUAL "C")
    _MPI_env_unset_ifnot(I_MPI_CC)
    _MPI_env_unset_ifnot(MPICH_CC)
  elseif(LANG STREQUAL "CXX")
    _MPI_env_unset_ifnot(I_MPI_CXX)
    _MPI_env_unset_ifnot(MPICH_CXX)
  elseif(LANG STREQUAL "Fortran")
    _MPI_env_unset_ifnot(I_MPI_FC)
    _MPI_env_unset_ifnot(MPICH_FC)
    _MPI_env_unset_ifnot(I_MPI_F77)
    _MPI_env_unset_ifnot(MPICH_F77)
    _MPI_env_unset_ifnot(I_MPI_F90)
    _MPI_env_unset_ifnot(MPICH_F90)
  endif()

  _MPI_env_unset_ifnot(I_MPI_DEBUG_INFO_STRIP)
  _MPI_env_unset_ifnot(I_MPI_FORT_BIND)

  if (NOT (MPI_COMPILER_RETURN EQUAL "0") OR NOT (DEFINED MPI_COMPILE_CMDLINE))
    # Cannot interrogate this compiler, so exit.
    set(MPI_${LANG}_WRAPPER_FOUND FALSE PARENT_SCOPE)
    return()
  endif()
  unset(MPI_COMPILER_RETURN)

  # We have our command lines, but we might need to copy MPI_COMPILE_CMDLINE
  # into MPI_LINK_CMDLINE, if we didn't find the link line.
  if (NOT DEFINED MPI_LINK_CMDLINE)
    set(MPI_LINK_CMDLINE "${MPI_COMPILE_CMDLINE}")
  endif()

  # Visual Studio parsers permit each flag prefixed by either / or -.
  # We'll normalize this to the - syntax we use for CMake purposes anyways.
  if(MSVC)
    foreach(_MPI_VARIABLE IN ITEMS COMPILE LINK)
      # The Intel MPI wrappers on Windows prefix their output with some copyright boilerplate.
      # To prevent possible problems, we discard this text before proceeding with any further matching.
      string(REGEX REPLACE "^[^ ]+ for the Intel\\(R\\) MPI Library [^\n]+ for Windows\\*\nCopyright\\(C\\) [^\n]+, Intel Corporation\\. All rights reserved\\.\n\n" ""
        MPI_${_MPI_VARIABLE}_CMDLINE "${MPI_${_MPI_VARIABLE}_CMDLINE}")
      string(REGEX REPLACE "(^| )/" "\\1-" MPI_${_MPI_VARIABLE}_CMDLINE "${MPI_${_MPI_VARIABLE}_CMDLINE}")
      string(REPLACE "-libpath:" "-LIBPATH:" MPI_${_MPI_VARIABLE}_CMDLINE "${MPI_${_MPI_VARIABLE}_CMDLINE}")
    endforeach()
  endif()

  # For MSVC and cl-compatible compilers, the keyword /link indicates a point after which
  # everything following is passed to the linker. In this case, we drop all prior information
  # from the link line and treat any unknown extra flags as linker flags.
  set(_MPI_FILTERED_LINK_INFORMATION FALSE)
  if(MSVC)
    if(MPI_LINK_CMDLINE MATCHES " -(link|LINK) ")
      string(REGEX REPLACE ".+-(link|LINK) +" "" MPI_LINK_CMDLINE "${MPI_LINK_CMDLINE}")
      set(_MPI_FILTERED_LINK_INFORMATION TRUE)
    endif()
    string(REGEX REPLACE " +-(link|LINK) .+" "" MPI_COMPILE_CMDLINE "${MPI_COMPILE_CMDLINE}")
  endif()

  if(UNIX)
    # At this point, we obtained some output from a compiler wrapper that works.
    # We'll now try to parse it into variables with meaning to us.
    if(LANG STREQUAL "Fortran")
      # noqa: spellcheck off
      # If MPICH (and derivates) didn't recognize the Fortran compiler include flag during configuration,
      # they'll return a set of three commands, consisting out of a symlink command for mpif.h,
      # the actual compiler command and deletion of the created symlink.
      # Especially with M(VA)PICH-1, this appears to happen erroneously, and therefore we should translate
      # this output into an additional include directory and then drop it from the output.
      # noqa: spellcheck on
      if(MPI_COMPILE_CMDLINE MATCHES "^ln -s ([^\"\n ]+|\"[^\"]+\") mpif.h")
        get_filename_component(MPI_INCLUDE_DIRS_WORK "${CMAKE_MATCH_1}" DIRECTORY)
        string(REGEX REPLACE "^ln -s ([^\"\n ]+|\"[^\"]+\") mpif.h\n" "" MPI_COMPILE_CMDLINE "${MPI_COMPILE_CMDLINE}")
        string(REGEX REPLACE "^ln -s ([^\"\n ]+|\"[^\"]+\") mpif.h\n" "" MPI_LINK_CMDLINE "${MPI_LINK_CMDLINE}")
        string(REGEX REPLACE "\nrm -f mpif.h$" "" MPI_COMPILE_CMDLINE "${MPI_COMPILE_CMDLINE}")
        string(REGEX REPLACE "\nrm -f mpif.h$" "" MPI_LINK_CMDLINE "${MPI_LINK_CMDLINE}")
      endif()
    endif()

    # If Intel MPI was configured for static linkage with -static_mpi, the wrapper will by default strip
    # debug information from resulting binaries (see I_MPI_DEBUG_INFO_STRIP).
    # Since we cannot process this information into CMake logic, we need to discard the resulting objcopy
    # commands from the output.
    string(REGEX REPLACE "(^|\n)objcopy[^\n]+(\n|$)" "" MPI_COMPILE_CMDLINE "${MPI_COMPILE_CMDLINE}")
    string(REGEX REPLACE "(^|\n)objcopy[^\n]+(\n|$)" "" MPI_LINK_CMDLINE "${MPI_LINK_CMDLINE}")
  endif()

  # For Visual C++, extracting compiler options in a generic fashion isn't easy. However, no MPI implementation
  # on Windows seems to require any specific ones, either.
  if(NOT MSVC)
    # Extract compile options from the compile command line.
    string(REGEX MATCHALL "(^| )-f([^\"\n ]+|\"[^\"]+\")" MPI_ALL_COMPILE_OPTIONS "${MPI_COMPILE_CMDLINE}")

    foreach(_MPI_COMPILE_OPTION IN LISTS MPI_ALL_COMPILE_OPTIONS)
      string(REGEX REPLACE "^ " "" _MPI_COMPILE_OPTION "${_MPI_COMPILE_OPTION}")

      # Ignore -fstack-protector directives: These occur on MPICH and MVAPICH when the libraries
      # themselves were built with this flag. However, this flag is unrelated to using MPI, and
      # we won't match the accompanying --param-ssp-size and -Wp,-D_FORTIFY_SOURCE flags and therefore
      # produce inconsistent results with the regularly flags.
      # Similarly, aliasing flags do not belong into our flag array.
      # Also strip out `-framework` flags.
      if(NOT _MPI_COMPILE_OPTION MATCHES "^-f((no-|)(stack-protector|strict-aliasing)|PI[CE]|pi[ce]|ramework)")
        list(APPEND MPI_COMPILE_OPTIONS_WORK "${_MPI_COMPILE_OPTION}")
      endif()
    endforeach()
  endif()

  # For GNU-style compilers, it's possible to prefix includes and definitions with certain flags to pass them
  # only to the preprocessor. For CMake purposes, we need to treat, but ignore such scopings.
  # Note that we do not support spaces between the arguments, i.e. -Wp,-I -Wp,/opt/mympi will not be parsed
  # correctly. This form does not seem to occur in any common MPI implementation, however.
  if(NOT MSVC)
    set(_MPI_PREPROCESSOR_FLAG_REGEX "(-Wp,|-Xpreprocessor )?")
  else()
    set(_MPI_PREPROCESSOR_FLAG_REGEX "")
  endif()

  # Same deal as above, for the definitions.
  string(REGEX MATCHALL "(^| )${_MPI_PREPROCESSOR_FLAG_REGEX}-D *([^\"\n ]+|\"[^\"]+\")" MPI_ALL_COMPILE_DEFINITIONS "${MPI_COMPILE_CMDLINE}")

  foreach(_MPI_COMPILE_DEFINITION IN LISTS MPI_ALL_COMPILE_DEFINITIONS)
    string(REGEX REPLACE "^ ?${_MPI_PREPROCESSOR_FLAG_REGEX}-D *" "" _MPI_COMPILE_DEFINITION "${_MPI_COMPILE_DEFINITION}")
    string(REPLACE "\"" "" _MPI_COMPILE_DEFINITION "${_MPI_COMPILE_DEFINITION}")
    if(NOT _MPI_COMPILE_DEFINITION MATCHES "^_FORTIFY_SOURCE.*")
      list(APPEND MPI_COMPILE_DEFINITIONS_WORK "${_MPI_COMPILE_DEFINITION}")
    endif()
  endforeach()

  # Extract include paths from compile command line
  string(REGEX MATCHALL "(^|\n| )${_MPI_PREPROCESSOR_FLAG_REGEX}${CMAKE_INCLUDE_FLAG_${LANG}} *([^\"\n ]+|\"[^\"]+\")"
    MPI_ALL_INCLUDE_PATHS "${MPI_COMPILE_CMDLINE}")

  # If extracting failed to work, we'll try using -showme:incdirs.
  # Unlike before, we do this without the environment variables set up, but since only MPICH derivatives are affected by any of them, and
  # -showme:... is only supported by Open MPI and LAM/MPI, this isn't a concern.
  if (NOT MPI_ALL_INCLUDE_PATHS)
    _MPI_check_compiler(${LANG} "-showme:incdirs" MPI_INCDIRS_CMDLINE MPI_INCDIRS_COMPILER_RETURN)
    if(MPI_INCDIRS_COMPILER_RETURN)
      separate_arguments(MPI_ALL_INCLUDE_PATHS NATIVE_COMMAND "${MPI_INCDIRS_CMDLINE}")
    endif()
  endif()

  foreach(_MPI_INCLUDE_PATH IN LISTS MPI_ALL_INCLUDE_PATHS)
    string(REGEX REPLACE "^ ?${_MPI_PREPROCESSOR_FLAG_REGEX}${CMAKE_INCLUDE_FLAG_${LANG}} *" "" _MPI_INCLUDE_PATH "${_MPI_INCLUDE_PATH}")
    string(REPLACE "\n" "" _MPI_INCLUDE_PATH "${_MPI_INCLUDE_PATH}")
    string(REPLACE "\"" "" _MPI_INCLUDE_PATH "${_MPI_INCLUDE_PATH}")
    string(REPLACE "'" "" _MPI_INCLUDE_PATH "${_MPI_INCLUDE_PATH}")
    get_filename_component(_MPI_INCLUDE_PATH "${_MPI_INCLUDE_PATH}" REALPATH)
    list(APPEND MPI_INCLUDE_DIRS_WORK "${_MPI_INCLUDE_PATH}")
  endforeach()

  # The next step are linker flags and library directories. Here, we first take the flags given in raw -L or -LIBPATH: syntax.
  string(REGEX MATCHALL "(^| )${CMAKE_LIBRARY_PATH_FLAG} *([^\"\n ]+|\"[^\"]+\")" MPI_DIRECT_LINK_PATHS "${MPI_LINK_CMDLINE}")
  foreach(_MPI_LPATH IN LISTS MPI_DIRECT_LINK_PATHS)
    string(REGEX REPLACE "(^| )${CMAKE_LIBRARY_PATH_FLAG} *" "" _MPI_LPATH "${_MPI_LPATH}")
    list(APPEND MPI_ALL_LINK_PATHS "${_MPI_LPATH}")
  endforeach()

  # If the link commandline hasn't been filtered (e.g. when using MSVC and /link), we need to extract the relevant parts first.
  if(NOT _MPI_FILTERED_LINK_INFORMATION)
    string(REGEX MATCHALL "(^| )(-Wl,|-Xlinker +)([^\"\n ]+|\"[^\"]+\")" MPI_LINK_FLAGS "${MPI_LINK_CMDLINE}")

    # In this case, we could also find some indirectly given linker paths, e.g. prefixed by -Xlinker or -Wl,
    # Since syntaxes like -Wl,-L -Wl,/my/path/to/lib are also valid, we parse these paths by first removing -Wl, and -Xlinker
    # from the list of filtered flags and then parse the remainder of the output.
    string(REGEX REPLACE "(-Wl,|-Xlinker +)" "" MPI_LINK_FLAGS_RAW "${MPI_LINK_FLAGS}")

    # Now we can parse the leftover output. Note that spaces can now be handled since the above example would reduce to
    # -L /my/path/to/lib and can be extracted correctly.
    string(REGEX MATCHALL "^(${CMAKE_LIBRARY_PATH_FLAG},? *|--library-path=)([^\"\n ]+|\"[^\"]+\")"
      MPI_INDIRECT_LINK_PATHS "${MPI_LINK_FLAGS_RAW}")

    foreach(_MPI_LPATH IN LISTS MPI_INDIRECT_LINK_PATHS)
      string(REGEX REPLACE "^(${CMAKE_LIBRARY_PATH_FLAG},? *|--library-path=)" "" _MPI_LPATH "${_MPI_LPATH}")
      list(APPEND MPI_ALL_LINK_PATHS "${_MPI_LPATH}")
    endforeach()

    # We need to remove the flags we extracted from the linker flag list now.
    string(REGEX REPLACE "(^| )(-Wl,|-Xlinker +)(${CMAKE_LIBRARY_PATH_FLAG},? *(-Wl,|-Xlinker +)?|--library-path=)([^\"\n ]+|\"[^\"]+\")" ""
      MPI_LINK_CMDLINE_FILTERED "${MPI_LINK_CMDLINE}")

    # Some MPI implementations pass on options they themselves were built with. Since -z,noexecstack is a common
    # hardening, we should strip it. In general, the -z options should be undesirable.
    string(REGEX REPLACE "(^| )-Wl,-z(,[^ ]+| +-Wl,[^ ]+)" "" MPI_LINK_CMDLINE_FILTERED "${MPI_LINK_CMDLINE_FILTERED}")
    string(REGEX REPLACE "(^| )-Xlinker +-z +-Xlinker +[^ ]+" "" MPI_LINK_CMDLINE_FILTERED "${MPI_LINK_CMDLINE_FILTERED}")

    # We only consider options of the form -Wl or -Xlinker:
    string(REGEX MATCHALL "(^| )(-Wl,|-Xlinker +)([^\"\n ]+|\"[^\"]+\")" MPI_ALL_LINK_FLAGS "${MPI_LINK_CMDLINE_FILTERED}")

    # As a next step, we assemble the linker flags extracted in a preliminary flags string
    foreach(_MPI_LINK_FLAG IN LISTS MPI_ALL_LINK_FLAGS)
      string(STRIP "${_MPI_LINK_FLAG}" _MPI_LINK_FLAG)
      if (MPI_LINK_FLAGS_WORK)
        string(APPEND MPI_LINK_FLAGS_WORK " ${_MPI_LINK_FLAG}")
      else()
        set(MPI_LINK_FLAGS_WORK "${_MPI_LINK_FLAG}")
      endif()
    endforeach()
  else()
    # In the filtered case, we obtain the link time flags by just stripping the library paths.
    string(REGEX REPLACE "(^| )${CMAKE_LIBRARY_PATH_FLAG} *([^\"\n ]+|\"[^\"]+\")" "" MPI_LINK_CMDLINE_FILTERED "${MPI_LINK_CMDLINE}")
  endif()

  # If we failed to extract any linker paths, we'll try using the -showme:libdirs option with the MPI compiler.
  # This will return a list of folders, not a set of flags!
  if (NOT MPI_ALL_LINK_PATHS)
    _MPI_check_compiler(${LANG} "-showme:libdirs" MPI_LIBDIRS_CMDLINE MPI_LIBDIRS_COMPILER_RETURN)
    if(MPI_LIBDIRS_COMPILER_RETURN)
      separate_arguments(MPI_ALL_LINK_PATHS NATIVE_COMMAND "${MPI_LIBDIRS_CMDLINE}")
    endif()
  endif()

  # We need to remove potential quotes and convert the paths to CMake syntax while resolving them, too.
  foreach(_MPI_LPATH IN LISTS MPI_ALL_LINK_PATHS)
    string(REPLACE "\"" "" _MPI_LPATH "${_MPI_LPATH}")
    get_filename_component(_MPI_LPATH "${_MPI_LPATH}" REALPATH)
    list(APPEND MPI_LINK_DIRECTORIES_WORK "${_MPI_LPATH}")
  endforeach()

  # Extract the set of libraries to link against from the link command line
  # This only makes sense if CMAKE_LINK_LIBRARY_FLAG is defined, i.e. a -lxxxx syntax is supported by the compiler.
  if(CMAKE_LINK_LIBRARY_FLAG)
    string(REGEX MATCHALL "(^| )${CMAKE_LINK_LIBRARY_FLAG}([^\"\n ]+|\"[^\"]+\")"
      MPI_LIBNAMES "${MPI_LINK_CMDLINE}")

    foreach(_MPI_LIB_NAME IN LISTS MPI_LIBNAMES)
      # also match flags starting with "-l:" here
      string(REGEX REPLACE "^ ?${CMAKE_LINK_LIBRARY_FLAG}(:lib|:)?" "" _MPI_LIB_NAME "${_MPI_LIB_NAME}")
      string(REPLACE "\"" "" _MPI_LIB_NAME "${_MPI_LIB_NAME}")
      list(APPEND MPI_LIB_NAMES_WORK "${_MPI_LIB_NAME}")
    endforeach()
  endif()

  # Treat linker objects given by full path, for example static libraries, import libraries
  # or shared libraries if there aren't any import libraries in use on the system.
  # Note that we do not consider CMAKE_<TYPE>_LIBRARY_PREFIX intentionally here: The linker will for a given file
  # decide how to link it based on file type, not based on a prefix like 'lib'.
  set(_MPI_LIB_SUFFIX_REGEX "${CMAKE_STATIC_LIBRARY_SUFFIX}")
  if(DEFINED CMAKE_IMPORT_LIBRARY_SUFFIX)
    if(NOT (CMAKE_IMPORT_LIBRARY_SUFFIX STREQUAL CMAKE_STATIC_LIBRARY_SUFFIX))
      string(APPEND _MPI_LIB_SUFFIX_REGEX "|${CMAKE_IMPORT_LIBRARY_SUFFIX}")
    endif()
  else()
    string(APPEND _MPI_LIB_SUFFIX_REGEX "|${CMAKE_SHARED_LIBRARY_SUFFIX}")
  endif()
  set(_MPI_LIB_NAME_REGEX "(([^\"\n ]+(${_MPI_LIB_SUFFIX_REGEX}))|(\"[^\"]+(${_MPI_LIB_SUFFIX_REGEX})\"))( +|$)")
  string(REPLACE "." "\\." _MPI_LIB_NAME_REGEX "${_MPI_LIB_NAME_REGEX}")

  string(REGEX MATCHALL "${_MPI_LIB_NAME_REGEX}" MPI_LIBNAMES "${MPI_LINK_CMDLINE}")
  foreach(_MPI_LIB_NAME IN LISTS MPI_LIBNAMES)
    # Do not match "-l:" flags
    string(REGEX MATCH "^ ?${CMAKE_LINK_LIBRARY_FLAG}:" _MPI_LIB_NAME_TEST "${_MPI_LIB_NAME}")
    if(_MPI_LIB_NAME_TEST STREQUAL "")
      string(REGEX REPLACE "^ +\"?|\"? +$" "" _MPI_LIB_NAME "${_MPI_LIB_NAME}")
      get_filename_component(_MPI_LIB_PATH "${_MPI_LIB_NAME}" DIRECTORY)
      if(NOT _MPI_LIB_PATH STREQUAL "")
        list(APPEND MPI_LIB_FULLPATHS_WORK "${_MPI_LIB_NAME}")
      else()
        list(APPEND MPI_LIB_NAMES_WORK "${_MPI_LIB_NAME}")
      endif()
    endif()
  endforeach()

  # Save the explicitly given link directories
  set(MPI_LINK_DIRECTORIES_LEFTOVER "${MPI_LINK_DIRECTORIES_WORK}")

  # An MPI compiler wrapper could have its MPI libraries in the implicitly
  # linked directories of the compiler itself.
  if(DEFINED CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES)
    list(APPEND MPI_LINK_DIRECTORIES_WORK "${CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES}")
  endif()

  # Determine full path names for all of the libraries that one needs
  # to link against in an MPI program
  unset(MPI_PLAIN_LIB_NAMES_WORK)
  foreach(_MPI_LIB_NAME IN LISTS MPI_LIB_NAMES_WORK)
    get_filename_component(_MPI_PLAIN_LIB_NAME "${_MPI_LIB_NAME}" NAME_WE)
    list(APPEND MPI_PLAIN_LIB_NAMES_WORK "${_MPI_PLAIN_LIB_NAME}")
    find_library(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY
      NAMES "${_MPI_LIB_NAME}" "lib${_MPI_LIB_NAME}"
      HINTS ${MPI_LINK_DIRECTORIES_WORK}
      DOC "Location of the ${_MPI_PLAIN_LIB_NAME} library for MPI"
    )
    mark_as_advanced(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY)
    # Remove the directory from the remainder list.
    if(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY)
      get_filename_component(_MPI_TAKEN_DIRECTORY "${MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY}" DIRECTORY)
      list(REMOVE_ITEM MPI_LINK_DIRECTORIES_LEFTOVER "${_MPI_TAKEN_DIRECTORY}")
    endif()
  endforeach()

  # Add the link directories given explicitly that we haven't used back as linker directories.
  if(NOT WIN32)
    foreach(_MPI_LINK_DIRECTORY IN LISTS MPI_LINK_DIRECTORIES_LEFTOVER)
      file(TO_NATIVE_PATH "${_MPI_LINK_DIRECTORY}" _MPI_LINK_DIRECTORY_ACTUAL)
      string(FIND "${_MPI_LINK_DIRECTORY_ACTUAL}" " " _MPI_LINK_DIRECTORY_CONTAINS_SPACE)
      if(NOT _MPI_LINK_DIRECTORY_CONTAINS_SPACE EQUAL "-1")
        set(_MPI_LINK_DIRECTORY_ACTUAL "\"${_MPI_LINK_DIRECTORY_ACTUAL}\"")
      endif()
      if(MPI_LINK_FLAGS_WORK)
        string(APPEND MPI_LINK_FLAGS_WORK " ${CMAKE_LIBRARY_PATH_FLAG}${_MPI_LINK_DIRECTORY_ACTUAL}")
      else()
        set(MPI_LINK_FLAGS_WORK "${CMAKE_LIBRARY_PATH_FLAG}${_MPI_LINK_DIRECTORY_ACTUAL}")
      endif()
    endforeach()
  endif()

  # Deal with the libraries given with full path next
  unset(MPI_DIRECT_LIB_NAMES_WORK)
  foreach(_MPI_LIB_FULLPATH IN LISTS MPI_LIB_FULLPATHS_WORK)
    get_filename_component(_MPI_PLAIN_LIB_NAME "${_MPI_LIB_FULLPATH}" NAME_WE)
    list(APPEND MPI_DIRECT_LIB_NAMES_WORK "${_MPI_PLAIN_LIB_NAME}")
    set(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY "${_MPI_LIB_FULLPATH}" CACHE FILEPATH "Location of the ${_MPI_PLAIN_LIB_NAME} library for MPI")
    mark_as_advanced(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY)
  endforeach()
  # Directly linked objects should be linked first in case some generic linker flags are needed for them.
  if(MPI_DIRECT_LIB_NAMES_WORK)
    set(MPI_PLAIN_LIB_NAMES_WORK "${MPI_DIRECT_LIB_NAMES_WORK};${MPI_PLAIN_LIB_NAMES_WORK}")
  endif()

  # MPI might require pthread to work. The above mechanism wouldn't detect it, but we need to
  # link it in that case. -lpthread is covered by the normal library treatment on the other hand.
  if(MPI_COMPILE_CMDLINE MATCHES "-pthread")
    list(APPEND MPI_COMPILE_OPTIONS_WORK "-pthread")
    if(MPI_LINK_FLAGS_WORK)
      string(APPEND MPI_LINK_FLAGS_WORK " -pthread")
    else()
      set(MPI_LINK_FLAGS_WORK "-pthread")
    endif()
  endif()

  if(MPI_${LANG}_EXTRA_COMPILE_DEFINITIONS)
    list(APPEND MPI_COMPILE_DEFINITIONS_WORK "${MPI_${LANG}_EXTRA_COMPILE_DEFINITIONS}")
  endif()
  if(MPI_${LANG}_EXTRA_COMPILE_OPTIONS)
    list(APPEND MPI_COMPILE_OPTIONS_WORK "${MPI_${LANG}_EXTRA_COMPILE_OPTIONS}")
  endif()
  if(MPI_${LANG}_EXTRA_LIB_NAMES)
    list(APPEND MPI_PLAIN_LIB_NAMES_WORK "${MPI_${LANG}_EXTRA_LIB_NAMES}")
  endif()

  # If we found MPI, set up all of the appropriate cache entries
  if(NOT MPI_${LANG}_COMPILE_OPTIONS)
    set(MPI_${LANG}_COMPILE_OPTIONS          ${MPI_COMPILE_OPTIONS_WORK}     CACHE STRING "MPI ${LANG} compilation options"            FORCE)
  endif()
  if(NOT MPI_${LANG}_COMPILE_DEFINITIONS)
    set(MPI_${LANG}_COMPILE_DEFINITIONS      ${MPI_COMPILE_DEFINITIONS_WORK} CACHE STRING "MPI ${LANG} compilation definitions"        FORCE)
  endif()
  if(NOT MPI_${LANG}_COMPILER_INCLUDE_DIRS)
    set(MPI_${LANG}_COMPILER_INCLUDE_DIRS    ${MPI_INCLUDE_DIRS_WORK}        CACHE STRING "MPI ${LANG} compiler wrapper include directories" FORCE)
  endif()
  if(NOT MPI_${LANG}_LINK_FLAGS)
    set(MPI_${LANG}_LINK_FLAGS               ${MPI_LINK_FLAGS_WORK}          CACHE STRING "MPI ${LANG} linker flags"                   FORCE)
  endif()
  if(NOT MPI_${LANG}_LIB_NAMES)
    set(MPI_${LANG}_LIB_NAMES                ${MPI_PLAIN_LIB_NAMES_WORK}     CACHE STRING "MPI ${LANG} libraries to link against"      FORCE)
  endif()
  set(MPI_${LANG}_WRAPPER_FOUND TRUE PARENT_SCOPE)
endfunction()

function(_MPI_guess_settings LANG)
  set(MPI_GUESS_FOUND FALSE)
  # Currently only MSMPI and MPICH2 on Windows are supported, so we can skip this search if we're not targeting that.
  if(WIN32)
    # MSMPI

    # The environment variables MSMPI_INC and MSMPILIB32/64 are the only ways of locating the MSMPI_SDK,
    # which is installed separately from the runtime. Thus it's possible to have mpiexec but not MPI headers
    # or import libraries and vice versa.
    if(NOT MPI_GUESS_LIBRARY_NAME OR MPI_GUESS_LIBRARY_NAME STREQUAL "MSMPI")
      # We first attempt to locate the msmpi.lib. Should be find it, we'll assume that the MPI present is indeed
      # Microsoft MPI.
      if(CMAKE_SIZEOF_VOID_P EQUAL "8")
        file(TO_CMAKE_PATH "$ENV{MSMPI_LIB64}" MPI_MSMPI_LIB_PATH)
        file(TO_CMAKE_PATH "$ENV{MSMPI_INC}/x64" MPI_MSMPI_INC_PATH_EXTRA)
      else()
        file(TO_CMAKE_PATH "$ENV{MSMPI_LIB32}" MPI_MSMPI_LIB_PATH)
        file(TO_CMAKE_PATH "$ENV{MSMPI_INC}/x86" MPI_MSMPI_INC_PATH_EXTRA)
      endif()

      find_library(MPI_msmpi_LIBRARY
        NAMES msmpi
        HINTS ${MPI_MSMPI_LIB_PATH}
        DOC "Location of the msmpi library for Microsoft MPI")
      mark_as_advanced(MPI_msmpi_LIBRARY)

      if(MPI_msmpi_LIBRARY)
        # Next, we attempt to locate the MPI header. Note that for Fortran we know that mpif.h is a way
        # MSMPI can be used and therefore that header has to be present.
        if(NOT MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS)
          get_filename_component(MPI_MSMPI_INC_DIR "$ENV{MSMPI_INC}" REALPATH)
          set(MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS "${MPI_MSMPI_INC_DIR}" CACHE STRING "MPI ${LANG} additional include directories" FORCE)
          unset(MPI_MSMPI_INC_DIR)
        endif()

        # For MSMPI, one can compile the MPI module by building the mpi.f90 shipped with the MSMPI SDK,
        # thus it might be present or provided by the user. Figuring out which is supported is done later on.
        # The PGI Fortran compiler for instance ships a prebuilt set of modules in its own include folder.
        # Should a user be employing PGI or have built its own set and provided it via cache variables, the
        # splitting routine would have located the module files.

        # noqa: spellcheck off
        # For C and C++, we're done here (MSMPI does not ship the MPI-2 C++ bindings) - however, for Fortran
        # we need some extra library to glue Fortran support together:
        # MSMPI ships 2-4 Fortran libraries, each for different Fortran compiler behaviors. The library names
        # ending with a c are using the cdecl calling convention, whereas those ending with an s are for Fortran
        # implementations using stdcall. Therefore, the 64-bit MSMPI only ships those ending in 'c', whereas the 32-bit
        # has both variants available.
        # The second difference is the last but one letter, if it's an e(nd), the length of a string argument is
        # passed by the Fortran compiler after all other arguments on the parameter list, if it's an m(ixed),
        # it's passed immediately after the string address.
        # noqa: spellcheck on

        # To summarize:
        #   - msmpifec: CHARACTER length passed after the parameter list and using cdecl calling convention
        #   - msmpifmc: CHARACTER length passed directly after string address and using cdecl calling convention
        #   - msmpifes: CHARACTER length passed after the parameter list and using stdcall calling convention
        #   - msmpifms: CHARACTER length passed directly after string address and using stdcall calling convention
        # 32-bit MSMPI ships all four libraries, 64-bit MSMPI ships only the first two.

        # As is, Intel Fortran and PGI Fortran both use the 'ec' variant of the calling convention, whereas
        # the old Compaq Visual Fortran compiler defaulted to the 'ms' version. It's possible to make Intel Fortran
        # use the CVF calling convention using /iface:cvf, but we assume - and this is also assumed in FortranCInterface -
        # this isn't the case. It's also possible to make CVF use the 'ec' variant, using /iface=(cref,nomixed_str_len_arg).

        # Our strategy is now to locate all libraries, but enter msmpifec into the LIB_NAMES array.
        # Should this not be adequate it's a straightforward way for a user to change the LIB_NAMES array and
        # have their library found. Still, this should not be necessary outside of exceptional cases, as reasoned.
        if (LANG STREQUAL "Fortran")
          set(MPI_MSMPI_CALLINGCONVS c)
          if(CMAKE_SIZEOF_VOID_P EQUAL "4")
            list(APPEND MPI_MSMPI_CALLINGCONVS s)
          endif()
          foreach(mpistrlenpos IN ITEMS e m)
            foreach(mpicallingconv IN LISTS MPI_MSMPI_CALLINGCONVS)
              find_library(MPI_msmpif${mpistrlenpos}${mpicallingconv}_LIBRARY
                NAMES msmpif${mpistrlenpos}${mpicallingconv}
                HINTS "${MPI_MSMPI_LIB_PATH}"
                DOC "Location of the msmpi${mpistrlenpos}${mpicallingconv} library for Microsoft MPI")
              mark_as_advanced(MPI_msmpif${mpistrlenpos}${mpicallingconv}_LIBRARY)
            endforeach()
          endforeach()
          if(NOT MPI_${LANG}_LIB_NAMES)
            set(MPI_${LANG}_LIB_NAMES "msmpi;msmpifec" CACHE STRING "MPI ${LANG} libraries to link against" FORCE)
          endif()

          # At this point we're *not* done. MSMPI requires an additional include file for Fortran giving the value
          # of MPI_AINT. This file is called mpifptr.h located in the x64 and x86 subfolders, respectively.
          find_path(MPI_mpifptr_INCLUDE_DIR
            NAMES "mpifptr.h"
            HINTS "${MPI_MSMPI_INC_PATH_EXTRA}"
            DOC "Location of the mpifptr.h extra header for Microsoft MPI")
          if(NOT MPI_${LANG}_ADDITIONAL_INCLUDE_VARS)
            set(MPI_${LANG}_ADDITIONAL_INCLUDE_VARS "mpifptr" CACHE STRING "MPI ${LANG} additional include directory variables, given in the form MPI_<name>_INCLUDE_DIR." FORCE)
          endif()
          mark_as_advanced(MPI_${LANG}_ADDITIONAL_INCLUDE_VARS MPI_mpifptr_INCLUDE_DIR)
        else()
          if(NOT MPI_${LANG}_LIB_NAMES)
            set(MPI_${LANG}_LIB_NAMES "msmpi" CACHE STRING "MPI ${LANG} libraries to link against" FORCE)
          endif()
        endif()
        mark_as_advanced(MPI_${LANG}_LIB_NAMES)
        set(MPI_GUESS_FOUND TRUE)

        if(_MPIEXEC_NOT_GIVEN)
          unset(MPIEXEC_EXECUTABLE CACHE)
        endif()

        find_program(MPIEXEC_EXECUTABLE
          NAMES mpiexec
          HINTS $ENV{MSMPI_BIN} "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\MPI;InstallRoot]/Bin"
          DOC "Executable for running MPI programs.")
      endif()
    endif()

    # At this point there's not many MPIs that we could still consider.
    # OpenMPI 1.6.x and below supported Windows, but these ship compiler wrappers that still work.
    # The only other relevant MPI implementation without a wrapper is MPICH2, which had Windows support in 1.4.1p1 and older.
    if(NOT MPI_GUESS_FOUND AND (NOT MPI_GUESS_LIBRARY_NAME OR MPI_GUESS_LIBRARY_NAME STREQUAL "MPICH2"))
      set(MPI_MPICH_PREFIX_PATHS
        "$ENV{ProgramW6432}/MPICH2/lib"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MPICH\\SMPD;binary]/../lib"
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MPICH2;Path]/lib"
      )

      # All of C, C++ and Fortran will need mpi.lib, so we'll look for this first
      find_library(MPI_mpi_LIBRARY
        NAMES mpi
        HINTS ${MPI_MPICH_PREFIX_PATHS})
      mark_as_advanced(MPI_mpi_LIBRARY)
      # If we found mpi.lib, we detect the rest of MPICH2
      if(MPI_mpi_LIBRARY)
        set(MPI_MPICH_LIB_NAMES "mpi")
        # If MPI-2 C++ bindings are requested, we need to locate cxx.lib as well.
        # Otherwise, MPICH_SKIP_MPICXX will be defined and these bindings aren't needed.
        if(LANG STREQUAL "CXX" AND NOT MPI_CXX_SKIP_MPICXX)
          find_library(MPI_cxx_LIBRARY
            NAMES cxx
            HINTS ${MPI_MPICH_PREFIX_PATHS})
          mark_as_advanced(MPI_cxx_LIBRARY)
          list(APPEND MPI_MPICH_LIB_NAMES "cxx")
        # For Fortran, MPICH2 provides three different libraries:
        #   fmpich2.lib which uses uppercase symbols and cdecl,
        #   fmpich2s.lib which uses uppercase symbols and stdcall (32-bit only),
        #   fmpich2g.lib which uses lowercase symbols with double underscores and cdecl.
        # fmpich2s.lib would be useful for Compaq Visual Fortran, fmpich2g.lib has to be used with GNU g77 and is also
        # provided in the form of an .a archive for MinGW and Cygwin. From our perspective, fmpich2.lib is the only one
        # we need to try, and if it doesn't work with the given Fortran compiler we'd find out later on during validation
        elseif(LANG STREQUAL "Fortran")
          find_library(MPI_fmpich2_LIBRARY
            NAMES fmpich2
            HINTS ${MPI_MPICH_PREFIX_PATHS})
          find_library(MPI_fmpich2s_LIBRARY
            NAMES fmpich2s
            HINTS ${MPI_MPICH_PREFIX_PATHS})
          find_library(MPI_fmpich2g_LIBRARY
            NAMES fmpich2g
            HINTS ${MPI_MPICH_PREFIX_PATHS})
          mark_as_advanced(MPI_fmpich2_LIBRARY MPI_fmpich2s_LIBRARY MPI_fmpich2g_LIBRARY)
          list(APPEND MPI_MPICH_LIB_NAMES "fmpich2")
        endif()

        if(NOT MPI_${LANG}_LIB_NAMES)
          set(MPI_${LANG}_LIB_NAMES "${MPI_MPICH_LIB_NAMES}" CACHE STRING "MPI ${LANG} libraries to link against" FORCE)
        endif()
        unset(MPI_MPICH_LIB_NAMES)

        if(NOT MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS)
          # For MPICH2, the include folder would be in ../include relative to the library folder.
          get_filename_component(MPI_MPICH_ROOT_DIR "${MPI_mpi_LIBRARY}" DIRECTORY)
          get_filename_component(MPI_MPICH_ROOT_DIR "${MPI_MPICH_ROOT_DIR}" DIRECTORY)
          if(IS_DIRECTORY "${MPI_MPICH_ROOT_DIR}/include")
            set(MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS "${MPI_MPICH_ROOT_DIR}/include" CACHE STRING "MPI ${LANG} additional include directory variables, given in the form MPI_<name>_INCLUDE_DIR." FORCE)
          endif()
          unset(MPI_MPICH_ROOT_DIR)
        endif()
        set(MPI_GUESS_FOUND TRUE)

        if(_MPIEXEC_NOT_GIVEN)
          unset(MPIEXEC_EXECUTABLE CACHE)
        endif()

        find_program(MPIEXEC_EXECUTABLE
          NAMES ${_MPIEXEC_NAMES}
          HINTS "$ENV{ProgramW6432}/MPICH2/bin"
                "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MPICH\\SMPD;binary]"
                "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MPICH2;Path]/bin"
          DOC "Executable for running MPI programs.")
      endif()
      unset(MPI_MPICH_PREFIX_PATHS)
    endif()
  endif()
  set(MPI_${LANG}_GUESS_FOUND "${MPI_GUESS_FOUND}" PARENT_SCOPE)
endfunction()

function(_MPI_adjust_compile_definitions LANG)
  if(LANG STREQUAL "CXX")
    # To disable the C++ bindings, we need to pass some definitions since the mpi.h header has to deal with both C and C++
    # bindings in MPI-2.
    if(MPI_CXX_SKIP_MPICXX AND NOT MPI_${LANG}_COMPILE_DEFINITIONS MATCHES "SKIP_MPICXX")
      # MPICH_SKIP_MPICXX is being used in MPICH and derivatives like MVAPICH or Intel MPI
      # OMPI_SKIP_MPICXX is being used in Open MPI
      # _MPICC_H is being used for IBM Platform MPI
      list(APPEND MPI_${LANG}_COMPILE_DEFINITIONS "MPICH_SKIP_MPICXX" "OMPI_SKIP_MPICXX" "_MPICC_H")
      set(MPI_${LANG}_COMPILE_DEFINITIONS "${MPI_${LANG}_COMPILE_DEFINITIONS}" CACHE STRING "MPI ${LANG} compilation definitions" FORCE)
    endif()
  endif()
endfunction()

macro(_MPI_assemble_libraries LANG)
  set(MPI_${LANG}_LIBRARIES "")
  # Only for libraries do we need to check whether the compiler's linking stage is separate.
  if(NOT MPI_${LANG}_COMPILER STREQUAL CMAKE_${LANG}_COMPILER OR NOT MPI_${LANG}_WORKS_IMPLICIT)
    foreach(mpilib IN LISTS MPI_${LANG}_LIB_NAMES)
      list(APPEND MPI_${LANG}_LIBRARIES ${MPI_${mpilib}_LIBRARY})
    endforeach()
  endif()
endmacro()

macro(_MPI_assemble_include_dirs LANG)
  set(MPI_${LANG}_INCLUDE_DIRS
    ${MPI_${LANG}_COMPILER_INCLUDE_DIRS}
    ${MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS}
    )
  if(LANG MATCHES "^(C|CXX)$")
    if(MPI_${LANG}_HEADER_DIR)
      list(APPEND MPI_${LANG}_INCLUDE_DIRS "${MPI_${LANG}_HEADER_DIR}")
    endif()
  else() # Fortran
    if(MPI_${LANG}_F77_HEADER_DIR)
      list(APPEND MPI_${LANG}_INCLUDE_DIRS "${MPI_${LANG}_F77_HEADER_DIR}")
    endif()
    if(MPI_${LANG}_MODULE_DIR)
      list(APPEND MPI_${LANG}_INCLUDE_DIRS "${MPI_${LANG}_MODULE_DIR}")
    endif()
  endif()
  if(MPI_${LANG}_INCLUDE_DIRS)
    list(REMOVE_DUPLICATES MPI_${LANG}_INCLUDE_DIRS)
  endif()
endmacro()

macro(_MPI_split_include_dirs LANG)
  # Backwards compatibility: Search INCLUDE_PATH if given.
  if(MPI_${LANG}_INCLUDE_PATH)
    list(APPEND MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS "${MPI_${LANG}_INCLUDE_PATH}")
  endif()

  # We try to find the headers/modules among those paths (and system paths)
  # For C/C++, we just need to have a look for mpi.h.
  if(LANG MATCHES "^(C|CXX)$")
    find_path(MPI_${LANG}_HEADER_DIR "mpi.h"
      HINTS
        ${MPI_${LANG}_COMPILER_INCLUDE_DIRS}
        ${MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS}
    )
    mark_as_advanced(MPI_${LANG}_HEADER_DIR)
    if(MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS)
      list(REMOVE_ITEM MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS "${MPI_${LANG}_HEADER_DIR}")
    endif()

  # Fortran is more complicated here: An implementation could provide
  # any of the Fortran 77/90/2008 APIs for MPI. For example, MSMPI
  # only provides Fortran 77 and - if mpi.f90 is built - potentially
  # a Fortran 90 module.
  elseif(LANG STREQUAL "Fortran")
    find_path(MPI_${LANG}_F77_HEADER_DIR "mpif.h"
      HINTS
        ${MPI_${LANG}_COMPILER_INCLUDE_DIRS}
        ${MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS}
    )
    find_path(MPI_${LANG}_MODULE_DIR
      NAMES "mpi.mod" "mpi_f08.mod"
      HINTS
        ${MPI_${LANG}_COMPILER_INCLUDE_DIRS}
        ${MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS}
    )
    if(MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS)
      list(REMOVE_ITEM MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS
        "${MPI_${LANG}_F77_HEADER_DIR}"
        "${MPI_${LANG}_MODULE_DIR}"
      )
    endif()
    mark_as_advanced(MPI_${LANG}_F77_HEADER_DIR MPI_${LANG}_MODULE_DIR)
  endif()

  # Remove duplicates and default system directories from the list.
  if(MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS)
    list(REMOVE_DUPLICATES MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS)
    foreach(MPI_IMPLICIT_INC_DIR IN LISTS CMAKE_${LANG}_IMPLICIT_LINK_DIRECTORIES)
      list(REMOVE_ITEM MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS ${MPI_IMPLICIT_INC_DIR})
    endforeach()
  endif()

  set(MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS ${MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS} CACHE STRING "MPI ${LANG} additional include directories" FORCE)
endmacro()

macro(_MPI_create_imported_target LANG)
  if(NOT TARGET MPI::MPI_${LANG})
    add_library(MPI::MPI_${LANG} INTERFACE IMPORTED)
  endif()

  # When this is consumed for compiling CUDA, use '-Xcompiler' to wrap '-pthread' and '-fexceptions'.
  string(REPLACE "-pthread" "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler >-pthread"
    _MPI_${LANG}_COMPILE_OPTIONS "${MPI_${LANG}_COMPILE_OPTIONS}")
  string(REPLACE "-fexceptions" "$<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:SHELL:-Xcompiler >-fexceptions"
    _MPI_${LANG}_COMPILE_OPTIONS "${_MPI_${LANG}_COMPILE_OPTIONS}")
  set_property(TARGET MPI::MPI_${LANG} PROPERTY INTERFACE_COMPILE_OPTIONS "${_MPI_${LANG}_COMPILE_OPTIONS}")
  unset(_MPI_${LANG}_COMPILE_OPTIONS)

  set_property(TARGET MPI::MPI_${LANG} PROPERTY INTERFACE_COMPILE_DEFINITIONS "${MPI_${LANG}_COMPILE_DEFINITIONS}")

  if(MPI_${LANG}_LINK_FLAGS)
    string(REPLACE "," "$<COMMA>" _MPI_${LANG}_LINK_FLAGS "${MPI_${LANG}_LINK_FLAGS}")
    string(PREPEND _MPI_${LANG}_LINK_FLAGS "$<HOST_LINK:SHELL:")
    string(APPEND _MPI_${LANG}_LINK_FLAGS ">")
    set_property(TARGET MPI::MPI_${LANG} PROPERTY INTERFACE_LINK_OPTIONS "${_MPI_${LANG}_LINK_FLAGS}")
  endif()
  # If the compiler links MPI implicitly, no libraries will be found as they're contained within
  # CMAKE_<LANG>_IMPLICIT_LINK_LIBRARIES already.
  if(MPI_${LANG}_LIBRARIES)
    set_property(TARGET MPI::MPI_${LANG} PROPERTY INTERFACE_LINK_LIBRARIES "${MPI_${LANG}_LIBRARIES}")
  endif()
  # Given the new design of FindMPI, INCLUDE_DIRS will always be located, even under implicit linking.
  set_property(TARGET MPI::MPI_${LANG} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MPI_${LANG}_INCLUDE_DIRS}")
endmacro()

function(_MPI_try_staged_settings LANG MPI_TEST_FILE_NAME MODE RUN_BINARY SUPPRESS_ERRORS)
  set(WORK_DIR "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindMPI")
  set(SRC_DIR "${CMAKE_ROOT}/Modules/FindMPI")
  set(BIN_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindMPI/${MPI_TEST_FILE_NAME}_${LANG}.bin")
  unset(MPI_TEST_COMPILE_DEFINITIONS)
  if(LANG STREQUAL "Fortran")
    if(MODE STREQUAL "F90_MODULE")
      set(MPI_Fortran_INCLUDE_LINE "use mpi\n      implicit none")
    elseif(MODE STREQUAL "F08_MODULE")
      set(MPI_Fortran_INCLUDE_LINE "use mpi_f08\n      implicit none")
    else() # F77 header
      set(MPI_Fortran_INCLUDE_LINE "implicit none\n      include 'mpif.h'")
    endif()
    file(READ "${SRC_DIR}/${MPI_TEST_FILE_NAME}.f90.in" MPI_TEST_SOURCE_CONTENT)
    string(CONFIGURE "${MPI_TEST_SOURCE_CONTENT}" MPI_TEST_SOURCE_CONTENT)
    set(MPI_TEST_SOURCE_FILE "${MPI_TEST_FILE_NAME}.f90")
  elseif(LANG STREQUAL "CXX")
    file(READ "${SRC_DIR}/${MPI_TEST_FILE_NAME}.c" MPI_TEST_SOURCE_CONTENT)
    set(MPI_TEST_SOURCE_FILE "${MPI_TEST_FILE_NAME}.cpp")
    if(MODE STREQUAL "TEST_MPICXX")
      set(MPI_TEST_COMPILE_DEFINITIONS TEST_MPI_MPICXX)
    endif()
  else() # C
    file(READ "${SRC_DIR}/${MPI_TEST_FILE_NAME}.c" MPI_TEST_SOURCE_CONTENT)
    set(MPI_TEST_SOURCE_FILE "${MPI_TEST_FILE_NAME}.c")
  endif()
  if(SUPPRESS_ERRORS)
    set(maybe_no_log NO_LOG)
  else()
    set(maybe_no_log "")
  endif()
  if(RUN_BINARY)
    try_run(MPI_RUN_RESULT_${LANG}_${MPI_TEST_FILE_NAME}_${MODE} MPI_RESULT_${LANG}_${MPI_TEST_FILE_NAME}_${MODE}
      SOURCE_FROM_VAR "${MPI_TEST_SOURCE_FILE}" MPI_TEST_SOURCE_CONTENT
      ${maybe_no_log}
      LOG_DESCRIPTION "The MPI test ${MPI_TEST_FILE_NAME} for ${LANG} in mode ${MODE}"
      COMPILE_DEFINITIONS ${MPI_TEST_COMPILE_DEFINITIONS}
      LINK_LIBRARIES MPI::MPI_${LANG}
      RUN_OUTPUT_VARIABLE MPI_RUN_OUTPUT_${LANG}_${MPI_TEST_FILE_NAME}_${MODE}
      COMPILE_OUTPUT_VARIABLE _MPI_TRY_${MPI_TEST_FILE_NAME}_${MODE}_OUTPUT)
    set(MPI_RUN_OUTPUT_${LANG}_${MPI_TEST_FILE_NAME}_${MODE} "${MPI_RUN_OUTPUT_${LANG}_${MPI_TEST_FILE_NAME}_${MODE}}" PARENT_SCOPE)
  else()
    try_compile(MPI_RESULT_${LANG}_${MPI_TEST_FILE_NAME}_${MODE}
      SOURCE_FROM_VAR "${MPI_TEST_SOURCE_FILE}" MPI_TEST_SOURCE_CONTENT
      ${maybe_no_log}
      LOG_DESCRIPTION "The MPI test ${MPI_TEST_FILE_NAME} for ${LANG} in mode ${MODE}"
      COMPILE_DEFINITIONS ${MPI_TEST_COMPILE_DEFINITIONS}
      LINK_LIBRARIES MPI::MPI_${LANG}
      COPY_FILE "${BIN_FILE}"
      OUTPUT_VARIABLE _MPI_TRY_${MPI_TEST_FILE_NAME}_${MODE}_OUTPUT)
  endif()
endfunction()

macro(_MPI_check_lang_works LANG SUPPRESS_ERRORS)
  # For Fortran we may have by the MPI-3 standard an implementation that provides:
  #   - the mpi_f08 module
  #   - *both*, the mpi module and 'mpif.h'
  # Since older MPI standards (MPI-1) did not define anything but 'mpif.h', we need to check all three individually.
  if( NOT MPI_${LANG}_WORKS )
    if(LANG STREQUAL "Fortran")
      set(MPI_Fortran_INTEGER_LINE "(kind=MPI_INTEGER_KIND)")
      _MPI_try_staged_settings(${LANG} test_mpi F77_HEADER FALSE ${SUPPRESS_ERRORS})
      _MPI_try_staged_settings(${LANG} test_mpi F90_MODULE FALSE ${SUPPRESS_ERRORS})
      _MPI_try_staged_settings(${LANG} test_mpi F08_MODULE FALSE ${SUPPRESS_ERRORS})

      set(MPI_${LANG}_WORKS FALSE)

      foreach(mpimethod IN ITEMS F77_HEADER F08_MODULE F90_MODULE)
        if(MPI_RESULT_${LANG}_test_mpi_${mpimethod})
          set(MPI_${LANG}_WORKS TRUE)
          set(MPI_${LANG}_HAVE_${mpimethod} TRUE)
        else()
          set(MPI_${LANG}_HAVE_${mpimethod} FALSE)
        endif()
      endforeach()
      # MPI-1 versions had no MPI_INTEGER_KIND defined, so we need to try without it.
      # However, MPI-1 also did not define the Fortran 90 and 08 modules, so we only try the F77 header.
      unset(MPI_Fortran_INTEGER_LINE)
      if(NOT MPI_${LANG}_WORKS)
        _MPI_try_staged_settings(${LANG} test_mpi F77_HEADER_NOKIND FALSE ${SUPPRESS_ERRORS})
        if(MPI_RESULT_${LANG}_test_mpi_F77_HEADER_NOKIND)
          set(MPI_${LANG}_WORKS TRUE)
          set(MPI_${LANG}_HAVE_F77_HEADER TRUE)
        endif()
      endif()
    else()
      _MPI_try_staged_settings(${LANG} test_mpi normal FALSE ${SUPPRESS_ERRORS})
      # If 'test_mpi' built correctly, we've found valid MPI settings. There might not be MPI-2 C++ support, but there can't
      # be MPI-2 C++ support without the C bindings being present, so checking for them is sufficient.
      set(MPI_${LANG}_WORKS "${MPI_RESULT_${LANG}_test_mpi_normal}")
    endif()
  endif()
endmacro()

# Some systems install various MPI implementations in separate folders in some MPI prefix
# This macro enumerates all such subfolders and adds them to the list of hints that will be searched.
macro(MPI_search_mpi_prefix_folder PREFIX_FOLDER)
  if(EXISTS "${PREFIX_FOLDER}")
    file(GLOB _MPI_folder_children RELATIVE "${PREFIX_FOLDER}" "${PREFIX_FOLDER}/*")
    foreach(_MPI_folder_child IN LISTS _MPI_folder_children)
      if(IS_DIRECTORY "${PREFIX_FOLDER}/${_MPI_folder_child}")
        list(APPEND MPI_HINT_DIRS "${PREFIX_FOLDER}/${_MPI_folder_child}")
      endif()
    endforeach()
  endif()
endmacro()

set(MPI_HINT_DIRS ${MPI_HOME} $ENV{MPI_HOME} $ENV{I_MPI_ROOT})
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  # SUSE Linux Enterprise Server stores its MPI implementations under /usr/lib64/mpi/gcc/<name>
  # We enumerate the subfolders and append each as a prefix
  MPI_search_mpi_prefix_folder("/usr/lib64/mpi/gcc")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "FreeBSD")
  # FreeBSD ships mpich under the normal system paths - but available openmpi implementations
  # will be found in /usr/local/mpi/<name>
  MPI_search_mpi_prefix_folder("/usr/local/mpi")
endif()

# Most MPI distributions have some form of mpiexec or mpirun which gives us something we can look for.
# The MPI standard does not mandate the existence of either, but instead only makes requirements if a distribution
# ships an mpiexec program (mpirun executables are not regulated by the standard).

# We defer searching for mpiexec binaries belonging to guesses until later. By doing so, mismatches between mpiexec
# and the MPI we found should be reduced.
if(NOT MPIEXEC_EXECUTABLE)
  set(_MPIEXEC_NOT_GIVEN TRUE)
else()
  set(_MPIEXEC_NOT_GIVEN FALSE)
endif()

find_program(MPIEXEC_EXECUTABLE
  NAMES ${_MPIEXEC_NAMES}
  PATH_SUFFIXES bin sbin
  HINTS ${MPI_HINT_DIRS}
  DOC "Executable for running MPI programs.")

# call get_filename_component twice to remove mpiexec and the directory it exists in (typically bin).
# This gives us a fairly reliable base directory to search for /bin /lib and /include from.
get_filename_component(_MPI_BASE_DIR "${MPIEXEC_EXECUTABLE}" PATH)
get_filename_component(_MPI_BASE_DIR "${_MPI_BASE_DIR}" PATH)

# According to the MPI standard, section 8.8 -n is a guaranteed, and the only guaranteed way to
# launch an MPI process using mpiexec if such a program exists.
set(MPIEXEC_NUMPROC_FLAG "-n"  CACHE STRING "Flag used by MPI to specify the number of processes for mpiexec; the next option will be the number of processes.")
set(MPIEXEC_PREFLAGS     ""    CACHE STRING "These flags will be directly before the executable that is being run by mpiexec.")
set(MPIEXEC_POSTFLAGS    ""    CACHE STRING "These flags will be placed after all flags passed to mpiexec.")

# Set the number of processes to the physical processor count
cmake_host_system_information(RESULT _MPIEXEC_NUMPROCS QUERY NUMBER_OF_PHYSICAL_CORES)
set(MPIEXEC_MAX_NUMPROCS "${_MPIEXEC_NUMPROCS}" CACHE STRING "Maximum number of processors available to run MPI applications.")
unset(_MPIEXEC_NUMPROCS)
mark_as_advanced(MPIEXEC_EXECUTABLE MPIEXEC_NUMPROC_FLAG MPIEXEC_PREFLAGS MPIEXEC_POSTFLAGS MPIEXEC_MAX_NUMPROCS)

#=============================================================================
# Backward compatibility input hacks.  Propagate the FindMPI hints to C and
# CXX if the respective new versions are not defined.  Translate the old
# MPI_LIBRARY and MPI_EXTRA_LIBRARY to respective MPI_${LANG}_LIBRARIES.
#
# Once we find the new variables, we translate them back into their old
# equivalents below.
if(NOT MPI_IGNORE_LEGACY_VARIABLES)
  foreach (LANG IN ITEMS C CXX)
    # Old input variables.
    set(_MPI_OLD_INPUT_VARS COMPILER COMPILE_FLAGS INCLUDE_PATH LINK_FLAGS)

    # Set new vars based on their old equivalents, if the new versions are not already set.
    foreach (var ${_MPI_OLD_INPUT_VARS})
      if (NOT MPI_${LANG}_${var} AND MPI_${var})
        set(MPI_${LANG}_${var} "${MPI_${var}}")
      endif()
    endforeach()

    # Chop the old compile flags into options and definitions

    unset(MPI_${LANG}_EXTRA_COMPILE_DEFINITIONS)
    unset(MPI_${LANG}_EXTRA_COMPILE_OPTIONS)
    if(MPI_${LANG}_COMPILE_FLAGS)
      separate_arguments(MPI_SEPARATE_FLAGS NATIVE_COMMAND "${MPI_${LANG}_COMPILE_FLAGS}")
      foreach(_MPI_FLAG IN LISTS MPI_SEPARATE_FLAGS)
        if(_MPI_FLAG MATCHES "^ *-D([^ ]+)")
          list(APPEND MPI_${LANG}_EXTRA_COMPILE_DEFINITIONS "${CMAKE_MATCH_1}")
        else()
          list(APPEND MPI_${LANG}_EXTRA_COMPILE_OPTIONS "${_MPI_FLAG}")
        endif()
      endforeach()
      unset(MPI_SEPARATE_FLAGS)
    endif()

    # If a list of libraries was given, we'll split it into new-style cache variables
    unset(MPI_${LANG}_EXTRA_LIB_NAMES)
    if(NOT MPI_${LANG}_LIB_NAMES)
      foreach(_MPI_LIB IN LISTS MPI_${LANG}_LIBRARIES MPI_LIBRARY MPI_EXTRA_LIBRARY)
        if(_MPI_LIB)
          get_filename_component(_MPI_PLAIN_LIB_NAME "${_MPI_LIB}" NAME_WE)
          get_filename_component(_MPI_LIB_NAME "${_MPI_LIB}" NAME)
          get_filename_component(_MPI_LIB_DIR "${_MPI_LIB}" DIRECTORY)
          list(APPEND MPI_${LANG}_EXTRA_LIB_NAMES "${_MPI_PLAIN_LIB_NAME}")
          find_library(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY
            NAMES "${_MPI_LIB_NAME}" "lib${_MPI_LIB_NAME}"
            HINTS ${_MPI_LIB_DIR} $ENV{MPI_LIB}
            DOC "Location of the ${_MPI_PLAIN_LIB_NAME} library for MPI"
          )
          mark_as_advanced(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY)
        endif()
      endforeach()
    endif()
  endforeach()
endif()
#=============================================================================

unset(MPI_VERSION)
unset(MPI_VERSION_MAJOR)
unset(MPI_VERSION_MINOR)

unset(_MPI_MIN_VERSION)

# If the user specified a library name we assume they prefer that library over a wrapper. If not, they can disable skipping manually.
if(NOT DEFINED MPI_SKIP_COMPILER_WRAPPER AND MPI_GUESS_LIBRARY_NAME)
  set(MPI_SKIP_COMPILER_WRAPPER TRUE)
endif()

# This loop finds the compilers and sends them off for interrogation.
foreach(LANG IN ITEMS C CXX Fortran)
  if(CMAKE_${LANG}_COMPILER_LOADED)
    if(NOT MPI_FIND_COMPONENTS)
      set(_MPI_FIND_${LANG} TRUE)
    elseif( LANG IN_LIST MPI_FIND_COMPONENTS)
      set(_MPI_FIND_${LANG} TRUE)
    elseif( LANG STREQUAL "CXX" AND NOT MPI_CXX_SKIP_MPICXX AND MPICXX IN_LIST MPI_FIND_COMPONENTS )
      set(_MPI_FIND_${LANG} TRUE)
    else()
      set(_MPI_FIND_${LANG} FALSE)
    endif()
  else()
    set(_MPI_FIND_${LANG} FALSE)
    if(LANG IN_LIST MPI_FIND_COMPONENTS)
      string(APPEND _MPI_FAIL_REASON "MPI component '${LANG}' was requested, but language ${LANG} is not enabled.  ")
    endif()
  endif()
  if(_MPI_FIND_${LANG})
    if( LANG STREQUAL "CXX" AND NOT MPICXX IN_LIST MPI_FIND_COMPONENTS )
      option(MPI_CXX_SKIP_MPICXX "If true, the MPI-2 C++ bindings are disabled using definitions." FALSE)
      mark_as_advanced(MPI_CXX_SKIP_MPICXX)
    endif()
    _MPI_adjust_compile_definitions(${LANG})
    if(NOT (MPI_${LANG}_LIB_NAMES AND (MPI_${LANG}_INCLUDE_PATH OR MPI_${LANG}_INCLUDE_DIRS OR MPI_${LANG}_COMPILER_INCLUDE_DIRS)))
      set(MPI_${LANG}_TRIED_IMPLICIT FALSE)
      set(MPI_${LANG}_WORKS_IMPLICIT FALSE)
      if(NOT MPI_${LANG}_COMPILER AND NOT MPI_ASSUME_NO_BUILTIN_MPI)
        # Should the imported targets be empty, we effectively try whether the compiler supports MPI on its own, which is the case on e.g.
        # Cray PrgEnv.
        _MPI_create_imported_target(${LANG})
        _MPI_check_lang_works(${LANG} TRUE)

        # If the compiler can build MPI code on its own, it functions as an MPI compiler and we'll set the variable to point to it.
        if(MPI_${LANG}_WORKS)
          set(MPI_${LANG}_COMPILER "${CMAKE_${LANG}_COMPILER}" CACHE FILEPATH "MPI compiler for ${LANG}" FORCE)
          set(MPI_${LANG}_WORKS_IMPLICIT TRUE)
        endif()
        set(MPI_${LANG}_TRIED_IMPLICIT TRUE)
      endif()

      if(NOT MPI_${LANG}_COMPILER STREQUAL CMAKE_${LANG}_COMPILER OR NOT MPI_${LANG}_WORKS)
        set(MPI_${LANG}_WRAPPER_FOUND FALSE)
        set(MPI_PINNED_COMPILER FALSE)

        if(NOT MPI_SKIP_COMPILER_WRAPPER)
          if(MPI_${LANG}_COMPILER)
            # If the user supplies a compiler *name* instead of an absolute path, assume that we need to find THAT compiler.
            if (NOT IS_ABSOLUTE "${MPI_${LANG}_COMPILER}")
              # Get rid of our default list of names and just search for the name the user wants.
              set(_MPI_${LANG}_COMPILER_NAMES "${MPI_${LANG}_COMPILER}")
              unset(MPI_${LANG}_COMPILER CACHE)
            endif()
            # If the user specifies a compiler, we don't want to try to search libraries either.
            set(MPI_PINNED_COMPILER TRUE)
          endif()

          # If we have an MPI base directory, we'll try all compiler names in that one first.
          # This should prevent mixing different MPI environments
          if(_MPI_BASE_DIR)
            find_program(MPI_${LANG}_COMPILER
              NAMES  ${_MPI_${LANG}_COMPILER_NAMES}
              PATH_SUFFIXES bin sbin
              HINTS  ${_MPI_BASE_DIR}
              NO_DEFAULT_PATH
              DOC    "MPI compiler for ${LANG}"
            )
          endif()

          # If the base directory did not help (for example because the mpiexec isn't in the same directory as the compilers),
          # we shall try searching in the default paths.
          find_program(MPI_${LANG}_COMPILER
            NAMES  ${_MPI_${LANG}_COMPILER_NAMES}
            PATH_SUFFIXES bin sbin
            DOC    "MPI compiler for ${LANG}"
          )

          if(MPI_${LANG}_COMPILER STREQUAL CMAKE_${LANG}_COMPILER)
            set(MPI_PINNED_COMPILER TRUE)

            # If we haven't made the implicit compiler test yet, perform it now.
            if(NOT MPI_${LANG}_TRIED_IMPLICIT)
              _MPI_create_imported_target(${LANG})
              _MPI_check_lang_works(${LANG} TRUE)
            endif()

            # Should the MPI compiler not work implicitly for MPI, still interrogate it.
            # Otherwise, MPI compilers for which CMake has separate linking stages, e.g. Intel MPI on Windows where link.exe is being used
            # directly during linkage instead of CMAKE_<LANG>_COMPILER will not work.
            if(NOT MPI_${LANG}_WORKS)
              set(MPI_${LANG}_WORKS_IMPLICIT FALSE)
              _MPI_interrogate_compiler(${LANG})
            else()
              set(MPI_${LANG}_WORKS_IMPLICIT TRUE)
            endif()
          elseif(MPI_${LANG}_COMPILER)
            _MPI_interrogate_compiler(${LANG})
          endif()
        endif()

        # We are on a Cray, environment identifier: PE_ENV is set (CRAY), and
        # have NOT found an mpic++-like compiler wrapper (previous block),
        # and we do NOT use the Cray cc/CC compiler wrappers as CC/CXX CMake
        # compiler.
        # So as a last resort, we now interrogate cc/CC/ftn for MPI flags.
        if(DEFINED ENV{PE_ENV} AND NOT "${MPI_${LANG}_COMPILER}")
          set(MPI_PINNED_COMPILER TRUE)
          find_program(MPI_${LANG}_COMPILER
            NAMES  ${_MPI_Cray_${LANG}_COMPILER_NAMES}
            PATH_SUFFIXES bin sbin
            DOC    "MPI compiler for ${LANG}"
          )

          # If we haven't made the implicit compiler test yet, perform it now.
          if(NOT MPI_${LANG}_TRIED_IMPLICIT)
            _MPI_create_imported_target(${LANG})
            _MPI_check_lang_works(${LANG} TRUE)
          endif()

          set(MPI_${LANG}_WORKS_IMPLICIT TRUE)
          _MPI_interrogate_compiler(${LANG})
        endif()

        if(NOT MPI_PINNED_COMPILER AND NOT MPI_${LANG}_WRAPPER_FOUND)
          # If MPI_PINNED_COMPILER wasn't given, and the MPI compiler we potentially found didn't work, we withdraw it.
          set(MPI_${LANG}_COMPILER "MPI_${LANG}_COMPILER-NOTFOUND" CACHE FILEPATH "MPI compiler for ${LANG}" FORCE)

          if(LANG STREQUAL "C")
            set(_MPI_PKG "mpi-c")
          elseif(LANG STREQUAL "CXX")
            set(_MPI_PKG "mpi-cxx")
          elseif(LANG STREQUAL "Fortran")
            set(_MPI_PKG "mpi-fort")
          else()
            set(_MPI_PKG "")
          endif()
          if(_MPI_PKG AND PkgConfig_FOUND)
            pkg_check_modules("MPI_${LANG}_PKG" "${_MPI_PKG}")
            if(MPI_${LANG}_PKG_FOUND)
              set(MPI_${LANG}_COMPILE_OPTIONS  ${MPI_${LANG}_PKG_CFLAGS}        CACHE STRING "MPI ${LANG} compilation options"       FORCE)
              set(MPI_${LANG}_INCLUDE_PATH     ${MPI_${LANG}_PKG_INCLUDE_DIRS}  CACHE STRING "MPI ${LANG} include directories"       FORCE)
              set(MPI_${LANG}_LINK_FLAGS       ${MPI_${LANG}_PKG_LDFLAGS}       CACHE STRING "MPI ${LANG} linker flags"              FORCE)
              set(MPI_${LANG}_LIB_NAMES        ${MPI_${LANG}_PKG_LIBRARIES}     CACHE STRING "MPI ${LANG} libraries to link against" FORCE)
              foreach(_MPI_LIB IN LISTS MPI_${LANG}_LIB_NAMES)
                if(_MPI_LIB)
                  get_filename_component(_MPI_PLAIN_LIB_NAME "${_MPI_LIB}" NAME_WE)
                  get_filename_component(_MPI_LIB_NAME "${_MPI_LIB}" NAME)
                  get_filename_component(_MPI_LIB_DIR "${_MPI_LIB}" DIRECTORY)
                  find_library(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY
                    NAMES "${_MPI_LIB_NAME}" "lib${_MPI_LIB_NAME}"
                    HINTS ${_MPI_LIB_DIR}
                    DOC "Location of the ${_MPI_PLAIN_LIB_NAME} library for MPI"
                  )
                  mark_as_advanced(MPI_${_MPI_PLAIN_LIB_NAME}_LIBRARY)
                endif()
              endforeach()
            endif()
          endif()

          if(NOT MPI_SKIP_GUESSING AND NOT MPI_${LANG}_PKG_FOUND)
            # For C++, we may use the settings for C. Should a given compiler wrapper for C++ not exist, but one for C does, we copy over the
            # settings for C. An MPI distribution that is in this situation would be IBM Platform MPI.
            if(LANG STREQUAL "CXX" AND MPI_C_WRAPPER_FOUND)
              set(MPI_${LANG}_COMPILE_OPTIONS          ${MPI_C_COMPILE_OPTIONS}     CACHE STRING "MPI ${LANG} compilation options"           )
              set(MPI_${LANG}_COMPILE_DEFINITIONS      ${MPI_C_COMPILE_DEFINITIONS} CACHE STRING "MPI ${LANG} compilation definitions"       )
              set(MPI_${LANG}_COMPILER_INCLUDE_DIRS    ${MPI_C_INCLUDE_DIRS}        CACHE STRING "MPI ${LANG} compiler wrapper include directories")
              set(MPI_${LANG}_LINK_FLAGS               ${MPI_C_LINK_FLAGS}          CACHE STRING "MPI ${LANG} linker flags"                  )
              set(MPI_${LANG}_LIB_NAMES                ${MPI_C_LIB_NAMES}           CACHE STRING "MPI ${LANG} libraries to link against"     )
            else()
              _MPI_guess_settings(${LANG})
            endif()
          endif()
        endif()
      endif()
    endif()

    if(NOT MPI_${LANG}_COMPILER STREQUAL CMAKE_${LANG}_COMPILER)
      _MPI_split_include_dirs(${LANG})
      _MPI_assemble_include_dirs(${LANG})
    else()
      set(MPI_${LANG}_INCLUDE_DIRS "")
    endif()
    _MPI_assemble_libraries(${LANG})

    # We always create imported targets even if they're empty
    _MPI_create_imported_target(${LANG})

    if(NOT MPI_${LANG}_WORKS)
      _MPI_check_lang_works(${LANG} FALSE)
    endif()

    # Next, we'll initialize the MPI variables that have not been previously set.
    set(MPI_${LANG}_COMPILE_OPTIONS          "" CACHE STRING "MPI ${LANG} compilation flags"             )
    set(MPI_${LANG}_COMPILE_DEFINITIONS      "" CACHE STRING "MPI ${LANG} compilation definitions"       )
    set(MPI_${LANG}_COMPILER_INCLUDE_DIRS    "" CACHE STRING "MPI ${LANG} compiler wrapper include directories")
    set(MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS  "" CACHE STRING "MPI ${LANG} additional include directories")
    set(MPI_${LANG}_LINK_FLAGS               "" CACHE STRING "MPI ${LANG} linker flags"                  )
    if(NOT MPI_${LANG}_COMPILER STREQUAL CMAKE_${LANG}_COMPILER)
      set(MPI_${LANG}_LIB_NAMES                "" CACHE STRING "MPI ${LANG} libraries to link against"   )
    endif()
    mark_as_advanced(
      MPI_${LANG}_COMPILE_OPTIONS
      MPI_${LANG}_COMPILE_DEFINITIONS
      MPI_${LANG}_LINK_FLAGS
      MPI_${LANG}_LIB_NAMES
      MPI_${LANG}_COMPILER_INCLUDE_DIRS
      MPI_${LANG}_ADDITIONAL_INCLUDE_DIRS
      MPI_${LANG}_COMPILER
      )

    # If we've found MPI, then we'll perform additional analysis: Determine the MPI version, MPI library version, supported
    # MPI APIs (i.e. MPI-2 C++ bindings). For Fortran we also need to find specific parameters if we're under MPI-3.
    if(MPI_${LANG}_WORKS)
      if(LANG STREQUAL "CXX" AND NOT DEFINED MPI_MPICXX_FOUND)
        if(NOT MPI_CXX_SKIP_MPICXX AND NOT MPI_CXX_VALIDATE_SKIP_MPICXX)
          _MPI_try_staged_settings(${LANG} test_mpi MPICXX FALSE FALSE)
          if(MPI_RESULT_${LANG}_test_mpi_MPICXX)
            set(MPI_MPICXX_FOUND TRUE)
          else()
            set(MPI_MPICXX_FOUND FALSE)
          endif()
        else()
          set(MPI_MPICXX_FOUND FALSE)
        endif()
      endif()

      # At this point, we know the bindings present but not the MPI version or anything else.
      if(NOT DEFINED MPI_${LANG}_VERSION)
        unset(MPI_${LANG}_VERSION_MAJOR)
        unset(MPI_${LANG}_VERSION_MINOR)
      endif()
      set(MPI_BIN_FOLDER ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/FindMPI)

      # For Fortran, we'll want to use the most modern MPI binding to test capabilities other than the
      # Fortran parameters, since those depend on the method of consumption.
      # For C++, we can always use the C bindings, and should do so, since the C++ bindings do not exist in MPI-3
      # whereas the C bindings do, and the C++ bindings never offered any feature advantage over their C counterparts.
      if(LANG STREQUAL "Fortran")
        if(MPI_${LANG}_HAVE_F08_MODULE)
          set(MPI_${LANG}_HIGHEST_METHOD F08_MODULE)
        elseif(MPI_${LANG}_HAVE_F90_MODULE)
          set(MPI_${LANG}_HIGHEST_METHOD F90_MODULE)
        else()
          set(MPI_${LANG}_HIGHEST_METHOD F77_HEADER)
        endif()

        # Another difference between C and Fortran is that we can't use the preprocessor to determine whether MPI_VERSION
        # and MPI_SUBVERSION are provided. These defines did not exist in MPI 1.0 and 1.1 and therefore might not
        # exist. For C/C++, test_mpi.c will handle the MPI_VERSION extraction, but for Fortran, we need mpiver.f90.
        if(NOT DEFINED MPI_${LANG}_VERSION)
          _MPI_try_staged_settings(${LANG} mpiver ${MPI_${LANG}_HIGHEST_METHOD} FALSE FALSE)
          if(MPI_RESULT_${LANG}_mpiver_${MPI_${LANG}_HIGHEST_METHOD})
            file(STRINGS ${MPI_BIN_FOLDER}/mpiver_${LANG}.bin _MPI_VERSION_STRING LIMIT_COUNT 1 REGEX "INFO:MPI-VER")
            if(_MPI_VERSION_STRING MATCHES ".*INFO:MPI-VER\\[([0-9]+)\\.([0-9]+)\\].*")
              set(MPI_${LANG}_VERSION_MAJOR "${CMAKE_MATCH_1}")
              set(MPI_${LANG}_VERSION_MINOR "${CMAKE_MATCH_2}")
              set(MPI_${LANG}_VERSION "${MPI_${LANG}_VERSION_MAJOR}.${MPI_${LANG}_VERSION_MINOR}")
            endif()
          endif()
        endif()

        # Finally, we want to find out which capabilities a given interface supports, compare the MPI-3 standard.
        # This is determined by interface specific parameters MPI_SUBARRAYS_SUPPORTED and MPI_ASYNC_PROTECTS_NONBLOCKING
        # and might vary between the different methods of consumption.
        if(MPI_DETERMINE_Fortran_CAPABILITIES AND NOT MPI_Fortran_CAPABILITIES_DETERMINED)
          foreach(mpimethod IN ITEMS F08_MODULE F90_MODULE F77_HEADER)
            if(MPI_${LANG}_HAVE_${mpimethod})
              set(MPI_${LANG}_${mpimethod}_SUBARRAYS FALSE)
              set(MPI_${LANG}_${mpimethod}_ASYNCPROT FALSE)
              _MPI_try_staged_settings(${LANG} fortranparam_mpi ${mpimethod} TRUE FALSE)
              if(MPI_RESULT_${LANG}_fortranparam_mpi_${mpimethod} AND
                NOT "${MPI_RUN_RESULT_${LANG}_fortranparam_mpi_${mpimethod}}" STREQUAL "FAILED_TO_RUN")
                if(MPI_RUN_OUTPUT_${LANG}_fortranparam_mpi_${mpimethod} MATCHES
                  ".*INFO:SUBARRAYS\\[ *([TF]) *\\]-ASYNCPROT\\[ *([TF]) *\\].*")
                  if(CMAKE_MATCH_1 STREQUAL "T")
                    set(MPI_${LANG}_${mpimethod}_SUBARRAYS TRUE)
                  endif()
                  if(CMAKE_MATCH_2 STREQUAL "T")
                    set(MPI_${LANG}_${mpimethod}_ASYNCPROT TRUE)
                  endif()
                endif()
              endif()
            endif()
          endforeach()
          set(MPI_Fortran_CAPABILITIES_DETERMINED TRUE)
        endif()
      else()
        set(MPI_${LANG}_HIGHEST_METHOD normal)

        # By the MPI-2 standard, MPI_VERSION and MPI_SUBVERSION are valid for both C and C++ bindings.
        if(NOT DEFINED MPI_${LANG}_VERSION)
          file(STRINGS ${MPI_BIN_FOLDER}/test_mpi_${LANG}.bin _MPI_VERSION_STRING LIMIT_COUNT 1 REGEX "INFO:MPI-VER")
          if(_MPI_VERSION_STRING MATCHES ".*INFO:MPI-VER\\[([0-9]+)\\.([0-9]+)\\].*")
            set(MPI_${LANG}_VERSION_MAJOR "${CMAKE_MATCH_1}")
            set(MPI_${LANG}_VERSION_MINOR "${CMAKE_MATCH_2}")
            set(MPI_${LANG}_VERSION "${MPI_${LANG}_VERSION_MAJOR}.${MPI_${LANG}_VERSION_MINOR}")
          endif()
        endif()
      endif()

      unset(MPI_BIN_FOLDER)

      # At this point, we have dealt with determining the MPI version and parameters for each Fortran method available.
      # The one remaining issue is to determine which MPI library is installed.
      # Determining the version and vendor of the MPI library is only possible via MPI_Get_library_version() at runtime,
      # and therefore we cannot do this while cross-compiling (a user may still define MPI_<lang>_LIBRARY_VERSION_STRING
      # themselves and we'll attempt splitting it, which is equivalent to provide the try_run output).
      # It's also worth noting that the installed version string can depend on the language, or on the system the binary
      # runs on if MPI is not statically linked.
      if(MPI_DETERMINE_LIBRARY_VERSION AND NOT MPI_${LANG}_LIBRARY_VERSION_STRING)
        _MPI_try_staged_settings(${LANG} libver_mpi ${MPI_${LANG}_HIGHEST_METHOD} TRUE FALSE)
        if(MPI_RESULT_${LANG}_libver_mpi_${MPI_${LANG}_HIGHEST_METHOD} AND
          MPI_RUN_RESULT_${LANG}_libver_mpi_${MPI_${LANG}_HIGHEST_METHOD} EQUAL "0")
          string(STRIP "${MPI_RUN_OUTPUT_${LANG}_libver_mpi_${MPI_${LANG}_HIGHEST_METHOD}}"
            MPI_${LANG}_LIBRARY_VERSION_STRING)
        else()
          set(MPI_${LANG}_LIBRARY_VERSION_STRING "NOTFOUND")
        endif()
      endif()
    endif()

    set(MPI_${LANG}_FIND_QUIETLY ${MPI_FIND_QUIETLY})
    set(MPI_${LANG}_FIND_VERSION ${MPI_FIND_VERSION})
    set(MPI_${LANG}_FIND_VERSION_EXACT ${MPI_FIND_VERSION_EXACT})

    unset(MPI_${LANG}_REQUIRED_VARS)
    if (NOT MPI_${LANG}_COMPILER STREQUAL CMAKE_${LANG}_COMPILER)
      foreach(mpilibname IN LISTS MPI_${LANG}_LIB_NAMES)
        list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${mpilibname}_LIBRARY")
      endforeach()
      list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${LANG}_LIB_NAMES")
      if(LANG STREQUAL "Fortran")
        # For Fortran we only need one of the module or header directories to have *some* support for MPI.
        if(NOT MPI_${LANG}_MODULE_DIR)
          list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${LANG}_F77_HEADER_DIR")
        endif()
        if(NOT MPI_${LANG}_F77_HEADER_DIR)
          list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${LANG}_MODULE_DIR")
        endif()
      else()
        list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${LANG}_HEADER_DIR")
      endif()
      if(MPI_${LANG}_ADDITIONAL_INCLUDE_VARS)
        foreach(mpiincvar IN LISTS MPI_${LANG}_ADDITIONAL_INCLUDE_VARS)
          list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${mpiincvar}_INCLUDE_DIR")
        endforeach()
      endif()
      # Append the works variable now. If the settings did not work, this will show up properly.
      list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${LANG}_WORKS")
    else()
      # If the compiler worked implicitly, use its path as output.
      # Should the compiler variable be set, we also require it to work.
      list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${LANG}_COMPILER")
      if(MPI_${LANG}_COMPILER)
        list(APPEND MPI_${LANG}_REQUIRED_VARS "MPI_${LANG}_WORKS")
      endif()
    endif()
    find_package_handle_standard_args(MPI_${LANG} NAME_MISMATCHED
      REQUIRED_VARS ${MPI_${LANG}_REQUIRED_VARS}
      VERSION_VAR MPI_${LANG}_VERSION)

    if(DEFINED MPI_${LANG}_VERSION)
      if(NOT _MPI_MIN_VERSION OR _MPI_MIN_VERSION VERSION_GREATER MPI_${LANG}_VERSION)
        set(_MPI_MIN_VERSION MPI_${LANG}_VERSION)
      endif()
    endif()
  endif()
endforeach()

unset(_MPI_REQ_VARS)
foreach(LANG IN ITEMS C CXX Fortran)
  if((NOT MPI_FIND_COMPONENTS AND CMAKE_${LANG}_COMPILER_LOADED) OR LANG IN_LIST MPI_FIND_COMPONENTS)
    list(APPEND _MPI_REQ_VARS "MPI_${LANG}_FOUND")
  endif()
endforeach()

if(MPICXX IN_LIST MPI_FIND_COMPONENTS)
  list(APPEND _MPI_REQ_VARS "MPI_MPICXX_FOUND")
endif()

find_package_handle_standard_args(MPI
    REQUIRED_VARS ${_MPI_REQ_VARS}
    VERSION_VAR ${_MPI_MIN_VERSION}
    REASON_FAILURE_MESSAGE "${_MPI_FAIL_REASON}"
    HANDLE_COMPONENTS)

#=============================================================================
# More backward compatibility stuff

# For compatibility reasons, we also define MPIEXEC
set(MPIEXEC "${MPIEXEC_EXECUTABLE}")

# Copy over MPI_<LANG>_INCLUDE_PATH from the assembled INCLUDE_DIRS.
foreach(LANG IN ITEMS C CXX Fortran)
  if(MPI_${LANG}_FOUND)
    set(MPI_${LANG}_INCLUDE_PATH "${MPI_${LANG}_INCLUDE_DIRS}")
    unset(MPI_${LANG}_COMPILE_FLAGS)
    if(MPI_${LANG}_COMPILE_OPTIONS)
      list(JOIN MPI_${LANG}_COMPILE_OPTIONS " " MPI_${LANG}_COMPILE_FLAGS)
    endif()
    if(MPI_${LANG}_COMPILE_DEFINITIONS)
      foreach(_MPI_DEF IN LISTS MPI_${LANG}_COMPILE_DEFINITIONS)
        string(APPEND MPI_${LANG}_COMPILE_FLAGS " -D${_MPI_DEF}")
      endforeach()
    endif()
  endif()
endforeach()

# Bare MPI sans ${LANG} vars are set to CXX then C, depending on what was found.
# This mimics the behavior of the old language-oblivious FindMPI.
set(_MPI_OLD_VARS COMPILER INCLUDE_PATH COMPILE_FLAGS LINK_FLAGS LIBRARIES)
if (MPI_CXX_FOUND)
  foreach (var ${_MPI_OLD_VARS})
    set(MPI_${var} ${MPI_CXX_${var}})
  endforeach()
elseif (MPI_C_FOUND)
  foreach (var ${_MPI_OLD_VARS})
    set(MPI_${var} ${MPI_C_${var}})
  endforeach()
endif()

# Chop MPI_LIBRARIES into the old-style MPI_LIBRARY and MPI_EXTRA_LIBRARY, and set them in cache.
if (MPI_LIBRARIES)
  list(GET MPI_LIBRARIES 0 MPI_LIBRARY_WORK)
  set(MPI_LIBRARY "${MPI_LIBRARY_WORK}")
  unset(MPI_LIBRARY_WORK)
else()
  set(MPI_LIBRARY "MPI_LIBRARY-NOTFOUND")
endif()

list(LENGTH MPI_LIBRARIES MPI_NUMLIBS)
if (MPI_NUMLIBS GREATER "1")
  set(MPI_EXTRA_LIBRARY_WORK "${MPI_LIBRARIES}")
  list(REMOVE_AT MPI_EXTRA_LIBRARY_WORK 0)
  set(MPI_EXTRA_LIBRARY "${MPI_EXTRA_LIBRARY_WORK}")
  unset(MPI_EXTRA_LIBRARY_WORK)
else()
  set(MPI_EXTRA_LIBRARY "MPI_EXTRA_LIBRARY-NOTFOUND")
endif()
set(MPI_IGNORE_LEGACY_VARIABLES TRUE)
#=============================================================================

# unset these vars to cleanup namespace
unset(_MPI_OLD_VARS)
unset(_MPI_PREFIX_PATH)
unset(_MPI_BASE_DIR)
foreach (lang C CXX Fortran)
  unset(_MPI_${LANG}_COMPILER_NAMES)
endforeach()

cmake_policy(POP)
