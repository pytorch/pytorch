# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
ProcessorCount
--------------

This module provides a command to determine the number of processors/cores.

Load this module in CMake with:

.. code-block:: cmake

  include(ProcessorCount)

Commands
^^^^^^^^

This module provides the following command:

.. command:: ProcessorCount

  Determines the number of logical CPU cores available on the machine:

  .. code-block:: cmake

    ProcessorCount(<variable>)

  This command sets a variable named ``<variable>`` to the number of logical
  CPU cores available on the machine, if the information can be determined.
  If successful, the variable is guaranteed to be set to a positive integer
  (>=1).  If the processor count cannot be determined, it is set to 0.

  Currently, this functionality is implemented for AIX, Cygwin, FreeBSD, Haiku,
  HPUX, Linux, macOS, QNX, Sun and Windows.

  This command provides an approximation of the number of compute cores
  available on the current machine, making it useful for parallel building and
  testing.  It is meant to help utilize as much of the machine as seems
  reasonable, though users should consider other workloads running on the
  machine before using its full capacity for parallel tasks.

  .. versionchanged:: 3.15
    On Linux, returns the container CPU count instead of the host CPU count.

.. note::

  This module relies on system-dependent commands to determine the number of
  processors, which may not always provide accurate information in certain
  environments.  A more generally accurate logical CPU count can be also
  obtained with the :command:`cmake_host_system_information`:

  .. code-block:: cmake

    cmake_host_system_information(RESULT n QUERY NUMBER_OF_LOGICAL_CORES)

Examples
^^^^^^^^

In the following example this module is used in a :option:`ctest -S`
dashboard script to determine number of cores to use for a parallel
:ref:`CTest Build Step`:

.. code-block:: cmake

  include(ProcessorCount)
  ProcessorCount(n)
  if(NOT n EQUAL 0)
    set(CTEST_BUILD_FLAGS -j${n})
    set(ctest_test_args ${ctest_test_args} PARALLEL_LEVEL ${n})
  endif()
#]=======================================================================]

# A more reliable way might be to compile a small C program that uses the CPUID
# instruction, but that again requires compiler support or compiling assembler
# code.

function(ProcessorCount var)
  # Unknown:
  set(count 0)

  if(WIN32)
    # Windows:
    set(count "$ENV{NUMBER_OF_PROCESSORS}")
    #message("ProcessorCount: WIN32, trying environment variable")
  endif()

  if(NOT count)
    # Mac, FreeBSD, OpenBSD (systems with sysctl):
    find_program(ProcessorCount_cmd_sysctl sysctl
      PATHS /usr/sbin /sbin)
    mark_as_advanced(ProcessorCount_cmd_sysctl)
    if(ProcessorCount_cmd_sysctl)
      execute_process(COMMAND ${ProcessorCount_cmd_sysctl} -n hw.ncpu
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE count)
      #message("ProcessorCount: trying sysctl '${ProcessorCount_cmd_sysctl}'")
    endif()
  endif()

  if(NOT count)
    # Linux (systems with nproc):
    # Prefer nproc to getconf if available as getconf may return the host CPU count in Linux containers
    find_program(ProcessorCount_cmd_nproc nproc)
    mark_as_advanced(ProcessorCount_cmd_nproc)
    if(ProcessorCount_cmd_nproc)
      execute_process(COMMAND ${ProcessorCount_cmd_nproc}
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE count)
      #message("ProcessorCount: trying nproc '${ProcessorCount_cmd_nproc}'")
    endif()
  endif()

  if(NOT count)
    # Linux (systems with getconf):
    find_program(ProcessorCount_cmd_getconf getconf)
    mark_as_advanced(ProcessorCount_cmd_getconf)
    if(ProcessorCount_cmd_getconf)
      execute_process(COMMAND ${ProcessorCount_cmd_getconf} _NPROCESSORS_ONLN
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE count)
      #message("ProcessorCount: trying getconf '${ProcessorCount_cmd_getconf}'")
    endif()
  endif()

  if(NOT count)
    # HPUX (systems with machinfo):
    find_program(ProcessorCount_cmd_machinfo machinfo
      PATHS /usr/contrib/bin)
    mark_as_advanced(ProcessorCount_cmd_machinfo)
    if(ProcessorCount_cmd_machinfo)
      execute_process(COMMAND ${ProcessorCount_cmd_machinfo}
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE machinfo_output)
      string(REGEX MATCHALL "Number of CPUs = ([0-9]+)" procs "${machinfo_output}")
      set(count "${CMAKE_MATCH_1}")
      if(NOT count)
        string(REGEX MATCHALL "([0-9]+) logical processors" procs "${machinfo_output}")
        set(count "${CMAKE_MATCH_1}")
      endif()
      #message("ProcessorCount: trying machinfo '${ProcessorCount_cmd_machinfo}'")
    else()
      find_program(ProcessorCount_cmd_mpsched mpsched)
      mark_as_advanced(ProcessorCount_cmd_mpsched)
      if(ProcessorCount_cmd_mpsched)
        execute_process(COMMAND ${ProcessorCount_cmd_mpsched} -s
          OUTPUT_QUIET
          ERROR_STRIP_TRAILING_WHITESPACE
          ERROR_VARIABLE mpsched_output)
        string(REGEX MATCHALL "Processor Count *: *([0-9]+)" procs "${mpsched_output}")
        set(count "${CMAKE_MATCH_1}")
        #message("ProcessorCount: trying mpsched -s '${ProcessorCount_cmd_mpsched}'")
      endif()
    endif()
  endif()

  if(NOT count)
    # AIX (systems with lsconf):
    find_program(ProcessorCount_cmd_lsconf lsconf
      PATHS /usr/sbin)
    mark_as_advanced(ProcessorCount_cmd_lsconf)
    if(ProcessorCount_cmd_lsconf)
      execute_process(COMMAND ${ProcessorCount_cmd_lsconf}
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE lsconf_output)
      string(REGEX MATCHALL "Number Of Processors: ([0-9]+)" procs "${lsconf_output}")
      set(count "${CMAKE_MATCH_1}")
      #message("ProcessorCount: trying lsconf '${ProcessorCount_cmd_lsconf}'")
    endif()
  endif()

  if(NOT count)
    # QNX (systems with pidin):
    find_program(ProcessorCount_cmd_pidin pidin)
    mark_as_advanced(ProcessorCount_cmd_pidin)
    if(ProcessorCount_cmd_pidin)
      execute_process(COMMAND ${ProcessorCount_cmd_pidin} info
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE pidin_output)
      string(REGEX MATCHALL "Processor[0-9]+: " procs "${pidin_output}")
      list(LENGTH procs count)
      #message("ProcessorCount: trying pidin '${ProcessorCount_cmd_pidin}'")
    endif()
  endif()

  if(NOT count)
    # Sun (systems where psrinfo tool is available)
    find_program(ProcessorCount_cmd_psrinfo psrinfo PATHS /usr/sbin /sbin)
    mark_as_advanced(ProcessorCount_cmd_psrinfo)
    if (ProcessorCount_cmd_psrinfo)
      execute_process(COMMAND ${ProcessorCount_cmd_psrinfo} -p -v
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE psrinfo_output)
      string(REGEX MATCHALL "has [0-9]+ virtual processor" procs "${psrinfo_output}")
      set(count "")
      foreach(proc ${procs})
        string(REGEX MATCH "has ([0-9]+) virtual" res ${proc})
        math(EXPR count "${count} + ${CMAKE_MATCH_1}")
      endforeach()
      #message("ProcessorCount: trying '${ProcessorCount_cmd_psrinfo}' -p -v")
    else()
      # Sun (systems where uname -X emits "NumCPU" in its output):
      find_program(ProcessorCount_cmd_uname uname)
      mark_as_advanced(ProcessorCount_cmd_uname)
      if(ProcessorCount_cmd_uname)
        execute_process(COMMAND ${ProcessorCount_cmd_uname} -X
          ERROR_QUIET
          OUTPUT_STRIP_TRAILING_WHITESPACE
          OUTPUT_VARIABLE uname_X_output)
        string(REGEX MATCHALL "NumCPU = ([0-9]+)" procs "${uname_X_output}")
        set(count "${CMAKE_MATCH_1}")
        #message("ProcessorCount: trying uname -X '${ProcessorCount_cmd_uname}'")
      endif()
    endif()
  endif()

  # Execute this code when all previously attempted methods return empty
  # output:
  #
  if(NOT count)
    # Systems with /proc/cpuinfo:
    set(cpuinfo_file /proc/cpuinfo)
    if(EXISTS "${cpuinfo_file}")
      cmake_policy(PUSH)
      cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
      file(STRINGS "${cpuinfo_file}" procs REGEX "^processor.: [0-9]+$")
      cmake_policy(POP)
      list(LENGTH procs count)
      #message("ProcessorCount: trying cpuinfo '${cpuinfo_file}'")
    endif()
  endif()

  if(NOT count)
    # Haiku
    find_program(ProcessorCount_cmd_sysinfo sysinfo)
    mark_as_advanced(ProcessorCount_cmd_sysinfo)
    if(ProcessorCount_cmd_sysinfo)
      execute_process(COMMAND ${ProcessorCount_cmd_sysinfo}
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE sysinfo_X_output)
      string(REGEX MATCHALL "\nCPU #[0-9]+:" procs "\n${sysinfo_X_output}")
      list(LENGTH procs count)
      #message("ProcessorCount: trying sysinfo '${ProcessorCount_cmd_sysinfo}'")
    endif()
  endif()

  # Since cygwin builds of CMake do not define WIN32 anymore, but they still
  # run on Windows, and will still have this env var defined:
  #
  if(NOT count)
    set(count "$ENV{NUMBER_OF_PROCESSORS}")
    #message("ProcessorCount: last fallback, trying environment variable")
  endif()

  # Ensure an integer return (avoid inadvertently returning an empty string
  # or an error string)... If it's not a decimal integer, return 0:
  #
  if(NOT count MATCHES "^[0-9]+$")
    set(count 0)
  endif()

  set(${var} ${count} PARENT_SCOPE)
endfunction()
