# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLAPACK
----------

Finds the installed Linear Algebra PACKage (LAPACK) Fortran library that
implements the `LAPACK linear-algebra interface`_:

.. code-block:: cmake

  find_package(LAPACK [...])

At least one of the ``C``, ``CXX``, or ``Fortran`` languages must be enabled.

.. _`LAPACK linear-algebra interface`: https://netlib.org/lapack/

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``LAPACK::LAPACK``
  .. versionadded:: 3.18

  Target encapsulating the LAPACK usage requirements, available only if
  LAPACK is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LAPACK_FOUND``
  Boolean indicating whether the library implementing the LAPACK interface
  was found.
``LAPACK_LINKER_FLAGS``
  Uncached list of required linker flags (excluding ``-l`` and ``-L``).
``LAPACK_LIBRARIES``
  Uncached list of libraries (using full path name) to link against to use
  LAPACK.
``LAPACK95_LIBRARIES``
  Uncached list of libraries (using full path name) to link against to use
  LAPACK95.
``LAPACK95_FOUND``
  Boolean indicating whether the library implementing the LAPACK95 interface
  was found.

Input Variables
^^^^^^^^^^^^^^^

The following variables may be set to influence this module's behavior:

``BLA_STATIC``
  If ``ON``, the static linkage will be used.

``BLA_VENDOR``
  Set to one of the :ref:`BLAS/LAPACK Vendors` to search for BLAS only
  from the specified vendor.  If not set, all vendors are considered.

``BLA_F95``
  If ``ON``, the module tries to find the BLAS95/LAPACK95 interfaces.

``BLA_PREFER_PKGCONFIG``
  .. versionadded:: 3.20

  If set, ``pkg-config`` will be used to search for a LAPACK library first
  and if one is found that is preferred.

``BLA_PKGCONFIG_LAPACK``
  .. versionadded:: 3.25

  If set, the ``pkg-config`` method will look for this module name instead of
  just ``lapack``.

``BLA_SIZEOF_INTEGER``
  .. versionadded:: 3.22

  Specify the BLAS/LAPACK library integer size:

  ``4``
    Search for a BLAS/LAPACK with 32-bit integer interfaces.
  ``8``
    Search for a BLAS/LAPACK with 64-bit integer interfaces.
  ``ANY``
    Search for any BLAS/LAPACK.
    Most likely, a BLAS/LAPACK with 32-bit integer interfaces will be found.

``BLA_THREAD``
  .. versionadded:: 4.1

  Specify the BLAS/LAPACK threading model:

  ``SEQ``
    Sequential model
  ``OMP``
    OpenMP model
  ``ANY``
    Search for any BLAS/LAPACK, if both are available most likely ``OMP`` will
    be found.

  This is currently only supported by NVIDIA NVPL.

Intel MKL
^^^^^^^^^

To use the Intel MKL implementation of LAPACK, a project must enable at least
one of the ``C`` or ``CXX`` languages.  Set ``BLA_VENDOR`` to an Intel MKL
variant either on the command-line as ``-DBLA_VENDOR=Intel10_64lp`` or in
project code:

.. code-block:: cmake

  set(BLA_VENDOR Intel10_64lp)
  find_package(LAPACK)

In order to build a project using Intel MKL, and end user must first
establish an Intel MKL environment.  See the :module:`FindBLAS` module
section on :ref:`Intel MKL` for details.

Examples
^^^^^^^^

Finding LAPACK and linking it to a project target:

.. code-block:: cmake

  find_package(LAPACK)
  target_link_libraries(project_target PRIVATE LAPACK::LAPACK)
#]=======================================================================]

# The approach follows that of the ``autoconf`` macro file, ``acx_lapack.m4``
# (distributed at http://ac-archive.sourceforge.net/ac-archive/acx_lapack.html).

if(CMAKE_Fortran_COMPILER_LOADED)
  include(${CMAKE_CURRENT_LIST_DIR}/CheckFortranFunctionExists.cmake)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/CheckFunctionExists.cmake)
endif()
include(FindPackageHandleStandardArgs)

function(_add_lapack_target)
  if(LAPACK_FOUND AND NOT TARGET LAPACK::LAPACK)
    add_library(LAPACK::LAPACK INTERFACE IMPORTED)

    # Filter out redundant BLAS info and replace with the BLAS target
    set(_lapack_libs "${LAPACK_LIBRARIES}")
    set(_lapack_flags "${LAPACK_LINKER_FLAGS}")
    if(TARGET BLAS::BLAS)
      if(_lapack_libs AND BLAS_LIBRARIES)
        foreach(_blas_lib IN LISTS BLAS_LIBRARIES)
          list(REMOVE_ITEM _lapack_libs "${_blas_lib}")
        endforeach()
      endif()
      if(_lapack_flags AND BLAS_LINKER_FLAGS)
        foreach(_blas_flag IN LISTS BLAS_LINKER_FLAGS)
          list(REMOVE_ITEM _lapack_flags "${_blas_flag}")
        endforeach()
      endif()
      list(APPEND _lapack_libs BLAS::BLAS)
    endif()
    if(_lapack_libs)
      set_target_properties(LAPACK::LAPACK PROPERTIES
        INTERFACE_LINK_LIBRARIES "${_lapack_libs}"
      )
    endif()
    if(_lapack_flags)
      set_target_properties(LAPACK::LAPACK PROPERTIES
        INTERFACE_LINK_OPTIONS "${_lapack_flags}"
      )
    endif()
  endif()
endfunction()

# TODO: move this stuff to a separate module

function(CHECK_LAPACK_LIBRARIES LIBRARIES _prefix _name _flags _list _deps _addlibdir _subdirs _blas)
  # This function checks for the existence of the combination of libraries
  # given by _list.  If the combination is found, this checks whether can link
  # against that library combination using the name of a routine given by _name
  # using the linker flags given by _flags.  If the combination of libraries is
  # found and passes the link test, ${LIBRARIES} is set to the list of complete
  # library paths that have been found.  Otherwise, ${LIBRARIES} is set to FALSE.

  set(_libraries_work TRUE)
  set(_libraries)
  set(_combined_name)

  if(BLA_STATIC)
    if(WIN32)
      set(CMAKE_FIND_LIBRARY_SUFFIXES .lib ${CMAKE_FIND_LIBRARY_SUFFIXES})
    else()
      set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
    endif()
  else()
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      # for ubuntu's libblas3gf and liblapack3gf packages
      set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES} .so.3gf)
    endif()
  endif()

  set(_extaddlibdir "${_addlibdir}")
  if(WIN32)
    list(APPEND _extaddlibdir ENV LIB)
  elseif(APPLE)
    list(APPEND _extaddlibdir ENV DYLD_LIBRARY_PATH)
  else()
    list(APPEND _extaddlibdir ENV LD_LIBRARY_PATH)
  endif()
  list(APPEND _extaddlibdir "${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}")

  foreach(_library ${_list})
    if(_library MATCHES "^-")
      # Respect linker flags as-is (required by MKL)
      list(APPEND _libraries "${_library}")
    else()
      string(REGEX REPLACE "[^A-Za-z0-9]" "_" _lib_var "${_library}")
      string(APPEND _combined_name "_${_lib_var}")
      if(NOT "${_deps}" STREQUAL "")
        string(APPEND _combined_name "_deps")
      endif()
      if(_libraries_work)
        find_library(${_prefix}_${_lib_var}_LIBRARY
          NAMES ${_library}
          NAMES_PER_DIR
          PATHS ${_extaddlibdir}
          PATH_SUFFIXES ${_subdirs}
        )
        mark_as_advanced(${_prefix}_${_lib_var}_LIBRARY)
        list(APPEND _libraries ${${_prefix}_${_lib_var}_LIBRARY})
        set(_libraries_work ${${_prefix}_${_lib_var}_LIBRARY})
      endif()
    endif()
  endforeach()

  foreach(_flag ${_flags})
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" _flag_var "${_flag}")
    string(APPEND _combined_name "_${_flag_var}")
  endforeach()
  if(_libraries_work)
    # Test this combination of libraries.
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${_libraries} ${_blas} ${_deps})
    set(CMAKE_REQUIRED_QUIET ${LAPACK_FIND_QUIETLY})
    if(CMAKE_Fortran_COMPILER_LOADED)
      check_fortran_function_exists("${_name}" ${_prefix}${_combined_name}_WORKS)
    else()
      check_function_exists("${_name}_" ${_prefix}${_combined_name}_WORKS)
    endif()
    set(CMAKE_REQUIRED_LIBRARIES)
    set(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  endif()

  if(_libraries_work)
    if("${_list}${_blas}" STREQUAL "")
      set(_libraries "${LIBRARIES}-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
    else()
      list(APPEND _libraries ${_blas} ${_deps})
    endif()
  else()
    set(_libraries FALSE)
  endif()
  set(${LIBRARIES} "${_libraries}" PARENT_SCOPE)
endfunction()

macro(_lapack_find_dependency dep)
  set(_lapack_quiet_arg)
  if(LAPACK_FIND_QUIETLY)
    set(_lapack_quiet_arg QUIET)
  endif()
  set(_lapack_required_arg)
  if(LAPACK_FIND_REQUIRED)
    set(_lapack_required_arg REQUIRED)
  endif()
  find_package(${dep} ${ARGN}
    ${_lapack_quiet_arg}
    ${_lapack_required_arg}
  )
  if (NOT ${dep}_FOUND)
    set(LAPACK_NOT_FOUND_MESSAGE "LAPACK could not be found because dependency ${dep} could not be found.")
  endif()

  set(_lapack_required_arg)
  set(_lapack_quiet_arg)
endmacro()

set(LAPACK_LINKER_FLAGS)
set(LAPACK_LIBRARIES)
set(LAPACK95_LIBRARIES)
set(_lapack_fphsa_req_var LAPACK_LIBRARIES)

# Check the language being used
if(NOT (CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED OR CMAKE_Fortran_COMPILER_LOADED))
  set(LAPACK_NOT_FOUND_MESSAGE
    "FindLAPACK requires Fortran, C, or C++ to be enabled.")
endif()

if(NOT BLA_SIZEOF_INTEGER)
  # in the reality we do not know which API of BLAS/LAPACK is masked in library
  set(_lapack_sizeof_integer "ANY")
elseif((BLA_SIZEOF_INTEGER STREQUAL "ANY") OR
       (BLA_SIZEOF_INTEGER STREQUAL "4") OR
       (BLA_SIZEOF_INTEGER STREQUAL "8"))
  set(_lapack_sizeof_integer ${BLA_SIZEOF_INTEGER})
else()
  message(FATAL_ERROR "BLA_SIZEOF_INTEGER can have only <no value>, ANY, 4, or 8 values")
endif()

if(NOT BLA_THREAD)
  set(_lapack_thread "ANY")
elseif((BLA_THREAD STREQUAL "ANY") OR
       (BLA_THREAD STREQUAL "SEQ") OR
       (BLA_THREAD STREQUAL "OMP"))
  set(_lapack_thread ${BLA_THREAD})
else()
  message(FATAL_ERROR "BLA_THREAD can have only <no value>, ANY, SEQ, or OMP values")
endif()

# Load BLAS
if(NOT LAPACK_NOT_FOUND_MESSAGE)
  _lapack_find_dependency(BLAS)
endif()

# Search with pkg-config if specified
if(BLA_PREFER_PKGCONFIG)
  if(NOT BLA_PKGCONFIG_LAPACK)
    set(BLA_PKGCONFIG_LAPACK "lapack")
  endif()
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(PKGC_LAPACK QUIET ${BLA_PKGCONFIG_LAPACK})
    if(PKGC_LAPACK_FOUND)
      set(LAPACK_FOUND TRUE)
      set(LAPACK_LIBRARIES "${PKGC_LAPACK_LINK_LIBRARIES}")
      if (BLAS_LIBRARIES)
        list(APPEND LAPACK_LIBRARIES "${BLAS_LIBRARIES}")
      endif()
      _add_lapack_target()
      return()
    endif()
  endif()
endif()

# Search for different LAPACK distributions if BLAS is found
if(NOT LAPACK_NOT_FOUND_MESSAGE)
  set(LAPACK_LINKER_FLAGS ${BLAS_LINKER_FLAGS})
  if(NOT BLA_VENDOR)
    if(NOT "$ENV{BLA_VENDOR}" STREQUAL "")
      set(BLA_VENDOR "$ENV{BLA_VENDOR}")
    else()
      set(BLA_VENDOR "All")
    endif()
  endif()

  # LAPACK in the Intel MKL 10+ library?
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "Intel" OR BLA_VENDOR STREQUAL "All")
      AND (CMAKE_C_COMPILER_LOADED OR CMAKE_CXX_COMPILER_LOADED))
    # System-specific settings
    if(NOT WIN32)
      set(LAPACK_mkl_LM "-lm")
      set(LAPACK_mkl_LDL "-ldl")
    endif()

    _lapack_find_dependency(Threads)

    if(_lapack_sizeof_integer EQUAL 8)
      set(LAPACK_mkl_ILP_MODE "ilp64")
    elseif(_lapack_sizeof_integer EQUAL 4)
      set(LAPACK_mkl_ILP_MODE "lp64")
    else()
      if(BLA_VENDOR MATCHES "_64ilp")
        set(LAPACK_mkl_ILP_MODE "ilp64")
      else()
        set(LAPACK_mkl_ILP_MODE "lp64")
      endif()
    endif()

    set(LAPACK_SEARCH_LIBS "")

    if(BLA_F95)
      set(LAPACK_mkl_SEARCH_SYMBOL "cheev_f95")
      set(_LAPACK_LIBRARIES LAPACK95_LIBRARIES)
      set(_BLAS_LIBRARIES ${BLAS95_LIBRARIES})

      # old
      list(APPEND LAPACK_SEARCH_LIBS
        "mkl_lapack95")
      # new >= 10.3
      list(APPEND LAPACK_SEARCH_LIBS
        "mkl_intel_c")
      list(APPEND LAPACK_SEARCH_LIBS
        "mkl_lapack95_${LAPACK_mkl_ILP_MODE}")
    else()
      set(LAPACK_mkl_SEARCH_SYMBOL "cheev")
      set(_LAPACK_LIBRARIES LAPACK_LIBRARIES)
      set(_BLAS_LIBRARIES ${BLAS_LIBRARIES})

      # old and new >= 10.3
      list(APPEND LAPACK_SEARCH_LIBS
        "mkl_lapack")
    endif()

    # MKL uses a multitude of partially platform-specific subdirectories:
    if(BLA_VENDOR STREQUAL "Intel10_32")
      set(LAPACK_mkl_ARCH_NAME "ia32")
    else()
      set(LAPACK_mkl_ARCH_NAME "intel64")
    endif()
    if(WIN32)
      set(LAPACK_mkl_OS_NAME "win")
    elseif(APPLE)
      set(LAPACK_mkl_OS_NAME "mac")
    else()
      set(LAPACK_mkl_OS_NAME "lin")
    endif()
    if(DEFINED ENV{MKLROOT})
      file(TO_CMAKE_PATH "$ENV{MKLROOT}" LAPACK_mkl_MKLROOT)
      # If MKLROOT points to the subdirectory 'mkl', use the parent directory instead
      # so we can better detect other relevant libraries in 'compiler' or 'tbb':
      get_filename_component(LAPACK_mkl_MKLROOT_LAST_DIR "${LAPACK_mkl_MKLROOT}" NAME)
      if(LAPACK_mkl_MKLROOT_LAST_DIR STREQUAL "mkl")
          get_filename_component(LAPACK_mkl_MKLROOT "${LAPACK_mkl_MKLROOT}" DIRECTORY)
      endif()
    endif()
    set(LAPACK_mkl_LIB_PATH_SUFFIXES
        "compiler/lib" "compiler/lib/${LAPACK_mkl_ARCH_NAME}_${LAPACK_mkl_OS_NAME}"
        "compiler/lib/${LAPACK_mkl_ARCH_NAME}"
        "mkl/lib" "mkl/lib/${LAPACK_mkl_ARCH_NAME}_${LAPACK_mkl_OS_NAME}"
        "mkl/lib/${LAPACK_mkl_ARCH_NAME}"
        "lib" "lib/${LAPACK_mkl_ARCH_NAME}_${LAPACK_mkl_OS_NAME}"
        "lib/${LAPACK_mkl_ARCH_NAME}"
        )

    # First try empty lapack libs (implicitly linked or automatic from BLAS)
    if(NOT ${_LAPACK_LIBRARIES})
      check_lapack_libraries(
        ${_LAPACK_LIBRARIES}
        LAPACK
        ${LAPACK_mkl_SEARCH_SYMBOL}
        ""
        ""
        "${CMAKE_THREAD_LIBS_INIT};${LAPACK_mkl_LM};${LAPACK_mkl_LDL}"
        "${LAPACK_mkl_MKLROOT}"
        "${LAPACK_mkl_LIB_PATH_SUFFIXES}"
        "${_BLAS_LIBRARIES}"
      )
      if(LAPACK_WORKS AND NOT _BLAS_LIBRARIES)
        # Give a more helpful "found" message
        set(LAPACK_WORKS "implicitly linked")
        set(_lapack_fphsa_req_var LAPACK_WORKS)
      endif()
    endif()

    # Then try the search libs
    foreach(_search ${LAPACK_SEARCH_LIBS})
      string(REPLACE " " ";" _search ${_search})
      if(NOT ${_LAPACK_LIBRARIES})
        check_lapack_libraries(
          ${_LAPACK_LIBRARIES}
          LAPACK
          ${LAPACK_mkl_SEARCH_SYMBOL}
          ""
          "${_search}"
          "${CMAKE_THREAD_LIBS_INIT};${LAPACK_mkl_LM};${LAPACK_mkl_LDL}"
          "${LAPACK_mkl_MKLROOT}"
          "${LAPACK_mkl_LIB_PATH_SUFFIXES}"
          "${_BLAS_LIBRARIES}"
        )
      endif()
    endforeach()

    unset(_search)
    unset(LAPACK_mkl_ILP_MODE)
    unset(LAPACK_mkl_SEARCH_SYMBOL)
    unset(LAPACK_mkl_LM)
    unset(LAPACK_mkl_LDL)
    unset(LAPACK_mkl_MKLROOT)
    unset(LAPACK_mkl_ARCH_NAME)
    unset(LAPACK_mkl_OS_NAME)
    unset(LAPACK_mkl_LIB_PATH_SUFFIXES)
  endif()

  # gotoblas? (http://www.tacc.utexas.edu/tacc-projects/gotoblas2)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR STREQUAL "Goto" OR BLA_VENDOR STREQUAL "All"))
    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "goto2"
      ""
      ""
      ""
      "${BLAS_LIBRARIES}"
    )
  endif()

  # FlexiBLAS? (http://www.mpi-magdeburg.mpg.de/mpcsc/software/FlexiBLAS/)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR STREQUAL "FlexiBLAS" OR BLA_VENDOR STREQUAL "All"))
    set(_lapack_flexiblas_lib "flexiblas")

    if(_lapack_sizeof_integer EQUAL 8)
      string(APPEND _lapack_flexiblas_lib "64")
    endif()

    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "${_lapack_flexiblas_lib}"
      ""
      ""
      ""
      "${BLAS_LIBRARIES}"
    )

    unset(_lapack_flexiblas_lib)
  endif()

  # OpenBLAS? (http://www.openblas.net)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR STREQUAL "OpenBLAS" OR BLA_VENDOR STREQUAL "All"))
    set(_lapack_openblas_lib "openblas")

    if(_lapack_sizeof_integer EQUAL 8)
      if(MINGW)
        string(APPEND _lapack_openblas_lib "_64")
      else()
        string(APPEND _lapack_openblas_lib "64")
      endif()
    endif()

    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "${_lapack_openblas_lib}"
      ""
      ""
      ""
      "${BLAS_LIBRARIES}"
    )

    unset(_lapack_openblas_lib)
  endif()

  # ArmPL? (https://developer.arm.com/tools-and-software/server-and-hpc/compile/arm-compiler-for-linux/arm-performance-libraries)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "Arm" OR BLA_VENDOR STREQUAL "All"))
    # Check for 64bit Integer support
    if(_lapack_sizeof_integer EQUAL 8)
      set(LAPACK_armpl_LIB "armpl_ilp64")
    elseif(_lapack_sizeof_integer EQUAL 4)
      set(LAPACK_armpl_LIB "armpl_lp64")
    else()
      if(BLA_VENDOR MATCHES "_ilp64")
        set(LAPACK_armpl_LIB "armpl_ilp64")
      else()
        set(LAPACK_armpl_LIB "armpl_lp64")
      endif()
    endif()

    # Check for OpenMP support, VIA BLA_VENDOR of Arm_mp or Arm_ipl64_mp
    if(BLA_VENDOR MATCHES "_mp")
      string(APPEND LAPACK_armpl_LIB "_mp")
    endif()

    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "${LAPACK_armpl_LIB}"
      ""
      ""
      ""
      "${BLAS_LIBRARIES}"
    )
  endif()

  # FLAME's blis library? (https://github.com/flame/blis)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR STREQUAL "FLAME" OR BLA_VENDOR STREQUAL "All"))
    if(_lapack_sizeof_integer EQUAL 8)
      if(BLA_VENDOR STREQUAL "FLAME")
        message(FATAL_ERROR "libFLAME does not support Int64 type")
      endif()
    else()
      check_lapack_libraries(
        LAPACK_LIBRARIES
        LAPACK
        cheev
        ""
        "flame"
        ""
        ""
        ""
        "${BLAS_LIBRARIES}"
      )
    endif()
  endif()

  # AOCL? (https://developer.amd.com/amd-aocl/)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "AOCL" OR BLA_VENDOR STREQUAL "All"))
    if(_lapack_sizeof_integer EQUAL 8)
      set(_lapack_aocl_subdir "ILP64")
    else()
      set(_lapack_aocl_subdir "LP64")
    endif()

    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "flame"
      "-fopenmp"
      ""
      "${_lapack_aocl_subdir}"
      "${BLAS_LIBRARIES}"
    )
    unset(_lapack_aocl_subdir)
  endif()

  # LAPACK in SCSL library? (SGI/Cray Scientific Library)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "SCSL" OR BLA_VENDOR STREQUAL "All"))
    set(_lapack_scsl_lib "scs")

    if(_lapack_sizeof_integer EQUAL 8)
      string(APPEND _lapack_scsl_lib "_i8")
    endif()
    # Check for OpenMP support, VIA BLA_VENDOR of scs_mp
    if(BLA_VENDOR MATCHES "_mp")
      string(APPEND _lapack_scsl_lib "_mp")
    endif()

    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "${_lapack_scsl_lib}"
      ""
      ""
      ""
      "${BLAS_LIBRARIES}"
    )
    unset(_lapack_scsl_lib)
  endif()

  # BLAS in acml library?
  if(BLA_VENDOR MATCHES "ACML" OR BLA_VENDOR STREQUAL "All")
    if(BLAS_LIBRARIES MATCHES ".+acml.+")
      set(LAPACK_LIBRARIES ${BLAS_LIBRARIES})
    endif()
  endif()

  # Apple LAPACK library?
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR STREQUAL "Apple" OR BLA_VENDOR STREQUAL "All"))
    if(_lapack_sizeof_integer EQUAL 8)
      if(BLA_VENDOR STREQUAL "Apple")
        message(FATAL_ERROR "Accelerate Framework does not support Int64 type")
      endif()
    else()
      check_lapack_libraries(
        LAPACK_LIBRARIES
        LAPACK
        cheev
        ""
        "Accelerate"
        ""
        ""
        ""
        "${BLAS_LIBRARIES}"
      )
    endif()
  endif()

  # Apple NAS (vecLib) library?
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR STREQUAL "NAS" OR BLA_VENDOR STREQUAL "All"))
    if(_lapack_sizeof_integer EQUAL 8)
      if(BLA_VENDOR STREQUAL "NAS")
        message(FATAL_ERROR "Accelerate Framework does not support Int64 type")
      endif()
    else()
      check_lapack_libraries(
        LAPACK_LIBRARIES
        LAPACK
        cheev
        ""
        "vecLib"
        ""
        ""
        ""
        "${BLAS_LIBRARIES}"
      )
    endif()
  endif()

  # Elbrus Math Library?
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "EML" OR BLA_VENDOR STREQUAL "All"))
    if(BLAS_LIBRARIES MATCHES "eml.+")
      set(LAPACK_LIBRARIES ${BLAS_LIBRARIES})
    endif()
  endif()

  # Fujitsu SSL2 Library?
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "Fujitsu_SSL2" OR BLA_VENDOR STREQUAL "All"))
    if(BLAS_LIBRARIES MATCHES "fjlapack.+")
      set(LAPACK_LIBRARIES ${BLAS_LIBRARIES})
      set(LAPACK_LINKER_FLAGS ${BLAS_LINKER_FLAGS})
    endif()
  endif()

  # LAPACK in IBM ESSL library?
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "IBMESSL" OR BLA_VENDOR STREQUAL "All"))
    if(BLAS_LIBRARIES MATCHES "essl.+")
      set(LAPACK_LIBRARIES ${BLAS_LIBRARIES})
    endif()
  endif()

  # nVidia NVPL? (https://developer.nvidia.com/nvpl)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "NVPL" OR BLA_VENDOR STREQUAL "All"))
    # Prefer lp64 unless ilp64 is requested.
    if((_lapack_sizeof_integer EQUAL 4) OR (_lapack_sizeof_integer STREQUAL "ANY"))
      list(APPEND _lapack_nvpl_ints "_lp64")
    endif()
    if((_lapack_sizeof_integer EQUAL 8) OR (_lapack_sizeof_integer STREQUAL "ANY"))
      list(APPEND _lapack_nvpl_ints "_ilp64")
    endif()

    # Prefer OMP if available
    if((_lapack_thread STREQUAL "OMP") OR (_lapack_thread STREQUAL "ANY"))
      list(APPEND _lapack_nvpl_threads "_omp")
    endif()
    if((_lapack_thread STREQUAL "SEQ") OR (_lapack_thread STREQUAL "ANY"))
      list(APPEND _lapack_nvpl_threads "_seq")
    endif()

    find_package(nvpl)
    if(nvpl_FOUND)
      foreach(_nvpl_thread IN LISTS _lapack_nvpl_threads)
        foreach(_nvpl_int IN LISTS _lapack_nvpl_ints)

          set(_lapack_lib "nvpl::lapack${_nvpl_int}${_nvpl_thread}")

          if(TARGET ${_lapack_lib})
            set(LAPACK_LIBRARIES ${_lapack_lib})
            break()
          endif()

        endforeach()

        if(LAPACK_LIBRARIES)
          break()
        endif()

      endforeach()
    endif()

    unset(_lapack_lib)
    unset(_lapack_nvpl_ints)
    unset(_lapack_nvpl_threads)
  endif()

  # NVHPC Library?

  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR MATCHES "NVHPC" OR BLA_VENDOR STREQUAL "All"))
    set(_lapack_nvhpc_lib "lapack")

    if(_lapack_sizeof_integer EQUAL 8)
      string(APPEND _lapack_nvhpc_lib "_ilp64")
    elseif(_lapack_sizeof_integer EQUAL 4)
      string(APPEND _lapack_nvhpc_lib "_lp64")
    endif()
    set(_lapack_nvhpc_flags)
    if(";${CMAKE_C_COMPILER_ID};${CMAKE_CXX_COMPILER_ID};${CMAKE_Fortran_COMPILER_ID};" MATCHES ";(NVHPC|PGI);")
      set(_lapack_nvhpc_flags "-fortranlibs")
    endif()

    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "${_lapack_nvhpc_lib}"
      "${_lapack_nvhpc_flags}"
      ""
      ""
      "${BLAS_LIBRARIES}"
    )

    # an additional check for NVHPC 2020
    # which does not have differentiation
    # between lp64 and ilp64 modes
    if(NOT LAPACK_LIBRARIES AND NOT _lapack_sizeof_integer EQUAL 8)
      set(_lapack_nvhpc_lib "lapack")

      check_lapack_libraries(
        LAPACK_LIBRARIES
        LAPACK
        cheev
        ""
        "${_lapack_nvhpc_lib}"
        "${_lapack_nvhpc_flags}"
        ""
        ""
        "${BLAS_LIBRARIES}"
      )
    endif()

    unset(_lapack_nvhpc_lib)
    unset(_lapack_nvhpc_flags)
  endif()

  # libblastrampoline? (https://github.com/JuliaLinearAlgebra/libblastrampoline/tree/main)
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR STREQUAL "libblastrampoline" OR BLA_VENDOR STREQUAL "All"))
    if(BLAS_LIBRARIES MATCHES "blastrampoline.+")
      set(LAPACK_LIBRARIES ${BLAS_LIBRARIES})
    endif()
  endif()

  # Generic LAPACK library?
  if(NOT LAPACK_LIBRARIES
      AND (BLA_VENDOR STREQUAL "Generic"
           OR BLA_VENDOR STREQUAL "ATLAS"
           OR BLA_VENDOR STREQUAL "All"))
    set(_lapack_generic_lib "lapack")
    if(BLA_STATIC)
      # We do not know for sure how the LAPACK reference implementation
      # is built on this host.  Guess typical dependencies.
      set(_lapack_generic_deps "-lgfortran;-lm")
    else()
      set(_lapack_generic_deps "")
    endif()

    if(_lapack_sizeof_integer EQUAL 8)
      string(APPEND _lapack_generic_lib "64")
    endif()

    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "${_lapack_generic_lib}"
      "${_lapack_generic_deps}"
      ""
      ""
      "${BLAS_LIBRARIES}"
    )

    unset(_lapack_generic_deps)
    unset(_lapack_generic_lib)
  endif()
endif()

if(BLA_F95)
  set(LAPACK_LIBRARIES "${LAPACK95_LIBRARIES}")
endif()

if(LAPACK_NOT_FOUND_MESSAGE)
  set(LAPACK_NOT_FOUND_MESSAGE
    REASON_FAILURE_MESSAGE ${LAPACK_NOT_FOUND_MESSAGE})
endif()
find_package_handle_standard_args(LAPACK REQUIRED_VARS ${_lapack_fphsa_req_var}
  ${LAPACK_NOT_FOUND_MESSAGE})
unset(LAPACK_NOT_FOUND_MESSAGE)

if(BLA_F95)
  set(LAPACK95_FOUND ${LAPACK_FOUND})
endif()

# On compilers that implicitly link LAPACK (such as ftn, cc, and CC on Cray HPC machines)
# we used a placeholder for empty LAPACK_LIBRARIES to get through our logic above.
if(LAPACK_LIBRARIES STREQUAL "LAPACK_LIBRARIES-PLACEHOLDER-FOR-EMPTY-LIBRARIES")
  set(LAPACK_LIBRARIES "")
endif()

_add_lapack_target()
unset(_lapack_fphsa_req_var)
unset(_lapack_sizeof_integer)
unset(_LAPACK_LIBRARIES)
