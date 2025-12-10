# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGTest
---------

Finds GoogleTest, the Google C++ testing and mocking framework:

.. code-block:: cmake

  find_package(GTest [...])

The GoogleTest framework also includes GoogleMock, a library for writing
and using C++ mock classes.  On some systems, GoogleMock may be distributed
as a separate package.

When both debug and release (optimized) variants of the GoogleTest and
GoogleMock libraries are available, this module selects the appropriate
variants based on the current :ref:`Build Configuration <Build Configurations>`.

.. versionadded:: 3.20
  If GoogleTest is built and installed using its CMake-based build system, it
  provides a :ref:`package configuration file <Config File Packages>`
  (``GTestConfig.cmake``) that can be used with :command:`find_package` in
  :ref:`Config mode`.  By default, this module now searches for that
  configuration file and, if found, returns the results without further
  action.  If the upstream configuration file is not found, this module falls
  back to :ref:`Module mode` and searches standard locations.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``GTest::gtest``
  .. versionadded:: 3.20

  Target encapsulating the usage requirements of the GoogleTest ``gtest``
  library, available if GoogleTest is found.  The ``gtest`` library provides
  the core GoogleTest testing framework functionality.

``GTest::gtest_main``
  .. versionadded:: 3.20

  Target encapsulating the usage requirements of the GoogleTest ``gtest_main``
  library, available if GoogleTest is found.  The ``gtest_main`` library
  provides a ``main()`` function, allowing tests to be run without defining
  one manually.

  Only link to ``GTest::gtest_main`` if GoogleTest should supply the
  ``main()`` function for the executable.  If the project is supplying its
  own ``main()`` implementation, link only to ``GTest::gtest``.

``GTest::gmock``
  .. versionadded:: 3.23

  Target encapsulating the usage requirements of the GoogleMock ``gmock``
  library, available if GoogleTest and its Mock library are found.  The
  ``gmock`` library provides facilities for writing and using mock classes
  in C++.

``GTest::gmock_main``
  .. versionadded:: 3.23

  Target encapsulating the usage requirements of the GoogleMock ``gmock_main``
  library, available if GoogleTest and ``gmock_main`` are found.  The
  ``gmock_main`` library provides a ``main()`` function, allowing GoogleMock
  tests to be run without defining one manually.

  Only link to ``GTest::gmock_main`` if GoogleTest should supply the
  ``main()`` function for the executable.  If project is supplying its own
  ``main()`` implementation, link only to ``GTest::gmock``.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``GTest_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether GoogleTest was found.

Hints
^^^^^

This module accepts the following variables before calling
``find_package(GTest)``:

``GTEST_ROOT``
  The root directory of the GoogleTest installation (may also be set as an
  environment variable).  This variable is used only when GoogleTest is found
  in :ref:`Module mode`.

``GTEST_MSVC_SEARCH``
  When compiling with MSVC, this variable controls which GoogleTest build
  variant to search for, based on the runtime library linkage model.  This
  variable is used only when GoogleTest is found in :ref:`Module mode` and
  accepts one of the following values:

  ``MD``
    (Default) Searches for shared library variants of GoogleTest that are
    built to link against the dynamic C runtime.  These libraries are
    typically compiled with the MSVC runtime flags ``/MD`` or ``/MDd`` (for
    Release or Debug, respectively).

  ``MT``
    Searches for static library variants of GoogleTest that are built to
    link against the static C runtime.  These libraries are typically
    compiled with the MSVC runtime flags ``/MT`` or ``/MTd``.

Deprecated Items
^^^^^^^^^^^^^^^^

Deprecated Variables
""""""""""""""""""""

The following variables are provided for backward compatibility:

``GTEST_INCLUDE_DIRS``
  .. deprecated:: 4.1
    Use the ``GTest::gtest`` imported target instead, which exposes the
    required include directories through its
    :prop_tgt:`INTERFACE_INCLUDE_DIRECTORIES` target property.

  Result variable that provides include directories containing headers
  needed to use GoogleTest.  This variable is only guaranteed to be available
  when GoogleTest is found in :ref:`Module mode`.

``GTEST_LIBRARIES``
  .. deprecated:: 4.1
    Use the ``GTest::gtest`` imported target instead.

  Result variable providing libraries needed to link against to use the
  GoogleTest ``gtest`` library.  Note that projects are also responsible
  for linking with an appropriate thread library in addition to the libraries
  specified by this variable.

``GTEST_MAIN_LIBRARIES``
  .. deprecated:: 4.1
    Use the ``GTest::gtest_main`` imported target instead.

  Result variable providing libraries needed to link against to use the
  GoogleTest ``gtest_main`` library.

``GTEST_BOTH_LIBRARIES``
  .. deprecated:: 4.1
    Use the ``GTest::gtest`` and ``GTest::gtest_main`` imported targets
    instead.

  Result variable providing both ``gtest`` and ``gtest_main`` libraries
  combined.

``GTEST_FOUND``
  .. deprecated:: 4.2
    Use ``GTest_FOUND``, which has the same value.

  Boolean indicating whether GoogleTest was found.

Deprecated Imported Targets
"""""""""""""""""""""""""""

For backward compatibility, this module also provides the following imported
targets (available since CMake 3.5):

``GTest::GTest``
  .. deprecated:: 3.20
    Use the ``GTest::gtest`` imported target instead.

  Imported target linking the ``GTest::gtest`` library.

``GTest::Main``
  .. deprecated:: 3.20
    Use the ``GTest::gtest_main`` imported target instead.

  Imported target linking the ``GTest::gtest_main`` library.

Examples
^^^^^^^^

Examples: Finding GoogleTest
""""""""""""""""""""""""""""

Finding GoogleTest:

.. code-block:: cmake

  find_package(GoogleTest)

Or, finding GoogleTest and making it required (if not found, processing stops
with an error message):

.. code-block:: cmake

  find_package(GoogleTest REQUIRED)

Examples: Using Imported Targets
""""""""""""""""""""""""""""""""

In the following example, the ``GTest::gtest`` imported target is linked to
a project target, which enables using the core GoogleTest testing framework:

.. code-block:: cmake

  find_package(GTest REQUIRED)

  target_link_libraries(foo PRIVATE GTest::gtest)

In the next example, the ``GTest::gtest_main`` imported target is also linked
to the executable, and a test is registered.  The ``GTest::gtest_main`` library
provides a ``main()`` function, so there is no need to write one manually.
The ``GTest::gtest`` library is still linked because the test code directly
uses things provided by ``GTest::gtest``, and good practice is to link directly
to libraries used directly.

.. code-block:: cmake

  enable_testing()

  find_package(GTest REQUIRED)

  add_executable(foo foo.cc)
  target_link_libraries(foo PRIVATE GTest::gtest GTest::gtest_main)

  add_test(NAME AllTestsInFoo COMMAND foo)

Deeper Integration With CTest
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This module is commonly used with the :module:`GoogleTest` module, which
provides :command:`gtest_discover_tests` and :command:`gtest_add_tests`
commands to help integrate GoogleTest infrastructure with CTest:

.. code-block:: cmake

  find_package(GTest)
  target_link_libraries(example PRIVATE GTest::gtest GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(example)

  # ...

.. versionchanged:: 3.9
  Previous CMake versions defined the :command:`gtest_add_tests` command in
  this module.
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/GoogleTest.cmake)

function(__gtest_append_debugs _endvar _library)
    if(${_library} AND ${_library}_DEBUG)
        set(_output optimized ${${_library}} debug ${${_library}_DEBUG})
    else()
        set(_output ${${_library}})
    endif()
    set(${_endvar} ${_output} PARENT_SCOPE)
endfunction()

function(__gtest_find_library _name)
    find_library(${_name}
        NAMES ${ARGN}
        HINTS
            ENV GTEST_ROOT
            ${GTEST_ROOT}
        PATH_SUFFIXES ${_gtest_libpath_suffixes}
    )
    mark_as_advanced(${_name})
endfunction()

macro(__gtest_determine_windows_library_type _var)
    if(EXISTS "${${_var}}")
        file(TO_NATIVE_PATH "${${_var}}" _lib_path)
        get_filename_component(_name "${${_var}}" NAME_WE)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
        file(STRINGS "${${_var}}" _match REGEX "${_name}\\.dll" LIMIT_COUNT 1)
        cmake_policy(POP)
        if(NOT _match STREQUAL "")
            set(${_var}_TYPE SHARED PARENT_SCOPE)
        else()
            set(${_var}_TYPE UNKNOWN PARENT_SCOPE)
        endif()
        return()
    endif()
endmacro()

function(__gtest_determine_library_type _var)
    if(WIN32)
        # For now, at least, only Windows really needs to know the library type
        __gtest_determine_windows_library_type(${_var})
        __gtest_determine_windows_library_type(${_var}_RELEASE)
        __gtest_determine_windows_library_type(${_var}_DEBUG)
    endif()
    # If we get here, no determination was made from the above checks
    set(${_var}_TYPE UNKNOWN PARENT_SCOPE)
endfunction()

function(__gtest_import_library _target _var _config)
    if(_config)
        set(_config_suffix "_${_config}")
    else()
        set(_config_suffix "")
    endif()

    set(_lib "${${_var}${_config_suffix}}")
    if(EXISTS "${_lib}")
        if(_config)
            set_property(TARGET ${_target} APPEND PROPERTY
                IMPORTED_CONFIGURATIONS ${_config})
        endif()
        set_target_properties(${_target} PROPERTIES
            IMPORTED_LINK_INTERFACE_LANGUAGES${_config_suffix} "CXX")
        if(WIN32 AND ${_var}_TYPE STREQUAL SHARED)
            set_target_properties(${_target} PROPERTIES
                IMPORTED_IMPLIB${_config_suffix} "${_lib}")
        else()
            set_target_properties(${_target} PROPERTIES
                IMPORTED_LOCATION${_config_suffix} "${_lib}")
        endif()
    endif()
endfunction()

function(__gtest_define_backwards_compatible_library_targets)
    set(GTEST_BOTH_LIBRARIES ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} PARENT_SCOPE)

    # Add targets mapping the same library names as defined in
    # older versions of CMake's FindGTest
    if(NOT TARGET GTest::GTest)
        add_library(GTest::GTest INTERFACE IMPORTED)
        target_link_libraries(GTest::GTest INTERFACE GTest::gtest)
    endif()
    if(NOT TARGET GTest::Main)
        add_library(GTest::Main INTERFACE IMPORTED)
        target_link_libraries(GTest::Main INTERFACE GTest::gtest_main)
    endif()
endfunction()

#

include(FindPackageHandleStandardArgs)

# first specifically look for the CMake version of GTest
find_package(GTest QUIET NO_MODULE)

# if we found the GTest cmake package then we are done, and
# can print what we found and return.
if(GTest_FOUND)
    find_package_handle_standard_args(GTest HANDLE_COMPONENTS CONFIG_MODE)

    set(GTEST_LIBRARIES      GTest::gtest)
    set(GTEST_MAIN_LIBRARIES GTest::gtest_main)

    __gtest_define_backwards_compatible_library_targets()

    return()
endif()

if(NOT DEFINED GTEST_MSVC_SEARCH)
    set(GTEST_MSVC_SEARCH MD)
endif()

set(_gtest_libpath_suffixes lib)
if(MSVC)
    if(GTEST_MSVC_SEARCH STREQUAL "MD")
        list(APPEND _gtest_libpath_suffixes
            msvc/gtest-md/Debug
            msvc/gtest-md/Release
            msvc/x64/Debug
            msvc/x64/Release
            msvc/2010/gtest-md/Win32-Debug
            msvc/2010/gtest-md/Win32-Release
            msvc/2010/gtest-md/x64-Debug
            msvc/2010/gtest-md/x64-Release
            )
    elseif(GTEST_MSVC_SEARCH STREQUAL "MT")
        list(APPEND _gtest_libpath_suffixes
            msvc/gtest/Debug
            msvc/gtest/Release
            msvc/x64/Debug
            msvc/x64/Release
            msvc/2010/gtest/Win32-Debug
            msvc/2010/gtest/Win32-Release
            msvc/2010/gtest/x64-Debug
            msvc/2010/gtest/x64-Release
            )
    endif()
endif()


find_path(GTEST_INCLUDE_DIR gtest/gtest.h
    HINTS
        $ENV{GTEST_ROOT}/include
        ${GTEST_ROOT}/include
)
mark_as_advanced(GTEST_INCLUDE_DIR)

if(MSVC AND GTEST_MSVC_SEARCH STREQUAL "MD")
    # The provided /MD project files for Google Test add -md suffixes to the
    # library names.
    __gtest_find_library(GTEST_LIBRARY            gtest-md  gtest)
    __gtest_find_library(GTEST_LIBRARY_DEBUG      gtest-mdd gtestd)
    __gtest_find_library(GTEST_MAIN_LIBRARY       gtest_main-md  gtest_main)
    __gtest_find_library(GTEST_MAIN_LIBRARY_DEBUG gtest_main-mdd gtest_maind)
    __gtest_find_library(GMOCK_LIBRARY            gmock-md  gmock)
    __gtest_find_library(GMOCK_LIBRARY_DEBUG      gmock-mdd gmockd)
    __gtest_find_library(GMOCK_MAIN_LIBRARY       gmock_main-md  gmock_main)
    __gtest_find_library(GMOCK_MAIN_LIBRARY_DEBUG gmock_main-mdd gmock_maind)
else()
    __gtest_find_library(GTEST_LIBRARY            gtest)
    __gtest_find_library(GTEST_LIBRARY_DEBUG      gtestd)
    __gtest_find_library(GTEST_MAIN_LIBRARY       gtest_main)
    __gtest_find_library(GTEST_MAIN_LIBRARY_DEBUG gtest_maind)
    __gtest_find_library(GMOCK_LIBRARY            gmock)
    __gtest_find_library(GMOCK_LIBRARY_DEBUG      gmockd)
    __gtest_find_library(GMOCK_MAIN_LIBRARY       gmock_main)
    __gtest_find_library(GMOCK_MAIN_LIBRARY_DEBUG gmock_maind)
endif()

find_package_handle_standard_args(GTest DEFAULT_MSG GTEST_LIBRARY GTEST_INCLUDE_DIR GTEST_MAIN_LIBRARY)

if(GMOCK_LIBRARY AND GMOCK_MAIN_LIBRARY)
    set(GMock_FOUND True)
else()
    set(GMock_FOUND False)
endif()

if(GTest_FOUND)
    set(GTEST_INCLUDE_DIRS ${GTEST_INCLUDE_DIR})
    __gtest_append_debugs(GTEST_LIBRARIES      GTEST_LIBRARY)
    __gtest_append_debugs(GTEST_MAIN_LIBRARIES GTEST_MAIN_LIBRARY)

    find_package(Threads QUIET)

    if(NOT TARGET GTest::gtest)
        __gtest_determine_library_type(GTEST_LIBRARY)
        add_library(GTest::gtest ${GTEST_LIBRARY_TYPE} IMPORTED)
        if(TARGET Threads::Threads)
            set_target_properties(GTest::gtest PROPERTIES
                INTERFACE_LINK_LIBRARIES Threads::Threads)
        endif()
        if(GTEST_LIBRARY_TYPE STREQUAL "SHARED")
            set_target_properties(GTest::gtest PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS "GTEST_LINKED_AS_SHARED_LIBRARY=1")
        endif()
        if(GTEST_INCLUDE_DIRS)
            set_target_properties(GTest::gtest PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}")
        endif()
        __gtest_import_library(GTest::gtest GTEST_LIBRARY "")
        __gtest_import_library(GTest::gtest GTEST_LIBRARY "RELEASE")
        __gtest_import_library(GTest::gtest GTEST_LIBRARY "DEBUG")
    endif()
    if(NOT TARGET GTest::gtest_main)
        __gtest_determine_library_type(GTEST_MAIN_LIBRARY)
        add_library(GTest::gtest_main ${GTEST_MAIN_LIBRARY_TYPE} IMPORTED)
        set_target_properties(GTest::gtest_main PROPERTIES
            INTERFACE_LINK_LIBRARIES "GTest::gtest")
        __gtest_import_library(GTest::gtest_main GTEST_MAIN_LIBRARY "")
        __gtest_import_library(GTest::gtest_main GTEST_MAIN_LIBRARY "RELEASE")
        __gtest_import_library(GTest::gtest_main GTEST_MAIN_LIBRARY "DEBUG")
    endif()

    __gtest_define_backwards_compatible_library_targets()
endif()

if(GMock_FOUND AND GTest_FOUND)
    if(NOT TARGET GTest::gmock)
        __gtest_determine_library_type(GMOCK_LIBRARY)
        add_library(GTest::gmock ${GMOCK_LIBRARY_TYPE} IMPORTED)
        set(_gmock_link_libraries "GTest::gtest")
        if(TARGET Threads::Threads)
            list(APPEND _gmock_link_libraries Threads::Threads)
        endif()
        set_target_properties(GTest::gmock PROPERTIES
            INTERFACE_LINK_LIBRARIES "${_gmock_link_libraries}")
        if(GMOCK_LIBRARY_TYPE STREQUAL "SHARED")
            set_target_properties(GTest::gmock PROPERTIES
                INTERFACE_COMPILE_DEFINITIONS "GMOCK_LINKED_AS_SHARED_LIBRARY=1")
        endif()
        if(GTEST_INCLUDE_DIRS)
            set_target_properties(GTest::gmock PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIRS}")
        endif()
        __gtest_import_library(GTest::gmock GMOCK_LIBRARY "")
        __gtest_import_library(GTest::gmock GMOCK_LIBRARY "RELEASE")
        __gtest_import_library(GTest::gmock GMOCK_LIBRARY "DEBUG")
    endif()
    if(NOT TARGET GTest::gmock_main)
        __gtest_determine_library_type(GMOCK_MAIN_LIBRARY)
        add_library(GTest::gmock_main ${GMOCK_MAIN_LIBRARY_TYPE} IMPORTED)
        set_target_properties(GTest::gmock_main PROPERTIES
            INTERFACE_LINK_LIBRARIES "GTest::gmock")
        __gtest_import_library(GTest::gmock_main GMOCK_MAIN_LIBRARY "")
        __gtest_import_library(GTest::gmock_main GMOCK_MAIN_LIBRARY "RELEASE")
        __gtest_import_library(GTest::gmock_main GMOCK_MAIN_LIBRARY "DEBUG")
    endif()
endif()
