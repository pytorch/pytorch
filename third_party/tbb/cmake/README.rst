.. contents::

Introduction
------------
Many developers use CMake to manage their development projects, so the Intel(R) Threading Building Blocks (Intel(R) TBB)
team created the set of CMake modules to simplify integration of the Intel TBB library into a CMake project.
The modules are available starting from Intel TBB 2017 U7 in `<tbb_root>/cmake <https://github.com/01org/tbb/tree/tbb_2017/cmake>`_.

About Intel TBB
^^^^^^^^^^^^^^^
Intel TBB is a library that supports scalable parallel programming using standard ISO C++ code. It does not require special languages or compilers. It is designed to promote scalable data parallel programming. Additionally, it fully supports nested parallelism, so you can build larger parallel components from smaller parallel components. To use the library, you specify tasks, not threads, and let the library map tasks onto threads in an efficient manner.

Many of the library interfaces employ generic programming, in which interfaces are defined by requirements on types and not specific types. The C++ Standard Template Library (STL) is an example of generic programming. Generic programming enables Intel TBB to be flexible yet efficient. The generic interfaces enable you to customize components to your specific needs.

The net result is that Intel TBB enables you to specify parallelism far more conveniently than using raw threads, and at the same time can improve performance.

References
^^^^^^^^^^
* `Official Intel TBB open source site <https://www.threadingbuildingblocks.org/>`_
* `Official GitHub repository <https://github.com/01org/tbb>`_

Engineering team contacts
^^^^^^^^^^^^^^^^^^^^^^^^^
The Intel TBB team is very interested in convenient integration of the Intel TBB library into customer projects. These CMake modules were created to provide such a possibility for CMake projects using a simple but powerful interface. We hope you will try these modules and we are looking forward to receiving your feedback!

E-mail us: `inteltbbdevelopers@intel.com <mailto:inteltbbdevelopers@intel.com>`_.

Visit our `forum <https://software.intel.com/en-us/forums/intel-threading-building-blocks/>`_.

Release Notes
-------------
* Minimum supported CMake version: ``3.0.0``.
* Intel TBB versioning via `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_ has restricted functionality: compatibility of update numbers (as well as interface versions) is not checked. Supported versioning: ``find_package(TBB <major>.<minor> ...)``. Intel TBB interface version can be obtained in the customer project via the ``TBB_INTERFACE_VERSION`` variable.

Use cases of Intel TBB integration into CMake-aware projects
------------------------------------------------------------
There are two types of Intel TBB packages:
 * Binary packages with pre-built binaries for Windows* OS, Linux* OS and macOS*. They are available on the releases page of the Github repository: https://github.com/01org/tbb/releases. The main purpose of the binary package integration is the ability to build Intel TBB header files and binaries into your CMake-aware project.
 * A source package is also available to download from the release page via the "Source code" link. In addition, it can be cloned from the repository by ``git clone https://github.com/01org/tbb.git``. The main purpose of the source package integration is to allow you to do a custom build of the Intel TBB library from the source files and then build that into your CMake-aware project.

There are four types of CMake modules that can be used to integrate Intel TBB: `TBBConfig`, `TBBGet`, `TBBMakeConfig` and `TBBBuild`. See `Technical documentation for CMake modules`_ section for additional details.

Binary package integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following use case is valid for packages starting from Intel TBB 2017 U7:

* Download package manually and make integration.

 Pre-condition: Location of TBBConfig.cmake is available via ``TBB_DIR`` or ``CMAKE_PREFIX_PATH`` contains path to Intel TBB root.

 CMake code for integration:

  .. code:: cmake

   find_package(TBB <options>)

The following use case is valid for all Intel TBB 2017 packages.

* Download package using TBBGet_ and make integration.

 Pre-condition: Intel TBB CMake modules are available via <path-to-tbb-cmake-modules>.

 CMake code for integration:
  .. code:: cmake

   include(<path-to-tbb-cmake-modules>/TBBGet.cmake)
   tbb_get(TBB_ROOT tbb_root CONFIG_DIR TBB_DIR)
   find_package(TBB <options>)

Source package integration
^^^^^^^^^^^^^^^^^^^^^^^^^^
* Build Intel TBB from existing source files using TBBBuild_ and make integration.

 Pre-condition: Intel TBB source code is available via <tbb_root> and Intel TBB CMake modules are available via <path-to-tbb-cmake-modules>.

 CMake code for integration:
  .. code:: cmake

   include(<path-to-tbb-cmake-modules>/TBBBuild.cmake)
   tbb_build(TBB_ROOT <tbb_root> CONFIG_DIR TBB_DIR)
   find_package(TBB <options>)

* Download Intel TBB source files using TBBGet_, build it using TBBBuild_ and make integration.

 Pre-condition: Intel TBB CMake modules are available via <path-to-tbb-cmake-modules>.

 CMake code for integration:
  .. code:: cmake

   include(<path-to-tbb-cmake-modules>/TBBGet.cmake)
   include(<path-to-tbb-cmake-modules>/TBBBuild.cmake)
   tbb_get(TBB_ROOT tbb_root SOURCE_CODE)
   tbb_build(TBB_ROOT ${tbb_root} CONFIG_DIR TBB_DIR)
   find_package(TBB <options>)

Tutorials: Intel TBB integration using CMake
--------------------------------------------
Binary Intel TBB integration to the sub_string_finder sample (Windows* OS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will integrate binary Intel TBB package into the sub_string_finder sample on Windows* OS (Microsoft* Visual Studio).
This example is also applicable for other platforms with slight changes.
Place holders <version> and <date> should be replaced with the actual values for the Intel TBB package being used. The example is written for `CMake 3.7.1`.

Precondition:
  * `Microsoft* Visual Studio 11` or higher.
  * `CMake 3.0.0` or higher.

#. Download the latest binary package for Windows from `this page <https://github.com/01org/tbb/releases/latest>`_ and unpack it to the directory ``C:\demo_tbb_cmake``.
#. In the directory ``C:\demo_tbb_cmake\tbb<version>_<date>oss\examples\GettingStarted\sub_string_finder`` create ``CMakeLists.txt`` file with the following content:
    .. code:: cmake

        cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

        project(sub_string_finder CXX)
        add_executable(sub_string_finder sub_string_finder.cpp)

        # find_package will search for available TBBConfig using variables CMAKE_PREFIX_PATH and TBB_DIR.
        find_package(TBB REQUIRED tbb)

        # Link Intel TBB imported targets to the executable;
        # "TBB::tbb" can be used instead of "${TBB_IMPORTED_TARGETS}".
        target_link_libraries(sub_string_finder ${TBB_IMPORTED_TARGETS})
#. Run CMake GUI and:
    * Fill the following fields (you can use the buttons ``Browse Source...`` and ``Browse Build...`` accordingly)

     * Where is the source code: ``C:/demo_tbb_cmake/tbb<version>_<date>oss/examples/GettingStarted/sub_string_finder``
     * Where to build the binaries: ``C:/demo_tbb_cmake/tbb<version>_<date>oss/examples/GettingStarted/sub_string_finder/build``

    * Add new cache entry using button ``Add Entry`` to let CMake know where to search for TBBConfig:

     * Name: ``CMAKE_PREFIX_PATH``
     * Type: ``PATH``
     * Value: ``C:/demo_tbb_cmake/tbb<version>_<date>oss``

    * Push the button ``Generate`` and choose a proper generator for your Microsoft* Visual Studio version.
#. Now you can open the generated solution ``C:/demo_tbb_cmake/tbb<version>_<date>oss/examples/GettingStarted/sub_string_finder/build/sub_string_finder.sln`` in your Microsoft* Visual Studio and build it.

Source code integration of Intel TBB to the sub_string_finder sample (Linux* OS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we will build Intel TBB from source code with enabled Community Preview Features and link the sub_string_finder sample with the built library.
This example is also applicable for other platforms with slight changes.

Precondition:
  * `CMake 3.0.0` or higher.
  * `Git` (to clone the Intel TBB repository from GitHub)

#. Create the directory ``~/demo_tbb_cmake``, go to the created directory and clone the Intel TBB repository there:
    ``mkdir ~/demo_tbb_cmake ; cd ~/demo_tbb_cmake ; git clone https://github.com/01org/tbb.git``
#. In the directory ``~/demo_tbb_cmake/tbb/examples/GettingStarted/sub_string_finder`` create ``CMakeLists.txt`` file with following content:
    .. code:: cmake

     cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

     project(sub_string_finder CXX)
     add_executable(sub_string_finder sub_string_finder.cpp)

     include(${TBB_ROOT}/cmake/TBBBuild.cmake)

     # Build Intel TBB with enabled Community Preview Features (CPF).
     tbb_build(TBB_ROOT ${TBB_ROOT} CONFIG_DIR TBB_DIR MAKE_ARGS tbb_cpf=1)

     find_package(TBB REQUIRED tbb_preview)

     # Link Intel TBB imported targets to the executable;
     # "TBB::tbb_preview" can be used instead of "${TBB_IMPORTED_TARGETS}".
     target_link_libraries(sub_string_finder ${TBB_IMPORTED_TARGETS})
#. Create a build directory for the sub_string_finder sample to perform build out of source, go to the created directory
    ``mkdir ~/demo_tbb_cmake/tbb/examples/GettingStarted/sub_string_finder/build ; cd ~/demo_tbb_cmake/tbb/examples/GettingStarted/sub_string_finder/build``
#. Run CMake to prepare Makefile for the sub_string_finder sample and provide Intel TBB location (root) where to perform build:
    ``cmake -DTBB_ROOT=${HOME}/demo_tbb_cmake/tbb ..``
#. Make an executable and run it:
    ``make ; ./sub_string_finder``

Technical documentation for CMake modules
-----------------------------------------
TBBConfig
^^^^^^^^^

Configuration module for ``Intel(R) Threading Building Blocks (Intel(R) TBB)`` library.

How to use this module in your CMake project:
 #. Add location of Intel TBB (root) to `CMAKE_PREFIX_PATH <https://cmake.org/cmake/help/latest/variable/CMAKE_PREFIX_PATH.html>`_
    or specify location of TBBConfig.cmake in ``TBB_DIR``.
 #. Use `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_ to configure Intel TBB.
 #. Use provided variables and/or imported targets (described below) to work with Intel TBB.

Intel TBB components can be passed to `find_package <https://cmake.org/cmake/help/latest/command/find_package.html>`_
after keyword ``COMPONENTS`` or ``REQUIRED``.
Use basic names of components (``tbb``, ``tbbmalloc``, ``tbb_preview``, etc.).

If components are not specified then default are used: ``tbb``, ``tbbmalloc`` and ``tbbmalloc_proxy``.

If ``tbbmalloc_proxy`` is requested, ``tbbmalloc`` component will also be added and set as dependency for ``tbbmalloc_proxy``.

TBBConfig creates `imported targets <https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#imported-targets>`_ as
shared libraries using the following format: ``TBB::<component>`` (for example, ``TBB::tbb``, ``TBB::tbbmalloc``).

Variables set during Intel TBB configuration:

=========================  ================================================
         Variable                            Description
=========================  ================================================
``TBB_FOUND``              Intel TBB library is found
``TBB_<component>_FOUND``  specific Intel TBB component is found
``TBB_IMPORTED_TARGETS``   all created Intel TBB imported targets
``TBB_VERSION``            Intel TBB version (format: ``<major>.<minor>``)
``TBB_INTERFACE_VERSION``  Intel TBB interface version
=========================  ================================================

TBBGet
^^^^^^

Module for getting ``Intel(R) Threading Building Blocks (Intel(R) TBB)`` library from `GitHub <https://github.com/01org/tbb>`_.

Provides the following functions:
 ``tbb_get(TBB_ROOT <variable> [RELEASE_TAG <release_tag>|LATEST] [SAVE_TO <path>] [SYSTEM_NAME Linux|Windows|Darwin] [CONFIG_DIR <variable> | SOURCE_CODE])``
  downloads Intel TBB from GitHub and creates TBBConfig for the downloaded binary package if there is no TBBConfig.

  ====================================  ====================================
                     Parameter                       Description
  ====================================  ====================================
  ``TBB_ROOT <variable>``               a variable to save Intel TBB root in, ``<variable>-NOTFOUND`` will be provided in case ``tbb_get`` is unsuccessful
  ``RELEASE_TAG <release_tag>|LATEST``  Intel TBB release tag to be downloaded (for example, ``2017_U6``), ``LATEST`` is used by default
  ``SAVE_TO <path>``                    path to location at which to unpack downloaded Intel TBB, ``${CMAKE_CURRENT_BINARY_DIR}/tbb_downloaded`` is used by default
  ``SYSTEM_NAME Linux|Windows|Darwin``  operating system name to download a binary package for,
                                        value of `CMAKE_SYSTEM_NAME <https://cmake.org/cmake/help/latest/variable/CMAKE_SYSTEM_NAME.html>`_ is used by default
  ``CONFIG_DIR <variable>``             a variable to save location of TBBConfig.cmake and TBBConfigVersion.cmake. Ignored if ``SOURCE_CODE`` specified
  ``SOURCE_CODE``                       flag to get Intel TBB source code (instead of binary package)
  ====================================  ====================================

TBBMakeConfig
^^^^^^^^^^^^^

Module for making TBBConfig in ``Intel(R) Threading Building Blocks (Intel(R) TBB)`` binary package.

This module is to be used for packages that do not have TBBConfig.

Provides the following functions:
 ``tbb_make_config(TBB_ROOT <path> CONFIG_DIR <variable> [SYSTEM_NAME Linux|Windows|Darwin])``
  creates CMake configuration files (TBBConfig.cmake and TBBConfigVersion.cmake) for Intel TBB binary package.

  ====================================  ====================================
                     Parameter                       Description
  ====================================  ====================================
  ``TBB_ROOT <variable>``               path to Intel TBB root
  ``CONFIG_DIR <variable>``             a variable to store location of the created configuration files
  ``SYSTEM_NAME Linux|Windows|Darwin``  operating system name of the binary Intel TBB package,
                                        value of `CMAKE_SYSTEM_NAME <https://cmake.org/cmake/help/latest/variable/CMAKE_SYSTEM_NAME.html>`_ is used by default
  ====================================  ====================================

TBBBuild
^^^^^^^^

Module for building ``Intel(R) Threading Building Blocks (Intel(R) TBB)`` library from the source code.

Provides the following functions:
 ``tbb_build(TBB_ROOT <tbb_root> CONFIG_DIR <variable> [MAKE_ARGS <custom_make_arguments>])``
  builds Intel TBB from source code using the ``Makefile``, creates and provides the location of the CMake configuration files (TBBConfig.cmake and TBBConfigVersion.cmake) .

  =====================================  ====================================
                Parameter                             Description
  =====================================  ====================================
  ``TBB_ROOT <variable>``                path to Intel TBB root
  ``CONFIG_DIR <variable>``              a variable to store location of the created configuration files,
                                         ``<variable>-NOTFOUND`` will be provided in case ``tbb_build`` is unsuccessful
  ``MAKE_ARGS <custom_make_arguments>``  custom arguments to be passed to ``make`` tool.

                                         The following arguments are always passed with automatically detected values to
                                         ``make`` tool if they are not redefined in ``<custom_make_arguments>``:

                                           - ``compiler=<compiler>``
                                           - ``tbb_build_dir=<tbb_build_dir>``
                                           - ``tbb_build_prefix=<tbb_build_prefix>``
                                           - ``-j<n>``
  =====================================  ====================================


------------

Intel and the Intel logo are trademarks of Intel Corporation or its subsidiaries in the U.S. and/or other countries.

``*`` Other names and brands may be claimed as the property of others.
