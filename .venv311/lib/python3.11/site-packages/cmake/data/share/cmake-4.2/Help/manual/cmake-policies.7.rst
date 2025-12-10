.. cmake-manual-description: CMake Policies Reference

cmake-policies(7)
*****************

.. only:: html

   .. contents::

Introduction
============

CMake policies introduce behavior changes while preserving compatibility
for existing project releases.  Policies are deprecation mechanisms, not
feature toggles.  Each policy documents a deprecated ``OLD`` behavior and
a preferred ``NEW`` behavior.  Projects must be updated over time to
use the ``NEW`` behavior, but their existing releases will continue to
work with the ``OLD`` behavior.

Updating Projects
-----------------

When policies are newly introduced by a version of CMake, their ``OLD``
behaviors are immediately deprecated by that version of CMake and later.
Projects should be updated to use the ``NEW`` behaviors of the policies
as soon as possible.

Use the :command:`cmake_minimum_required` command to record the latest
version of CMake for which a project has been updated.
For example:

..
  Sync this cmake_minimum_required example with ``Help/dev/maint.rst``.

.. code-block:: cmake

  cmake_minimum_required(VERSION 3.10...4.1)

This uses the ``<min>...<max>`` syntax to enable the ``NEW`` behaviors
of policies introduced in CMake 4.1 and earlier while only requiring a
minimum version of CMake 3.10.  The project is expected to work with
both the ``OLD`` and ``NEW`` behaviors of policies introduced between
those versions.

Transition Schedule
-------------------

To help projects port to the ``NEW`` behaviors of policies on their own
schedule, CMake offers a transition period:

* If a policy is not set by a project, CMake uses its ``OLD`` behavior,
  but may warn that the policy has not been set.

  * Users running CMake may silence the warning without modifying a
    project by setting the :variable:`CMAKE_POLICY_DEFAULT_CMP<NNNN>`
    variable as a cache entry on the :manual:`cmake(1)` command line:

    .. code-block:: shell

      cmake -DCMAKE_POLICY_DEFAULT_CMP0990=OLD ...

  * Projects may silence the warning by using the :command:`cmake_policy`
    command to explicitly set the policy to ``OLD`` or ``NEW`` behavior:

    .. code-block:: cmake

      if(POLICY CMP0990)
        cmake_policy(SET CMP0990 NEW)
      endif()

    .. note::

      A policy should almost never be set to ``OLD``, except to silence
      warnings in an otherwise frozen or stable codebase, or temporarily
      as part of a larger migration path.

* If a policy is set to ``OLD`` by a project, CMake versions released
  at least |POLICY_OLD_DELAY_WARNING| after the version that introduced
  a policy may issue a warning that the policy's ``OLD`` behavior will
  be removed from a future version of CMake.

* If a policy is not set to ``NEW`` by a project, CMake versions released
  at least |POLICY_OLD_DELAY_ERROR| after the version that introduced a
  policy, and whose major version number is higher, may issue an error
  that the policy's ``OLD`` behavior has been removed.

.. |POLICY_OLD_DELAY_WARNING| replace:: 2 years
.. |POLICY_OLD_DELAY_ERROR| replace:: 6 years

Supported Policies
==================

The following policies are supported.

Policies Introduced by CMake 4.2
--------------------------------

.. toctree::
   :maxdepth: 1

   CMP0204: A character set is always defined when targeting the MSVC ABI. </policy/CMP0204>
   CMP0203: _WINDLL is defined for shared libraries targeting the MSVC ABI. </policy/CMP0203>
   CMP0202: PDB file names always include their target's per-config POSTFIX. </policy/CMP0202>
   CMP0201: Python::NumPy does not depend on Python::Development.Module. </policy/CMP0201>
   CMP0200: Location and configuration selection for imported targets is more consistent. </policy/CMP0200>
   CMP0199: $<CONFIG> only matches the configuration of the consumed target. </policy/CMP0199>
   CMP0198: CMAKE_PARENT_LIST_FILE is not defined in CMakeLists.txt. </policy/CMP0198>

Policies Introduced by CMake 4.1
--------------------------------

.. toctree::
   :maxdepth: 1

   CMP0197: MSVC link -machine: flag is not in CMAKE_*_LINKER_FLAGS. </policy/CMP0197>
   CMP0196: The CMakeDetermineVSServicePack module is removed. </policy/CMP0196>
   CMP0195: Swift modules in build trees use the Swift module directory structure. </policy/CMP0195>
   CMP0194: MSVC is not an assembler for language ASM. </policy/CMP0194>
   CMP0193: GNUInstallDirs caches CMAKE_INSTALL_* with leading 'usr/' for install prefix '/'. </policy/CMP0193>
   CMP0192: GNUInstallDirs uses absolute SYSCONFDIR, LOCALSTATEDIR, and RUNSTATEDIR in special prefixes. </policy/CMP0192>
   CMP0191: The FindCABLE module is removed. </policy/CMP0191>
   CMP0190: FindPython enforce consistency in cross-compiling mode. </policy/CMP0190>
   CMP0189: TARGET_PROPERTY evaluates LINK_LIBRARIES properties transitively. </policy/CMP0189>
   CMP0188: The FindGCCXML module is removed. </policy/CMP0188>
   CMP0187: Include source file without an extension after the same name with an extension. </policy/CMP0187>
   CMP0186: Regular expressions match ^ at most once in repeated searches. </policy/CMP0186>

Policies Introduced by CMake 4.0
--------------------------------

.. toctree::
   :maxdepth: 1

   CMP0185: FindRuby no longer provides upper-case RUBY_* variables. </policy/CMP0185>
   CMP0184: MSVC runtime checks flags are selected by an abstraction. </policy/CMP0184>
   CMP0183: add_feature_info() supports full Condition Syntax. </policy/CMP0183>
   CMP0182: Create shared library archives by default on AIX. </policy/CMP0182>
   CMP0181: Link command-line fragment variables are parsed and re-quoted. </policy/CMP0181>

Policies Introduced by CMake 3.31
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0180: project() always sets <PROJECT-NAME>_* as normal variables. </policy/CMP0180>
   CMP0179: De-duplication of static libraries on link lines keeps first occurrence. </policy/CMP0179>
   CMP0178: Test command lines preserve empty arguments. </policy/CMP0178>
   CMP0177: install() DESTINATION paths are normalized. </policy/CMP0177>
   CMP0176: execute_process() ENCODING is UTF-8 by default. </policy/CMP0176>
   CMP0175: add_custom_command() rejects invalid arguments. </policy/CMP0175>
   CMP0174: cmake_parse_arguments(PARSE_ARGV) defines a variable for an empty string after a single-value keyword. </policy/CMP0174>
   CMP0173: The CMakeFindFrameworks module is removed. </policy/CMP0173>
   CMP0172: The CPack module enables per-machine installation by default in the CPack WIX Generator. </policy/CMP0172>
   CMP0171: 'codegen' is a reserved target name. </policy/CMP0171>

Policies Introduced by CMake 3.30
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0170: FETCHCONTENT_FULLY_DISCONNECTED requirements are enforced. </policy/CMP0170>
   CMP0169: FetchContent_Populate(depName) single-argument signature is deprecated. </policy/CMP0169>
   CMP0168: FetchContent implements steps directly instead of through a sub-build. </policy/CMP0168>
   CMP0167: The FindBoost module is removed. </policy/CMP0167>
   CMP0166: TARGET_PROPERTY evaluates link properties transitively over private dependencies of static libraries. </policy/CMP0166>
   CMP0165: enable_language() must not be called before project(). </policy/CMP0165>
   CMP0164: add_library() rejects SHARED libraries when not supported by the platform. </policy/CMP0164>
   CMP0163: The GENERATED source file property is now visible in all directories. </policy/CMP0163>
   CMP0162: Visual Studio generators add UseDebugLibraries indicators by default. </policy/CMP0162>

Policies Introduced by CMake 3.29
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0161: CPACK_PRODUCTBUILD_DOMAINS defaults to true. </policy/CMP0161>
   CMP0160: More read-only target properties now error when trying to set them. </policy/CMP0160>
   CMP0159: file(STRINGS) with REGEX updates CMAKE_MATCH_<n>. </policy/CMP0159>
   CMP0158: add_test() honors CMAKE_CROSSCOMPILING_EMULATOR only when cross-compiling. </policy/CMP0158>
   CMP0157: Swift compilation mode is selected by an abstraction. </policy/CMP0157>
   CMP0156: De-duplicate libraries on link lines based on linker capabilities. </policy/CMP0156>

Policies Introduced by CMake 3.28
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0155: C++ sources in targets with at least C++20 are scanned for imports when supported. </policy/CMP0155>
   CMP0154: Generated files are private by default in targets using file sets. </policy/CMP0154>
   CMP0153: The exec_program command should not be called. </policy/CMP0153>
   CMP0152: file(REAL_PATH) resolves symlinks before collapsing ../ components.  </policy/CMP0152>

Policies Introduced by CMake 3.27
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0151: AUTOMOC include directory is a system include directory by default. </policy/CMP0151>
   CMP0150: ExternalProject_Add and FetchContent_Declare treat relative git repository paths as being relative to parent project's remote. </policy/CMP0150>
   CMP0149: Visual Studio generators select latest Windows SDK by default. </policy/CMP0149>
   CMP0148: The FindPythonInterp and FindPythonLibs modules are removed. </policy/CMP0148>
   CMP0147: Visual Studio generators build custom commands in parallel. </policy/CMP0147>
   CMP0146: The FindCUDA module is removed. </policy/CMP0146>
   CMP0145: The Dart and FindDart modules are removed. </policy/CMP0145>
   CMP0144: find_package uses upper-case PACKAGENAME_ROOT variables. </policy/CMP0144>

Policies Introduced by CMake 3.26
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0143: USE_FOLDERS global property is treated as ON by default. </policy/CMP0143>

Policies Introduced by CMake 3.25
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0142: The Xcode generator does not append per-config suffixes to library search paths. </policy/CMP0142>
   CMP0141: MSVC debug information format flags are selected by an abstraction. </policy/CMP0141>
   CMP0140: The return() command checks its arguments. </policy/CMP0140>

Policies Introduced by CMake 3.24
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0139: The if() command supports path comparisons using PATH_EQUAL operator. </policy/CMP0139>
   CMP0138: CheckIPOSupported uses flags from calling project. </policy/CMP0138>
   CMP0137: try_compile() passes platform variables in project mode. </policy/CMP0137>
   CMP0136: Watcom runtime library flags are selected by an abstraction. </policy/CMP0136>
   CMP0135: ExternalProject and FetchContent ignore timestamps in archives by default for the URL download method. </policy/CMP0135>
   CMP0134: Fallback to "HOST" Windows registry view when "TARGET" view is not usable. </policy/CMP0134>
   CMP0133: The CPack module disables SLA by default in the CPack DragNDrop Generator. </policy/CMP0133>
   CMP0132: Do not set compiler environment variables on first run. </policy/CMP0132>
   CMP0131: LINK_LIBRARIES supports the LINK_ONLY generator expression. </policy/CMP0131>
   CMP0130: while() diagnoses condition evaluation errors. </policy/CMP0130>

Policies Introduced by CMake 3.23
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0129: Compiler id for MCST LCC compilers is now LCC, not GNU. </policy/CMP0129>

Policies Introduced by CMake 3.22
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0128: Selection of language standard and extension flags improved. </policy/CMP0128>
   CMP0127: cmake_dependent_option() supports full Condition Syntax. </policy/CMP0127>

Policies Introduced by CMake 3.21
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0126: set(CACHE) does not remove a normal variable of the same name. </policy/CMP0126>
   CMP0125: find_(path|file|library|program) have consistent behavior for cache variables. </policy/CMP0125>
   CMP0124: foreach() loop variables are only available in the loop scope. </policy/CMP0124>
   CMP0123: ARMClang cpu/arch compile and link flags must be set explicitly. </policy/CMP0123>
   CMP0122: UseSWIG use standard library name conventions for csharp language. </policy/CMP0122>
   CMP0121: The list command detects invalid indices. </policy/CMP0121>

Policies Introduced by CMake 3.20
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0120: The WriteCompilerDetectionHeader module is removed. </policy/CMP0120>
   CMP0119: LANGUAGE source file property explicitly compiles as language. </policy/CMP0119>
   CMP0118: GENERATED sources may be used across directories without manual marking. </policy/CMP0118>
   CMP0117: MSVC RTTI flag /GR is not added to CMAKE_CXX_FLAGS by default. </policy/CMP0117>
   CMP0116: Ninja generators transform DEPFILEs from add_custom_command(). </policy/CMP0116>
   CMP0115: Source file extensions must be explicit. </policy/CMP0115>

Policies Introduced by CMake 3.19
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0114: ExternalProject step targets fully adopt their steps. </policy/CMP0114>
   CMP0113: Makefile generators do not repeat custom commands from target dependencies. </policy/CMP0113>
   CMP0112: Target file component generator expressions do not add target dependencies. </policy/CMP0112>
   CMP0111: An imported target missing its location property fails during generation. </policy/CMP0111>
   CMP0110: add_test() supports arbitrary characters in test names. </policy/CMP0110>
   CMP0109: find_program() requires permission to execute but not to read. </policy/CMP0109>

Policies Introduced by CMake 3.18
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0108: A target cannot link to itself through an alias. </policy/CMP0108>
   CMP0107: An ALIAS target cannot overwrite another target. </policy/CMP0107>
   CMP0106: The Documentation module is removed. </policy/CMP0106>
   CMP0105: Device link step uses the link options. </policy/CMP0105>
   CMP0104: CMAKE_CUDA_ARCHITECTURES now detected for NVCC, empty CUDA_ARCHITECTURES not allowed. </policy/CMP0104>
   CMP0103: Multiple export() with same FILE without APPEND is not allowed. </policy/CMP0103>

Policies Introduced by CMake 3.17
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0102: mark_as_advanced() does nothing if a cache entry does not exist. </policy/CMP0102>
   CMP0101: target_compile_options honors BEFORE keyword in all scopes. </policy/CMP0101>
   CMP0100: Let AUTOMOC and AUTOUIC process .hh header files. </policy/CMP0100>
   CMP0099: Link properties are transitive over private dependencies of static libraries. </policy/CMP0099>
   CMP0098: FindFLEX runs flex in CMAKE_CURRENT_BINARY_DIR when executing. </policy/CMP0098>

Policies Introduced by CMake 3.16
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0097: ExternalProject_Add with GIT_SUBMODULES "" initializes no submodules. </policy/CMP0097>
   CMP0096: project() preserves leading zeros in version components. </policy/CMP0096>
   CMP0095: RPATH entries are properly escaped in the intermediary CMake install script. </policy/CMP0095>

Policies Introduced by CMake 3.15
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0094: FindPython3, FindPython2 and FindPython use LOCATION for lookup strategy. </policy/CMP0094>
   CMP0093: FindBoost reports Boost_VERSION in x.y.z format. </policy/CMP0093>
   CMP0092: MSVC warning flags are not in CMAKE_{C,CXX}_FLAGS by default. </policy/CMP0092>
   CMP0091: MSVC runtime library flags are selected by an abstraction. </policy/CMP0091>
   CMP0090: export(PACKAGE) does not populate package registry by default. </policy/CMP0090>
   CMP0089: Compiler id for IBM Clang-based XL compilers is now XLClang. </policy/CMP0089>

Policies Introduced by CMake 3.14
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0088: FindBISON runs bison in CMAKE_CURRENT_BINARY_DIR when executing. </policy/CMP0088>
   CMP0087: install(SCRIPT | CODE) supports generator expressions. </policy/CMP0087>
   CMP0086: UseSWIG honors SWIG_MODULE_NAME via -module flag. </policy/CMP0086>
   CMP0085: IN_LIST generator expression handles empty list items. </policy/CMP0085>
   CMP0084: The FindQt module does not exist for find_package(). </policy/CMP0084>
   CMP0083: Add PIE options when linking executable. </policy/CMP0083>
   CMP0082: Install rules from add_subdirectory() are interleaved with those in caller. </policy/CMP0082>


Policies Introduced by CMake 3.13
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0081: Relative paths not allowed in LINK_DIRECTORIES target property. </policy/CMP0081>
   CMP0080: BundleUtilities cannot be included at configure time. </policy/CMP0080>
   CMP0079: target_link_libraries allows use with targets in other directories. </policy/CMP0079>
   CMP0078: UseSWIG generates standard target names. </policy/CMP0078>
   CMP0077: option() honors normal variables. </policy/CMP0077>
   CMP0076: target_sources() command converts relative paths to absolute. </policy/CMP0076>

Policies Introduced by CMake 3.12
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0075: Include file check macros honor CMAKE_REQUIRED_LIBRARIES. </policy/CMP0075>
   CMP0074: find_package uses PackageName_ROOT variables. </policy/CMP0074>
   CMP0073: Do not produce legacy _LIB_DEPENDS cache entries. </policy/CMP0073>

Policies Introduced by CMake 3.11
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0072: FindOpenGL prefers GLVND by default when available. </policy/CMP0072>

Policies Introduced by CMake 3.10
---------------------------------

.. toctree::
   :maxdepth: 1

   CMP0071: Let AUTOMOC and AUTOUIC process GENERATED files. </policy/CMP0071>
   CMP0070: Define file(GENERATE) behavior for relative paths. </policy/CMP0070>

Policies Introduced by CMake 3.9
--------------------------------

.. toctree::
   :maxdepth: 1

   CMP0069: INTERPROCEDURAL_OPTIMIZATION is enforced when enabled. </policy/CMP0069>
   CMP0068: RPATH settings on macOS do not affect install_name. </policy/CMP0068>

Policies Introduced by CMake 3.8
--------------------------------

.. toctree::
   :maxdepth: 1

   CMP0067: Honor language standard in try_compile() source-file signature. </policy/CMP0067>

Policies Introduced by CMake 3.7
--------------------------------

.. toctree::
   :maxdepth: 1

   CMP0066: Honor per-config flags in try_compile() source-file signature. </policy/CMP0066>

Unsupported Policies
====================

The following policies are no longer supported.
Projects' calls to :command:`cmake_minimum_required(VERSION)` or
:command:`cmake_policy(VERSION)` must set them to ``NEW``.
Their ``OLD`` behaviors have been removed from CMake.

.. _`Policies Introduced by CMake 3.4`:

Policies Introduced by CMake 3.4, Removed by CMake 4.0
------------------------------------------------------

.. toctree::
   :maxdepth: 1

   CMP0065: Do not add flags to export symbols from executables without the ENABLE_EXPORTS target property. </policy/CMP0065>
   CMP0064: Support new TEST if() operator. </policy/CMP0064>

.. _`Policies Introduced by CMake 3.3`:

Policies Introduced by CMake 3.3, Removed by CMake 4.0
------------------------------------------------------

.. toctree::
   :maxdepth: 1

   CMP0063: Honor visibility properties for all target types. </policy/CMP0063>
   CMP0062: Disallow install() of export() result. </policy/CMP0062>
   CMP0061: CTest does not by default tell make to ignore errors (-i). </policy/CMP0061>
   CMP0060: Link libraries by full path even in implicit directories. </policy/CMP0060>
   CMP0059: Do not treat DEFINITIONS as a built-in directory property. </policy/CMP0059>
   CMP0058: Ninja requires custom command byproducts to be explicit. </policy/CMP0058>
   CMP0057: Support new IN_LIST if() operator. </policy/CMP0057>

.. _`Policies Introduced by CMake 3.2`:

Policies Introduced by CMake 3.2, Removed by CMake 4.0
------------------------------------------------------

.. toctree::
   :maxdepth: 1

   CMP0056: Honor link flags in try_compile() source-file signature. </policy/CMP0056>
   CMP0055: Strict checking for break() command. </policy/CMP0055>

.. _`Policies Introduced by CMake 3.1`:

Policies Introduced by CMake 3.1, Removed by CMake 4.0
------------------------------------------------------

.. toctree::
   :maxdepth: 1

   CMP0054: Only interpret if() arguments as variables or keywords when unquoted. </policy/CMP0054>
   CMP0053: Simplify variable reference and escape sequence evaluation. </policy/CMP0053>
   CMP0052: Reject source and build dirs in installed INTERFACE_INCLUDE_DIRECTORIES. </policy/CMP0052>
   CMP0051: List TARGET_OBJECTS in SOURCES target property. </policy/CMP0051>

.. _`Policies Introduced by CMake 3.0`:

Policies Introduced by CMake 3.0, Removed by CMake 4.0
------------------------------------------------------

.. toctree::
   :maxdepth: 1

   CMP0050: Disallow add_custom_command SOURCE signatures. </policy/CMP0050>
   CMP0049: Do not expand variables in target source entries. </policy/CMP0049>
   CMP0048: project() command manages VERSION variables. </policy/CMP0048>
   CMP0047: Use QCC compiler id for the qcc drivers on QNX. </policy/CMP0047>
   CMP0046: Error on non-existent dependency in add_dependencies. </policy/CMP0046>
   CMP0045: Error on non-existent target in get_target_property. </policy/CMP0045>
   CMP0044: Case sensitive Lang_COMPILER_ID generator expressions. </policy/CMP0044>
   CMP0043: Ignore COMPILE_DEFINITIONS_Config properties. </policy/CMP0043>
   CMP0042: MACOSX_RPATH is enabled by default. </policy/CMP0042>
   CMP0041: Error on relative include with generator expression. </policy/CMP0041>
   CMP0040: The target in the TARGET signature of add_custom_command() must exist. </policy/CMP0040>
   CMP0039: Utility targets may not have link dependencies. </policy/CMP0039>
   CMP0038: Targets may not link directly to themselves. </policy/CMP0038>
   CMP0037: Target names should not be reserved and should match a validity pattern. </policy/CMP0037>
   CMP0036: The build_name command should not be called. </policy/CMP0036>
   CMP0035: The variable_requires command should not be called. </policy/CMP0035>
   CMP0034: The utility_source command should not be called. </policy/CMP0034>
   CMP0033: The export_library_dependencies command should not be called. </policy/CMP0033>
   CMP0032: The output_required_files command should not be called. </policy/CMP0032>
   CMP0031: The load_command command should not be called. </policy/CMP0031>
   CMP0030: The use_mangled_mesa command should not be called. </policy/CMP0030>
   CMP0029: The subdir_depends command should not be called. </policy/CMP0029>
   CMP0028: Double colon in target name means ALIAS or IMPORTED target. </policy/CMP0028>
   CMP0027: Conditionally linked imported targets with missing include directories. </policy/CMP0027>
   CMP0026: Disallow use of the LOCATION target property. </policy/CMP0026>
   CMP0025: Compiler id for Apple Clang is now AppleClang. </policy/CMP0025>
   CMP0024: Disallow include export result. </policy/CMP0024>

.. _`Policies Introduced by CMake 2.8`:

Policies Introduced by CMake 2.8, Removed by CMake 4.0
------------------------------------------------------

.. toctree::
   :maxdepth: 1

   CMP0023: Plain and keyword target_link_libraries signatures cannot be mixed. </policy/CMP0023>
   CMP0022: INTERFACE_LINK_LIBRARIES defines the link interface. </policy/CMP0022>
   CMP0021: Fatal error on relative paths in INCLUDE_DIRECTORIES target property. </policy/CMP0021>
   CMP0020: Automatically link Qt executables to qtmain target on Windows. </policy/CMP0020>
   CMP0019: Do not re-expand variables in include and link information. </policy/CMP0019>
   CMP0018: Ignore CMAKE_SHARED_LIBRARY_Lang_FLAGS variable. </policy/CMP0018>
   CMP0017: Prefer files from the CMake module directory when including from there. </policy/CMP0017>
   CMP0016: target_link_libraries() reports error if its only argument is not a target. </policy/CMP0016>
   CMP0015: link_directories() treats paths relative to the source dir. </policy/CMP0015>
   CMP0014: Input directories must have CMakeLists.txt. </policy/CMP0014>
   CMP0013: Duplicate binary directories are not allowed. </policy/CMP0013>
   CMP0012: if() recognizes numbers and boolean constants. </policy/CMP0012>

.. _`Policies Introduced by CMake 2.6`:

Policies Introduced by CMake 2.6, Removed by CMake 4.0
------------------------------------------------------

.. toctree::
   :maxdepth: 1

   CMP0011: Included scripts do automatic cmake_policy PUSH and POP. </policy/CMP0011>
   CMP0010: Bad variable reference syntax is an error. </policy/CMP0010>
   CMP0009: FILE GLOB_RECURSE calls should not follow symlinks by default. </policy/CMP0009>
   CMP0008: Libraries linked by full-path must have a valid library file name. </policy/CMP0008>
   CMP0007: list command no longer ignores empty elements. </policy/CMP0007>
   CMP0006: Installing MACOSX_BUNDLE targets requires a BUNDLE DESTINATION. </policy/CMP0006>
   CMP0005: Preprocessor definition values are now escaped automatically. </policy/CMP0005>
   CMP0004: Libraries linked may not have leading or trailing whitespace. </policy/CMP0004>
   CMP0003: Libraries linked via full path no longer produce linker search paths. </policy/CMP0003>
   CMP0002: Logical target names must be globally unique. </policy/CMP0002>
   CMP0001: CMAKE_BACKWARDS_COMPATIBILITY should no longer be used. </policy/CMP0001>
   CMP0000: A minimum required CMake version must be specified. </policy/CMP0000>
