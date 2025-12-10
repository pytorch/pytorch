# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindDoxygen
-----------

Finds `Doxygen <https://www.doxygen.nl>`_, a source code documentation
generator, along with some optional supporting tools, and provides a command
for integrating Doxygen-based documentation into CMake projects:

.. code-block:: cmake

  find_package(Doxygen [<version>] [COMPONENTS <components>...] [...])

Components
^^^^^^^^^^

Additional Doxygen supporting tools, can be specified as components with the
:command:`find_package()` command:

.. code-block:: cmake

  find_package(Doxygen [COMPONENTS <components>...])

Supported components include:

``doxygen``
  .. versionadded:: 3.9

  Finds the ``doxygen`` executable.  This component is always automatically
  implied, even if not requested.

``dot``
  .. versionadded:: 3.9

  Finds the `Graphviz <https://graphviz.org>`_ ``dot`` utility, used for
  rendering graphs and diagrams as part of the documentation.

``mscgen``
  .. versionadded:: 3.9

  Finds the `Message Chart Generator <https://www.mcternan.me.uk/mscgen/>`_
  utility used by Doxygen's ``\msc`` and ``\mscfile`` commands.

``dia``
  .. versionadded:: 3.9

  Finds the `Dia <https://wiki.gnome.org/Apps/Dia>`_ diagram editor used by
  Doxygen's ``\diafile`` command.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`, each of which is
defined if the corresponding component was requested and its associated
executable was found:

``Doxygen::doxygen``
  .. versionadded:: 3.9

  Imported executable target encapsulating the ``doxygen`` executable usage
  requirements, available if Doxygen is found.

``Doxygen::dot``
  .. versionadded:: 3.9

  Imported executable target encapsulating the ``dot`` executable usage
  requirements, available if the above ``dot`` component is found.

``Doxygen::mscgen``
  .. versionadded:: 3.9

  Imported executable target encapsulating the ``mscgen`` executable usage
  requirements, available if the above ``mscgen`` component is found.

``Doxygen::dia``
  .. versionadded:: 3.9

  Imported executable target encapsulating the ``dia`` executable usage
  requirements, available if the above ``dia`` component is found.

These targets can be used in commands such as :command:`add_custom_command`
and are preferred over the older, now-deprecated variables like
``DOXYGEN_EXECUTABLE``.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Doxygen_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) ``doxygen`` executable
  and all requested required components were found.

``Doxygen_VERSION``
  .. versionadded:: 4.2

  The version of Doxygen found (as reported by ``doxygen --version``).

Commands
^^^^^^^^

This module provides the following command:

.. command:: doxygen_add_docs

  .. versionadded:: 3.9

  Adds a custom target for generating documentation with Doxygen during the
  build phase:

  .. code-block:: cmake

    doxygen_add_docs(
      <target-name>
      [<files-or-dirs>...]
      [ALL]
      [USE_STAMP_FILE]
      [WORKING_DIRECTORY <dir>]
      [COMMENT <comment>]
      [CONFIG_FILE <file>]
    )

  By default, this convenience command also generates a configuration file
  named ``Doxyfile.<target-name>`` in the current binary directory at
  CMake configuration phase.  It provides sensible defaults, so most projects
  only need to specify input files or directories.  Additional behavior and
  configuration can be customized using variables described in the following
  sections.

  .. rubric:: The arguments are:

  ``<target-name>``
    The name of the target to be created for generating documentation with
    Doxygen.

  ``<files-or-dirs>...``
    One or more paths (files or directories) that serve as input sources for
    documentation.

    These are passed to the ``INPUT`` Doxygen configuration tag in the
    generated ``Doxyfile.<target-name>``.  Files listed here are also added
    as ``SOURCES`` argument of the underlying :command:`add_custom_target`
    command so they appear in IDE project's source list.

    When using the ``USE_STAMP_FILE`` option, only files (not directories,
    symlinks, or wildcards) are allowed, and each must exist when this
    command is called.

  ``ALL``
    .. versionadded:: 3.12

    Adds the created documentation target to the default build target so
    that it runs automatically as part of the build phase.

  ``USE_STAMP_FILE``
    .. versionadded:: 3.16

    Enables use of a stamp file to avoid regenerating documentation unless
    source files have changed.

    Stamp file named ``<target-name>.stamp`` is created in the current binary
    directory by an underlying custom command.

    With this option present, all entries in ``<files-or-dirs>`` must be
    existing files (i.e. no directories, symlinks or wildcards) when this
    command is called.  An error is raised if any listed path is invalid.

    Without this option, CMake will re-run Doxygen every time the
    ``<target-name>`` target is built, regardless of whether any input source
    file listed in ``<files-or-dirs>`` has changed.

  ``WORKING_DIRECTORY <dir>``
    By default, the Doxygen working directory is the current source directory
    (:variable:`CMAKE_CURRENT_SOURCE_DIR`).  This aligns with using relative
    input paths.

    Use this option, to change and override the directory where Doxygen is
    being run.  The absolute path ``<dir>`` will then be used as the base
    point for relative paths.

    Note also that Doxygen's default behavior is to strip the working
    directory from relative paths in the generated documentation.  See the
    ``STRIP_FROM_PATH`` config tag in the `Doxygen manual
    <https://www.doxygen.nl/manual/config.html>`_ for more details.

  ``COMMENT <comment>``
    If provided, the ``<comment>`` string will be passed as the ``COMMENT``
    argument to the underlying :command:`add_custom_target` command used to
    create the custom target internally.  This appears in the build system
    output, when the target is built.

  ``CONFIG_FILE <file>``
    .. versionadded:: 3.27

    If specified, the given file provided with full-path will be used as
    Doxygen configuration file instead of the default
    ``Doxyfile.<target-name>``.

  .. rubric:: Variables for customizing Doxygen configuration

  The ``doxygen_add_docs()`` command generates a Doxygen configuration file
  containing configuration tags.  For example:

  .. code-block:: text
    :caption: ``Doxygen.<target-name>``

    DOXYFILE_ENCODING      = UTF-8
    PROJECT_NAME           = DoxygenExample
    PROJECT_NUMBER         = 1.2.3
    PROJECT_BRIEF          = "Example project using Doxygen"
    PROJECT_LOGO           =
    OUTPUT_DIRECTORY       = /home/user/doxygen-example/build
    GENERATE_HTML          = YES
    GENERATE_MAN           = NO
    # ...

  In CMake, these tags can be modified by setting input variables in form
  of ``DOXYGEN_<tag>``, where ``<tag>`` is one of the configuration tags
  listed in the `Doxygen manual
  <https://www.doxygen.nl/manual/config.html>`_.

  For example, to modify the ``GENERATE_HTML`` and ``GENERATE_MAN``
  configuration tags, the following variables can be set before calling
  ``doxygen_add_docs()``:

  .. code-block:: cmake
    :caption: ``CMakeLists.txt``

    find_package(Doxygen)

    if(Doxygen_FOUND)
      set(DOXYGEN_GENERATE_HTML NO)
      set(DOXYGEN_GENERATE_MAN YES)

      doxygen_add_docs(project_docs ${PROJECT_SOURCE_DIR})
    endif()

  .. rubric:: Default configuration

  By default, ``doxygen_add_docs()`` overrides several of Doxygen's settings
  to better suit typical CMake projects.  Each of the following variables is
  explicitly set unless already defined prior to calling
  ``doxygen_add_docs()``, with a few exceptions noted below:

  ``DOXYGEN_HAVE_DOT``
    Set to ``YES`` if the ``dot`` component was requested and found, ``NO``
    otherwise.  Any existing value of ``DOXYGEN_HAVE_DOT`` is ignored.

  ``DOXYGEN_DOT_MULTI_TARGETS``
    Set to ``YES`` by this module (note that this requires a ``dot`` version
    newer than 1.8.10).  This option is only meaningful if ``DOXYGEN_HAVE_DOT``
    is also set to ``YES``.

  ``DOXYGEN_GENERATE_LATEX``
    Set to ``NO`` by this module.

  ``DOXYGEN_WARN_FORMAT``
    For Visual Studio based generators, this is set to the form recognized by
    the Visual Studio IDE: ``$file($line) : $text``. For all other generators,
    Doxygen's default value is not overridden.

  ``DOXYGEN_PROJECT_NAME``
    Populated with the name of the current project (i.e.
    :variable:`PROJECT_NAME`).

  ``DOXYGEN_PROJECT_NUMBER``
    Populated with the version of the current project (i.e.
    :variable:`PROJECT_VERSION`).

  ``DOXYGEN_PROJECT_BRIEF``
    Populated with the description of the current project (i.e.
    :variable:`PROJECT_DESCRIPTION`).

  ``DOXYGEN_INPUT``
    This variable is automatically populated with the list of files and
    directories passed to ``doxygen_add_docs()``.  For consistent behavior
    with other built-in commands like :command:`add_executable`,
    :command:`add_library`, and :command:`add_custom_target`, projects should
    not set this variable manually.  If a variable named ``DOXYGEN_INPUT``
    is set by the project, it will be ignored and a warning will be issued.

  ``DOXYGEN_RECURSIVE``
    Set to ``YES`` by this module.

  ``DOXYGEN_EXCLUDE_PATTERNS``
    If the ``<files-or-dirs>`` argument of ``doxygen_add_docs()`` contains
    directories, this variable will specify patterns used to exclude files
    from them.  The following patterns are added by default to ensure
    CMake-specific files and directories are not included in the input.  If
    the project sets this variable, those contents are merged with these
    additional patterns rather than replacing them:

    ::

      */.git/*
      */.svn/*
      */.hg/*
      */CMakeFiles/*
      */_CPack_Packages/*
      DartConfiguration.tcl
      CMakeLists.txt
      CMakeCache.txt

  ``DOXYGEN_OUTPUT_DIRECTORY``
    Set to :variable:`CMAKE_CURRENT_BINARY_DIR` by this module.  If the
    project provides its own value for this and it is a relative path, it
    will be interpreted relative to the current binary directory
    (:variable:`CMAKE_CURRENT_BINARY_DIR`).  This is necessary because
    Doxygen will normally be run from a directory within the source tree so
    that relative source paths work as expected.  If this directory does not
    exist, it will be recursively created prior to executing Doxygen.

  .. rubric:: Lists

  A number of Doxygen config tags accept lists of values, and Doxygen
  requires them to be separated by whitespace, while in CMake a list is a
  string with items separated by :ref:`semicolons <CMake Language Lists>`.

  The ``doxygen_add_docs()`` specifically checks for the following Doxygen
  config tags and converts their associated CMake ``DOXYGEN_<tag>`` variable
  values into the Doxygen-formatted lists if set:

  .. hlist::

    - ``ABBREVIATE_BRIEF``
    - ``ALIASES``
    - ``CITE_BIB_FILES``
    - ``DIAFILE_DIRS``
    - ``DOTFILE_DIRS``
    - ``DOT_FONTPATH``
    - ``ENABLED_SECTIONS``
    - ``EXAMPLE_PATH``
    - ``EXAMPLE_PATTERNS``
    - ``EXCLUDE``
    - ``EXCLUDE_PATTERNS``
    - ``EXCLUDE_SYMBOLS``
    - ``EXPAND_AS_DEFINED``
    - ``EXTENSION_MAPPING``
    - ``EXTRA_PACKAGES``
    - ``EXTRA_SEARCH_MAPPINGS``
    - ``FILE_PATTERNS``
    - ``FILTER_PATTERNS``
    - ``FILTER_SOURCE_PATTERNS``
    - ``HTML_EXTRA_FILES``
    - ``HTML_EXTRA_STYLESHEET``
    - ``IGNORE_PREFIX``
    - ``IMAGE_PATH``
    - ``INCLUDE_FILE_PATTERNS``
    - ``INCLUDE_PATH``
    - ``INPUT``
    - ``LATEX_EXTRA_FILES``
    - ``LATEX_EXTRA_STYLESHEET``
    - ``MATHJAX_EXTENSIONS``
    - ``MSCFILE_DIRS``
    - ``PLANTUML_INCLUDE_PATH``
    - ``PREDEFINED``
    - ``QHP_CUST_FILTER_ATTRS``
    - ``QHP_SECT_FILTER_ATTRS``
    - ``STRIP_FROM_INC_PATH``
    - ``STRIP_FROM_PATH``
    - ``TAGFILES``
    - ``TCL_SUBST``

  For example, to customize the Doxygen file patterns, a usual
  :ref:`semicolon-separated list <CMake Language Lists>` can be set in CMake:

  .. code-block:: cmake
    :caption: ``CMakeLists.txt``

    find_package(Doxygen)

    if(Doxygen_FOUND)
      set(DOXYGEN_FILE_PATTERNS *.c *.cxx *.h *.hxx)
      doxygen_add_docs(example_docs ${CMAKE_CURRENT_SOURCE_DIR} ALL)
    endif()

  Which will produce a Doxygen list of patterns separated by spaces in the
  generated configuration file:

  .. code-block:: text
    :caption: ``Doxyfile.<target-name>``

    # ...
    FILE_PATTERNS          = *.c *.cxx *.h *.hxx

  .. rubric:: Automatic quoting

  If a Doxygen single-value tag contains spaces, their values must be
  surrounded by double quotes (``"..."``).  ``doxygen_add_docs()``
  automatically quotes values of the following Doxygen tags when generating
  the ``Doxyfile``, if they contain at least one space:

  .. hlist::

    - ``CHM_FILE``
    - ``DIA_PATH``
    - ``DOCBOOK_OUTPUT``
    - ``DOCSET_FEEDNAME``
    - ``DOCSET_PUBLISHER_NAME``
    - ``DOT_FONTNAME``
    - ``DOT_PATH``
    - ``EXTERNAL_SEARCH_ID``
    - ``FILE_VERSION_FILTER``
    - ``GENERATE_TAGFILE``
    - ``HHC_LOCATION``
    - ``HTML_FOOTER``
    - ``HTML_HEADER``
    - ``HTML_OUTPUT``
    - ``HTML_STYLESHEET``
    - ``INPUT_FILTER``
    - ``LATEX_FOOTER``
    - ``LATEX_HEADER``
    - ``LATEX_OUTPUT``
    - ``LAYOUT_FILE``
    - ``MAN_OUTPUT``
    - ``MAN_SUBDIR``
    - ``MATHJAX_CODEFILE``
    - ``MSCGEN_PATH``
    - ``OUTPUT_DIRECTORY``
    - ``PERL_PATH``
    - ``PLANTUML_JAR_PATH``
    - ``PROJECT_BRIEF``
    - ``PROJECT_LOGO``
    - ``PROJECT_NAME``
    - ``QCH_FILE``
    - ``QHG_LOCATION``
    - ``QHP_CUST_FILTER_NAME``
    - ``QHP_VIRTUAL_FOLDER``
    - ``RTF_EXTENSIONS_FILE``
    - ``RTF_OUTPUT``
    - ``RTF_STYLESHEET_FILE``
    - ``SEARCHDATA_FILE``
    - ``USE_MDFILE_AS_MAINPAGE``
    - ``WARN_FORMAT``
    - ``WARN_LOGFILE``
    - ``XML_OUTPUT``

  ``DOXYGEN_VERBATIM_VARS``
    .. versionadded:: 3.11

    A CMake input variable used by ``doxygen_add_docs()`` to specify a list
    of Doxygen input variables (including their leading ``DOXYGEN_`` prefix)
    whose values should be passed to the generated ``Doxyfile`` configuration
    without automatic quoting.

    When using this variable, the project is then responsible for ensuring
    that those variables' values make sense when placed directly in the
    generated ``Doxyfile`` configuration.  For list variables, items are
    still separated by spaces in the output, but no quoting is applied to the
    individual items.

    For certain Doxygen tags, such as ``ALIASES``, automatic quoting done by
    ``doxygen_add_docs()`` may interfere with correct syntax (e.g., embedded
    quotes).

    For example, the following will quote ``DOXYGEN_PROJECT_BRIEF``, but skip
    each item in the ``DOXYGEN_ALIASES`` list (:ref:`bracket syntax
    <Bracket Argument>` is used to make working with embedded quotes easier):

    .. code-block:: cmake
      :caption: ``CMakeLists.txt``

      find_package(Doxygen)

      if(Doxygen_FOUND)
        set(DOXYGEN_PROJECT_BRIEF "String with spaces")
        set(
          DOXYGEN_ALIASES
          [[somealias="@some_command param"]]
          "anotherAlias=@foobar"
        )
        set(DOXYGEN_VERBATIM_VARS DOXYGEN_ALIASES)

        add_doxygen_docs(project_docs ${PROJECT_SOURCE_DIR})
      endif()

    The resultant ``Doxyfile`` configuration will contain the following lines:

    .. code-block:: text
      :caption: ``Doxyfile.project_docs``

      PROJECT_BRIEF = "String with spaces"
      ALIASES       = somealias="@some_command param" anotherAlias=@foobar

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

For compatibility with previous versions of CMake, the following variables
are also defined but they are deprecated and should no longer be used:

``DOXYGEN_FOUND``
  .. deprecated:: 4.2
    Use ``Doxygen_FOUND``.

  Boolean indicating whether the (requested version of) ``doxygen`` executable
  and all requested required components were found.  It has a boolean value
  of ``YES`` or ``NO``.

``DOXYGEN_EXECUTABLE``
  .. deprecated:: 3.9
    Use ``Doxygen::doxygen`` imported target instead of referring to the
    ``doxygen`` executable directly.

  Cache variable containing the path to the ``doxygen`` command.

``DOXYGEN_DOT_FOUND``
  .. deprecated:: 3.9
    Use ``Doxygen_dot_FOUND``.

  Boolean result variable indicating whether ``dot`` executable was found.

``DOXYGEN_DOT_EXECUTABLE``
  .. deprecated:: 3.9
    Use ``Doxygen::dot`` imported target instead of referring to the ``dot``
    executable directly.

  Cache variable containing the path to the ``dot`` command-line executable.

``DOXYGEN_DOT_PATH``
  .. deprecated:: 3.9

  Result variable containing the path to the directory where the ``dot``
  executable is located as reported in ``DOXYGEN_DOT_EXECUTABLE``.  The path
  may have forward slashes even on Windows and is not suitable for direct
  substitution into a ``Doxyfile.in`` template.  If this value is needed, get
  the :prop_tgt:`IMPORTED_LOCATION` property of the ``Doxygen::dot`` target
  and use :command:`get_filename_component` to extract the directory part of
  that path.  Consider also using :command:`file(TO_NATIVE_PATH)` to prepare
  the path for a Doxygen configuration file.

``DOXYGEN_SKIP_DOT``
  .. deprecated:: 3.9

  This hint variable has no effect when specifying components in
  ``find_package(Doxygen COMPONENTS ...)``.  In backward-compatibility mode
  (i.e. without specifying components) it prevents this find module from
  searching for Graphviz's ``dot`` utility.

``DOXYGEN_VERSION``
  .. deprecated:: 4.2
    Superseded by the ``Doxygen_VERSION``.

  The version of Doxygen found.

Examples
^^^^^^^^

Examples: Finding Doxygen
"""""""""""""""""""""""""

Finding Doxygen:

.. code-block:: cmake

  find_package(Doxygen)

Or, finding Doxygen and specifying a minimum required version:

.. code-block:: cmake

  find_package(Doxygen 1.9)

Or, finding Doxygen and making it required (if not found, processing stops
with an error message):

.. code-block:: cmake

  find_package(Doxygen REQUIRED)

Or, finding Doxygen as required and specifying ``dot`` tool as required
component and ``mscgen`` and ``dia`` tools as optional components:

.. code-block:: cmake

  find_package(Doxygen REQUIRED COMPONENTS dot OPTIONAL_COMPONENTS mscgen dia)

Example: Using Doxygen in CMake
"""""""""""""""""""""""""""""""

The following example demonstrates how to find Doxygen and create documentation
from source files at build phase.  Once project is built, generated
documentation files will be located in the ``html`` directory inside the
project binary directory:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  cmake_minimum_required(VERSION 3.24)
  project(
    DoxygenExample
    DESCRIPTION "Example project using Doxygen"
    VERSION 1.2.3
  )

  add_executable(example example.c)

  find_package(Doxygen)

  if(Doxygen_FOUND)
    doxygen_add_docs(project_docs example.c ALL USE_STAMP_FILE)
  endif()

.. code-block:: c
  :caption: ``example.c``

  /**
   * @file example.c
   * @brief A simple example to demonstrate Doxygen.
   */

  #include <stdio.h>

  /**
   * @brief Calculates the sum of two integers.
   *
   * @param a First integer.
   * @param b Second integer.
   * @return Sum of a and b.
   *
   * @par Example
   * @code
   * int result = sum(3, 4);
   * printf("%d\n", result); // Outputs: 7
   * @endcode
   */
  int sum(int a, int b) { return a + b; }

  /**
   * @brief Main function.
   *
   * @return 0 on success.
   */
  int main(void)
  {
    int result = sum(5, 7);
    printf("Result: %d\n", result);
    return 0;
  }

Example: Configuring Doxygen With Variables
"""""""""""""""""""""""""""""""""""""""""""

In the following example, Doxygen configuration is customized using CMake
variables.  The configuration sets file patterns when using a directory as
the source input (:variable:`CMAKE_CURRENT_SOURCE_DIR`), enables a theme
toggle for switching between light and dark modes, suppresses Doxygen's
standard output during the build phase, specifies a Markdown file as the
main page, and disables warnings about undocumented code:

.. code-block:: cmake

  find_package(Doxygen)

  if(Doxygen_FOUND)
    set(DOXYGEN_FILE_PATTERNS *.c *.cxx *.md)
    set(DOXYGEN_HTML_COLORSTYLE "TOGGLE")
    set(DOXYGEN_QUIET YES)
    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
    set(DOXYGEN_WARN_IF_UNDOCUMENTED NO)

    doxygen_add_docs(example_docs ${CMAKE_CURRENT_SOURCE_DIR} ALL)
  endif()

Example: Custom Configuration File
""""""""""""""""""""""""""""""""""

In the following example, a custom ``Doxyfile`` configuration file is created
in the current binary directory (:variable:`CMAKE_CURRENT_BINARY_DIR`) prior
to calling the ``doxygen_add_docs()``.  This allows project-specific
configuration tags to be customized as needed:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(Doxygen)

  if(Doxygen_FOUND)
    configure_file(Doxyfile.in Doxyfile)

    doxygen_add_docs(
      example_docs
      foo.c bar.c
      ALL
      USE_STAMP_FILE
      COMMENT "Generating project documentation with custom Doxyfile"
      CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    )
  endif()

.. code-block:: text
  :caption: ``Doxyfile.in``

  PROJECT_NAME           = "Customized project name"
  OUTPUT_DIRECTORY       = "@CMAKE_CURRENT_BINARY_DIR@"
  # ...
#]=======================================================================]

# For backwards compatibility support
if(Doxygen_FIND_QUIETLY)
    set(DOXYGEN_FIND_QUIETLY TRUE)
endif()

# ===== Rationale for OS X AppBundle mods below =====
#  With the OS X GUI version, Doxygen likes to be installed to /Applications
#  and it contains the doxygen executable in the bundle. In the versions I've
#  seen, it is located in Resources, but in general, more often binaries are
#  located in MacOS.
#
#  NOTE: The official Doxygen.app distributed for OS X uses non-standard
#  conventions. Instead of the command-line "doxygen" tool being placed in
#  Doxygen.app/Contents/MacOS, "Doxywizard" is placed there instead and
#  "doxygen" is placed in Contents/Resources.  This is most likely done
#  so that something happens when people double-click on the Doxygen.app
#  package. Unfortunately, CMake gets confused by this as when it sees the
#  bundle it uses "Doxywizard" as the executable to use instead of
#  "doxygen". Therefore to work-around this issue we temporarily disable
#  the app-bundle feature, just for this CMake module:
#
if(APPLE)
    # Save the old setting
    set(TEMP_DOXYGEN_SAVE_CMAKE_FIND_APPBUNDLE ${CMAKE_FIND_APPBUNDLE})
    # Disable the App-bundle detection feature
    set(CMAKE_FIND_APPBUNDLE "NEVER")
endif()
# FYI:
# In older versions of OS X Doxygen, dot was included with the Doxygen bundle,
# but newer versions require you to download Graphviz.app which contains "dot"
# or use something like homebrew.
# ============== End OSX stuff ================

include(FindPackageHandleStandardArgs)

#
# Find Doxygen...
#
function(_Doxygen_get_version doxy_version result_var doxy_path)
        execute_process(
            COMMAND "${doxy_path}" --version
            OUTPUT_VARIABLE full_doxygen_version
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE version_result
        )

        # Ignore any commit hashes, etc.
        string(REGEX MATCH [[^[0-9]+\.[0-9]+\.[0-9]+]] sem_doxygen_version "${full_doxygen_version}")

        set(${result_var} ${version_result} PARENT_SCOPE)
        set(${doxy_version} ${sem_doxygen_version} PARENT_SCOPE)
endfunction()

function(_Doxygen_version_validator version_match doxy_path)
    if(NOT DEFINED Doxygen_FIND_VERSION)
        set(${is_valid_version} TRUE PARENT_SCOPE)
    else()
        _Doxygen_get_version(candidate_version version_result "${doxy_path}")

        find_package_check_version("${candidate_version}" valid_doxy_version
            HANDLE_VERSION_RANGE
        )

        set(${version_match} "${valid_doxy_version}" PARENT_SCOPE)
    endif()
endfunction()

macro(_Doxygen_find_doxygen)
    find_program(
        DOXYGEN_EXECUTABLE
        NAMES doxygen
        PATHS
            "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\doxygen_is1;Inno Setup: App Path]/bin"
            /Applications/Doxygen.app/Contents/Resources
            /Applications/Doxygen.app/Contents/MacOS
            /Applications/Utilities/Doxygen.app/Contents/Resources
            /Applications/Utilities/Doxygen.app/Contents/MacOS
        DOC "Doxygen documentation generation tool (https://www.doxygen.nl)"
        VALIDATOR _Doxygen_version_validator
    )
    mark_as_advanced(DOXYGEN_EXECUTABLE)

    if(DOXYGEN_EXECUTABLE)
        _Doxygen_get_version(Doxygen_VERSION _Doxygen_version_result "${DOXYGEN_EXECUTABLE}")
        set(DOXYGEN_VERSION "${Doxygen_VERSION}")

        if(_Doxygen_version_result)
            if(NOT Doxygen_FIND_QUIETLY)
                message(WARNING "Doxygen executable failed unexpected while determining version (exit status: ${_Doxygen_version_result}). Disabling Doxygen.")
            endif()
            set(DOXYGEN_EXECUTABLE "${DOXYGEN_EXECUTABLE}-FAILED_EXECUTION-NOTFOUND")
        else()
            # Create an imported target for Doxygen
            if(NOT TARGET Doxygen::doxygen)
                add_executable(Doxygen::doxygen IMPORTED GLOBAL)
                set_target_properties(Doxygen::doxygen PROPERTIES
                    IMPORTED_LOCATION "${DOXYGEN_EXECUTABLE}"
                )
            endif()
        endif()
    endif()
endmacro()

#
# Find Diagram Editor...
#
macro(_Doxygen_find_dia)
    set(_x86 "(x86)")
    find_program(
        DOXYGEN_DIA_EXECUTABLE
        NAMES dia
        PATHS
            "$ENV{ProgramFiles}/Dia"
            "$ENV{ProgramFiles${_x86}}/Dia"
        DOC "Diagram Editor tool for use with Doxygen"
    )
    mark_as_advanced(DOXYGEN_DIA_EXECUTABLE)

    if(DOXYGEN_DIA_EXECUTABLE)
        # The Doxyfile wants the path to the utility, not the entire path
        # including file name
        get_filename_component(DOXYGEN_DIA_PATH
                              "${DOXYGEN_DIA_EXECUTABLE}"
                              DIRECTORY)
        if(WIN32)
            file(TO_NATIVE_PATH "${DOXYGEN_DIA_PATH}" DOXYGEN_DIA_PATH)
        endif()

        # Create an imported target for component
        if(NOT TARGET Doxygen::dia)
            add_executable(Doxygen::dia IMPORTED GLOBAL)
            set_target_properties(Doxygen::dia PROPERTIES
                IMPORTED_LOCATION "${DOXYGEN_DIA_EXECUTABLE}"
            )
        endif()
    endif()

    unset(_x86)
endmacro()

#
# Find Graphviz Dot...
#
macro(_Doxygen_find_dot)
    if(WIN32)
        set(_x86 "(x86)")
        file(
            GLOB _Doxygen_GRAPHVIZ_BIN_DIRS
            "$ENV{ProgramFiles}/Graphviz*/bin"
            "$ENV{ProgramFiles${_x86}}/Graphviz*/bin"
        )
        unset(_x86)
    else()
        set(_Doxygen_GRAPHVIZ_BIN_DIRS "")
    endif()

    find_program(
        DOXYGEN_DOT_EXECUTABLE
        NAMES dot
        PATHS
            ${_Doxygen_GRAPHVIZ_BIN_DIRS}
            "$ENV{ProgramFiles}/ATT/Graphviz/bin"
            "C:/Program Files/ATT/Graphviz/bin"
            [HKEY_LOCAL_MACHINE\\SOFTWARE\\ATT\\Graphviz;InstallPath]/bin
            /Applications/Graphviz.app/Contents/MacOS
            /Applications/Utilities/Graphviz.app/Contents/MacOS
            /Applications/Doxygen.app/Contents/Resources
            /Applications/Doxygen.app/Contents/MacOS
            /Applications/Utilities/Doxygen.app/Contents/Resources
            /Applications/Utilities/Doxygen.app/Contents/MacOS
        DOC "Dot tool for use with Doxygen"
    )
    mark_as_advanced(DOXYGEN_DOT_EXECUTABLE)

    if(DOXYGEN_DOT_EXECUTABLE)
        # The Doxyfile wants the path to the utility, not the entire path
        # including file name
        get_filename_component(DOXYGEN_DOT_PATH
                               "${DOXYGEN_DOT_EXECUTABLE}"
                               DIRECTORY)
        if(WIN32)
            file(TO_NATIVE_PATH "${DOXYGEN_DOT_PATH}" DOXYGEN_DOT_PATH)
        endif()

        # Create an imported target for component
        if(NOT TARGET Doxygen::dot)
            add_executable(Doxygen::dot IMPORTED GLOBAL)
            set_target_properties(Doxygen::dot PROPERTIES
                IMPORTED_LOCATION "${DOXYGEN_DOT_EXECUTABLE}"
            )
        endif()
    endif()

    unset(_Doxygen_GRAPHVIZ_BIN_DIRS)
endmacro()

#
# Find Message Sequence Chart...
#
macro(_Doxygen_find_mscgen)
    set(_x86 "(x86)")
    find_program(
        DOXYGEN_MSCGEN_EXECUTABLE
        NAMES mscgen
        PATHS
            "$ENV{ProgramFiles}/Mscgen"
            "$ENV{ProgramFiles${_x86}}/Mscgen"
        DOC "Message sequence chart tool for use with Doxygen"
    )
    mark_as_advanced(DOXYGEN_MSCGEN_EXECUTABLE)

    if(DOXYGEN_MSCGEN_EXECUTABLE)
        # The Doxyfile wants the path to the utility, not the entire path
        # including file name
        get_filename_component(DOXYGEN_MSCGEN_PATH
                               "${DOXYGEN_MSCGEN_EXECUTABLE}"
                               DIRECTORY)
        if(WIN32)
            file(TO_NATIVE_PATH "${DOXYGEN_MSCGEN_PATH}" DOXYGEN_MSCGEN_PATH)
        endif()

        # Create an imported target for component
        if(NOT TARGET Doxygen::mscgen)
            add_executable(Doxygen::mscgen IMPORTED GLOBAL)
            set_target_properties(Doxygen::mscgen PROPERTIES
                IMPORTED_LOCATION "${DOXYGEN_MSCGEN_EXECUTABLE}"
            )
        endif()
    endif()

    unset(_x86)
endmacro()

# Make sure `doxygen` is one of the components to find
set(_Doxygen_keep_backward_compat FALSE)
if(NOT Doxygen_FIND_COMPONENTS)
    # Search at least for `doxygen` executable
    set(Doxygen_FIND_COMPONENTS doxygen)
    # Preserve backward compatibility:
    # search for `dot` also if `DOXYGEN_SKIP_DOT` is not explicitly disable this.
    if(NOT DOXYGEN_SKIP_DOT)
        list(APPEND Doxygen_FIND_COMPONENTS dot)
    endif()
    set(_Doxygen_keep_backward_compat TRUE)
elseif(NOT doxygen IN_LIST Doxygen_FIND_COMPONENTS)
    list(INSERT Doxygen_FIND_COMPONENTS 0 doxygen)
endif()

#
# Find all requested components of Doxygen...
#
foreach(_comp IN LISTS Doxygen_FIND_COMPONENTS)
    if(_comp STREQUAL "doxygen")
        _Doxygen_find_doxygen()
    elseif(_comp STREQUAL "dia")
        _Doxygen_find_dia()
    elseif(_comp STREQUAL "dot")
        _Doxygen_find_dot()
    elseif(_comp STREQUAL "mscgen")
        _Doxygen_find_mscgen()
    else()
        message(WARNING "${_comp} is not a valid Doxygen component")
        set(Doxygen_${_comp}_FOUND FALSE)
        continue()
    endif()

    if(TARGET Doxygen::${_comp})
        set(Doxygen_${_comp}_FOUND TRUE)
    else()
        set(Doxygen_${_comp}_FOUND FALSE)
    endif()
endforeach()
unset(_comp)

# Verify find results
find_package_handle_standard_args(
    Doxygen
    REQUIRED_VARS DOXYGEN_EXECUTABLE
    VERSION_VAR Doxygen_VERSION
    HANDLE_VERSION_RANGE
    HANDLE_COMPONENTS
)

#
# Backwards compatibility...
#
if(APPLE)
    # Restore the old app-bundle setting
    set(CMAKE_FIND_APPBUNDLE ${TEMP_DOXYGEN_SAVE_CMAKE_FIND_APPBUNDLE})
endif()

# Maintain the _FOUND variables as "YES" or "NO" for backwards
# compatibility. This allows people to substitute them directly into
# Doxyfile with configure_file().
if(Doxygen_FOUND)
    set(DOXYGEN_FOUND "YES")
else()
    set(DOXYGEN_FOUND "NO")
endif()
if(_Doxygen_keep_backward_compat)
    if(Doxygen_dot_FOUND)
        set(DOXYGEN_DOT_FOUND "YES")
    else()
        set(DOXYGEN_DOT_FOUND "NO")
    endif()

    # For backwards compatibility support for even older CMake versions
    set(DOXYGEN ${DOXYGEN_EXECUTABLE})
    set(DOT ${DOXYGEN_DOT_EXECUTABLE})

    # No need to keep any backward compatibility for `DOXYGEN_MSCGEN_XXX`
    # and `DOXYGEN_DIA_XXX` since they were not supported before component
    # support was added
endif()
unset(_Doxygen_keep_backward_compat)

#
# Allow full control of Doxygen from CMakeLists.txt
#

# Prepare a template Doxyfile and Doxygen's default values CMake file
if(TARGET Doxygen::doxygen)
    # If doxygen was found, use it to generate a minimal default Doxyfile.
    # We will delete this file after we have finished using it below to
    # generate the other files that doxygen_add_docs() will use.
    set(_Doxygen_tpl "${CMAKE_BINARY_DIR}/CMakeDoxyfile.tpl")
    execute_process(
        COMMAND "${DOXYGEN_EXECUTABLE}" -s -g "${_Doxygen_tpl}"
        OUTPUT_QUIET
        RESULT_VARIABLE _Doxygen_tpl_result
    )
    if(_Doxygen_tpl_result)
        message(FATAL_ERROR
                "Unable to generate Doxyfile template: ${_Doxygen_tpl_result}")
    elseif(NOT EXISTS "${_Doxygen_tpl}")
        message(FATAL_ERROR
                "Doxygen has failed to generate a Doxyfile template")
    endif()

    # Write a do-not-edit header to files we are going to generate...
    set(_Doxygen_do_not_edit_header
[[
#
# DO NOT EDIT! THIS FILE WAS GENERATED BY CMAKE!
#

]]
    )
    # We only need one copy of these across the whole build, since their
    # content is only dependent on the version of Doxygen being used. Therefore
    # we always put them at the top of the build tree so that they are in a
    # predictable location.
    set(_doxyfile_in       "${CMAKE_BINARY_DIR}/CMakeDoxyfile.in")
    set(_doxyfile_defaults "${CMAKE_BINARY_DIR}/CMakeDoxygenDefaults.cmake")

    set(_doxyfile_in_contents "")
    set(_doxyfile_defaults_contents "")

    # Get strings containing a configuration key from the template Doxyfile
    # we obtained from this version of Doxygen. Because some options are split
    # across multiple lines by ending lines with backslashes, we cannot just
    # use file(STRINGS...) with a REGEX. Instead, read lines without a REGEX
    # so that file(STRINGS...) handles the trailing backslash as a line
    # continuation. It stores multi-lines as lists, so we then have to replace
    # the ";" list separator with backslashed newlines again so that we get the
    # original content stored back as the value part.
    file(STRINGS "${_Doxygen_tpl}" _file_lines)
    unset(_Doxygen_tpl_params)
    foreach(_line IN LISTS _file_lines)
        if(_line MATCHES "([A-Z][A-Z0-9_]+)( *=)(.*)")
            set(_key "${CMAKE_MATCH_1}")
            set(_eql "${CMAKE_MATCH_2}")
            set(_value "${CMAKE_MATCH_3}")
            string(REPLACE "\\" "\\\\" _value "${_value}")
            string(REPLACE ";" "\\\n" _value "${_value}")
            list(APPEND _Doxygen_tpl_params "${_key}${_eql}${_value}")
        endif()
    endforeach()

    # Build up a Doxyfile that provides @configVar@ substitutions for each
    # Doxygen config option as well as a separate CMake script which provides
    # the default value for each of those options if the project doesn't supply
    # them. Each config option will support substitution of a CMake variable
    # of the same name except with DOXYGEN_ prepended.
    foreach(_Doxygen_param IN LISTS _Doxygen_tpl_params)
        if(_Doxygen_param MATCHES "([A-Z][A-Z0-9_]+)( *)=( (.*))?")
            # Ok, this is a config key with a value
            if(CMAKE_MATCH_COUNT EQUAL 4)
                string(APPEND _doxyfile_in_contents
                       "${CMAKE_MATCH_1}${CMAKE_MATCH_2}= @DOXYGEN_${CMAKE_MATCH_1}@\n")
                # Remove the backslashes we had to preserve to handle newlines
                string(REPLACE "\\\n" "\n" _value "${CMAKE_MATCH_4}")
                string(APPEND _doxyfile_defaults_contents
"if(NOT DEFINED DOXYGEN_${CMAKE_MATCH_1})
    set(DOXYGEN_${CMAKE_MATCH_1} ${_value})
endif()
")
            # Ok, this is a config key with empty default value
            elseif(CMAKE_MATCH_COUNT EQUAL 2)
                string(APPEND _doxyfile_in_contents
                       "${CMAKE_MATCH_1}${CMAKE_MATCH_2}= @DOXYGEN_${CMAKE_MATCH_1}@\n")
            else()
                message(AUTHOR_WARNING
"Unexpected line format! Code review required!\nFault line: ${_Doxygen_param}")
            endif()
        else()
            message(AUTHOR_WARNING
"Unexpected line format! Code review required!\nFault line: ${_Doxygen_param}")
        endif()
    endforeach()
    file(WRITE "${_doxyfile_defaults}" "${_Doxygen_do_not_edit_header}"
                                       "${_doxyfile_defaults_contents}")
    file(WRITE "${_doxyfile_in}"       "${_Doxygen_do_not_edit_header}"
                                       "${_doxyfile_in_contents}")

    # Ok, dumped defaults are not needed anymore...
    file(REMOVE "${_Doxygen_tpl}")

    unset(_Doxygen_param)
    unset(_Doxygen_tpl_params)
    unset(_Doxygen_do_not_edit_header)
    unset(_Doxygen_tpl)

endif()

function(doxygen_quote_value VARIABLE)
    # Quote a value of the given variable if:
    # - VARIABLE parameter was really given
    # - the variable it names is defined and is not present in the list
    #   specified by DOXYGEN_VERBATIM_VARS (if set)
    # - the value of the named variable isn't already quoted
    # - the value has spaces
    if(VARIABLE AND DEFINED ${VARIABLE} AND
       NOT ${VARIABLE} MATCHES "^\".* .*\"$" AND ${VARIABLE} MATCHES " " AND
       NOT (DEFINED DOXYGEN_VERBATIM_VARS AND
            "${VARIABLE}" IN_LIST DOXYGEN_VERBATIM_VARS))
        set(${VARIABLE} "\"${${VARIABLE}}\"" PARENT_SCOPE)
    endif()
endfunction()

function(doxygen_list_to_quoted_strings LIST_VARIABLE)
    if(LIST_VARIABLE AND DEFINED ${LIST_VARIABLE})
        unset(_inputs)
        unset(_sep)
        unset(_verbatim)
        # Have to test if list items should be treated as verbatim here
        # because we lose the variable name when we pass just one list item
        # to doxygen_quote_value() below
        if(DEFINED DOXYGEN_VERBATIM_VARS AND
           "${LIST_VARIABLE}" IN_LIST DOXYGEN_VERBATIM_VARS)
            set(_verbatim True)
        endif()
        foreach(_in IN LISTS ${LIST_VARIABLE})
            if(NOT _verbatim)
                doxygen_quote_value(_in)
            endif()
            string(APPEND _inputs "${_sep}${_in}")
            set(_sep " ")
        endforeach()
        set(${LIST_VARIABLE} "${_inputs}" PARENT_SCOPE)
    endif()
endfunction()

function(doxygen_add_docs targetName)
    set(_options ALL USE_STAMP_FILE)
    set(_one_value_args WORKING_DIRECTORY COMMENT CONFIG_FILE)
    set(_multi_value_args)
    cmake_parse_arguments(_args
                          "${_options}"
                          "${_one_value_args}"
                          "${_multi_value_args}"
                          ${ARGN})

    if(NOT _args_COMMENT)
        set(_args_COMMENT "Generating API documentation for ${targetName}")
    endif()

    if(NOT _args_WORKING_DIRECTORY)
        set(_args_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    if(DEFINED DOXYGEN_INPUT)
        message(WARNING
"DOXYGEN_INPUT is set but it will be ignored. Pass the files and directories \
directly to the doxygen_add_docs() command instead.")
    endif()
    set(DOXYGEN_INPUT ${_args_UNPARSED_ARGUMENTS})

    if(NOT TARGET Doxygen::doxygen)
        message(FATAL_ERROR "Doxygen was not found, needed by \
doxygen_add_docs() for target ${targetName}")
    endif()

    # If not already defined, set some relevant defaults based on the
    # assumption that the documentation is for the whole project. Details
    # specified in the project() command will be used to populate a number of
    # these defaults.

    if(NOT DEFINED DOXYGEN_PROJECT_NAME)
        # The PROJECT_NAME tag is a single word (or a sequence of words
        # surrounded by double-quotes, unless you are using Doxywizard) that
        # should identify the project for which the documentation is generated.
        # This name is used in the title of most generated pages and in a few
        # other places. The default value is: My Project.
        set(DOXYGEN_PROJECT_NAME ${PROJECT_NAME})
    endif()

    if(NOT DEFINED DOXYGEN_PROJECT_NUMBER)
        # The PROJECT_NUMBER tag can be used to enter a project or revision
        # number. This could be handy for archiving the generated documentation
        # or if some version control system is used.
        set(DOXYGEN_PROJECT_NUMBER ${PROJECT_VERSION})
    endif()

    if(NOT DEFINED DOXYGEN_PROJECT_BRIEF)
        # Using the PROJECT_BRIEF tag one can provide an optional one line
        # description for a project that appears at the top of each page and
        # should give viewer a quick idea about the purpose of the project.
        # Keep the description short.
        set(DOXYGEN_PROJECT_BRIEF "${PROJECT_DESCRIPTION}")
    endif()

    if(NOT DEFINED DOXYGEN_RECURSIVE)
        # The RECURSIVE tag can be used to specify whether or not
        # subdirectories should be searched for input files as well. CMake
        # projects generally evolve to span multiple directories, so it makes
        # more sense for this to be on by default. Doxygen's default value
        # has this setting turned off, so we override it.
        set(DOXYGEN_RECURSIVE YES)
    endif()

    if(NOT DEFINED DOXYGEN_OUTPUT_DIRECTORY)
        # The OUTPUT_DIRECTORY tag is used to specify the (relative or
        # absolute) path into which the generated documentation will be
        # written. If a relative path is used, Doxygen will interpret it as
        # being relative to the location where doxygen was started, but we need
        # to run Doxygen in the source tree so that relative input paths work
        # intuitively. Therefore, we ensure that the output directory is always
        # an absolute path and if the project provided a relative path, we
        # treat it as relative to the current BINARY directory so that output
        # is not generated inside the source tree.
        set(DOXYGEN_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    elseif(NOT IS_ABSOLUTE "${DOXYGEN_OUTPUT_DIRECTORY}")
        get_filename_component(DOXYGEN_OUTPUT_DIRECTORY
                               "${DOXYGEN_OUTPUT_DIRECTORY}"
                               ABSOLUTE
                               BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    if(NOT DEFINED DOXYGEN_HAVE_DOT)
        # If you set the HAVE_DOT tag to YES then doxygen will assume the dot
        # tool is available from the path. This tool is part of Graphviz (see:
        # https://www.graphviz.org/), a graph visualization toolkit from AT&T
        # and Lucent Bell Labs. The other options in this section have no
        # effect if this option is set to NO.
        # Doxygen's default value is: NO.
        if(Doxygen_dot_FOUND)
          set(DOXYGEN_HAVE_DOT "YES")
        else()
          set(DOXYGEN_HAVE_DOT "NO")
        endif()
    endif()

    if(NOT DEFINED DOXYGEN_DOT_MULTI_TARGETS)
        # Set the DOT_MULTI_TARGETS tag to YES to allow dot to generate
        # multiple output files in one run (i.e. multiple -o and -T options on
        # the command line). This makes dot run faster, but since only newer
        # versions of dot (>1.8.10) support this, Doxygen disables this feature
        # by default.
        # This tag requires that the tag HAVE_DOT is set to YES.
        set(DOXYGEN_DOT_MULTI_TARGETS YES)
    endif()

    if(NOT DEFINED DOXYGEN_GENERATE_LATEX)
        # If the GENERATE_LATEX tag is set to YES, doxygen will generate LaTeX
        # output. We only want the HTML output enabled by default, so we turn
        # this off if the project hasn't specified it.
        set(DOXYGEN_GENERATE_LATEX NO)
    endif()

    if(NOT DEFINED DOXYGEN_WARN_FORMAT)
        if(CMAKE_VS_MSBUILD_COMMAND OR CMAKE_VS_DEVENV_COMMAND)
            # The WARN_FORMAT tag determines the format of the warning messages
            # that doxygen can produce. The string should contain the $file,
            # $line and $text tags, which will be replaced by the file and line
            # number from which the warning originated and the warning text.
            # Optionally, the format may contain $version, which will be
            # replaced by the version of the file (if it could be obtained via
            # FILE_VERSION_FILTER).
            # Doxygen's default value is: $file:$line: $text
            set(DOXYGEN_WARN_FORMAT "$file($line) : $text ")
        endif()
    endif()

    if(DEFINED DOXYGEN_WARN_LOGFILE AND NOT IS_ABSOLUTE "${DOXYGEN_WARN_LOGFILE}")
        # The WARN_LOGFILE tag can be used to specify a file to which warning and error
        # messages should be written. If left blank the output is written to standard
        # error (stderr).
        get_filename_component(DOXYGEN_WARN_LOGFILE
                               "${DOXYGEN_WARN_LOGFILE}"
                               ABSOLUTE
                               BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    endif()

    # Any files from the INPUT that match any of the EXCLUDE_PATTERNS will be
    # excluded from the set of input files. We provide some additional patterns
    # to prevent commonly unwanted things from CMake builds being pulled in.
    #
    # Note that the wildcards are matched against the file with absolute path,
    # so to exclude all test directories for example use the pattern */test/*
    list(
        APPEND
        DOXYGEN_EXCLUDE_PATTERNS
        "*/.git/*"
        "*/.svn/*"
        "*/.hg/*"
        "*/CMakeFiles/*"
        "*/_CPack_Packages/*"
        "DartConfiguration.tcl"
        "CMakeLists.txt"
        "CMakeCache.txt"
    )

    # Now bring in Doxygen's defaults for those things the project has not
    # already set and we have not provided above
    include("${CMAKE_BINARY_DIR}/CMakeDoxygenDefaults.cmake" OPTIONAL)

    # Cleanup built HTMLs on "make clean"
    # TODO Any other dirs?
    if(DOXYGEN_GENERATE_HTML)
        if(IS_ABSOLUTE "${DOXYGEN_HTML_OUTPUT}")
            set(_args_clean_html_dir "${DOXYGEN_HTML_OUTPUT}")
        else()
            set(_args_clean_html_dir
                "${DOXYGEN_OUTPUT_DIRECTORY}/${DOXYGEN_HTML_OUTPUT}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY
            ADDITIONAL_CLEAN_FILES "${_args_clean_html_dir}")
    endif()

    # Build up a list of files we can identify from the inputs so we can list
    # them as DEPENDS and SOURCES in the custom command/target (the latter
    # makes them display in IDEs). This must be done before we transform the
    # various DOXYGEN_... variables below because we need to process
    # DOXYGEN_INPUT as a list first.
    unset(_sources)
    foreach(_item IN LISTS DOXYGEN_INPUT)
        get_filename_component(_abs_item "${_item}" ABSOLUTE
                               BASE_DIR "${_args_WORKING_DIRECTORY}")
        get_source_file_property(_isGenerated "${_abs_item}" GENERATED)
        if(_isGenerated OR
           (EXISTS "${_abs_item}" AND
            NOT IS_DIRECTORY "${_abs_item}" AND
            NOT IS_SYMLINK "${_abs_item}"))
            list(APPEND _sources "${_abs_item}")
        elseif(_args_USE_STAMP_FILE)
            message(FATAL_ERROR "Source does not exist or is not a file:\n"
                "    ${_abs_item}\n"
                "Only existing files may be specified when the "
                "USE_STAMP_FILE option is given.")
        endif()
    endforeach()

    # Transform known list type options into space separated strings.
    set(_doxygen_list_options
        ABBREVIATE_BRIEF
        ALIASES
        CITE_BIB_FILES
        DIAFILE_DIRS
        DOTFILE_DIRS
        DOT_FONTPATH
        ENABLED_SECTIONS
        EXAMPLE_PATH
        EXAMPLE_PATTERNS
        EXCLUDE
        EXCLUDE_PATTERNS
        EXCLUDE_SYMBOLS
        EXPAND_AS_DEFINED
        EXTENSION_MAPPING
        EXTRA_PACKAGES
        EXTRA_SEARCH_MAPPINGS
        FILE_PATTERNS
        FILTER_PATTERNS
        FILTER_SOURCE_PATTERNS
        HTML_EXTRA_FILES
        HTML_EXTRA_STYLESHEET
        IGNORE_PREFIX
        IMAGE_PATH
        INCLUDE_FILE_PATTERNS
        INCLUDE_PATH
        INPUT
        LATEX_EXTRA_FILES
        LATEX_EXTRA_STYLESHEET
        MATHJAX_EXTENSIONS
        MSCFILE_DIRS
        PLANTUML_INCLUDE_PATH
        PREDEFINED
        QHP_CUST_FILTER_ATTRS
        QHP_SECT_FILTER_ATTRS
        STRIP_FROM_INC_PATH
        STRIP_FROM_PATH
        TAGFILES
        TCL_SUBST
    )
    foreach(_item IN LISTS _doxygen_list_options)
        doxygen_list_to_quoted_strings(DOXYGEN_${_item})
    endforeach()

    # Transform known single value variables which may contain spaces, such as
    # paths or description strings.
    set(_doxygen_quoted_options
        CHM_FILE
        DIA_PATH
        DOCBOOK_OUTPUT
        DOCSET_FEEDNAME
        DOCSET_PUBLISHER_NAME
        DOT_FONTNAME
        DOT_PATH
        EXTERNAL_SEARCH_ID
        FILE_VERSION_FILTER
        GENERATE_TAGFILE
        HHC_LOCATION
        HTML_FOOTER
        HTML_HEADER
        HTML_OUTPUT
        HTML_STYLESHEET
        INPUT_FILTER
        LATEX_FOOTER
        LATEX_HEADER
        LATEX_OUTPUT
        LAYOUT_FILE
        MAN_OUTPUT
        MAN_SUBDIR
        MATHJAX_CODEFILE
        MSCGEN_PATH
        OUTPUT_DIRECTORY
        PERL_PATH
        PLANTUML_JAR_PATH
        PROJECT_BRIEF
        PROJECT_LOGO
        PROJECT_NAME
        QCH_FILE
        QHG_LOCATION
        QHP_CUST_FILTER_NAME
        QHP_VIRTUAL_FOLDER
        RTF_EXTENSIONS_FILE
        RTF_OUTPUT
        RTF_STYLESHEET_FILE
        SEARCHDATA_FILE
        USE_MDFILE_AS_MAINPAGE
        WARN_FORMAT
        WARN_LOGFILE
        XML_OUTPUT
    )

    # Store the unmodified value of DOXYGEN_OUTPUT_DIRECTORY prior to invoking
    # doxygen_quote_value() below. This will mutate the string specifically for
    # consumption by Doxygen's config file, which we do not want when we use it
    # later in the custom target's commands.
    set( _original_doxygen_output_dir ${DOXYGEN_OUTPUT_DIRECTORY} )

    foreach(_item IN LISTS _doxygen_quoted_options)
        doxygen_quote_value(DOXYGEN_${_item})
    endforeach()

    # Prepare doxygen configuration file
    set(_doxyfile_template "${CMAKE_BINARY_DIR}/CMakeDoxyfile.in")
    if(_args_CONFIG_FILE)
        if(NOT EXISTS "${_args_CONFIG_FILE}")
            message(FATAL_ERROR "Option CONFIG_FILE specifies file:\n ${_args_CONFIG_FILE}\nbut it does not exist.")
        endif()
        set(_target_doxyfile "${_args_CONFIG_FILE}")
    else()
        set(_target_doxyfile "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.${targetName}")
        configure_file("${_doxyfile_template}" "${_target_doxyfile}")
    endif()

    unset(_all)
    if(${_args_ALL})
        set(_all ALL)
    endif()

    # Only create the stamp file if asked to. If we don't create it,
    # the target will always be considered out-of-date.
    if(_args_USE_STAMP_FILE)
        set(__stamp_file "${CMAKE_CURRENT_BINARY_DIR}/${targetName}.stamp")
        add_custom_command(
            VERBATIM
            OUTPUT ${__stamp_file}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${_original_doxygen_output_dir}
            COMMAND "${DOXYGEN_EXECUTABLE}" "${_target_doxyfile}"
            COMMAND ${CMAKE_COMMAND} -E touch ${__stamp_file}
            WORKING_DIRECTORY "${_args_WORKING_DIRECTORY}"
            DEPENDS "${_target_doxyfile}" ${_sources}
            COMMENT "${_args_COMMENT}"
        )
        add_custom_target(${targetName} ${_all}
            DEPENDS ${__stamp_file}
            SOURCES ${_sources}
        )
        unset(__stamp_file)
    else()
        add_custom_target( ${targetName} ${_all} VERBATIM
            COMMAND ${CMAKE_COMMAND} -E make_directory ${_original_doxygen_output_dir}
            COMMAND "${DOXYGEN_EXECUTABLE}" "${_target_doxyfile}"
            WORKING_DIRECTORY "${_args_WORKING_DIRECTORY}"
            DEPENDS "${_target_doxyfile}" ${_sources}
            COMMENT "${_args_COMMENT}"
            SOURCES ${_sources}
        )
    endif()

endfunction()
