# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGettext
-----------

Finds the GNU gettext tools and provides commands for producing multi-lingual
messages:

.. code-block:: cmake

  find_package(Gettext [<version>] [...])

GNU gettext is a system for internationalization (i18n) and localization
(l10n), consisting of command-line tools and a runtime library (``libintl``).
This module finds the gettext tools (such as ``msgmerge`` and ``msgfmt``),
while the runtime library can be found using the separate :module:`FindIntl`
module, which abstracts ``libintl`` handling across various implementations.

Common file types used with gettext:

POT files
  Portable Object Template (``.pot``) files used as the source template for
  translations.

PO files
  Portable Object (``.po``) files containing human-readable translations.

MO files
  Machine Object (``.mo``) files compiled from ``.po`` files for runtime use.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Gettext_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) gettext was found.

``Gettext_VERSION``
  .. versionadded:: 4.2

  The version of gettext found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GETTEXT_MSGMERGE_EXECUTABLE``
  The full path to the ``msgmerge`` tool for merging message catalog and
  template.

``GETTEXT_MSGFMT_EXECUTABLE``
  The full path to the ``msgfmt`` tool for compiling message catalog to a binary
  format.

Commands
^^^^^^^^

This module provides the following commands to work with gettext in CMake, if
gettext is found:

.. command:: gettext_process_po_files

  Creates a build rule that processes one or more ``.po`` translation files
  into binary ``.mo`` files for a specified translatable language locale:

  .. code-block:: cmake

    gettext_process_po_files(
      <language>
      [ALL]
      [INSTALL_DESTINATION <destdir>]
      PO_FILES <po-files>...
    )

  This command defines a custom target that compiles the given ``<po-files>``
  into ``.mo`` files for the specified ``<language>``.  On first invocation,
  it also creates a global custom target named ``pofiles``, to which all
  subsequent invocations contribute.  This target can be used to build all
  translation files collectively or referenced in other CMake commands.

  This command should be invoked separately for each language locale to
  generate the appropriate ``.mo`` files per locale.

  The arguments are:

  ``<language>``
    The target translatable language locale of the PO files.

    This string is typically formatted as a locale identifier (e.g., ``de_DE``
    for German as used in Germany, or ``de_AT`` for German as used in Austria,
    etc.).  The part before the underscore specifies the language, and the
    part after specifies the country or regional variant.  In some cases, a
    shorter form using only the language code (e.g., ``de``) may also be used.

  ``ALL``
    This option adds the generated target to the default CMake build target so
    that translations are built by default.

  ``INSTALL_DESTINATION <destdir>``
    Specifies the installation directory for the generated ``.mo`` files at
    the install phase.  If specified, files are installed to:
    ``<destdir>/<language>/LC_MESSAGES/*.mo``.  If not specified, files are
    not installed.

  ``PO_FILES <po-files>...``
    A list of one or more ``.po`` translation files to be compiled into
    ``.mo`` files at build phase for the specified ``<language>``.  Relative
    paths will be interpreted relative to the current source directory
    (:variable:`CMAKE_CURRENT_SOURCE_DIR`).

.. command:: gettext_process_pot_file

  Creates a build rule that processes a gettext Portable Object Template
  (``.pot``) file and associated ``.po`` files into compiled gettext Machine
  Object (``.mo``) files:

  .. code-block:: cmake

    gettext_process_pot_file(
      <pot-file>
      [ALL]
      [INSTALL_DESTINATION <destdir>]
      LANGUAGES <languages>...
    )

  This command defines a custom target named ``potfiles`` that compiles the
  given ``<pot-file>`` and language-specific ``.po`` files into binary ``.mo``
  files for each specified language.  The corresponding ``<language>.po``
  files must exist in the current binary directory
  (:variable:`CMAKE_CURRENT_BINARY_DIR`) before this command is invoked.

  The arguments are:

  ``<pot-file>``
    The path to the gettext Portable Object Template file (``.pot``) serving
    as the source for translations.  If given as a relative path, it will be
    interpreted relative to the current source directory
    (:variable:`CMAKE_CURRENT_SOURCE_DIR`).

  ``ALL``
    Adds the generated target to the default CMake build target so that the
    files are built by default.

  ``INSTALL_DESTINATION <destdir>``
    Specifies the installation directory for the generated ``.mo`` files at
    the install phase.  If specified, files are installed to:
    ``<destdir>/<language>/LC_MESSAGES/<pot-base-filename>.mo``.  If not
    specified, files are not installed.

  ``LANGUAGES <languages>...``
    A list of one or more translatable language locales (e.g., ``en_US``,
    ``fr``, ``de_DE``, ``zh_CN``, etc.).

.. command:: gettext_create_translations

  Creates a build rule that processes a given ``.pot`` template file and
  associated ``.po`` translation files into compiled Machine Object (``.mo``)
  files:

  .. code-block:: cmake

    gettext_create_translations(<pot-file> [ALL] <po-files>...)

  This command defines a custom target named ``translations`` that compiles
  the specified ``<pot-file>`` and ``<po-files>`` into binary ``.mo`` files.
  It also automatically adds an install rule for the generated ``.mo`` files,
  installing them into the default
  ``share/locale/<language>/LC_MESSAGES/<pot-base-filename>.mo`` path during
  the install phase.

  The arguments are:

  ``<pot-file>``
    The path to the gettext Portable Object Template file (``.pot``) serving
    as the source for translations.  If given as a relative path, it will be
    interpreted relative to the current source directory
    (:variable:`CMAKE_CURRENT_SOURCE_DIR`).

  ``ALL``
    Adds the generated target to the default CMake build target so that
    translations are created by default during the build.

  ``<po-files>...``
    A list of one or more translation source files in ``.po`` format, whose
    filenames must follow the format ``<language>.po``.  Relative paths will
    be interpreted relative to the current source directory
    (:variable:`CMAKE_CURRENT_SOURCE_DIR`).

  .. note::
    For better control over build and installation behavior, use
    :command:`gettext_process_po_files` instead.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``GETTEXT_FOUND``
  .. deprecated:: 4.2
    Use ``Gettext_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) gettext was found.

``GETTEXT_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``Gettext_VERSION``.

  The version of gettext found.

Examples
^^^^^^^^

Examples: Finding gettext
"""""""""""""""""""""""""

Finding the GNU gettext tools:

.. code-block:: cmake

  find_package(Gettext)

Or, finding gettext and specifying a minimum required version:

.. code-block:: cmake

  find_package(Gettext 0.21)

Or, finding gettext and making it required (if not found, processing stops
with an error message):

.. code-block:: cmake

  find_package(Gettext REQUIRED)

Example: Working With gettext in CMake
""""""""""""""""""""""""""""""""""""""

When starting with gettext, ``.pot`` file is considered to be created manually.
For example, using a ``xgettext`` tool on the provided ``main.cxx`` source
code file:

.. code-block:: c++
  :caption: ``main.cxx``
  :emphasize-lines: 18

  #include <iostream>
  #include <libintl.h>
  #include <locale.h>

  int main()
  {
    // Set locale from environment
    setlocale(LC_ALL, "");

    // Bind the text domain
    const char* dir = std::getenv("TEXTDOMAINDIR");
    if (!dir) {
      dir = "/usr/local/share/locale";
    }
    bindtextdomain("MyApp", dir);
    textdomain("MyApp");

    std::cout << gettext("Hello, World") << std::endl;

    return 0;
  }

The ``xgettext`` tool extracts all strings from ``gettext()`` calls in provided
source code and creates translatable strings:

.. code-block:: console

  $ xgettext -o MyApp.pot main.cxx

Translatable files can be initialized by the project manually using
``msginit`` tool:

.. code-block:: console

  $ mkdir -p locale/de_DE
  $ msginit -l de_DE.UTF8 -o locale/de_DE/MyApp.po -i MyApp.pot --no-translator

which creates a human-readable file that can be translated into a desired
language (adjust as needed):

.. code-block:: po
  :caption: ``locale/de_DE/MyApp.po``
  :emphasize-lines: 9

  msgid ""
  msgstr ""
  "Language: de\n"
  "Content-Type: text/plain; charset=UTF-8\n"
  "Content-Transfer-Encoding: 8bit\n"
  "Plural-Forms: nplurals=2; plural=(n != 1);\n"

  msgid "Hello, World"
  msgstr "Hallo, Welt"

In CMake, the :command:`gettext_process_po_files` command provided by this
module automatically creates the needed ``.mo`` files that application loads
at runtime depending on the system environment variables such as ``LANG``.
In the following example, also the :module:`GNUInstallDirs` module is used
to provide the ``CMAKE_INSTALL_LOCALEDIR`` variable:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``
  :emphasize-lines: 9-14

  cmake_minimum_required(VERSION 3.24)
  project(GettextExample)
  include(GNUInstallDirs)

  find_package(Gettext)

  if(Gettext_FOUND)
    foreach(language IN ITEMS de_DE)
      gettext_process_po_files(
        ${language}
        ALL
        PO_FILES locale/${language}/MyApp.po
        INSTALL_DESTINATION ${CMAKE_INSTALL_LOCALEDIR}
      )
    endforeach()
  endif()

  add_executable(example main.cxx)

  # Find and link Intl library to use gettext() from libintl.h
  find_package(Intl)
  target_link_libraries(example PRIVATE Intl::Intl)

  install(TARGETS example)

.. code-block:: console

  $ cmake -B build
  $ cmake --build build
  $ DESTDIR=$(pwd)/stage cmake --install build

To utilize the translations, the ``de_DE`` locale needs to be enabled on the
system (see ``locale -a``).  And then the localized output can be run:

.. code-block:: console

  $ LANG=de_DE.UTF-8 TEXTDOMAINDIR=./stage/usr/local/share/locale \
    ./stage/usr/local/bin/example

Example: Processing POT File
""""""""""""""""""""""""""""

The :command:`gettext_process_pot_file` command processes ``.po`` translation
files located in the current binary directory into ``.mo`` files:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(Gettext)

  if(Gettext_FOUND)
    set(languages de_DE fr zh_CN)

    foreach(language IN LISTS languages)
      configure_file(locale/${language}.po ${language}.po COPYONLY)
    endforeach()

    gettext_process_pot_file(
      MyApp.pot
      ALL
      INSTALL_DESTINATION ${CMAKE_INSTALL_LOCALEDIR}
      LANGUAGES ${languages}
    )
  endif()

Example: Creating Translations
""""""""""""""""""""""""""""""

Using a simplified :command:`gettext_create_translations` command to create
``.mo`` files:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(Gettext)

  if(Gettext_FOUND)
    gettext_create_translations(
      MyApp.pot
      ALL
      locale/de_DE.po
      locale/fr.po
      locale/zh_CN.po
    )
  endif()

See Also
^^^^^^^^

* The :module:`FindIntl` module to find the Gettext runtime library (libintl).
#]=======================================================================]

find_program(GETTEXT_MSGMERGE_EXECUTABLE msgmerge)

find_program(GETTEXT_MSGFMT_EXECUTABLE msgfmt)

if(GETTEXT_MSGMERGE_EXECUTABLE)
  execute_process(COMMAND ${GETTEXT_MSGMERGE_EXECUTABLE} --version
                  OUTPUT_VARIABLE gettext_version
                  ERROR_QUIET
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(msgmerge_name ${GETTEXT_MSGMERGE_EXECUTABLE} NAME)
  get_filename_component(msgmerge_namewe ${GETTEXT_MSGMERGE_EXECUTABLE} NAME_WE)
  if(gettext_version MATCHES "^(${msgmerge_name}|${msgmerge_namewe}) \\([^\\)]*\\) ([0-9\\.]+[^ \n]*)")
    set(Gettext_VERSION "${CMAKE_MATCH_2}")
    set(GETTEXT_VERSION_STRING "${Gettext_VERSION}")
  endif()
  unset(gettext_version)
  unset(msgmerge_name)
  unset(msgmerge_namewe)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gettext
                                  REQUIRED_VARS GETTEXT_MSGMERGE_EXECUTABLE GETTEXT_MSGFMT_EXECUTABLE
                                  VERSION_VAR Gettext_VERSION)

function(_GETTEXT_GET_UNIQUE_TARGET_NAME _name _unique_name)
  set(propertyName "_GETTEXT_UNIQUE_COUNTER_${_name}")
  get_property(currentCounter GLOBAL PROPERTY "${propertyName}")
  if(NOT currentCounter)
    set(currentCounter 1)
  endif()
  set(${_unique_name} "${_name}_${currentCounter}" PARENT_SCOPE)
  math(EXPR currentCounter "${currentCounter} + 1")
  set_property(GLOBAL PROPERTY ${propertyName} ${currentCounter})
endfunction()

macro(GETTEXT_CREATE_TRANSLATIONS _potFile _firstPoFileArg)
  # make it a real variable, so we can modify it here
  set(_firstPoFile "${_firstPoFileArg}")

  set(_gmoFiles)
  get_filename_component(_potName ${_potFile} NAME)
  string(REGEX REPLACE "^(.+)(\\.[^.]+)$" "\\1" _potBasename ${_potName})
  get_filename_component(_absPotFile ${_potFile} ABSOLUTE)

  set(_addToAll)
  if(${_firstPoFile} STREQUAL "ALL")
    set(_addToAll "ALL")
    set(_firstPoFile)
  endif()

  foreach(_currentPoFile ${_firstPoFile} ${ARGN})
    get_filename_component(_absFile ${_currentPoFile} ABSOLUTE)
    get_filename_component(_abs_PATH ${_absFile} PATH)
    get_filename_component(_lang ${_absFile} NAME_WE)
    set(_gmoFile ${CMAKE_CURRENT_BINARY_DIR}/${_lang}.gmo)

    add_custom_command(
      OUTPUT ${_gmoFile}
      COMMAND ${GETTEXT_MSGMERGE_EXECUTABLE} --quiet --update --backup=none ${_absFile} ${_absPotFile}
      COMMAND ${GETTEXT_MSGFMT_EXECUTABLE} -o ${_gmoFile} ${_absFile}
      DEPENDS ${_absPotFile} ${_absFile}
    )

    install(FILES ${_gmoFile} DESTINATION share/locale/${_lang}/LC_MESSAGES RENAME ${_potBasename}.mo)
    set(_gmoFiles ${_gmoFiles} ${_gmoFile})

  endforeach()

  if(NOT TARGET translations)
    add_custom_target(translations)
  endif()

  _GETTEXT_GET_UNIQUE_TARGET_NAME(translations uniqueTargetName)

  add_custom_target(${uniqueTargetName} ${_addToAll} DEPENDS ${_gmoFiles})

  add_dependencies(translations ${uniqueTargetName})

endmacro()


function(GETTEXT_PROCESS_POT_FILE _potFile)
  set(_gmoFiles)
  set(_options ALL)
  set(_oneValueArgs INSTALL_DESTINATION)
  set(_multiValueArgs LANGUAGES)

  cmake_parse_arguments(_parsedArguments "${_options}" "${_oneValueArgs}" "${_multiValueArgs}" ${ARGN})

  get_filename_component(_potName ${_potFile} NAME)
  string(REGEX REPLACE "^(.+)(\\.[^.]+)$" "\\1" _potBasename ${_potName})
  get_filename_component(_absPotFile ${_potFile} ABSOLUTE)

  foreach(_lang ${_parsedArguments_LANGUAGES})
    set(_poFile  "${CMAKE_CURRENT_BINARY_DIR}/${_lang}.po")
    set(_gmoFile "${CMAKE_CURRENT_BINARY_DIR}/${_lang}.gmo")

    add_custom_command(
      OUTPUT "${_poFile}"
      COMMAND ${GETTEXT_MSGMERGE_EXECUTABLE} --quiet --update --backup=none ${_poFile} ${_absPotFile}
      DEPENDS ${_absPotFile}
    )

    add_custom_command(
      OUTPUT "${_gmoFile}"
      COMMAND ${GETTEXT_MSGFMT_EXECUTABLE} -o ${_gmoFile} ${_poFile}
      DEPENDS ${_absPotFile} ${_poFile}
    )

    if(_parsedArguments_INSTALL_DESTINATION)
      install(FILES ${_gmoFile} DESTINATION ${_parsedArguments_INSTALL_DESTINATION}/${_lang}/LC_MESSAGES RENAME ${_potBasename}.mo)
    endif()
    list(APPEND _gmoFiles ${_gmoFile})
  endforeach()

  if(NOT TARGET potfiles)
    add_custom_target(potfiles)
  endif()

  _GETTEXT_GET_UNIQUE_TARGET_NAME( potfiles uniqueTargetName)

  if(_parsedArguments_ALL)
    add_custom_target(${uniqueTargetName} ALL DEPENDS ${_gmoFiles})
  else()
    add_custom_target(${uniqueTargetName} DEPENDS ${_gmoFiles})
  endif()

  add_dependencies(potfiles ${uniqueTargetName})

endfunction()


function(GETTEXT_PROCESS_PO_FILES _lang)
  set(_options ALL)
  set(_oneValueArgs INSTALL_DESTINATION)
  set(_multiValueArgs PO_FILES)
  set(_gmoFiles)

  cmake_parse_arguments(_parsedArguments "${_options}" "${_oneValueArgs}" "${_multiValueArgs}" ${ARGN})

  foreach(_current_PO_FILE ${_parsedArguments_PO_FILES})
    get_filename_component(_name ${_current_PO_FILE} NAME)
    string(REGEX REPLACE "^(.+)(\\.[^.]+)$" "\\1" _basename ${_name})
    set(_gmoFile ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.gmo)
    add_custom_command(OUTPUT ${_gmoFile}
      COMMAND ${GETTEXT_MSGFMT_EXECUTABLE} -o ${_gmoFile} ${_current_PO_FILE}
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      DEPENDS ${_current_PO_FILE}
    )

    if(_parsedArguments_INSTALL_DESTINATION)
      install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${_basename}.gmo DESTINATION ${_parsedArguments_INSTALL_DESTINATION}/${_lang}/LC_MESSAGES/ RENAME ${_basename}.mo)
    endif()
    list(APPEND _gmoFiles ${_gmoFile})
  endforeach()


  if(NOT TARGET pofiles)
    add_custom_target(pofiles)
  endif()

  _GETTEXT_GET_UNIQUE_TARGET_NAME(pofiles uniqueTargetName)

  if(_parsedArguments_ALL)
    add_custom_target(${uniqueTargetName} ALL DEPENDS ${_gmoFiles})
  else()
    add_custom_target(${uniqueTargetName} DEPENDS ${_gmoFiles})
  endif()

  add_dependencies(pofiles ${uniqueTargetName})

endfunction()
