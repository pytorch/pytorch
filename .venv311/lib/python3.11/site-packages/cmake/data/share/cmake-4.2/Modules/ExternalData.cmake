# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
ExternalData
------------

.. only:: html

   .. contents::

This module provides commands to manage data files stored outside source
tree.

Load this module in a CMake project with:

.. code-block:: cmake

  include(ExternalData)

Introduction
^^^^^^^^^^^^

Use this module to unambiguously reference data files stored outside
the source tree and fetch them at build time from arbitrary local and
remote content-addressed locations.  Functions provided by this module
recognize arguments with the syntax ``DATA{<name>}`` as references to
external data, replace them with full paths to local copies of those
data, and create build rules to fetch and update the local copies.

For example:

.. code-block:: cmake

 include(ExternalData)
 set(ExternalData_URL_TEMPLATES "file:///local/%(algo)/%(hash)"
                                "file:////host/share/%(algo)/%(hash)"
                                "http://data.org/%(algo)/%(hash)")
 ExternalData_Add_Test(MyData
   NAME MyTest
   COMMAND MyExe DATA{MyInput.png}
   )
 ExternalData_Add_Target(MyData)

When test ``MyTest`` runs the ``DATA{MyInput.png}`` argument will be
replaced by the full path to a real instance of the data file
``MyInput.png`` on disk.  If the source tree contains a content link
such as ``MyInput.png.md5`` then the ``MyData`` target creates a real
``MyInput.png`` in the build tree.

Module Functions
^^^^^^^^^^^^^^^^

.. command:: ExternalData_Expand_Arguments

  The ``ExternalData_Expand_Arguments`` function evaluates ``DATA{}``
  references in its arguments and constructs a new list of arguments:

  .. code-block:: cmake

    ExternalData_Expand_Arguments(
      <target>   # Name of data management target
      <outVar>   # Output variable
      [args...]  # Input arguments, DATA{} allowed
      )

  It replaces each ``DATA{}`` reference in an argument with the full path of
  a real data file on disk that will exist after the ``<target>`` builds.

.. command:: ExternalData_Add_Test

  The ``ExternalData_Add_Test`` function wraps around the CMake
  :command:`add_test` command but supports ``DATA{}`` references in
  its arguments:

  .. code-block:: cmake

    ExternalData_Add_Test(
      <target>   # Name of data management target
      ...        # Arguments of add_test(), DATA{} allowed
      )

  It passes its arguments through ``ExternalData_Expand_Arguments`` and then
  invokes the :command:`add_test` command using the results.

  .. versionchanged:: 3.31
    If the arguments after ``<target>`` define a test with an executable
    that is a CMake target, empty values in the :prop_tgt:`TEST_LAUNCHER`
    and :prop_tgt:`CROSSCOMPILING_EMULATOR` properties of that target are
    preserved.  See policy :policy:`CMP0178`.

.. command:: ExternalData_Add_Target

  The ``ExternalData_Add_Target`` function creates a custom target to
  manage local instances of data files stored externally:

  .. code-block:: cmake

    ExternalData_Add_Target(
      <target>                  # Name of data management target
      [SHOW_PROGRESS <ON|OFF>]  # Show progress during the download
      )

  It creates custom commands in the target as necessary to make data
  files available for each ``DATA{}`` reference previously evaluated by
  other functions provided by this module.
  Data files may be fetched from one of the URL templates specified in
  the ``ExternalData_URL_TEMPLATES`` variable, or may be found locally
  in one of the paths specified in the ``ExternalData_OBJECT_STORES``
  variable.

  .. versionadded:: 3.20
    The ``SHOW_PROGRESS`` argument may be passed to suppress progress information
    during the download of objects. If not provided, it defaults to ``OFF`` for
    :generator:`Ninja` and :generator:`Ninja Multi-Config` generators and ``ON``
    otherwise.

  Typically only one target is needed to manage all external data within
  a project.  Call this function once at the end of configuration after
  all data references have been processed.

Module Variables
^^^^^^^^^^^^^^^^

The following variables configure behavior.  They should be set before
calling any of the functions provided by this module.

.. variable:: ExternalData_BINARY_ROOT

  The ``ExternalData_BINARY_ROOT`` variable may be set to the directory to
  hold the real data files named by expanded ``DATA{}`` references.  The
  default is ``CMAKE_BINARY_DIR``.  The directory layout will mirror that of
  content links under ``ExternalData_SOURCE_ROOT``.

.. variable:: ExternalData_CUSTOM_SCRIPT_<key>

  .. versionadded:: 3.2

  Specify a full path to a ``.cmake`` custom fetch script identified by
  ``<key>`` in entries of the ``ExternalData_URL_TEMPLATES`` list.
  See `Custom Fetch Scripts`_.

.. variable:: ExternalData_HTTPHEADERS

  .. versionadded:: 4.0

  The ``ExternalData_HTTPHEADERS`` variable may be used to supply a list of
  headers, each element containing one header with the form ``Key: Value``.
  See the :command:`file(DOWNLOAD)` command's ``HTTPHEADER`` option.

.. variable:: ExternalData_LINK_CONTENT

  The ``ExternalData_LINK_CONTENT`` variable may be set to the name of a
  supported hash algorithm to enable automatic conversion of real data
  files referenced by the ``DATA{}`` syntax into content links.  For each
  such ``<file>`` a content link named ``<file><ext>`` is created.  The
  original file is renamed to the form ``.ExternalData_<algo>_<hash>`` to
  stage it for future transmission to one of the locations in the list
  of URL templates (by means outside the scope of this module).  The
  data fetch rule created for the content link will use the staged
  object if it cannot be found using any URL template.

.. variable:: ExternalData_NO_SYMLINKS

  .. versionadded:: 3.3

  The real data files named by expanded ``DATA{}`` references may be made
  available under ``ExternalData_BINARY_ROOT`` using symbolic links on
  some platforms.  The ``ExternalData_NO_SYMLINKS`` variable may be set
  to disable use of symbolic links and enable use of copies instead.

.. variable:: ExternalData_OBJECT_STORES

  The ``ExternalData_OBJECT_STORES`` variable may be set to a list of local
  directories that store objects using the layout ``<dir>/%(algo)/%(hash)``.
  These directories will be searched first for a needed object.  If the
  object is not available in any store then it will be fetched remotely
  using the URL templates and added to the first local store listed.  If
  no stores are specified the default is a location inside the build
  tree.

.. variable:: ExternalData_SERIES_PARSE
              ExternalData_SERIES_PARSE_PREFIX
              ExternalData_SERIES_PARSE_NUMBER
              ExternalData_SERIES_PARSE_SUFFIX
              ExternalData_SERIES_MATCH

  See `Referencing File Series`_.

.. variable:: ExternalData_SOURCE_ROOT

  The ``ExternalData_SOURCE_ROOT`` variable may be set to the highest source
  directory containing any path named by a ``DATA{}`` reference.  The
  default is ``CMAKE_SOURCE_DIR``.  ``ExternalData_SOURCE_ROOT`` and
  ``CMAKE_SOURCE_DIR`` must refer to directories within a single source
  distribution (e.g.  they come together in one tarball).

.. variable:: ExternalData_TIMEOUT_ABSOLUTE

  The ``ExternalData_TIMEOUT_ABSOLUTE`` variable sets the download
  absolute timeout, in seconds, with a default of ``300`` seconds.
  Set to ``0`` to disable enforcement.

.. variable:: ExternalData_TIMEOUT_INACTIVITY

  The ``ExternalData_TIMEOUT_INACTIVITY`` variable sets the download
  inactivity timeout, in seconds, with a default of ``60`` seconds.
  Set to ``0`` to disable enforcement.

.. variable:: ExternalData_URL_ALGO_<algo>_<key>

  .. versionadded:: 3.3

  Specify a custom URL component to be substituted for URL template
  placeholders of the form ``%(algo:<key>)``, where ``<key>`` is a
  valid C identifier, when fetching an object referenced via hash
  algorithm ``<algo>``.  If not defined, the default URL component
  is just ``<algo>`` for any ``<key>``.

.. variable:: ExternalData_URL_TEMPLATES

  The ``ExternalData_URL_TEMPLATES`` may be set to provide a list
  of URL templates using the placeholders ``%(algo)`` and ``%(hash)``
  in each template.  Data fetch rules try each URL template in order
  by substituting the hash algorithm name for ``%(algo)`` and the hash
  value for ``%(hash)``.  Alternatively one may use ``%(algo:<key>)``
  with ``ExternalData_URL_ALGO_<algo>_<key>`` variables to gain more
  flexibility in remote URLs.

Referencing Files
^^^^^^^^^^^^^^^^^

Referencing Single Files
""""""""""""""""""""""""

The ``DATA{}`` syntax is literal and the ``<name>`` is a full or relative path
within the source tree.  The source tree must contain either a real
data file at ``<name>`` or a "content link" at ``<name><ext>`` containing a
hash of the real file using a hash algorithm corresponding to ``<ext>``.
For example, the argument ``DATA{img.png}`` may be satisfied by either a
real ``img.png`` file in the current source directory or a ``img.png.md5``
file containing its MD5 sum.

.. versionadded:: 3.8
  Multiple content links of the same name with different hash algorithms
  are supported (e.g. ``img.png.sha256`` and ``img.png.sha1``) so long as
  they all correspond to the same real file.  This allows objects to be
  fetched from sources indexed by different hash algorithms.

Referencing File Series
"""""""""""""""""""""""

The ``DATA{}`` syntax can be told to fetch a file series using the form
``DATA{<name>,:}``, where the ``:`` is literal.  If the source tree
contains a group of files or content links named like a series then a
reference to one member adds rules to fetch all of them.  Although all
members of a series are fetched, only the file originally named by the
``DATA{}`` argument is substituted for it.  The default configuration
recognizes file series names ending with ``#.ext``, ``_#.ext``, ``.#.ext``,
or ``-#.ext`` where ``#`` is a sequence of decimal digits and ``.ext`` is
any single extension.  Configure it with a regex that parses ``<number>``
and ``<suffix>`` parts from the end of ``<name>``:

  ``ExternalData_SERIES_PARSE`` - regex of the form ``(<number>)(<suffix>)$``.

For more complicated cases set:

* ``ExternalData_SERIES_PARSE`` - regex with at least two ``()`` groups.
* ``ExternalData_SERIES_PARSE_PREFIX`` - regex group number of the ``<prefix>``, if any.
* ``ExternalData_SERIES_PARSE_NUMBER`` - regex group number of the ``<number>``.
* ``ExternalData_SERIES_PARSE_SUFFIX`` - regex group number of the ``<suffix>``.

Configure series number matching with a regex that matches the
``<number>`` part of series members named ``<prefix><number><suffix>``:

  ``ExternalData_SERIES_MATCH`` - regex matching ``<number>`` in all series
  members

Note that the ``<suffix>`` of a series does not include a hash-algorithm
extension.

Referencing Associated Files
""""""""""""""""""""""""""""

The ``DATA{}`` syntax can alternatively match files associated with the
named file and contained in the same directory.  Associated files may
be specified by options using the syntax
``DATA{<name>,<opt1>,<opt2>,...}``.  Each option may specify one file by
name or specify a regular expression to match file names using the
syntax ``REGEX:<regex>``.  For example, the arguments::

 DATA{MyData/MyInput.mhd,MyInput.img}                   # File pair
 DATA{MyData/MyFrames00.png,REGEX:MyFrames[0-9]+\\.png} # Series

will pass ``MyInput.mha`` and ``MyFrames00.png`` on the command line but
ensure that the associated files are present next to them.

Referencing Directories
"""""""""""""""""""""""

The ``DATA{}`` syntax may reference a directory using a trailing slash and
a list of associated files.  The form ``DATA{<name>/,<opt1>,<opt2>,...}``
adds rules to fetch any files in the directory that match one of the
associated file options.  For example, the argument
``DATA{MyDataDir/,REGEX:.*}`` will pass the full path to a ``MyDataDir``
directory on the command line and ensure that the directory contains
files corresponding to every file or content link in the ``MyDataDir``
source directory.

.. versionadded:: 3.3
  In order to match associated files in subdirectories,
  specify a ``RECURSE:`` option, e.g. ``DATA{MyDataDir/,RECURSE:,REGEX:.*}``.

Hash Algorithms
^^^^^^^^^^^^^^^

The following hash algorithms are supported:

 ============ ============= ============
 %(algo)      <ext>         Description
 ============ ============= ============
 ``MD5``      ``.md5``      Message-Digest Algorithm 5, RFC 1321
 ``SHA1``     ``.sha1``     US Secure Hash Algorithm 1, RFC 3174
 ``SHA224``   ``.sha224``   US Secure Hash Algorithms, RFC 4634
 ``SHA256``   ``.sha256``   US Secure Hash Algorithms, RFC 4634
 ``SHA384``   ``.sha384``   US Secure Hash Algorithms, RFC 4634
 ``SHA512``   ``.sha512``   US Secure Hash Algorithms, RFC 4634
 ``SHA3_224`` ``.sha3-224`` Keccak SHA-3
 ``SHA3_256`` ``.sha3-256`` Keccak SHA-3
 ``SHA3_384`` ``.sha3-384`` Keccak SHA-3
 ``SHA3_512`` ``.sha3-512`` Keccak SHA-3
 ============ ============= ============

.. versionadded:: 3.8
  Added the ``SHA3_*`` hash algorithms.

Note that the hashes are used only for unique data identification and
download verification.

.. _`ExternalData Custom Fetch Scripts`:

Custom Fetch Scripts
^^^^^^^^^^^^^^^^^^^^

.. versionadded:: 3.2

When a data file must be fetched from one of the URL templates
specified in the ``ExternalData_URL_TEMPLATES`` variable, it is
normally downloaded using the :command:`file(DOWNLOAD)` command.
One may specify usage of a custom fetch script by using a URL
template of the form ``ExternalDataCustomScript://<key>/<loc>``.
The ``<key>`` must be a C identifier, and the ``<loc>`` must
contain the ``%(algo)`` and ``%(hash)`` placeholders.
A variable corresponding to the key, ``ExternalData_CUSTOM_SCRIPT_<key>``,
must be set to the full path to a ``.cmake`` script file.  The script
will be included to perform the actual fetch, and provided with
the following variables:

.. variable:: ExternalData_CUSTOM_LOCATION

  When a custom fetch script is loaded, this variable is set to the
  location part of the URL, which will contain the substituted hash
  algorithm name and content hash value.

.. variable:: ExternalData_CUSTOM_FILE

  When a custom fetch script is loaded, this variable is set to the
  full path to a file in which the script must store the fetched
  content.  The name of the file is unspecified and should not be
  interpreted in any way.

The custom fetch script is expected to store fetched content in the
file or set a variable:

.. variable:: ExternalData_CUSTOM_ERROR

  When a custom fetch script fails to fetch the requested content,
  it must set this variable to a short one-line message describing
  the reason for failure.

#]=======================================================================]

function(ExternalData_add_test target)
  # Expand all arguments as a single string to preserve escaped semicolons.
  ExternalData_expand_arguments("${target}" testArgs "${ARGN}")

  # We need the caller's CMP0178 policy setting to apply here
  cmake_policy(GET CMP0178 cmp0178
    PARENT_SCOPE  # undocumented, do not use outside of CMake
  )

  # ExternalData_expand_arguments() escapes semicolons, so we should still be
  # preserving empty elements from ARGN here. But CMP0178 is still important
  # for correctly handling TEST_LAUNCHER and CROSSCOMPILING_EMULATOR target
  # properties that contain empty elements.
  add_test(${testArgs} __CMP0178 "${cmp0178}")
endfunction()

function(ExternalData_add_target target)
  if(NOT ExternalData_URL_TEMPLATES AND NOT ExternalData_OBJECT_STORES)
    message(FATAL_ERROR
      "Neither ExternalData_URL_TEMPLATES nor ExternalData_OBJECT_STORES is set!")
  endif()
  if(NOT ExternalData_OBJECT_STORES)
    set(ExternalData_OBJECT_STORES ${CMAKE_BINARY_DIR}/ExternalData/Objects)
  endif()
  set(_ExternalData_CONFIG_CODE "")

  cmake_parse_arguments(PARSE_ARGV 1 _ExternalData_add_target
    ""
    "SHOW_PROGRESS"
    "")
  if (_ExternalData_add_target_UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING
      "Ignoring unrecognized arguments passed to ExternalData_add_target: "
      "`${_ExternalData_add_target_UNPARSED_ARGUMENTS}`")
  endif ()

  # Turn `SHOW_PROGRESS` into a boolean
  if (NOT DEFINED _ExternalData_add_target_SHOW_PROGRESS)
    # The default setting
    if (CMAKE_GENERATOR MATCHES "Ninja")
      set(_ExternalData_add_target_SHOW_PROGRESS OFF)
    else ()
      set(_ExternalData_add_target_SHOW_PROGRESS ON)
    endif ()
  elseif (_ExternalData_add_target_SHOW_PROGRESS)
    set(_ExternalData_add_target_SHOW_PROGRESS ON)
  else ()
    set(_ExternalData_add_target_SHOW_PROGRESS OFF)
  endif ()

  # Store custom script configuration.
  foreach(url_template IN LISTS ExternalData_URL_TEMPLATES)
    if("${url_template}" MATCHES "^ExternalDataCustomScript://([^/]*)/(.*)$")
      set(key "${CMAKE_MATCH_1}")
      if(key MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        if(ExternalData_CUSTOM_SCRIPT_${key})
          if(IS_ABSOLUTE "${ExternalData_CUSTOM_SCRIPT_${key}}")
            string(CONCAT _ExternalData_CONFIG_CODE "${_ExternalData_CONFIG_CODE}\n"
              "set(ExternalData_CUSTOM_SCRIPT_${key} \"${ExternalData_CUSTOM_SCRIPT_${key}}\")")
          else()
            message(FATAL_ERROR
              "No ExternalData_CUSTOM_SCRIPT_${key} is not set to a full path:\n"
              " ${ExternalData_CUSTOM_SCRIPT_${key}}")
          endif()
        else()
          message(FATAL_ERROR
            "No ExternalData_CUSTOM_SCRIPT_${key} is set for URL template:\n"
            " ${url_template}")
        endif()
      else()
        message(FATAL_ERROR
          "Bad ExternalDataCustomScript key '${key}' in URL template:\n"
          " ${url_template}\n"
          "The key must be a valid C identifier.")
      endif()
    endif()

    # Store custom algorithm name to URL component maps.
    if("${url_template}" MATCHES "%\\(algo:([^)]*)\\)")
      set(key "${CMAKE_MATCH_1}")
      if(key MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
        string(REPLACE "|" ";" _algos "${_ExternalData_REGEX_ALGO}")
        foreach(algo ${_algos})
          if(DEFINED ExternalData_URL_ALGO_${algo}_${key})
            string(CONCAT _ExternalData_CONFIG_CODE "${_ExternalData_CONFIG_CODE}\n"
              "set(ExternalData_URL_ALGO_${algo}_${key} \"${ExternalData_URL_ALGO_${algo}_${key}}\")")
          endif()
        endforeach()
      else()
        message(FATAL_ERROR
          "Bad %(algo:${key}) in URL template:\n"
          " ${url_template}\n"
          "The transform name must be a valid C identifier.")
      endif()
    endif()
  endforeach()

  # Store http headers.
  if(ExternalData_HTTPHEADERS)
    message(STATUS "${CMAKE_CURRENT_BINARY_DIR}/${target}_config.cmake")
    string(CONCAT _ExternalData_CONFIG_CODE "${_ExternalData_CONFIG_CODE}\n"
      "set(ExternalData_HTTPHEADERS)")
    foreach(h IN LISTS ExternalData_HTTPHEADERS)
      string(REPLACE "\\" "\\\\" tmp "${h}")
      string(REPLACE "\"" "\\\"" h "${tmp}")
      string(CONCAT _ExternalData_CONFIG_CODE "${_ExternalData_CONFIG_CODE}\n"
        "list(APPEND ExternalData_HTTPHEADERS \"${h}\")")
    endforeach()
  endif()

  # Store configuration for use by build-time script.
  set(config ${CMAKE_CURRENT_BINARY_DIR}/${target}_config.cmake)
  configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/ExternalData_config.cmake.in ${config} @ONLY)

  set(files "")

  # Set a "_ExternalData_FILE_${file}" variable for each output file to avoid
  # duplicate entries within this target.  Set a directory property of the same
  # name to avoid repeating custom commands with the same output in this directory.
  # Repeating custom commands with the same output across directories or across
  # targets in the same directory may be a race, but this is likely okay because
  # we use atomic replacement of output files.
  #
  # Use local data first to prefer real files over content links.

  # Custom commands to copy or link local data.
  get_property(data_local GLOBAL PROPERTY _ExternalData_${target}_LOCAL)
  foreach(entry IN LISTS data_local)
    string(REPLACE "|" ";" tuple "${entry}")
    list(GET tuple 0 file)
    list(GET tuple 1 name)
    if(NOT DEFINED "_ExternalData_FILE_${file}")
      set("_ExternalData_FILE_${file}" 1)
      get_property(added DIRECTORY PROPERTY "_ExternalData_FILE_${file}")
      if(NOT added)
        set_property(DIRECTORY PROPERTY "_ExternalData_FILE_${file}" 1)
        add_custom_command(
          COMMENT "Generating ${file}"
          OUTPUT "${file}"
          COMMAND ${CMAKE_COMMAND} -Drelative_top=${CMAKE_BINARY_DIR}
                                   -Dfile=${file} -Dname=${name}
                                   -DExternalData_ACTION=local
                                   -DExternalData_SHOW_PROGRESS=${_ExternalData_add_target_SHOW_PROGRESS}
                                   -DExternalData_CONFIG=${config}
                                   -P ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
          MAIN_DEPENDENCY "${name}"
          )
      endif()
      list(APPEND files "${file}")
    endif()
  endforeach()

  # Custom commands to fetch remote data.
  get_property(data_fetch GLOBAL PROPERTY _ExternalData_${target}_FETCH)
  foreach(entry IN LISTS data_fetch)
    string(REPLACE "|" ";" tuple "${entry}")
    list(GET tuple 0 file)
    list(GET tuple 1 name)
    list(GET tuple 2 exts)
    string(REPLACE "+" ";" exts_list "${exts}")
    list(GET exts_list 0 first_ext)
    set(stamp "-hash-stamp")
    if(NOT DEFINED "_ExternalData_FILE_${file}")
      set("_ExternalData_FILE_${file}" 1)
      get_property(added DIRECTORY PROPERTY "_ExternalData_FILE_${file}")
      if(NOT added)
        set_property(DIRECTORY PROPERTY "_ExternalData_FILE_${file}" 1)
        add_custom_command(
          # Users care about the data file, so hide the hash/timestamp file.
          COMMENT "Generating ${file}"
          # The hash/timestamp file is the output from the build perspective.
          # List the real file as a second output in case it is a broken link.
          # The files must be listed in this order so CMake can hide from the
          # make tool that a symlink target may not be newer than the input.
          OUTPUT "${file}${stamp}" "${file}"
          # Run the data fetch/update script.
          COMMAND ${CMAKE_COMMAND} -Drelative_top=${CMAKE_BINARY_DIR}
                                   -Dfile=${file} -Dname=${name} -Dexts=${exts}
                                   -DExternalData_ACTION=fetch
                                   -DExternalData_SHOW_PROGRESS=${_ExternalData_add_target_SHOW_PROGRESS}
                                   -DExternalData_CONFIG=${config}
                                   -P ${CMAKE_CURRENT_FUNCTION_LIST_FILE}
          # Update whenever the object hash changes.
          MAIN_DEPENDENCY "${name}${first_ext}"
          )
      endif()
      list(APPEND files "${file}${stamp}")
    endif()
  endforeach()

  # Custom target to drive all update commands.
  add_custom_target(${target} ALL DEPENDS ${files})
endfunction()

function(ExternalData_expand_arguments target outArgsVar)
  # Replace DATA{} references with real arguments.
  set(data_regex "DATA{([^;{}\r\n]*)}")
  set(other_regex "([^D]|D[^A]|DA[^T]|DAT[^A]|DATA[^{])+|.")
  set(outArgs "")
  # This list expansion un-escapes semicolons in list element values so we
  # must re-escape them below anywhere a new list expansion will occur.
  foreach(arg IN LISTS ARGN)
    if("x${arg}" MATCHES "${data_regex}")
      # Re-escape in-value semicolons before expansion in foreach below.
      string(REPLACE ";" "\\;" tmp "${arg}")
      # Split argument into DATA{}-pieces and other pieces.
      string(REGEX MATCHALL "${data_regex}|${other_regex}" pieces "${tmp}")
      # Compose output argument with DATA{}-pieces replaced.
      set(outArg "")
      foreach(piece IN LISTS pieces)
        if("x${piece}" MATCHES "^x${data_regex}$")
          # Replace this DATA{}-piece with a file path.
          _ExternalData_arg("${target}" "${piece}" "${CMAKE_MATCH_1}" file)
          string(APPEND outArg "${file}")
        else()
          # No replacement needed for this piece.
          string(APPEND outArg "${piece}")
        endif()
      endforeach()
    else()
      # No replacements needed in this argument.
      set(outArg "${arg}")
    endif()
    # Re-escape in-value semicolons in resulting list.
    string(REPLACE ";" "\\;" outArg "${outArg}")
    list(APPEND outArgs "${outArg}")
  endforeach()
  set("${outArgsVar}" "${outArgs}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------
# Private helper interface

set(_ExternalData_REGEX_ALGO "MD5|SHA1|SHA224|SHA256|SHA384|SHA512|SHA3_224|SHA3_256|SHA3_384|SHA3_512")
set(_ExternalData_REGEX_EXT "md5|sha1|sha224|sha256|sha384|sha512|sha3-224|sha3-256|sha3-384|sha3-512")

function(_ExternalData_compute_hash var_hash algo file)
  if("${algo}" MATCHES "^${_ExternalData_REGEX_ALGO}$")
    file("${algo}" "${file}" hash)
    set("${var_hash}" "${hash}" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Hash algorithm ${algo} unimplemented.")
  endif()
endfunction()

function(_ExternalData_random var)
  string(RANDOM LENGTH 6 random)
  set("${var}" "${random}" PARENT_SCOPE)
endfunction()

function(_ExternalData_exact_regex regex_var string)
  string(REGEX REPLACE "([][+.*()^])" "\\\\\\1" regex "${string}")
  set("${regex_var}" "${regex}" PARENT_SCOPE)
endfunction()

function(_ExternalData_atomic_write file content)
  _ExternalData_random(random)
  set(tmp "${file}.tmp${random}")
  file(WRITE "${tmp}" "${content}")
  file(RENAME "${tmp}" "${file}")
endfunction()

function(_ExternalData_link_content name var_ext)
  if("${ExternalData_LINK_CONTENT}" MATCHES "^(${_ExternalData_REGEX_ALGO})$")
    set(algo "${ExternalData_LINK_CONTENT}")
  else()
    message(FATAL_ERROR
      "Unknown hash algorithm specified by ExternalData_LINK_CONTENT:\n"
      "  ${ExternalData_LINK_CONTENT}")
  endif()
  _ExternalData_compute_hash(hash "${algo}" "${name}")
  get_filename_component(dir "${name}" PATH)
  set(staged "${dir}/.ExternalData_${algo}_${hash}")
  string(TOLOWER ".${algo}" ext)
  _ExternalData_atomic_write("${name}${ext}" "${hash}\n")
  file(RENAME "${name}" "${staged}")
  set("${var_ext}" "${ext}" PARENT_SCOPE)

  file(RELATIVE_PATH relname "${ExternalData_SOURCE_ROOT}" "${name}${ext}")
  message(STATUS "Linked ${relname} to ExternalData ${algo}/${hash}")
endfunction()

function(_ExternalData_arg target arg options var_file)
  # Separate data path from the options.
  string(REPLACE "," ";" options "${options}")
  list(GET options 0 data)
  list(REMOVE_AT options 0)

  # Interpret trailing slashes as directories.
  set(data_is_directory 0)
  if("x${data}" MATCHES "^x(.*)([/\\])$")
    set(data_is_directory 1)
    set(data "${CMAKE_MATCH_1}")
  endif()

  # Convert to full path.
  if(IS_ABSOLUTE "${data}")
    set(absdata "${data}")
  else()
    set(absdata "${CMAKE_CURRENT_SOURCE_DIR}/${data}")
  endif()
  get_filename_component(absdata "${absdata}" ABSOLUTE)

  # Convert to relative path under the source tree.
  if(NOT ExternalData_SOURCE_ROOT)
    set(ExternalData_SOURCE_ROOT "${CMAKE_SOURCE_DIR}")
  endif()
  set(top_src "${ExternalData_SOURCE_ROOT}")
  file(RELATIVE_PATH reldata "${top_src}" "${absdata}")
  if(IS_ABSOLUTE "${reldata}" OR "${reldata}" MATCHES "^\\.\\./")
    message(FATAL_ERROR "Data file referenced by argument\n"
      "  ${arg}\n"
      "does not lie under the top-level source directory\n"
      "  ${top_src}\n")
  endif()
  if(data_is_directory AND NOT IS_DIRECTORY "${top_src}/${reldata}")
    message(FATAL_ERROR "Data directory referenced by argument\n"
      "  ${arg}\n"
      "corresponds to source tree path\n"
      "  ${reldata}\n"
      "that does not exist as a directory!")
  endif()
  if(NOT ExternalData_BINARY_ROOT)
    set(ExternalData_BINARY_ROOT "${CMAKE_BINARY_DIR}")
  endif()
  set(top_bin "${ExternalData_BINARY_ROOT}")

  # Handle in-source builds gracefully.
  if("${top_src}" STREQUAL "${top_bin}")
    if(ExternalData_LINK_CONTENT)
      message(WARNING "ExternalData_LINK_CONTENT cannot be used in-source")
      set(ExternalData_LINK_CONTENT 0)
    endif()
    set(top_same 1)
  endif()

  set(external "") # Entries external to the source tree.
  set(internal "") # Entries internal to the source tree.
  set(have_original ${data_is_directory})
  set(have_original_as_dir 0)

  # Process options.
  set(series_option "")
  set(recurse_option "")
  set(associated_files "")
  set(associated_regex "")
  foreach(opt ${options})
    # Regular expression to match associated files.
    if("x${opt}" MATCHES "^xREGEX:([^:/]+)$")
      list(APPEND associated_regex "${CMAKE_MATCH_1}")
    elseif(opt STREQUAL ":")
      # Activate series matching.
      set(series_option "${opt}")
    elseif(opt STREQUAL "RECURSE:")
      # Activate recursive matching in directories.
      set(recurse_option "${opt}")
    elseif("x${opt}" MATCHES "^[^][:/*?]+$")
      # Specific associated file.
      list(APPEND associated_files "${opt}")
    else()
      message(FATAL_ERROR "Unknown option \"${opt}\" in argument\n"
        "  ${arg}\n")
    endif()
  endforeach()

  if(series_option)
    if(data_is_directory)
      message(FATAL_ERROR "Series option \"${series_option}\" not allowed with directories.")
    endif()
    if(associated_files OR associated_regex)
      message(FATAL_ERROR "Series option \"${series_option}\" not allowed with associated files.")
    endif()
    if(recurse_option)
      message(FATAL_ERROR "Recurse option \"${recurse_option}\" allowed only with directories.")
    endif()
    # Load a whole file series.
    _ExternalData_arg_series()
  elseif(data_is_directory)
    if(associated_files OR associated_regex)
      # Load listed/matching associated files in the directory.
      _ExternalData_arg_associated()
    else()
      message(FATAL_ERROR "Data directory referenced by argument\n"
        "  ${arg}\n"
        "must list associated files.")
    endif()
  else()
    if(recurse_option)
      message(FATAL_ERROR "Recurse option \"${recurse_option}\" allowed only with directories.")
    endif()
    # Load the named data file.
    _ExternalData_arg_single()
    if(associated_files OR associated_regex)
      # Load listed/matching associated files.
      _ExternalData_arg_associated()
    endif()
  endif()

  if(NOT have_original)
    if(have_original_as_dir)
      set(msg_kind FATAL_ERROR)
      set(msg "that is directory instead of a file!")
    else()
      set(msg_kind AUTHOR_WARNING)
      set(msg "that does not exist as a file (with or without an extension)!")
    endif()
    message(${msg_kind} "Data file referenced by argument\n"
      "  ${arg}\n"
      "corresponds to source tree path\n"
      "  ${reldata}\n"
      "${msg}")
  endif()

  if(external)
    # Make the series available in the build tree.
    set_property(GLOBAL APPEND PROPERTY
      _ExternalData_${target}_FETCH "${external}")
    set_property(GLOBAL APPEND PROPERTY
      _ExternalData_${target}_LOCAL "${internal}")
    set("${var_file}" "${top_bin}/${reldata}" PARENT_SCOPE)
  else()
    # The whole series is in the source tree.
    set("${var_file}" "${top_src}/${reldata}" PARENT_SCOPE)
  endif()
endfunction()

macro(_ExternalData_arg_associated)
  # Associated files lie in the same directory.
  if(data_is_directory)
    set(reldir "${reldata}")
  else()
    get_filename_component(reldir "${reldata}" PATH)
  endif()
  if(reldir)
    string(APPEND reldir "/")
  endif()
  _ExternalData_exact_regex(reldir_regex "${reldir}")
  if(recurse_option)
    set(glob GLOB_RECURSE)
    string(APPEND reldir_regex "(.+/)?")
  else()
    set(glob GLOB)
  endif()

  # Find files named explicitly.
  foreach(file ${associated_files})
    _ExternalData_exact_regex(file_regex "${file}")
    _ExternalData_arg_find_files(${glob} "${reldir}${file}"
      "${reldir_regex}${file_regex}")
  endforeach()

  # Find files matching the given regular expressions.
  set(all "")
  set(sep "")
  foreach(regex ${associated_regex})
    string(APPEND all "${sep}${reldir_regex}${regex}")
    set(sep "|")
  endforeach()
  _ExternalData_arg_find_files(${glob} "${reldir}" "${all}")
endmacro()

macro(_ExternalData_arg_single)
  # Match only the named data by itself.
  _ExternalData_exact_regex(data_regex "${reldata}")
  _ExternalData_arg_find_files(GLOB "${reldata}" "${data_regex}")
endmacro()

macro(_ExternalData_arg_series)
  # Configure series parsing and matching.
  set(series_parse_prefix "")
  set(series_parse_number "\\1")
  set(series_parse_suffix "\\2")
  if(ExternalData_SERIES_PARSE)
    if(ExternalData_SERIES_PARSE_NUMBER AND ExternalData_SERIES_PARSE_SUFFIX)
      if(ExternalData_SERIES_PARSE_PREFIX)
        set(series_parse_prefix "\\${ExternalData_SERIES_PARSE_PREFIX}")
      endif()
      set(series_parse_number "\\${ExternalData_SERIES_PARSE_NUMBER}")
      set(series_parse_suffix "\\${ExternalData_SERIES_PARSE_SUFFIX}")
    elseif(NOT "x${ExternalData_SERIES_PARSE}" MATCHES "^x\\([^()]*\\)\\([^()]*\\)\\$$")
      message(FATAL_ERROR
        "ExternalData_SERIES_PARSE is set to\n"
        "  ${ExternalData_SERIES_PARSE}\n"
        "which is not of the form\n"
        "  (<number>)(<suffix>)$\n"
        "Fix the regular expression or set variables\n"
        "  ExternalData_SERIES_PARSE_PREFIX = <prefix> regex group number, if any\n"
        "  ExternalData_SERIES_PARSE_NUMBER = <number> regex group number\n"
        "  ExternalData_SERIES_PARSE_SUFFIX = <suffix> regex group number\n"
        )
    endif()
    set(series_parse "${ExternalData_SERIES_PARSE}")
  else()
    set(series_parse "([0-9]*)(\\.[^./]*)$")
  endif()
  if(ExternalData_SERIES_MATCH)
    set(series_match "${ExternalData_SERIES_MATCH}")
  else()
    set(series_match "[_.-]?[0-9]*")
  endif()

  # Parse the base, number, and extension components of the series.
  string(REGEX REPLACE "${series_parse}" "${series_parse_prefix};${series_parse_number};${series_parse_suffix}" tuple "${reldata}")
  list(LENGTH tuple len)
  if(NOT "${len}" EQUAL 3)
    message(FATAL_ERROR "Data file referenced by argument\n"
      "  ${arg}\n"
      "corresponds to path\n"
      "  ${reldata}\n"
      "that does not match regular expression\n"
      "  ${series_parse}")
  endif()
  list(GET tuple 0 relbase)
  list(GET tuple 2 ext)

  # Glob files that might match the series.
  # Then match base, number, and extension.
  _ExternalData_exact_regex(series_base "${relbase}")
  _ExternalData_exact_regex(series_ext "${ext}")
  _ExternalData_arg_find_files(GLOB "${relbase}*${ext}"
    "${series_base}${series_match}${series_ext}")
endmacro()

function(_ExternalData_arg_find_files glob pattern regex)
  file(${glob} globbed RELATIVE "${top_src}" "${top_src}/${pattern}*")
  set(externals_count -1)
  foreach(entry IN LISTS globbed)
    if("x${entry}" MATCHES "^x(.*)(\\.(${_ExternalData_REGEX_EXT}))$")
      set(relname "${CMAKE_MATCH_1}")
      set(alg "${CMAKE_MATCH_2}")
    else()
      set(relname "${entry}")
      set(alg "")
    endif()
    if("x${relname}" MATCHES "^x${regex}$" # matches
        AND NOT "x${relname}" MATCHES "(^x|/)\\.ExternalData_" # not staged obj
        )
      if(IS_DIRECTORY "${top_src}/${entry}")
        if("${relname}" STREQUAL "${reldata}")
          set(have_original_as_dir 1)
        endif()
      else()
        set(name "${top_src}/${relname}")
        set(file "${top_bin}/${relname}")
        if(alg)
          if(NOT "${external_${externals_count}_file_name}" STREQUAL "${file}|${name}")
            math(EXPR externals_count "${externals_count} + 1")
            set(external_${externals_count}_file_name "${file}|${name}")
          endif()
          list(APPEND external_${externals_count}_algs "${alg}")
        elseif(ExternalData_LINK_CONTENT)
          _ExternalData_link_content("${name}" alg)
          list(APPEND external "${file}|${name}|${alg}")
        elseif(NOT top_same)
          list(APPEND internal "${file}|${name}")
        endif()
        if("${relname}" STREQUAL "${reldata}")
          set(have_original 1)
        endif()
      endif()
    endif()
  endforeach()
  if(${externals_count} GREATER -1)
    foreach(ii RANGE ${externals_count})
      string(REPLACE ";" "+" algs_delim "${external_${ii}_algs}")
      list(APPEND external "${external_${ii}_file_name}|${algs_delim}")
      unset(external_${ii}_algs)
      unset(external_${ii}_file_name)
    endforeach()
  endif()
  set(external "${external}" PARENT_SCOPE)
  set(internal "${internal}" PARENT_SCOPE)
  set(have_original "${have_original}" PARENT_SCOPE)
  set(have_original_as_dir "${have_original_as_dir}" PARENT_SCOPE)
endfunction()

#-----------------------------------------------------------------------------
# Private script mode interface

if(CMAKE_GENERATOR OR NOT ExternalData_ACTION)
  return()
endif()

if(ExternalData_CONFIG)
  include(${ExternalData_CONFIG})
endif()
if(NOT ExternalData_URL_TEMPLATES AND NOT ExternalData_OBJECT_STORES)
  message(FATAL_ERROR
    "Neither ExternalData_URL_TEMPLATES nor ExternalData_OBJECT_STORES is set!")
endif()

function(_ExternalData_link_or_copy src dst)
  # Create a temporary file first.
  get_filename_component(dst_dir "${dst}" PATH)
  file(MAKE_DIRECTORY "${dst_dir}")
  _ExternalData_random(random)
  set(tmp "${dst}.tmp${random}")
  if(UNIX AND NOT ExternalData_NO_SYMLINKS)
    # Create a symbolic link.
    set(tgt "${src}")
    if(relative_top)
      # Use relative path if files are close enough.
      file(RELATIVE_PATH relsrc "${relative_top}" "${src}")
      file(RELATIVE_PATH relfile "${relative_top}" "${dst}")
      if(NOT IS_ABSOLUTE "${relsrc}" AND NOT "${relsrc}" MATCHES "^\\.\\./" AND
          NOT IS_ABSOLUTE "${reldst}" AND NOT "${reldst}" MATCHES "^\\.\\./")
        file(RELATIVE_PATH tgt "${dst_dir}" "${src}")
      endif()
    endif()
    # Create link (falling back to copying if there's a problem).
    file(CREATE_LINK "${tgt}" "${tmp}" RESULT result COPY_ON_ERROR SYMBOLIC)
  else()
    # Create a copy.
    file(COPY_FILE "${src}" "${tmp}" RESULT result INPUT_MAY_BE_RECENT)
  endif()
  if(result)
    file(REMOVE "${tmp}")
    message(FATAL_ERROR "Failed to create:\n  \"${tmp}\"\nfrom:\n  \"${obj}\"\nwith error:\n  ${result}")
  endif()

  # Atomically create/replace the real destination.
  file(RENAME "${tmp}" "${dst}")
endfunction()

function(_ExternalData_download_file url file err_var msg_var)
  set(retry 3)
  while(retry)
    math(EXPR retry "${retry} - 1")
    set(httpheader_args)
    if (ExternalData_HTTPHEADERS)
      foreach(h IN LISTS ExternalData_HTTPHEADERS)
        list(APPEND httpheader_args HTTPHEADER "${h}")
      endforeach()
    endif()
    if(ExternalData_TIMEOUT_INACTIVITY)
      set(inactivity_timeout INACTIVITY_TIMEOUT ${ExternalData_TIMEOUT_INACTIVITY})
    elseif(NOT "${ExternalData_TIMEOUT_INACTIVITY}" EQUAL 0)
      set(inactivity_timeout INACTIVITY_TIMEOUT 60)
    else()
      set(inactivity_timeout "")
    endif()
    if(ExternalData_TIMEOUT_ABSOLUTE)
      set(absolute_timeout TIMEOUT ${ExternalData_TIMEOUT_ABSOLUTE})
    elseif(NOT "${ExternalData_TIMEOUT_ABSOLUTE}" EQUAL 0)
      set(absolute_timeout TIMEOUT 300)
    else()
      set(absolute_timeout "")
    endif()
    set(show_progress_args)
    if (ExternalData_SHOW_PROGRESS)
      list(APPEND show_progress_args SHOW_PROGRESS)
    endif ()
    file(DOWNLOAD "${url}" "${file}" STATUS status LOG log ${httpheader_args} ${inactivity_timeout} ${absolute_timeout} ${show_progress_args})
    list(GET status 0 err)
    list(GET status 1 msg)
    if(err)
      if("${msg}" MATCHES "HTTP response code said error" AND
          "${log}" MATCHES "error: 503")
        set(msg "temporarily unavailable")
      endif()
    elseif("${log}" MATCHES "\nHTTP[^\n]* 503")
      set(err TRUE)
      set(msg "temporarily unavailable")
    endif()
    if(NOT err OR NOT "${msg}" MATCHES "partial|timeout|temporarily")
      break()
    elseif(retry)
      message(STATUS "[download terminated: ${msg}, retries left: ${retry}]")
    endif()
  endwhile()
  set("${err_var}" "${err}" PARENT_SCOPE)
  set("${msg_var}" "${msg}" PARENT_SCOPE)
endfunction()

function(_ExternalData_custom_fetch key loc file err_var msg_var)
  if(NOT ExternalData_CUSTOM_SCRIPT_${key})
    set(err 1)
    set(msg "No ExternalData_CUSTOM_SCRIPT_${key} set!")
  elseif(NOT EXISTS "${ExternalData_CUSTOM_SCRIPT_${key}}")
    set(err 1)
    set(msg "No '${ExternalData_CUSTOM_SCRIPT_${key}}' exists!")
  else()
    set(ExternalData_CUSTOM_LOCATION "${loc}")
    set(ExternalData_CUSTOM_FILE "${file}")
    unset(ExternalData_CUSTOM_ERROR)
    include("${ExternalData_CUSTOM_SCRIPT_${key}}")
    if(DEFINED ExternalData_CUSTOM_ERROR)
      set(err 1)
      set(msg "${ExternalData_CUSTOM_ERROR}")
    else()
      set(err 0)
      set(msg "no error")
    endif()
  endif()
  set("${err_var}" "${err}" PARENT_SCOPE)
  set("${msg_var}" "${msg}" PARENT_SCOPE)
endfunction()

function(_ExternalData_get_from_object_store hash algo var_obj var_success)
  # Search all object stores for an existing object.
  foreach(dir ${ExternalData_OBJECT_STORES})
    set(obj "${dir}/${algo}/${hash}")
    if(EXISTS "${obj}")
      message(STATUS "Found object: \"${obj}\"")
      set("${var_obj}" "${obj}" PARENT_SCOPE)
      set("${var_success}" 1 PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

function(_ExternalData_download_object name hash algo var_obj var_success var_errorMsg)
  # Search all object stores for an existing object.
  set(success 1)
  foreach(dir ${ExternalData_OBJECT_STORES})
    set(obj "${dir}/${algo}/${hash}")
    if(EXISTS "${obj}")
      message(STATUS "Found object: \"${obj}\"")
      set("${var_obj}" "${obj}" PARENT_SCOPE)
      set("${var_success}" "${success}" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  # Download object to the first store.
  list(GET ExternalData_OBJECT_STORES 0 store)
  set(obj "${store}/${algo}/${hash}")

  _ExternalData_random(random)
  set(tmp "${obj}.tmp${random}")
  set(found 0)
  set(tried "")
  foreach(url_template IN LISTS ExternalData_URL_TEMPLATES)
    string(REPLACE "%(hash)" "${hash}" url_tmp "${url_template}")
    string(REPLACE "%(algo)" "${algo}" url "${url_tmp}")
    if(url MATCHES "^(.*)%\\(algo:([A-Za-z_][A-Za-z0-9_]*)\\)(.*)$")
      set(lhs "${CMAKE_MATCH_1}")
      set(key "${CMAKE_MATCH_2}")
      set(rhs "${CMAKE_MATCH_3}")
      if(DEFINED ExternalData_URL_ALGO_${algo}_${key})
        set(url "${lhs}${ExternalData_URL_ALGO_${algo}_${key}}${rhs}")
      else()
        set(url "${lhs}${algo}${rhs}")
      endif()
    endif()
    string(REGEX REPLACE "((https?|ftp)://)([^@]+@)?(.*)" "\\1\\4" secured_url "${url}")
    message(STATUS "Fetching \"${secured_url}\"")
    if(url MATCHES "^ExternalDataCustomScript://([A-Za-z_][A-Za-z0-9_]*)/(.*)$")
      _ExternalData_custom_fetch("${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" "${tmp}" err errMsg)
    else()
      _ExternalData_download_file("${url}" "${tmp}" err errMsg)
    endif()
    string(APPEND tried "\n  ${url}")
    if(err)
      string(APPEND tried " (${errMsg})")
    else()
      # Verify downloaded object.
      _ExternalData_compute_hash(dl_hash "${algo}" "${tmp}")
      if("${dl_hash}" STREQUAL "${hash}")
        set(found 1)
        break()
      else()
        string(APPEND tried " (wrong hash ${algo}=${dl_hash})")
        if("$ENV{ExternalData_DEBUG_DOWNLOAD}" MATCHES ".")
          file(RENAME "${tmp}" "${store}/${algo}/${dl_hash}")
        endif()
      endif()
    endif()
    file(REMOVE "${tmp}")
  endforeach()

  get_filename_component(dir "${name}" PATH)
  set(staged "${dir}/.ExternalData_${algo}_${hash}")

  set(success 1)
  if(found)
    # Atomically create the object.  If we lose a race with another process,
    # do not replace it.  Content-addressing ensures it has what we expect.
    file(RENAME "${tmp}" "${obj}" NO_REPLACE RESULT result)
    if (result STREQUAL "NO_REPLACE")
      file(REMOVE "${tmp}")
    elseif (result)
      message(FATAL_ERROR "Failed to rename:\n  \"${tmp}\"\nto:\n  \"${obj}\"\nwith error:\n  ${result}")
    endif()
    message(STATUS "Downloaded object: \"${obj}\"")
  elseif(EXISTS "${staged}")
    set(obj "${staged}")
    message(STATUS "Staged object: \"${obj}\"")
  else()
    if(NOT tried)
      set(tried "\n  (No ExternalData_URL_TEMPLATES given)")
    endif()
    set(success 0)
    set("${var_errorMsg}" "Object ${algo}=${hash} not found at:${tried}" PARENT_SCOPE)
  endif()

  set("${var_obj}" "${obj}" PARENT_SCOPE)
  set("${var_success}" "${success}" PARENT_SCOPE)
endfunction()

if("${ExternalData_ACTION}" STREQUAL "fetch")
  foreach(v ExternalData_OBJECT_STORES file name exts)
    if(NOT DEFINED "${v}")
      message(FATAL_ERROR "No \"-D${v}=\" value provided!")
    endif()
  endforeach()

  string(REPLACE "+" ";" exts_list "${exts}")
  set(succeeded 0)
  set(errorMsg "")
  set(hash_list )
  set(algo_list )
  set(hash )
  set(algo )
  foreach(ext ${exts_list})
    file(READ "${name}${ext}" hash)
    string(STRIP "${hash}" hash)

    if("${ext}" MATCHES "^\\.(${_ExternalData_REGEX_EXT})$")
      string(TOUPPER "${CMAKE_MATCH_1}" algo)
      string(REPLACE "-" "_" algo "${algo}")
    else()
      message(FATAL_ERROR "Unknown hash algorithm extension \"${ext}\"")
    endif()

    list(APPEND hash_list ${hash})
    list(APPEND algo_list ${algo})
  endforeach()

  list(LENGTH exts_list num_extensions)
  math(EXPR exts_range "${num_extensions} - 1")
  foreach(ii RANGE 0 ${exts_range})
    list(GET hash_list ${ii} hash)
    list(GET algo_list ${ii} algo)
    _ExternalData_get_from_object_store("${hash}" "${algo}" obj succeeded)
    if(succeeded)
      break()
    endif()
  endforeach()
  if(NOT succeeded)
    foreach(ii RANGE 0 ${exts_range})
      list(GET hash_list ${ii} hash)
      list(GET algo_list ${ii} algo)
      _ExternalData_download_object("${name}" "${hash}" "${algo}"
        obj succeeded algoErrorMsg)
      string(APPEND errorMsg "\n${algoErrorMsg}")
      if(succeeded)
        break()
      endif()
    endforeach()
  endif()
  if(NOT succeeded)
    message(FATAL_ERROR "${errorMsg}")
  endif()
  # Check if file already corresponds to the object.
  set(stamp "-hash-stamp")
  set(file_up_to_date 0)
  if(EXISTS "${file}" AND EXISTS "${file}${stamp}")
    file(READ "${file}${stamp}" f_hash)
    string(STRIP "${f_hash}" f_hash)
    if("${f_hash}" STREQUAL "${hash}")
      set(file_up_to_date 1)
    endif()
  endif()

  if(file_up_to_date)
    # Touch the file to convince the build system it is up to date.
    file(TOUCH "${file}")
  else()
    _ExternalData_link_or_copy("${obj}" "${file}")
  endif()

  # Atomically update the hash/timestamp file to record the object referenced.
  _ExternalData_atomic_write("${file}${stamp}" "${hash}\n")
elseif("${ExternalData_ACTION}" STREQUAL "local")
  foreach(v file name)
    if(NOT DEFINED "${v}")
      message(FATAL_ERROR "No \"-D${v}=\" value provided!")
    endif()
  endforeach()
  _ExternalData_link_or_copy("${name}" "${file}")
else()
  message(FATAL_ERROR "Unknown ExternalData_ACTION=[${ExternalData_ACTION}]")
endif()
