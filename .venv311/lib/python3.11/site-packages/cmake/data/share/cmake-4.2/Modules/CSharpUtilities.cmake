# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CSharpUtilities
---------------

.. versionadded:: 3.8

This utility module is intended to simplify the configuration of CSharp/.NET
targets and provides a collection of commands for managing CSharp targets
with :ref:`Visual Studio Generators`, version 2010 and newer.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CSharpUtilities)

Commands
^^^^^^^^

This module provides the following commands:

.. rubric:: Main Commands

- :command:`csharp_set_windows_forms_properties`
- :command:`csharp_set_designer_cs_properties`
- :command:`csharp_set_xaml_cs_properties`

.. rubric:: Helper Commands

- :command:`csharp_get_filename_keys`
- :command:`csharp_get_filename_key_base`
- :command:`csharp_get_dependentupon_name`

Main Commands
"""""""""""""

.. command:: csharp_set_windows_forms_properties

  Sets source file properties for use of Windows Forms:

  .. code-block:: cmake

    csharp_set_windows_forms_properties([<files>...])

  ``<files>...``
    A list of zero or more source files which are relevant for setting the
    :prop_sf:`VS_CSHARP_<tagname>` source file properties. This typically
    includes files with ``.cs``, ``.resx``, and ``.Designer.cs`` extensions.

  Use this command when a CSharp target in the project uses Windows Forms.

  This command searches in the provided list of files for pairs of related
  files ending with ``.Designer.cs`` (*designer* files) or ``.resx``
  (*resource* files).  For each such file, a corresponding base ``.cs``
  file is searched (with the same base name).  When found, the
  :prop_sf:`VS_CSHARP_<tagname>` source file properties are set as follows:

  For the **.cs** file:
   - ``VS_CSHARP_SubType "Form"``

  For the **.Designer.cs** file (if it exists):
   - ``VS_CSHARP_DependentUpon <cs-filename>``
   - ``VS_CSHARP_DesignTime ""`` (tag is removed if previously defined)
   - ``VS_CSHARP_AutoGen ""`` (tag is removed if previously defined)

  For the **.resx** file (if it exists):
   - ``VS_RESOURCE_GENERATOR ""`` (tag is removed if previously defined)
   - ``VS_CSHARP_DependentUpon <cs-filename>``
   - ``VS_CSHARP_SubType "Designer"``

.. command:: csharp_set_designer_cs_properties

  Sets source file properties for ``.Designer.cs`` files depending on
  sibling filenames:

  .. code-block:: cmake

    csharp_set_designer_cs_properties([<files>...])

  ``<files>...``
    A list of zero or more source files which are relevant for setting the
    :prop_sf:`VS_CSHARP_<tagname>` source file properties.  This typically
    includes files with ``.resx``, ``.settings``, and ``.Designer.cs``
    extensions.

  Use this command, if the CSharp target does **not** use Windows Forms
  (for Windows Forms use :command:`csharp_set_windows_forms_properties`
  instead).

  This command searches through the provided list for files ending in
  ``.Designer.cs`` (*designer* files).  For each such file, it looks for
  sibling files with the same base name but different extensions.  If a
  matching file is found, the appropriate source file properties are set on
  the corresponding ``.Designer.cs`` file based on the matched extension:

  If match is **.resx** file:

  - ``VS_CSHARP_AutoGen "True"``
  - ``VS_CSHARP_DesignTime "True"``
  - ``VS_CSHARP_DependentUpon <resx-filename>``

  If match is **.cs** file:

  - ``VS_CSHARP_DependentUpon <cs-filename>``

  If match is **.settings** file:

  - ``VS_CSHARP_AutoGen "True"``
  - ``VS_CSHARP_DesignTimeSharedInput "True"``
  - ``VS_CSHARP_DependentUpon <settings-filename>``

.. note::

    Because the source file properties of the ``.Designer.cs`` file are set
    according to the found matches and every match sets the
    :prop_sf:`VS_CSHARP_DependentUpon <VS_CSHARP_<tagname>>`
    source file property, there should only be one match for
    each ``Designer.cs`` file.

.. command:: csharp_set_xaml_cs_properties

  Sets source file properties for use of Windows Presentation Foundation (WPF)
  and XAML:

  .. code-block:: cmake

    csharp_set_xaml_cs_properties([<files>...])

  Use this command, if the CSharp target uses WPF/XAML.

  ``<files>...``
    A list of zero or more source files which are relevant for setting the
    :prop_sf:`VS_CSHARP_<tagname>` source file properties.  This typically
    includes files with ``.cs``, ``.xaml``, and ``.xaml.cs`` extensions.

  This command searches the provided file list for files ending with
  ``.xaml.cs``.  For each such XAML code-behind file, a corresponding
  ``.xaml`` file with the same base name is searched.  If found, the
  following source file property is set on the ``.xaml.cs`` file:

  - ``VS_CSHARP_DependentUpon <xaml-filename>``

Helper Commands
"""""""""""""""

These commands are used by the above main commands and typically aren't
used directly:

.. command:: csharp_get_filename_keys

  Computes a normalized list of key values to identify source files
  independently of relative or absolute paths given in CMake and eliminates
  case sensitivity:

  .. code-block:: cmake

    csharp_get_filename_keys(<variable> [<files>...])

  ``<variable>``
    Name of the variable in which the list of computed keys is stored.

  ``<files>...``
    Zero or more source file paths as given to CSharp target using commands
    like :command:`add_library`, or :command:`add_executable`.

  This command canonicalizes file paths to ensure consistent identification
  of source files.  This is useful when source files are added to a target
  using different path forms.  Without normalization, CMake may treat paths
  like ``myfile.Designer.cs`` and
  ``${CMAKE_CURRENT_SOURCE_DIR}/myfile.Designer.cs`` as different files,
  which can cause issues when setting source file properties.

  For example, the following code will fail to set properties because the
  file paths do not match exactly:

  .. code-block:: cmake

    add_library(lib myfile.cs ${CMAKE_CURRENT_SOURCE_DIR}/myfile.Designer.cs)

    set_source_files_properties(
      myfile.Designer.cs
      PROPERTIES VS_CSHARP_DependentUpon myfile.cs
    )

.. command:: csharp_get_filename_key_base

  Returns the full filepath and name **without** extension of a key:

  .. code-block:: cmake

    csharp_get_filename_key_base(<base> <key>)

  ``<base>``
    Name of the variable with the computed base value of the ``<key>``
    without the file extension.

  ``<key>``
    The key of which the base will be computed.  Expected to be a
    uppercase full filename from :command:`csharp_get_filename_keys`.

.. command:: csharp_get_dependentupon_name

  Computes a string which can be used as value for the source file property
  :prop_sf:`VS_CSHARP_<tagname>` with ``<tagname>`` being ``DependentUpon``:

  .. code-block:: cmake

    csharp_get_dependentupon_name(<variable> <file>)

  ``<variable>``
    Name of the variable with the result value.  Value contains the name
    of the ``<file>`` without directory.

  ``<file>``
    Filename to convert for using in the value of the
    ``VS_CSHARP_DependentUpon`` source file property.
#]=======================================================================]

function(csharp_get_filename_keys OUT)
  set(${OUT} "")
  foreach(f ${ARGN})
    get_filename_component(f ${f} REALPATH)
    string(TOUPPER ${f} f)
    list(APPEND ${OUT} ${f})
  endforeach()
  set(${OUT} "${${OUT}}" PARENT_SCOPE)
endfunction()

function(csharp_get_filename_key_base base key)
  get_filename_component(dir ${key} DIRECTORY)
  get_filename_component(fil ${key} NAME_WE)
  set(${base} "${dir}/${fil}" PARENT_SCOPE)
endfunction()

function(csharp_get_dependentupon_name out in)
  get_filename_component(${out} ${in} NAME)
  set(${out} ${${out}} PARENT_SCOPE)
endfunction()

function(csharp_set_windows_forms_properties)
  csharp_get_filename_keys(fileKeys ${ARGN})
  foreach(key ${fileKeys})
    get_filename_component(ext ${key} EXT)
    if(${ext} STREQUAL ".DESIGNER.CS" OR
       ${ext} STREQUAL ".RESX")
      csharp_get_filename_key_base(NAME_BASE ${key})
      list(FIND fileKeys "${NAME_BASE}.CS" FILE_INDEX)
      if(NOT ${FILE_INDEX} EQUAL -1)
        list(GET ARGN ${FILE_INDEX} FILE_NAME)
        # set properties of main form file
        set_source_files_properties("${FILE_NAME}"
          PROPERTIES
          VS_CSHARP_SubType "Form")
        csharp_get_dependentupon_name(LINK "${FILE_NAME}")
        # set properties of designer file (if found)
        list(FIND fileKeys "${NAME_BASE}.DESIGNER.CS" FILE_INDEX)
        if(NOT ${FILE_INDEX} EQUAL -1)
          list(GET ARGN ${FILE_INDEX} FILE_NAME)
          set_source_files_properties("${FILE_NAME}"
            PROPERTIES
            VS_CSHARP_DependentUpon "${LINK}"
            VS_CSHARP_DesignTime ""
            VS_CSHARP_AutoGen "")
        endif()
        # set properties of corresponding resource file (if found)
        list(FIND fileKeys "${NAME_BASE}.RESX" FILE_INDEX)
        if(NOT ${FILE_INDEX} EQUAL -1)
          list(GET ARGN ${FILE_INDEX} FILE_NAME)
          set_source_files_properties("${FILE_NAME}"
            PROPERTIES
            VS_RESOURCE_GENERATOR ""
            VS_CSHARP_DependentUpon "${LINK}"
            VS_CSHARP_SubType "Designer")
        endif()
      endif()
    endif()
  endforeach()
endfunction()

function(csharp_set_designer_cs_properties)
  csharp_get_filename_keys(fileKeys ${ARGN})
  set(INDEX -1)
  foreach(key ${fileKeys})
    math(EXPR INDEX "${INDEX}+1")
    list(GET ARGN ${INDEX} source)
    get_filename_component(ext ${key} EXT)
    if(${ext} STREQUAL ".DESIGNER.CS")
      csharp_get_filename_key_base(NAME_BASE ${key})
      if("${NAME_BASE}.RESX" IN_LIST fileKeys)
        list(FIND fileKeys "${NAME_BASE}.RESX" FILE_INDEX)
        list(GET ARGN ${FILE_INDEX} FILE_NAME)
        csharp_get_dependentupon_name(LINK "${FILE_NAME}")
        set_source_files_properties("${source}"
          PROPERTIES
          VS_CSHARP_AutoGen "True"
          VS_CSHARP_DesignTime "True"
          VS_CSHARP_DependentUpon "${LINK}")
      elseif("${NAME_BASE}.CS" IN_LIST fileKeys)
        list(FIND fileKeys "${NAME_BASE}.CS" FILE_INDEX)
        list(GET ARGN ${FILE_INDEX} FILE_NAME)
        csharp_get_dependentupon_name(LINK "${FILE_NAME}")
        set_source_files_properties("${source}"
          PROPERTIES
          VS_CSHARP_DependentUpon "${LINK}")
      elseif("${NAME_BASE}.SETTINGS" IN_LIST fileKeys)
        list(FIND fileKeys "${NAME_BASE}.SETTINGS" FILE_INDEX)
        list(GET ARGN ${FILE_INDEX} FILE_NAME)
        csharp_get_dependentupon_name(LINK "${FILE_NAME}")
        set_source_files_properties("${source}"
          PROPERTIES
          VS_CSHARP_AutoGen "True"
          VS_CSHARP_DesignTimeSharedInput "True"
          VS_CSHARP_DependentUpon "${LINK}")
      endif()
    endif()
  endforeach()
endfunction()

function(csharp_set_xaml_cs_properties)
  csharp_get_filename_keys(fileKeys ${ARGN})
  set(INDEX -1)
  foreach(key ${fileKeys})
    math(EXPR INDEX "${INDEX}+1")
    list(GET ARGN ${INDEX} source)
    get_filename_component(ext ${key} EXT)
    if(${ext} STREQUAL ".XAML.CS")
      csharp_get_filename_key_base(NAME_BASE ${key})
      if("${NAME_BASE}.XAML" IN_LIST fileKeys)
        list(FIND fileKeys "${NAME_BASE}.XAML" FILE_INDEX)
        list(GET ARGN ${FILE_INDEX} FILE_NAME)
        csharp_get_dependentupon_name(LINK "${FILE_NAME}")
        set_source_files_properties("${source}"
          PROPERTIES
          VS_CSHARP_DependentUpon "${LINK}")
      endif()
    endif()
  endforeach()
endfunction()
