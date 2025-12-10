# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindProtobuf
------------

.. note::

  If the Protobuf library is built and installed using its CMake-based
  build system, it provides a :ref:`package configuration file
  <Config File Packages>` for use with the :command:`find_package` command
  in *config mode*:

  .. code-block:: cmake

    find_package(Protobuf CONFIG)

  In this case, imported targets and CMake commands such as
  :command:`protobuf_generate` are provided by the upstream package rather
  than this module.  Additionally, some variables documented here are not
  available in *config mode*, as imported targets are preferred.  For usage
  details, refer to the upstream documentation, which is the recommended
  way to use Protobuf with CMake.

  This module works only in *module mode*.

This module finds the Protocol Buffers library (Protobuf) in *module mode*:

.. code-block:: cmake

  find_package(Protobuf [<version>] [...])

Protobuf is an open-source, language-neutral, and platform-neutral mechanism
for serializing structured data, developed by Google.  It is commonly used
for data exchange between programs or across networks.

.. versionadded:: 3.6
  Support for the ``<version>`` argument in
  :command:`find_package(Protobuf \<version\>)`.

.. versionchanged:: 3.6
  All input and output variables use the ``Protobuf_`` prefix.  Variables
  with ``PROTOBUF_`` prefix are supported for backward compatibility.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``protobuf::libprotobuf``
  .. versionadded:: 3.9

  Target encapsulating the Protobuf library usage requirements, available if
  Protobuf library is found.

``protobuf::libprotobuf-lite``
  .. versionadded:: 3.9

  Target encapsulating the ``protobuf-lite`` library usage requirements,
  available if Protobuf and its lite library are found.

``protobuf::libprotoc``
  .. versionadded:: 3.9

  Target encapsulating the ``protoc`` library usage requirements, available
  if Protobuf and its ``protoc`` library are found.

``protobuf::protoc``
  .. versionadded:: 3.10

  Imported executable target encapsulating the ``protoc`` compiler usage
  requirements, available if Protobuf and ``protoc`` are found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Protobuf_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) Protobuf library
  was found.

``Protobuf_VERSION``
  .. versionadded:: 3.6

  The version of Protobuf found.
``Protobuf_INCLUDE_DIRS``
  Include directories needed to use Protobuf.
``Protobuf_LIBRARIES``
  Libraries needed to link against to use Protobuf.
``Protobuf_PROTOC_LIBRARIES``
  Libraries needed to link against to use the ``protoc`` library.
``Protobuf_LITE_LIBRARIES``
  Libraries needed to link against to use the ``protobuf-lite`` library.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``Protobuf_INCLUDE_DIR``
  The include directory containing Protobuf headers.
``Protobuf_LIBRARY``
  The path to the ``protobuf`` library.
``Protobuf_PROTOC_LIBRARY``
  The path to the ``protoc`` library.
``Protobuf_PROTOC_EXECUTABLE``
  The path to the ``protoc`` compiler.
``Protobuf_LIBRARY_DEBUG``
  The path to the ``protobuf`` debug library.
``Protobuf_PROTOC_LIBRARY_DEBUG``
  The path to the ``protoc`` debug library.
``Protobuf_LITE_LIBRARY``
  The path to the ``protobuf-lite`` library.
``Protobuf_LITE_LIBRARY_DEBUG``
  The path to the ``protobuf-lite`` debug library.
``Protobuf_SRC_ROOT_FOLDER``
  When compiling with MSVC, if this cache variable is set, the
  protobuf-default Visual Studio project build locations will be searched for
  libraries and binaries:

  * ``<Protobuf_SRC_ROOT_FOLDER>/vsprojects/{Debug,Release}``, or
  * ``<Protobuf_SRC_ROOT_FOLDER>/vsprojects/x64/{Debug,Release}``

Hints
^^^^^

This module accepts the following optional variables before calling the
``find_package(Protobuf)``:

``Protobuf_DEBUG``
  .. versionadded:: 3.6

  Boolean variable that enables debug messages of this module to be printed
  for debugging purposes.

``Protobuf_USE_STATIC_LIBS``
  .. versionadded:: 3.9

  Set to ON to force the use of the static libraries.  Default is OFF.

Commands
^^^^^^^^

This module provides the following commands if Protobuf is found:

Generating Source Files
"""""""""""""""""""""""

.. command:: protobuf_generate

  .. versionadded:: 3.13

  Automatically generates source files from ``.proto`` schema files at build
  time:

  .. code-block:: cmake

    protobuf_generate(
      [TARGET <target>]
      [LANGUAGE <lang>]
      [OUT_VAR <variable>]
      [EXPORT_MACRO <macro>]
      [PROTOC_OUT_DIR <out-dir>]
      [PLUGIN <plugin>]
      [PLUGIN_OPTIONS <plugin-options>]
      [DEPENDENCIES <dependencies>...]
      [PROTOS <proto-files>...]
      [IMPORT_DIRS <dirs>...]
      [APPEND_PATH]
      [GENERATE_EXTENSIONS <extensions>...]
      [PROTOC_OPTIONS <options>...]
      [PROTOC_EXE <executable>]
      [DESCRIPTORS]
    )

  ``TARGET <target>``
    The CMake target to which the generated files are added as sources.  This
    option is required when ``OUT_VAR <variable>`` is not used.

  ``LANGUAGE <lang>``
    A single value: ``cpp`` or ``python``.  Determines the kind of source
    files to generate.  Defaults to ``cpp``.  For other languages, use the
    ``GENERATE_EXTENSIONS`` option.

  ``OUT_VAR <variable>``
    The name of a CMake variable that will be populated with the paths to
    the generated source files.

  ``EXPORT_MACRO <macro>``
    The name of a preprocessor macro applied to all generated Protobuf message
    classes and extern variables.  This can be used, for example, to declare
    DLL exports.  The macro should expand to ``__declspec(dllexport)`` or
    ``__declspec(dllimport)``, depending on what is being compiled.

    This option is only used when ``LANGUAGE`` is ``cpp``.

  ``PROTOC_OUT_DIR <out-dir>``
    The output directory for generated source files.  Defaults to:
    :variable:`CMAKE_CURRENT_BINARY_DIR`.

  ``PLUGIN <plugin>``
    .. versionadded:: 3.21

    An optional plugin executable.  This could be, for example, the path to
    ``grpc_cpp_plugin``.

  ``PLUGIN_OPTIONS <plugin-options>``
    .. versionadded:: 3.28

    Additional options passed to the plugin, such as ``generate_mock_code=true``
    for the gRPC C++ plugin.

  ``DEPENDENCIES <dependencies>...``
    .. versionadded:: 3.28

    Dependencies on which the generation of files depends on.  These are
    forwarded to the underlying :command:`add_custom_command(DEPENDS)`
    invocation.

    .. versionchanged:: 4.1
      This argument now accepts multiple values (``DEPENDENCIES a b c...``).
      Previously, only a single value could be specified
      (``DEPENDENCIES "a;b;c;..."``).

  ``PROTOS <proto-files>...``
    A list of ``.proto`` schema files to process.  If ``<target>`` is also
    specified, these will be combined with all ``.proto`` source files from
    that target.

  ``IMPORT_DIRS <dirs>...``
    A list of one or more common parent directories for the schema files.
    For example, if the schema file is ``proto/helloworld/helloworld.proto``
    and the import directory is ``proto/``, then the generated files will be
    ``<out-dir>/helloworld/helloworld.pb.h`` and
    ``<out-dir>/helloworld/helloworld.pb.cc``.

  ``APPEND_PATH``
    If specified, the base paths of all proto schema files are appended to
    ``IMPORT_DIRS`` (it causes ``protoc`` to be invoked with ``-I`` argument
    for each directory containing a ``.proto`` file).

  ``GENERATE_EXTENSIONS <extensions>...``
    If ``LANGUAGE`` is omitted, this must be set to specify the extensions
    generated by ``protoc``.

  ``PROTOC_OPTIONS <options>...``
    .. versionadded:: 3.28

    A list of additional command-line options passed directly to the
    ``protoc`` compiler.

  ``PROTOC_EXE <executable>``
    .. versionadded:: 4.0

    The command-line program, path, or CMake executable used to generate
    Protobuf bindings.  If omitted, ``protobuf::protoc`` imported target is
    used by default.

  ``DESCRIPTORS``
    If specified, a command-line option ``--descriptor_set_out=<proto-file>``
    is appended to ``protoc`` compiler for each ``.proto`` source file,
    enabling the creation of self-describing messages.  This option can only
    be used when ``<lang>`` is ``cpp`` and Protobuf is found in *module mode*.

    .. note::

      This option is not available when Protobuf is found in *config mode*.

Deprecated Commands
"""""""""""""""""""

The following commands are provided for backward compatibility.

.. note::

  The ``protobuf_generate_cpp()`` and ``protobuf_generate_python()``
  commands work correctly only within the same directory scope, where
  ``find_package(Protobuf ...)`` is called.

.. note::

  If Protobuf is found in *config mode*, the ``protobuf_generate_cpp()`` and
  ``protobuf_generate_python()`` commands are **not available** as of
  Protobuf version 3.0.0, unless the upstream package configuration hint
  variable ``protobuf_MODULE_COMPATIBLE`` is set to boolean true before
  calling ``find_package(Protobuf ...)``.

.. command:: protobuf_generate_cpp

  .. deprecated:: 4.1
    Use :command:`protobuf_generate`.

  Automatically generates C++ source files from ``.proto`` schema files at
  build time:

  .. code-block:: cmake

    protobuf_generate_cpp(
      <sources-variable>
      <headers-variable>
      [DESCRIPTORS <variable>]
      [EXPORT_MACRO <macro>]
      <proto-files>...
    )

  ``<sources-variable>``
    Name of the variable to define, which will contain a list of generated
    C++ source files.

  ``<headers-variable>``
    Name of the variable to define, which will contain a list of generated
    header files.

  ``DESCRIPTORS <variable>``
    .. versionadded:: 3.10

    Name of the variable to define, which will contain a list of generated
    descriptor files if requested.

    .. note::
      This option is not available when Protobuf is found in *config mode*.

  ``EXPORT_MACRO <macro>``
    Name of a macro that should expand to ``__declspec(dllexport)`` or
    ``__declspec(dllimport)``, depending on what is being compiled.

  ``<proto-files>...``
    One of more ``.proto`` files to be processed.

.. command:: protobuf_generate_python

  .. deprecated:: 4.1
    Use :command:`protobuf_generate`.

  .. versionadded:: 3.4

  Automatically generates Python source files from ``.proto`` schema files at
  build time:

  .. code-block:: cmake

    protobuf_generate_python(<python-sources-variable> <proto-files>...)

  ``<python-sources-variable>``
    Name of the variable to define, which will contain a list of generated
    Python source files.

  ``<proto-files>...``
    One or more ``.proto`` files to be processed.

---------------------------------------------------------------------

The ``protobuf_generate_cpp()`` and ``protobuf_generate_python()`` commands
accept the following optional variables before being invoked:

``Protobuf_IMPORT_DIRS``
  .. deprecated:: 4.1

  A list of additional directories to search for imported ``.proto`` files.

``PROTOBUF_GENERATE_CPP_APPEND_PATH``
  .. deprecated:: 4.1
    Use :command:`protobuf_generate(APPEND_PATH)` command option.

  A boolean variable that, if set to boolean true, causes ``protoc`` to be
  invoked with ``-I`` argument for each directory containing a ``.proto``
  file.  By default, it is set to boolean true.

Examples
^^^^^^^^

Examples: Finding Protobuf
""""""""""""""""""""""""""

Finding Protobuf library:

.. code-block:: cmake

  find_package(Protobuf)

Or, finding Protobuf and specifying a minimum required version:

.. code-block:: cmake

  find_package(Protobuf 30)

Or, finding Protobuf and making it required (if not found, processing stops
with an error message):

.. code-block:: cmake

  find_package(Protobuf REQUIRED)

Example: Finding Protobuf in Config Mode
""""""""""""""""""""""""""""""""""""""""

When Protobuf library is built and installed using its CMake-based build
system, it can be found in *config mode*:

.. code-block:: cmake

  find_package(Protobuf CONFIG)

However, some Protobuf installations might still not provide package
configuration file.  The following example shows, how to use the
:variable:`CMAKE_FIND_PACKAGE_PREFER_CONFIG` variable to find Protobuf in
*config mode* and falling back to *module mode* if config file is not found:

.. code-block:: cmake

  set(CMAKE_FIND_PACKAGE_PREFER_CONFIG TRUE)
  find_package(Protobuf)
  unset(CMAKE_FIND_PACKAGE_PREFER_CONFIG)

Example: Using Protobuf
"""""""""""""""""""""""

Finding Protobuf and linking its imported library target to a project target:

.. code-block:: cmake

  find_package(Protobuf)
  target_link_libraries(example PRIVATE protobuf::libprotobuf)

Example: Processing Proto Schema Files
""""""""""""""""""""""""""""""""""""""

The following example demonstrates how to process all ``*.proto`` schema
source files added to a target into C++ source files:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  cmake_minimum_required(VERSION 3.24)
  project(ProtobufExample)

  add_executable(example main.cxx person.proto)

  find_package(Protobuf)

  if(Protobuf_FOUND)
    protobuf_generate(TARGET example)
  endif()

  target_link_libraries(example PRIVATE protobuf::libprotobuf)
  target_include_directories(example PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

.. code-block:: proto
  :caption: ``person.proto``

  syntax = "proto3";

  message Person {
    string name = 1;
    int32 id = 2;
  }

.. code-block:: c++
  :caption: ``main.cxx``

  #include <iostream>
  #include "person.pb.h"

  int main()
  {
    Person person;
    person.set_name("Alice");
    person.set_id(123);

    std::cout << "Name: " << person.name() << "\n";
    std::cout << "ID: " << person.id() << "\n";

    return 0;
  }

Example: Using Protobuf and gRPC
""""""""""""""""""""""""""""""""

The following example shows how to use Protobuf and gRPC:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(Protobuf REQUIRED)
  find_package(gRPC CONFIG REQUIRED)

  add_library(ProtoExample Example.proto)
  target_link_libraries(ProtoExample PUBLIC gRPC::grpc++)

  protobuf_generate(TARGET ProtoExample)
  protobuf_generate(
    TARGET ProtoExample
    LANGUAGE grpc
    PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    PLUGIN_OPTIONS generate_mock_code=true
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
  )

Examples: Upgrading Deprecated Commands
"""""""""""""""""""""""""""""""""""""""

The following example shows how to process ``.proto`` files to C++ code,
using a deprecated command and its modern replacement:

.. code-block:: cmake
  :caption: ``CMakeLists.txt`` with deprecated command

  find_package(Protobuf)

  if(Protobuf_FOUND)
    protobuf_generate_cpp(
      proto_sources
      proto_headers
      EXPORT_MACRO DLL_EXPORT
      DESCRIPTORS proto_descriptors
      src/protocol/Proto1.proto
      src/protocol/Proto2.proto
    )
  endif()

  target_sources(
    example
    PRIVATE ${proto_sources} ${proto_headers} ${proto_descriptors}
  )
  target_link_libraries(example PRIVATE protobuf::libprotobuf)

.. code-block:: cmake
  :caption: ``CMakeLists.txt`` with upgraded code

  find_package(Protobuf)

  if(Protobuf_FOUND)
    protobuf_generate(
      TARGET example
      EXPORT_MACRO DLL_EXPORT
      IMPORT_DIRS src/protocol
      DESCRIPTORS
      PROTOS
        src/protocol/Proto1.proto
        src/protocol/Proto2.proto
    )
  endif()

  target_link_libraries(example PRIVATE protobuf::libprotobuf)

The following example shows how to process ``.proto`` files to Python code,
using a deprecated command and its modern replacement:

.. code-block:: cmake
  :caption: ``CMakeLists.txt`` with deprecated command

  find_package(Protobuf)

  if(Protobuf_FOUND)
    protobuf_generate_python(python_sources foo.proto)
  endif()

  add_custom_target(proto_files DEPENDS ${python_sources})

.. code-block:: cmake
  :caption: ``CMakeLists.txt`` with upgraded code

  find_package(Protobuf)

  if(Protobuf_FOUND)
    protobuf_generate(
      LANGUAGE python
      PROTOS foo.proto
      OUT_VAR python_sources
    )
  endif()

  add_custom_target(proto_files DEPENDS ${python_sources})
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

function(protobuf_generate)
  set(_options APPEND_PATH DESCRIPTORS)
  set(_singleargs LANGUAGE OUT_VAR EXPORT_MACRO PROTOC_OUT_DIR PLUGIN PLUGIN_OPTIONS PROTOC_EXE)
  if(COMMAND target_sources)
    list(APPEND _singleargs TARGET)
  endif()
  set(_multiargs PROTOS IMPORT_DIRS GENERATE_EXTENSIONS PROTOC_OPTIONS DEPENDENCIES)

  cmake_parse_arguments(protobuf_generate "${_options}" "${_singleargs}" "${_multiargs}" "${ARGN}")

  if(NOT protobuf_generate_PROTOS AND NOT protobuf_generate_TARGET)
    message(SEND_ERROR "Error: protobuf_generate called without any targets or source files")
    return()
  endif()

  if(NOT protobuf_generate_OUT_VAR AND NOT protobuf_generate_TARGET)
    message(SEND_ERROR "Error: protobuf_generate called without a target or output variable")
    return()
  endif()

  if(NOT protobuf_generate_LANGUAGE)
    set(protobuf_generate_LANGUAGE cpp)
  endif()
  string(TOLOWER ${protobuf_generate_LANGUAGE} protobuf_generate_LANGUAGE)

  if(NOT protobuf_generate_PROTOC_OUT_DIR)
    set(protobuf_generate_PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  if(protobuf_generate_EXPORT_MACRO AND protobuf_generate_LANGUAGE STREQUAL cpp)
    set(_dll_export_decl "dllexport_decl=${protobuf_generate_EXPORT_MACRO}")
  endif()

  foreach(_option ${_dll_export_decl} ${protobuf_generate_PLUGIN_OPTIONS})
    # append comma - not using CMake lists and string replacement as users
    # might have semicolons in options
    if(_plugin_options)
      set( _plugin_options "${_plugin_options},")
    endif()
    set(_plugin_options "${_plugin_options}${_option}")
  endforeach()

  if(protobuf_generate_PLUGIN)
    set(_plugin "--plugin=${protobuf_generate_PLUGIN}")
  endif()

  if(NOT protobuf_generate_GENERATE_EXTENSIONS)
    if(protobuf_generate_LANGUAGE STREQUAL cpp)
      set(protobuf_generate_GENERATE_EXTENSIONS .pb.h .pb.cc)
    elseif(protobuf_generate_LANGUAGE STREQUAL python)
      set(protobuf_generate_GENERATE_EXTENSIONS _pb2.py)
    else()
      message(SEND_ERROR "Error: protobuf_generate given unknown Language ${LANGUAGE}, please provide a value for GENERATE_EXTENSIONS")
      return()
    endif()
  endif()

  if(protobuf_generate_TARGET)
    get_target_property(_source_list ${protobuf_generate_TARGET} SOURCES)
    foreach(_file ${_source_list})
      if(_file MATCHES "proto$")
        list(APPEND protobuf_generate_PROTOS ${_file})
      endif()
    endforeach()
  endif()

  if(NOT protobuf_generate_PROTOS)
    message(SEND_ERROR "Error: protobuf_generate could not find any .proto files")
    return()
  endif()

  if(NOT protobuf_generate_PROTOC_EXE)
    if(NOT TARGET protobuf::protoc)
      message(SEND_ERROR "protoc executable not found. "
        "Please define the Protobuf_PROTOC_EXECUTABLE variable, or pass PROTOC_EXE to protobuf_generate, or ensure that protoc is in CMake's search path.")
      return()
    endif()
    # Default to using the CMake executable
    set(protobuf_generate_PROTOC_EXE protobuf::protoc)
  endif()

  if(protobuf_generate_APPEND_PATH)
    # Create an include path for each file specified
    foreach(_file ${protobuf_generate_PROTOS})
      get_filename_component(_abs_file ${_file} ABSOLUTE)
      get_filename_component(_abs_dir ${_abs_file} DIRECTORY)
      list(FIND _protobuf_include_path ${_abs_dir} _contains_already)
      if(${_contains_already} EQUAL -1)
          list(APPEND _protobuf_include_path -I ${_abs_dir})
      endif()
    endforeach()
  endif()

  foreach(DIR ${protobuf_generate_IMPORT_DIRS})
    get_filename_component(ABS_PATH ${DIR} ABSOLUTE)
    list(FIND _protobuf_include_path ${ABS_PATH} _contains_already)
    if(${_contains_already} EQUAL -1)
        list(APPEND _protobuf_include_path -I ${ABS_PATH})
    endif()
  endforeach()

  if(NOT protobuf_generate_APPEND_PATH)
    list(APPEND _protobuf_include_path -I ${CMAKE_CURRENT_SOURCE_DIR})
  endif()

  set(_generated_srcs_all)
  foreach(_proto ${protobuf_generate_PROTOS})
    get_filename_component(_abs_file ${_proto} ABSOLUTE)
    get_filename_component(_abs_dir ${_abs_file} DIRECTORY)
    get_filename_component(_basename ${_proto} NAME_WLE)
    file(RELATIVE_PATH _rel_dir ${CMAKE_CURRENT_SOURCE_DIR} ${_abs_dir})

    set(_possible_rel_dir)
    if (NOT protobuf_generate_APPEND_PATH)
      foreach(DIR ${_protobuf_include_path})
        if(NOT DIR STREQUAL "-I")
          file(RELATIVE_PATH _rel_dir ${DIR} ${_abs_dir})
          if(_rel_dir STREQUAL _abs_dir)
            continue()
          endif()
          string(FIND "${_rel_dir}" "../" _is_in_parent_folder)
          if (NOT ${_is_in_parent_folder} EQUAL 0)
            break()
          endif()
        endif()
      endforeach()
      set(_possible_rel_dir ${_rel_dir}/)
    endif()

    set(_generated_srcs)
    foreach(_ext ${protobuf_generate_GENERATE_EXTENSIONS})
      list(APPEND _generated_srcs "${protobuf_generate_PROTOC_OUT_DIR}/${_possible_rel_dir}${_basename}${_ext}")
    endforeach()

    if(protobuf_generate_DESCRIPTORS AND protobuf_generate_LANGUAGE STREQUAL cpp)
      set(_descriptor_file "${CMAKE_CURRENT_BINARY_DIR}/${_basename}.desc")
      set(_dll_desc_out "--descriptor_set_out=${_descriptor_file}")
      list(APPEND _generated_srcs ${_descriptor_file})
    endif()
    list(APPEND _generated_srcs_all ${_generated_srcs})

    set(_comment "Running ${protobuf_generate_LANGUAGE} protocol buffer compiler on ${_proto}")
    if(protobuf_generate_PROTOC_OPTIONS)
      set(_comment "${_comment}, protoc-options: ${protobuf_generate_PROTOC_OPTIONS}")
    endif()
    if(_plugin_options)
      set(_comment "${_comment}, plugin-options: ${_plugin_options}")
    endif()

    add_custom_command(
      OUTPUT ${_generated_srcs}
      COMMAND ${protobuf_generate_PROTOC_EXE}
      ARGS ${protobuf_generate_PROTOC_OPTIONS} --${protobuf_generate_LANGUAGE}_out ${_plugin_options}:${protobuf_generate_PROTOC_OUT_DIR} ${_plugin} ${_dll_desc_out} ${_protobuf_include_path} ${_abs_file}
      DEPENDS ${_abs_file} protobuf::protoc ${protobuf_generate_DEPENDENCIES}
      COMMENT ${_comment}
      VERBATIM )
  endforeach()

  set_source_files_properties(${_generated_srcs_all} PROPERTIES GENERATED TRUE)
  if(protobuf_generate_OUT_VAR)
    set(${protobuf_generate_OUT_VAR} ${_generated_srcs_all} PARENT_SCOPE)
  endif()
  if(protobuf_generate_TARGET)
    target_sources(${protobuf_generate_TARGET} PRIVATE ${_generated_srcs_all})
  endif()
endfunction()

function(PROTOBUF_GENERATE_CPP SRCS HDRS)
  cmake_parse_arguments(protobuf_generate_cpp "" "EXPORT_MACRO;DESCRIPTORS" "" ${ARGN})

  set(_proto_files "${protobuf_generate_cpp_UNPARSED_ARGUMENTS}")
  if(NOT _proto_files)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_CPP() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    set(_append_arg APPEND_PATH)
  endif()

  if(protobuf_generate_cpp_DESCRIPTORS)
    set(_descriptors DESCRIPTORS)
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
    set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    set(_import_arg IMPORT_DIRS ${Protobuf_IMPORT_DIRS})
  endif()

  set(_outvar)
  protobuf_generate(${_append_arg} ${_descriptors} LANGUAGE cpp EXPORT_MACRO ${protobuf_generate_cpp_EXPORT_MACRO} OUT_VAR _outvar ${_import_arg} PROTOS ${_proto_files})

  set(${SRCS})
  set(${HDRS})
  if(protobuf_generate_cpp_DESCRIPTORS)
    set(${protobuf_generate_cpp_DESCRIPTORS})
  endif()

  foreach(_file ${_outvar})
    if(_file MATCHES "cc$")
      list(APPEND ${SRCS} ${_file})
    elseif(_file MATCHES "desc$")
      list(APPEND ${protobuf_generate_cpp_DESCRIPTORS} ${_file})
    else()
      list(APPEND ${HDRS} ${_file})
    endif()
  endforeach()
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
  if(protobuf_generate_cpp_DESCRIPTORS)
    set(${protobuf_generate_cpp_DESCRIPTORS} "${${protobuf_generate_cpp_DESCRIPTORS}}" PARENT_SCOPE)
  endif()
endfunction()

function(PROTOBUF_GENERATE_PYTHON SRCS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: PROTOBUF_GENERATE_PYTHON() called without any proto files")
    return()
  endif()

  if(PROTOBUF_GENERATE_CPP_APPEND_PATH)
    set(_append_arg APPEND_PATH)
  endif()

  if(DEFINED PROTOBUF_IMPORT_DIRS AND NOT DEFINED Protobuf_IMPORT_DIRS)
    set(Protobuf_IMPORT_DIRS "${PROTOBUF_IMPORT_DIRS}")
  endif()

  if(DEFINED Protobuf_IMPORT_DIRS)
    set(_import_arg IMPORT_DIRS ${Protobuf_IMPORT_DIRS})
  endif()

  set(_outvar)
  protobuf_generate(${_append_arg} LANGUAGE python OUT_VAR _outvar ${_import_arg} PROTOS ${ARGN})
  set(${SRCS} ${_outvar} PARENT_SCOPE)
endfunction()


if(Protobuf_DEBUG)
  # Output some of their choices
  message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                 "Protobuf_USE_STATIC_LIBS = ${Protobuf_USE_STATIC_LIBS}")
endif()


# Backwards compatibility
# Define camel case versions of input variables
foreach(UPPER
    PROTOBUF_SRC_ROOT_FOLDER
    PROTOBUF_IMPORT_DIRS
    PROTOBUF_DEBUG
    PROTOBUF_LIBRARY
    PROTOBUF_PROTOC_LIBRARY
    PROTOBUF_INCLUDE_DIR
    PROTOBUF_PROTOC_EXECUTABLE
    PROTOBUF_LIBRARY_DEBUG
    PROTOBUF_PROTOC_LIBRARY_DEBUG
    PROTOBUF_LITE_LIBRARY
    PROTOBUF_LITE_LIBRARY_DEBUG
    )
    if (DEFINED ${UPPER})
        string(REPLACE "PROTOBUF_" "Protobuf_" Camel ${UPPER})
        if (NOT DEFINED ${Camel})
            set(${Camel} ${${UPPER}})
        endif()
    endif()
endforeach()

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_PROTOBUF_ARCH_DIR x64/)
endif()


# Support preference of static libs by adjusting CMAKE_FIND_LIBRARY_SUFFIXES
if( Protobuf_USE_STATIC_LIBS )
  set( _protobuf_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES})
  if(WIN32)
    set(CMAKE_FIND_LIBRARY_SUFFIXES .lib .a ${CMAKE_FIND_LIBRARY_SUFFIXES})
  else()
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a )
  endif()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)

# Internal function: search for normal library as well as a debug one
#    if the debug one is specified also include debug/optimized keywords
#    in *_LIBRARIES variable
function(_protobuf_find_libraries name filename)
  if(${name}_LIBRARIES)
    # Use result recorded by a previous call.
    return()
  elseif(${name}_LIBRARY)
    # Honor cache entry used by CMake 3.5 and lower.
    set(${name}_LIBRARIES "${${name}_LIBRARY}" PARENT_SCOPE)
  else()
    find_library(${name}_LIBRARY_RELEASE
      NAMES ${filename}
      NAMES_PER_DIR
      PATHS ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Release)
    mark_as_advanced(${name}_LIBRARY_RELEASE)

    find_library(${name}_LIBRARY_DEBUG
      NAMES ${filename}d ${filename}
      NAMES_PER_DIR
      PATHS ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Debug)
    mark_as_advanced(${name}_LIBRARY_DEBUG)

    select_library_configurations(${name})

    if(UNIX AND Threads_FOUND AND ${name}_LIBRARY)
      list(APPEND ${name}_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
    endif()

    set(${name}_LIBRARY "${${name}_LIBRARY}" PARENT_SCOPE)
    set(${name}_LIBRARIES "${${name}_LIBRARIES}" PARENT_SCOPE)
  endif()
endfunction()

#
# Main.
#

# By default have PROTOBUF_GENERATE_CPP macro pass -I to protoc
# for each directory where a proto file is referenced.
if(NOT DEFINED PROTOBUF_GENERATE_CPP_APPEND_PATH)
  set(PROTOBUF_GENERATE_CPP_APPEND_PATH TRUE)
endif()


# Google's provided vcproj files generate libraries with a "lib"
# prefix on Windows
if(MSVC)
    set(Protobuf_ORIG_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
    set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "")

    find_path(Protobuf_SRC_ROOT_FOLDER protobuf.pc.in)
endif()

if(UNIX)
  # Protobuf headers may depend on threading.
  find_package(Threads QUIET)
endif()

# The Protobuf library
_protobuf_find_libraries(Protobuf protobuf)
#DOC "The Google Protocol Buffers RELEASE Library"

_protobuf_find_libraries(Protobuf_LITE protobuf-lite)

# The Protobuf Protoc Library
_protobuf_find_libraries(Protobuf_PROTOC protoc)

# Restore original find library prefixes
if(MSVC)
    set(CMAKE_FIND_LIBRARY_PREFIXES "${Protobuf_ORIG_FIND_LIBRARY_PREFIXES}")
endif()

# Find the include directory
find_path(Protobuf_INCLUDE_DIR
    google/protobuf/service.h
    PATHS ${Protobuf_SRC_ROOT_FOLDER}/src
)
mark_as_advanced(Protobuf_INCLUDE_DIR)

# Find the protoc Executable
find_program(Protobuf_PROTOC_EXECUTABLE
    NAMES protoc
    DOC "The Google Protocol Buffers Compiler"
    PATHS
    ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Release
    ${Protobuf_SRC_ROOT_FOLDER}/vsprojects/${_PROTOBUF_ARCH_DIR}Debug
)
mark_as_advanced(Protobuf_PROTOC_EXECUTABLE)

if(Protobuf_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
        "requested version of Google Protobuf is ${Protobuf_FIND_VERSION}")
endif()

if(Protobuf_INCLUDE_DIR)
  set(_PROTOBUF_COMMON_HEADER ${Protobuf_INCLUDE_DIR}/google/protobuf/stubs/common.h)

  if(Protobuf_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
                   "location of common.h: ${_PROTOBUF_COMMON_HEADER}")
  endif()

  set(Protobuf_VERSION "")
  set(Protobuf_LIB_VERSION "")
  file(STRINGS ${_PROTOBUF_COMMON_HEADER} _PROTOBUF_COMMON_H_CONTENTS REGEX "#define[ \t]+GOOGLE_PROTOBUF_VERSION[ \t]+")
  if(_PROTOBUF_COMMON_H_CONTENTS MATCHES "#define[ \t]+GOOGLE_PROTOBUF_VERSION[ \t]+([0-9]+)")
      set(Protobuf_LIB_VERSION "${CMAKE_MATCH_1}")
  endif()
  unset(_PROTOBUF_COMMON_H_CONTENTS)

  math(EXPR _PROTOBUF_MAJOR_VERSION "${Protobuf_LIB_VERSION} / 1000000")
  math(EXPR _PROTOBUF_MINOR_VERSION "${Protobuf_LIB_VERSION} / 1000 % 1000")
  math(EXPR _PROTOBUF_SUBMINOR_VERSION "${Protobuf_LIB_VERSION} % 1000")
  set(Protobuf_VERSION "${_PROTOBUF_MAJOR_VERSION}.${_PROTOBUF_MINOR_VERSION}.${_PROTOBUF_SUBMINOR_VERSION}")

  if(Protobuf_DEBUG)
    message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
        "${_PROTOBUF_COMMON_HEADER} reveals protobuf ${Protobuf_VERSION}")
  endif()

  if(Protobuf_PROTOC_EXECUTABLE)
    # Check Protobuf compiler version to be aligned with libraries version
    execute_process(COMMAND ${Protobuf_PROTOC_EXECUTABLE} --version
                    OUTPUT_VARIABLE _PROTOBUF_PROTOC_EXECUTABLE_VERSION)

    if("${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}" MATCHES "libprotoc ([0-9.]+)")
      set(_PROTOBUF_PROTOC_EXECUTABLE_VERSION "${CMAKE_MATCH_1}")
    endif()

    if(Protobuf_DEBUG)
      message(STATUS "[ ${CMAKE_CURRENT_LIST_FILE}:${CMAKE_CURRENT_LIST_LINE} ] "
          "${Protobuf_PROTOC_EXECUTABLE} reveals version ${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}")
    endif()

    # protoc version 22 and up don't print the major version any more
    if(NOT "${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}" VERSION_EQUAL "${Protobuf_VERSION}" AND
       NOT "${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}" VERSION_EQUAL "${_PROTOBUF_MINOR_VERSION}.${_PROTOBUF_SUBMINOR_VERSION}")
      message(WARNING "Protobuf compiler version ${_PROTOBUF_PROTOC_EXECUTABLE_VERSION}"
        " doesn't match library version ${Protobuf_VERSION}")
    endif()
  endif()

  if(Protobuf_LIBRARY)
      if(NOT TARGET protobuf::libprotobuf)
          add_library(protobuf::libprotobuf UNKNOWN IMPORTED)
          set_target_properties(protobuf::libprotobuf PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}")
          if(EXISTS "${Protobuf_LIBRARY}")
            set_target_properties(protobuf::libprotobuf PROPERTIES
              IMPORTED_LOCATION "${Protobuf_LIBRARY}")
          endif()
          if(EXISTS "${Protobuf_LIBRARY_RELEASE}")
            set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
              IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(protobuf::libprotobuf PROPERTIES
              IMPORTED_LOCATION_RELEASE "${Protobuf_LIBRARY_RELEASE}")
          endif()
          if(EXISTS "${Protobuf_LIBRARY_DEBUG}")
            set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
              IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(protobuf::libprotobuf PROPERTIES
              IMPORTED_LOCATION_DEBUG "${Protobuf_LIBRARY_DEBUG}")
          endif()
          if (Protobuf_VERSION VERSION_GREATER_EQUAL "3.6")
            set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
              INTERFACE_COMPILE_FEATURES cxx_std_11
            )
          endif()
          if (WIN32 AND NOT Protobuf_USE_STATIC_LIBS)
            set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
              INTERFACE_COMPILE_DEFINITIONS "PROTOBUF_USE_DLLS"
            )
          endif()
          if(UNIX AND TARGET Threads::Threads)
            set_property(TARGET protobuf::libprotobuf APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES Threads::Threads)
          endif()
      endif()
  endif()

  if(Protobuf_LITE_LIBRARY)
      if(NOT TARGET protobuf::libprotobuf-lite)
          add_library(protobuf::libprotobuf-lite UNKNOWN IMPORTED)
          set_target_properties(protobuf::libprotobuf-lite PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}")
          if(EXISTS "${Protobuf_LITE_LIBRARY}")
            set_target_properties(protobuf::libprotobuf-lite PROPERTIES
              IMPORTED_LOCATION "${Protobuf_LITE_LIBRARY}")
          endif()
          if(EXISTS "${Protobuf_LITE_LIBRARY_RELEASE}")
            set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY
              IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(protobuf::libprotobuf-lite PROPERTIES
              IMPORTED_LOCATION_RELEASE "${Protobuf_LITE_LIBRARY_RELEASE}")
          endif()
          if(EXISTS "${Protobuf_LITE_LIBRARY_DEBUG}")
            set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY
              IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(protobuf::libprotobuf-lite PROPERTIES
              IMPORTED_LOCATION_DEBUG "${Protobuf_LITE_LIBRARY_DEBUG}")
          endif()
          if (WIN32 AND NOT Protobuf_USE_STATIC_LIBS)
            set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY
              INTERFACE_COMPILE_DEFINITIONS "PROTOBUF_USE_DLLS"
            )
          endif()
          if(UNIX AND TARGET Threads::Threads)
            set_property(TARGET protobuf::libprotobuf-lite APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES Threads::Threads)
          endif()
      endif()
  endif()

  if(Protobuf_PROTOC_LIBRARY)
      if(NOT TARGET protobuf::libprotoc)
          add_library(protobuf::libprotoc UNKNOWN IMPORTED)
          set_target_properties(protobuf::libprotoc PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${Protobuf_INCLUDE_DIR}")
          if(EXISTS "${Protobuf_PROTOC_LIBRARY}")
            set_target_properties(protobuf::libprotoc PROPERTIES
              IMPORTED_LOCATION "${Protobuf_PROTOC_LIBRARY}")
          endif()
          if(EXISTS "${Protobuf_PROTOC_LIBRARY_RELEASE}")
            set_property(TARGET protobuf::libprotoc APPEND PROPERTY
              IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(protobuf::libprotoc PROPERTIES
              IMPORTED_LOCATION_RELEASE "${Protobuf_PROTOC_LIBRARY_RELEASE}")
          endif()
          if(EXISTS "${Protobuf_PROTOC_LIBRARY_DEBUG}")
            set_property(TARGET protobuf::libprotoc APPEND PROPERTY
              IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(protobuf::libprotoc PROPERTIES
              IMPORTED_LOCATION_DEBUG "${Protobuf_PROTOC_LIBRARY_DEBUG}")
          endif()
          if (Protobuf_VERSION VERSION_GREATER_EQUAL "3.6")
            set_property(TARGET protobuf::libprotoc APPEND PROPERTY
              INTERFACE_COMPILE_FEATURES cxx_std_11
            )
          endif()
          if (WIN32 AND NOT Protobuf_USE_STATIC_LIBS)
            set_property(TARGET protobuf::libprotoc APPEND PROPERTY
              INTERFACE_COMPILE_DEFINITIONS "PROTOBUF_USE_DLLS"
            )
          endif()
          if(UNIX AND TARGET Threads::Threads)
            set_property(TARGET protobuf::libprotoc APPEND PROPERTY
                INTERFACE_LINK_LIBRARIES Threads::Threads)
          endif()
      endif()
  endif()

  if(Protobuf_PROTOC_EXECUTABLE)
      if(NOT TARGET protobuf::protoc)
          add_executable(protobuf::protoc IMPORTED)
          if(EXISTS "${Protobuf_PROTOC_EXECUTABLE}")
            set_target_properties(protobuf::protoc PROPERTIES
              IMPORTED_LOCATION "${Protobuf_PROTOC_EXECUTABLE}")
          endif()
      endif()
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Protobuf
    REQUIRED_VARS Protobuf_LIBRARIES Protobuf_INCLUDE_DIR
    VERSION_VAR Protobuf_VERSION
)

if(Protobuf_FOUND)
    set(Protobuf_INCLUDE_DIRS ${Protobuf_INCLUDE_DIR})
endif()

# Restore the original find library ordering
if( Protobuf_USE_STATIC_LIBS )
  set(CMAKE_FIND_LIBRARY_SUFFIXES ${_protobuf_ORIG_CMAKE_FIND_LIBRARY_SUFFIXES})
endif()

# Backwards compatibility
# Define upper case versions of output variables
foreach(Camel
    Protobuf_SRC_ROOT_FOLDER
    Protobuf_IMPORT_DIRS
    Protobuf_DEBUG
    Protobuf_INCLUDE_DIRS
    Protobuf_LIBRARIES
    Protobuf_PROTOC_LIBRARIES
    Protobuf_LITE_LIBRARIES
    Protobuf_LIBRARY
    Protobuf_PROTOC_LIBRARY
    Protobuf_INCLUDE_DIR
    Protobuf_PROTOC_EXECUTABLE
    Protobuf_LIBRARY_DEBUG
    Protobuf_PROTOC_LIBRARY_DEBUG
    Protobuf_LITE_LIBRARY
    Protobuf_LITE_LIBRARY_DEBUG
    )
    string(TOUPPER ${Camel} UPPER)
    set(${UPPER} ${${Camel}})
endforeach()

cmake_policy(POP)
