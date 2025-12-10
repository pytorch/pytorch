.. cmake-manual-description: CMake Language Command Reference

cmake-commands(7)
*****************

.. only:: html

   .. contents::

Scripting Commands
==================

These commands are always available.

.. toctree::
   :maxdepth: 1

   /command/block
   /command/break
   /command/cmake_host_system_information
   /command/cmake_language
   /command/cmake_minimum_required
   /command/cmake_parse_arguments
   /command/cmake_path
   /command/cmake_pkg_config
   /command/cmake_policy
   /command/configure_file
   /command/continue
   /command/else
   /command/elseif
   /command/endblock
   /command/endforeach
   /command/endfunction
   /command/endif
   /command/endmacro
   /command/endwhile
   /command/execute_process
   /command/file
   /command/find_file
   /command/find_library
   /command/find_package
   /command/find_path
   /command/find_program
   /command/foreach
   /command/function
   /command/get_cmake_property
   /command/get_directory_property
   /command/get_filename_component
   /command/get_property
   /command/if
   /command/include
   /command/include_guard
   /command/list
   /command/load_cache
   /command/macro
   /command/mark_as_advanced
   /command/math
   /command/message
   /command/option
   /command/return
   /command/separate_arguments
   /command/set
   /command/set_directory_properties
   /command/set_property
   /command/site_name
   /command/string
   /command/unset
   /command/variable_watch
   /command/while

Project Commands
================

These commands are available only in CMake projects.

.. toctree::
   :maxdepth: 1

   /command/add_compile_definitions
   /command/add_compile_options
   /command/add_custom_command
   /command/add_custom_target
   /command/add_definitions
   /command/add_dependencies
   /command/add_executable
   /command/add_library
   /command/add_link_options
   /command/add_subdirectory
   /command/add_test
   /command/aux_source_directory
   /command/build_command
   /command/cmake_file_api
   /command/cmake_instrumentation
   /command/create_test_sourcelist
   /command/define_property
   /command/enable_language
   /command/enable_testing
   /command/export
   /command/fltk_wrap_ui
   /command/get_source_file_property
   /command/get_target_property
   /command/get_test_property
   /command/include_directories
   /command/include_external_msproject
   /command/include_regular_expression
   /command/install
   /command/link_directories
   /command/link_libraries
   /command/project
   /command/remove_definitions
   /command/set_source_files_properties
   /command/set_target_properties
   /command/set_tests_properties
   /command/source_group
   /command/target_compile_definitions
   /command/target_compile_features
   /command/target_compile_options
   /command/target_include_directories
   /command/target_link_directories
   /command/target_link_libraries
   /command/target_link_options
   /command/target_precompile_headers
   /command/target_sources
   /command/try_compile
   /command/try_run

.. _`CTest Commands`:

CTest Commands
==============

These commands are available only in CTest scripts.

.. toctree::
   :maxdepth: 1

   /command/ctest_build
   /command/ctest_configure
   /command/ctest_coverage
   /command/ctest_empty_binary_directory
   /command/ctest_memcheck
   /command/ctest_read_custom_files
   /command/ctest_run_script
   /command/ctest_sleep
   /command/ctest_start
   /command/ctest_submit
   /command/ctest_test
   /command/ctest_update
   /command/ctest_upload

Deprecated Commands
===================

These commands are deprecated and are only made available to maintain
backward compatibility.  The documentation of each command states the
CMake version in which it was deprecated.  Do not use these commands
in new code.

.. toctree::
   :maxdepth: 1

   /command/build_name
   /command/exec_program
   /command/export_library_dependencies
   /command/install_files
   /command/install_programs
   /command/install_targets
   /command/load_command
   /command/make_directory
   /command/output_required_files
   /command/qt_wrap_cpp
   /command/qt_wrap_ui
   /command/remove
   /command/subdir_depends
   /command/subdirs
   /command/use_mangled_mesa
   /command/utility_source
   /command/variable_requires
   /command/write_file
