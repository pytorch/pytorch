# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPHP4
--------

Finds PHP version 4, a general-purpose scripting language:

.. code-block:: cmake

  find_package(PHP4 [...])

.. note::

  This module is specifically for PHP version 4, which is obsolete and no longer
  supported.  For modern development, use a newer PHP version.

This module checks if PHP 4 is installed and determines the locations of the
include directories and the PHP command-line interpreter.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``PHP4_FOUND``
  Boolean indicating whether PHP 4 was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``PHP4_INCLUDE_PATH``
  The directory containing ``php.h`` and other headers needed to use PHP.
``PHP4_EXECUTABLE``
  The full path to the ``php`` command-line interpreter executable.

Examples
^^^^^^^^

Finding PHP:

.. code-block:: cmake

  find_package(PHP4)
#]=======================================================================]

set(PHP4_POSSIBLE_INCLUDE_PATHS
  /usr/include/php4
  /usr/local/include/php4
  /usr/include/php
  /usr/local/include/php
  /usr/local/apache/php
  )

set(PHP4_POSSIBLE_LIB_PATHS
  /usr/lib
  )

find_path(PHP4_FOUND_INCLUDE_PATH main/php.h
  ${PHP4_POSSIBLE_INCLUDE_PATHS})

if(PHP4_FOUND_INCLUDE_PATH)
  set(php4_paths "${PHP4_POSSIBLE_INCLUDE_PATHS}")
  foreach(php4_path Zend main TSRM)
    set(php4_paths ${php4_paths} "${PHP4_FOUND_INCLUDE_PATH}/${php4_path}")
  endforeach()
  set(PHP4_INCLUDE_PATH "${php4_paths}")
endif()

find_program(PHP4_EXECUTABLE NAMES php4 php )

mark_as_advanced(
  PHP4_EXECUTABLE
  PHP4_FOUND_INCLUDE_PATH
  )

if(APPLE)
  # this is a hack for now
  string(APPEND CMAKE_SHARED_MODULE_CREATE_C_FLAGS
   " -Wl,-flat_namespace")
  foreach(symbol
    __efree
    __emalloc
    __estrdup
    __object_init_ex
    __zend_get_parameters_array_ex
    __zend_list_find
    __zval_copy_ctor
    _add_property_zval_ex
    _alloc_globals
    _compiler_globals
    _convert_to_double
    _convert_to_long
    _zend_error
    _zend_hash_find
    _zend_register_internal_class_ex
    _zend_register_list_destructors_ex
    _zend_register_resource
    _zend_rsrc_list_get_rsrc_type
    _zend_wrong_param_count
    _zval_used_for_init
    )
    string(APPEND CMAKE_SHARED_MODULE_CREATE_C_FLAGS
      ",-U,${symbol}")
  endforeach()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PHP4 DEFAULT_MSG PHP4_EXECUTABLE PHP4_INCLUDE_PATH)
