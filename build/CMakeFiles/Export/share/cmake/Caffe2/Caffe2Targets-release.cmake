#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "c10" for configuration "Release"
set_property(TARGET c10 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(c10 PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libc10.so"
  IMPORTED_SONAME_RELEASE "libc10.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS c10 )
list(APPEND _IMPORT_CHECK_FILES_FOR_c10 "${_IMPORT_PREFIX}/lib/libc10.so" )

# Import target "torch" for configuration "Release"
set_property(TARGET torch APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(torch PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libtorch.so"
  IMPORTED_SONAME_RELEASE "libtorch.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS torch )
list(APPEND _IMPORT_CHECK_FILES_FOR_torch "${_IMPORT_PREFIX}/lib/libtorch.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
