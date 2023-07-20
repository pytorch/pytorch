
# Try to find the CUSPARSELT library and headers.
#  CUSPARSELT_FOUND        - system has CUSPARSELT
#  CUSPARSELT_INCLUDE_DIRS - the CUSPARSELT include directory

find_path(CUSPARSELT_INCLUDE_PATH
        HINTS "/home/jessecai/local/libcusparse_lt-linux-x86_64-0.4.0.7-archive/include"
        NAMES cusparseLt.h
        DOC "The directory where CUSPARSELT includes reside"
)

find_path(CUSPARSELT_LIBRARY_PATH
        HINTS "/home/jessecai/local/libcusparse_lt-linux-x86_64-0.4.0.7-archive/lib"
        NAMES libcusparseLt.so
        DOC "The directory where CUSPARSELT lib reside"
)


include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(CUSPARSELT
        FOUND_VAR CUSPARSELT_FOUND
        REQUIRED_VARS CUSPARSELT_INCLUDE_PATH CUSPARSELT_LIBRARY_PATH
)

mark_as_advanced(CUSPARSELT_FOUND)
