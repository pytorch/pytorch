# Try to find the pybind11 library and headers.
#  pybind11_FOUND        - system has pybind11
#  pybind11_INCLUDE_DIRS - the pybind11 include directory

find_path(pybind11_INCLUDE_DIR
        NAMES pybind11/pybind11.h
        DOC "The directory where pybind11 includes reside"
)

set(pybind11_INCLUDE_DIRS ${pybind11_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(pybind11
        FOUND_VAR pybind11_FOUND
        REQUIRED_VARS pybind11_INCLUDE_DIR
)

mark_as_advanced(pybind11_FOUND)
