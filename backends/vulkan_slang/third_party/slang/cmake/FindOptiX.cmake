set(Optix_ROOT_DIR "" CACHE PATH "Path to an installed OptiX SDK")

if(Optix_ROOT_DIR)
    find_path(
        OptiX_INCLUDE_DIRS
        NAMES optix.h
        PATH_SUFFIXES include
        PATHS "${Optix_ROOT_DIR}"
        NO_DEFAULT_PATH
    )
else()
    find_path(OptiX_INCLUDE_DIRS NAMES optix.h)
endif()

find_package_handle_standard_args(OptiX REQUIRED_VARS OptiX_INCLUDE_DIRS)
