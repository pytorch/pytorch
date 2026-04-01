set(Aftermath_ROOT_DIR "" CACHE PATH "Path to an installed Aftermath SDK")

if(Aftermath_ROOT_DIR)
    find_path(
        Aftermath_INCLUDE_DIRS
        NAMES GFSDK_Aftermath.h
        PATH_SUFFIXES include
        PATHS "${Aftermath_ROOT_DIR}"
        NO_DEFAULT_PATH
    )
else()
    find_path(Aftermath_INCLUDE_DIRS NAMES GFSDK_Aftermath.h)
endif()

# x86_64 only so far
find_library(
    Aftermath_LIBRARIES
    NAMES GFSDK_Aftermath_Lib.x64
    PATH_SUFFIXES x64
    PATHS "${Aftermath_ROOT_DIR}/lib"
)

find_package_handle_standard_args(
    Aftermath
    REQUIRED_VARS Aftermath_INCLUDE_DIRS Aftermath_LIBRARIES
)
