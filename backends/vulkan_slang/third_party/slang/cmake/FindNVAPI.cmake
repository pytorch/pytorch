set(NVAPI_ROOT_DIR "" CACHE PATH "Path to an installed NVAPI SDK")

if(NVAPI_ROOT_DIR)
    find_path(
        NVAPI_INCLUDE_DIRS
        NAMES nvapi.h
        PATHS "${NVAPI_ROOT_DIR}"
        NO_DEFAULT_PATH
    )
else()
    find_path(
        NVAPI_INCLUDE_DIRS
        NAMES nvapi.h
        PATHS "${slang-SOURCE_DIR}/external/nvapi"
        NO_DEFAULT_PATH
    )
    find_path(NVAPI_INCLUDE_DIRS NAMES nvapi.h)

    # The nvapi.h header is in the root, so we can populate that easily
    set(NVAPI_ROOT_DIR ${NVAPI_INCLUDE_DIRS})
endif()

# x86_64 only so far
find_library(
    NVAPI_LIBRARIES
    NAMES nvapi64
    PATH_SUFFIXES amd64
    PATHS ${NVAPI_ROOT_DIR}
)

find_package_handle_standard_args(
    NVAPI
    REQUIRED_VARS NVAPI_INCLUDE_DIRS NVAPI_LIBRARIES
)
