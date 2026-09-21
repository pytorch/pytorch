# xpupti: XPU implementation for kineto profiler
# outputs:
#  SYCL_INCLUDE_DIR   --  SYCL include dir
#  SYCL_LIBRARY       --  SYCL library file
#  XPU_xpupti_LIBRARY --  XPUPTI dependencies lib
#  XPUPTI_BUILD_FLAG  --  XPUPTI build flags

if((NOT SYCL_INCLUDE_DIR) OR (NOT SYCL_LIBRARY_DIR))
    include(FindSYCLToolkit.cmake)
    if(NOT SYCLTOOLKIT_FOUND)
        message(FATAL_ERROR
            "KINETO_BACKEND=xpu but SYCL toolkit was not found: "
            "${SYCL_NOT_FOUND_MESSAGE}. "
            "Install the SYCL toolkit or set KINETO_BACKEND=cpu.")
    endif()
endif()

message(STATUS " SYCL_INCLUDE_DIR = ${SYCL_INCLUDE_DIR}")
message(STATUS " SYCL_LIBRARY = ${SYCL_LIBRARY}")

set(XPUPTI_INCLUDE_DIR ${SYCL_INCLUDE_DIR})

# find xpupti sdk
find_package(Pti REQUIRED)
if(TARGET Pti::pti_view)
    message(STATUS " Found XPUPTI")

    set(pti_view_names)
    # For Linux, due to it has soft link original file name is fine.
    list(APPEND pti_view_names pti_view)
    # For Windows, file name to be appended with version number.
    foreach(ver_major RANGE 0 1)
        foreach(ver_minor RANGE 0 19)
        list(APPEND pti_view_names pti_view-${ver_major}-${ver_minor})
        endforeach()
    endforeach()

    get_target_property(PTI_INCLUDE_DIR Pti::pti_view INTERFACE_INCLUDE_DIRECTORIES)
    find_library(PTI_VIEW_LIBRARY NAMES ${pti_view_names} PATHS "${PTI_INCLUDE_DIR}/../lib")
    set(PTI_LIBRARY ${PTI_VIEW_LIBRARY} CACHE STRING "Imported PTI library.")
    set(PTI_INCLUDE_DIR ${PTI_INCLUDE_DIR} CACHE STRING "PTI include directory.")

    # find dependent lib
    set(XPU_xpupti_LIBRARY ${SYCL_LIBRARY})
    list(APPEND XPU_xpupti_LIBRARY ${PTI_LIBRARY})
    set(XPU_xpupti_LIBRARY ${XPU_xpupti_LIBRARY} PARENT_SCOPE)

    # find dependent include
    list(APPEND XPUPTI_INCLUDE_DIR ${PTI_INCLUDE_DIR})
    set(XPUPTI_INCLUDE_DIR ${XPUPTI_INCLUDE_DIR} PARENT_SCOPE)

    set(XPUPTI_BUILD_FLAG "-DHAS_XPUPTI")
    set(XPUPTI_BUILD_FLAG ${XPUPTI_BUILD_FLAG} PARENT_SCOPE)

    message(STATUS " XPU_xpupti_LIBRARY = ${XPU_xpupti_LIBRARY}")
    message(STATUS " XPUPTI_INCLUDE_DIR = ${XPUPTI_INCLUDE_DIR}")
    message(STATUS " XPUPTI_BUILD_FLAG = ${XPUPTI_BUILD_FLAG}")
else()
    message(FATAL_ERROR
        "KINETO_BACKEND=xpu but XPUPTI (Pti::pti_view) was not found. "
        "Install Intel PTI or set KINETO_BACKEND=cpu.")
endif()
