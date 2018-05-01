set(HAVE_HIP FALSE)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

include(${HIP_PATH}/cmake/FindHIP.cmake)

if(EXISTS ${HIP_ROOT_DIR})
	set(HAVE_HIP TRUE)
	list(APPEND Caffe2_HIP_DEPENDENCY_LIBS hip_hcc)
endif()