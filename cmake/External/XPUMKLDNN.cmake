if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(DNNL_DPCPP_HOST_COMPILER "g++")
    # g++ is soft linked to /usr/bin/cxx, oneDNN woule not treat it as an absolute path
else()
    set(DNNL_DPCPP_HOST_COMPILER DEFAULT)
endif()

set(MAKE_COMMAND "cmake" "--build" ".")
ExternalProject_Add(xpumkldnn_proj
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn
    BUILD_IN_SOURCE 0
    CMAKE_ARGS  -DCMAKE_C_COMPILER=icx 
    -DCMAKE_CXX_COMPILER=icpx 
    -DCMAKE_CXX_COMPILER_ID=IntelLLVM 
    -DDNNL_GPU_RUNTIME=SYCL 
    -DDNNL_CPU_RUNTIME=THREADPOOL 
    -DDNNL_BUILD_TESTS=OFF 
    -DDNNL_BUILD_EXAMPLES=OFF
    -DDNNL_LIBRARY_TYPE=STATIC
    -DDNNL_DPCPP_HOST_COMPILER=${DNNL_DPCPP_HOST_COMPILER} # Use global cxx compiler as host compiler
    -G ${CMAKE_GENERATOR} # Align Generator to Torch
    BUILD_COMMAND ${MAKE_COMMAND}
    INSTALL_COMMAND ""
)

ExternalProject_Get_Property(xpumkldnn_proj BINARY_DIR)
set(__XPU__MKLDNN_BUILD_DIR ${BINARY_DIR})
set(XPU_MKLDNN_LIBRARIES ${__XPU_MKLDNN_BUILD_DIR}/src/libdnnl.a)
set(XPU_MKLDNN_INCLUDE ${PROJECT_SOURCE_DIR}/third_party/ideep/mkl-dnn/include)
set(xpumkldnn_dep xpumkldnn_proj)
add_library(xpumkldnn INTERFACE)
add_dependencies(xpumkldnn xpumkldnn_dep)
target_link_libraries(xpumkldnn INTERFACE ${__XPU_MKLDNN_BUILD_DIR}/src/libdnnl.a)
target_include_directories(xpumkldnn INTERFACE ${XPU_MKLDNN_INCLUDE})

set(XPUMKLDNN_FOUND TRUE)
