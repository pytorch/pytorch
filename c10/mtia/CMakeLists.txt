# Build file for the C10 MTIA.
include(../../cmake/public/mtia.cmake)
set(C10_MTIA_HEADERS
    MTIADeviceProp.h
    MTIAMacros.h
)
add_library(c10_mtia INTERFACE)
target_compile_definitions(c10_mtia INTERFACE "C10_MTIA_BUILD_MAIN_LIB")
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(c10_mtia INTERFACE "-fvisibility=hidden")
endif()

# ---[ Dependency of c10_mtia
target_link_libraries(c10_mtia INTERFACE c10 torch::mtiart)
target_include_directories(
    c10_mtia INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

# ---[ Installation
install(TARGETS c10_mtia EXPORT Caffe2Targets DESTINATION lib)
foreach(file ${C10_MTIA_HEADERS})
  get_filename_component(dir ${file} DIRECTORY)
  install(FILES ${file} DESTINATION include/c10/mtia/${dir})
endforeach()
