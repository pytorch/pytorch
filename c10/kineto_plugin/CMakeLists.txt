# Build file for the Kineto Plugin Profilers.
#
# Kineto Plugin is a minimal library, but it does depend on third-party libraries
# on specific backends by cases.

set(KINETO_PLUGIN_MAIN_SRCS register_api.cpp)
set(KINETO_PLUGIN_MAIN_HEADERS register_api.h)
set(KINETO_PLUGIN_SRCS)
set(KINETO_PLUGIN_HEADERS)

set(KINETO_PLUGIN_DEP_LIBS kineto)
set(KINETO_PLUGIN_COMPILE_OPTIONS "-DUSE_KINETO")

# ---[ Path of XPU
if(USE_XPU)
find_package(Pti 0.7.0 QUIET)
if(TARGET Pti::pti_view)
  list(APPEND KINETO_PLUGIN_DEP_LIBS Pti::pti_view c10_xpu)
  list(APPEND KINETO_PLUGIN_COMPILE_OPTIONS "-DC10_XPU_BUILD_KINETO_PLUGIN")
  FILE(GLOB XPU_PLUGIN_SRCS "xpu/*.cpp")
  FILE(GLOB XPU_PLUGIN_HEADERS "xpu/*.h")
  list(APPEND KINETO_PLUGIN_SRCS ${XPU_PLUGIN_SRCS})
  list(APPEND KINETO_PLUGIN_HEADERS ${XPU_PLUGIN_HEADERS})
endif()
endif(USE_XPU)

# ---[ Path of more backends

# ---[ Main Lib of Kineto Plugin
add_library(kineto_plugin
	${KINETO_PLUGIN_MAIN_SRCS} ${KINETO_PLUGIN_SRCS}
	${KINETO_PLUGIN_MAIN_HEADERS} ${KINETO_PLUGIN_HEADERS})
target_compile_options(kineto_plugin PRIVATE ${KINETO_PLUGIN_COMPILE_OPTIONS})
# Enable hidden visibility if compiler supports it.
if(${COMPILER_SUPPORTS_HIDDEN_VISIBILITY})
  target_compile_options(kineto_plugin PRIVATE "-fvisibility=hidden")
endif()

# ---[ Dependency of Kineto Plugin
target_link_libraries(kineto_plugin PUBLIC ${KINETO_PLUGIN_DEP_LIBS})

target_include_directories(
    kineto_plugin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/kineto/libkineto/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
    )

# ---[ Installation
install(TARGETS kineto_plugin EXPORT Caffe2Targets DESTINATION lib)
foreach(file ${KINETO_PLUGIN_MAIN_HEADERS} ${KINETO_PLUGIN_HEADERS})
  get_filename_component(dir ${file} DIRECTORY)
  install(FILES ${file} DESTINATION include/c10/kineto_plugin/${dir})
endforeach()
