# ---[ CPU files.
file(GLOB_RECURSE NOMNI_SRCS *.cc)
file(GLOB NOMNI_TEST_SRCS test.cc)
exclude(NOMNI_SRCS "${NOMNI_SRCS}" "${NOMNI_TEST_SRCS}")

add_library(nomnigraph STATIC "${NOMNI_SRCS}")
add_dependencies(nomnigraph Caffe_PROTO Caffe2_PROTO)

target_include_directories(nomnigraph PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
list(APPEND Caffe2_DEPENDENCY_LIBS nomnigraph)
set(Caffe2_DEPENDENCY_LIBS ${Caffe2_DEPENDENCY_LIBS} PARENT_SCOPE)
