# find_package(Protobuf REQUIRED)

file(GLOB Caffe2_PROTOBUF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

caffe_protobuf_generate_cpp_py(${CMAKE_CURRENT_BINARY_DIR} Caffe2_PROTO_SRCS Caffe2_PROTO_HEADERS Caffe2_PROTO_PY ${Caffe2_PROTOBUF_FILES})

add_library(Caffe2_PROTO SHARED ${Caffe2_PROTO_HEADERS} ${Caffe2_PROTO_SRCS})
target_include_directories(Caffe2_PROTO PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

if (USE_SYSTEM_PROTOBUF)
  target_link_libraries(Caffe2_PROTO protobuf)
else()
  add_dependencies(Caffe2_PROTO protobuf_external)
endif()
