# find_package(Protobuf REQUIRED)

file(GLOB Caffe_PROTOBUF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

caffe_protobuf_generate_cpp_py(${CMAKE_CURRENT_BINARY_DIR} Caffe_PROTO_SRCS Caffe_PROTO_HEADERS Caffe_PROTO_PY ${Caffe_PROTOBUF_FILES})

add_library(Caffe_PROTO SHARED ${Caffe_PROTO_HEADERS} ${Caffe_PROTO_SRCS})
target_include_directories(Caffe_PROTO PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

if(USE_SYSTEM_PROTOBUF)
  target_link_libraries(Caffe_PROTO protobuf)
else()
  add_dependencies(Caffe_PROTO protobuf_external)
endif()
