# find_package(Protobuf REQUIRED)

file(GLOB Caffe2_PROTOBUF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")

caffe2_protobuf_generate_cpp_py(Caffe2_PROTO_SRCS Caffe2_PROTO_HEADERS Caffe2_PROTO_PY ${Caffe2_PROTOBUF_FILES})

add_library(Caffe2_PROTO OBJECT ${Caffe2_PROTO_HEADERS} ${Caffe2_PROTO_SRCS})
if (MSVC)
  target_compile_definitions(
      Caffe2_PROTO PRIVATE "-DCAFFE2_API=__declspec(dllexport)")
endif()
install(FILES ${Caffe2_PROTO_HEADERS} DESTINATION include/caffe2/proto)
