if(BUILD_CAFFE2)
  file(GLOB Caffe2_PROTOBUF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/*.proto")
else()
  set(Caffe2_PROTOBUF_FILES "${CMAKE_CURRENT_SOURCE_DIR}/torch.proto;${CMAKE_CURRENT_SOURCE_DIR}/caffe2.proto")
endif()

caffe2_protobuf_generate_cpp_py(Caffe2_PROTO_SRCS Caffe2_PROTO_HEADERS Caffe2_PROTO_PY ${Caffe2_PROTOBUF_FILES})

add_library(Caffe2_PROTO OBJECT ${Caffe2_PROTO_HEADERS} ${Caffe2_PROTO_SRCS})

if(MSVC)
  if(BUILD_SHARED_LIBS)
    set(TORCH_API_DEFINE "-DTORCH_API=__declspec(dllexport)")
  else()
    set(TORCH_API_DEFINE "-DTORCH_API=")
  endif()
else()
  set(TORCH_API_DEFINE "-DTORCH_API=")
endif()
target_compile_definitions(
    Caffe2_PROTO PRIVATE ${TORCH_API_DEFINE})

install(FILES ${Caffe2_PROTO_HEADERS} DESTINATION include/caffe2/proto)
