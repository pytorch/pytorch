#include "caffe2/utils/proto_wrap.h"
#include "caffe2/core/common.h"

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/generated_message_util.h>

namespace ONNX_NAMESPACE {

// ONNX wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
TORCH_API const ::std::string& GetEmptyStringAlreadyInited() {
  return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
}

}  // namespace ONNX_NAMESPACE

namespace caffe2 {

// Caffe2 wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
TORCH_API const ::std::string& GetEmptyStringAlreadyInited() {
  return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
}

void ShutdownProtobufLibrary() {
  ::google::protobuf::ShutdownProtobufLibrary();
}

}  // namespace caffe2

namespace torch {

// Caffe2 wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
TORCH_API const ::std::string& GetEmptyStringAlreadyInited() {
  return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
}

void ShutdownProtobufLibrary() {
  ::google::protobuf::ShutdownProtobufLibrary();
}

}  // namespace torch
