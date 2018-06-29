#include "caffe2/utils/proto_wrap.h"

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/generated_message_util.h>

namespace caffe {

// Caffe wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
google::protobuf::internal::ExplicitlyConstructed< ::std::string>& fixed_address_empty_string() {
  return ::google::protobuf::internal::fixed_address_empty_string();
}
const ::std::string& GetEmptyStringAlreadyInited() {
  return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
}
const ::std::string& GetEmptyString() {
  return ::google::protobuf::internal::GetEmptyString();
}

}  // namespace caffe

namespace ONNX_NAMESPACE {

// ONNX wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
google::protobuf::internal::ExplicitlyConstructed< ::std::string>& fixed_address_empty_string() {
  return ::google::protobuf::internal::fixed_address_empty_string();
}
const ::std::string& GetEmptyStringAlreadyInited() {
  return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
}
const ::std::string& GetEmptyString() {
  return ::google::protobuf::internal::GetEmptyString();
}

}  // namespace ONNX_NAMESPACE

namespace caffe2 {

// Caffe2 wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
google::protobuf::internal::ExplicitlyConstructed< ::std::string>& fixed_address_empty_string() {
  return ::google::protobuf::internal::fixed_address_empty_string();
}
const ::std::string& GetEmptyStringAlreadyInited() {
  return ::google::protobuf::internal::GetEmptyStringAlreadyInited();
}
const ::std::string& GetEmptyString() {
  return ::google::protobuf::internal::GetEmptyString();
}

void ShutdownProtobufLibrary() {
  ::google::protobuf::ShutdownProtobufLibrary();
}

}  // namespace caffe2
