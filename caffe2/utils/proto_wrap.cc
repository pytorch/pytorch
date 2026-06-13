#include "caffe2/utils/proto_wrap.h"

#include <google/protobuf/stubs/common.h>

namespace caffe2 {

void ShutdownProtobufLibrary() {
  ::google::protobuf::ShutdownProtobufLibrary();
}

}  // namespace caffe2

namespace torch {

void ShutdownProtobufLibrary() {
  ::google::protobuf::ShutdownProtobufLibrary();
}

}  // namespace torch
