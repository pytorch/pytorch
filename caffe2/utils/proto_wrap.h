#ifndef CAFFE2_UTILS_PROTO_WRAP_H_
#define CAFFE2_UTILS_PROTO_WRAP_H_

#include <c10/util/Logging.h>

namespace caffe2 {

// A wrapper function to shut down protobuf library (this is needed in ASAN
// testing and valgrind cases to avoid protobuf appearing to "leak" memory).
TORCH_API void ShutdownProtobufLibrary();

} // namespace caffe2

namespace torch {

void ShutdownProtobufLibrary();

} // namespace torch
#endif // CAFFE2_UTILS_PROTO_WRAP_H_
