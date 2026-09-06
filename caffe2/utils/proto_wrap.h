#ifndef CAFFE2_UTILS_PROTO_WRAP_H_
#define CAFFE2_UTILS_PROTO_WRAP_H_

#include <c10/util/Logging.h>

namespace ONNX_NAMESPACE {

// ONNX wrapper functions for protobuf's GetEmptyStringAlreadyInited() function
// used to avoid duplicated global variable in the case when protobuf
// is built with hidden visibility.
TORCH_API const ::std::string& GetEmptyStringAlreadyInited();

} // namespace ONNX_NAMESPACE
#endif // CAFFE2_UTILS_PROTO_WRAP_H_
