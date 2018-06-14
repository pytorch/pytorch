#ifndef CAFFE2_UTILS_PROTO_WRAP_H_
#define CAFFE2_UTILS_PROTO_WRAP_H_

namespace caffe2 {

// A wrapper function to shut down protobuf library (this is needed in ASAN
// testing and valgrind cases to avoid protobuf appearing to "leak" memory).
void ShutdownProtobufLibrary();

} // namespace caffe2

#endif // CAFFE2_UTILS_PROTO_WRAP_H_
