#ifndef CAFFE2_TENSOR_INT8_H_
#define CAFFE2_TENSOR_INT8_H_

#include "caffe2/core/context.h"
#include "caffe2/core/tensor.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {
namespace int8 {

struct Int8TensorCPU {
  float scale{1.0};
  int32_t zero_point{0};
  // Generally stores uint8_t data, but sometimes int32_t (e.g. bias
  // parameters).
  Tensor t{CPU};
};
} // namespace int8
} // namespace caffe2

#endif // CAFFE2_TENSOR_INT8_H_
