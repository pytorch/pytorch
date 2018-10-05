#ifndef CAFFE2_TENSOR_FACTORIES_H
#define CAFFE2_TENSOR_FACTORIES_H
#include <ATen/core/TensorOptions.h>
#include "caffe2/core/tensor.h"

namespace caffe2 {

CAFFE2_API Tensor
empty(const vector<int64_t>& dims, const at::TensorOptions& options);

} // namespace caffe2

#endif // CAFFE2_TENSOR_FACTORIES_H
