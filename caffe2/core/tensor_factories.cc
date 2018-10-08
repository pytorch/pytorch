#include "tensor_factories.h"

namespace caffe2 {

Tensor empty(const vector<int64_t>& dims, const at::TensorOptions& options) {
  auto tensor = Tensor(dims, options.device().type());
  // TODO: not go through raw_mutable_data to do allocation.
  tensor.raw_mutable_data(scalarTypeToTypeMeta(options.dtype()));
  return tensor;
} // namespace caffe2

} // namespace caffe2
