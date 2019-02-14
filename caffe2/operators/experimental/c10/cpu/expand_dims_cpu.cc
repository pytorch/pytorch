#include <ATen/core/dispatch/KernelRegistration.h>
#include "caffe2/operators/experimental/c10/schemas/expand_dims.h"
#include "caffe2/utils/math.h"
#include "caffe2/core/tensor.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {

struct Cache final : public c10::KernelCache {
  std::vector<int64_t> dims;
  bool initialized = false;
};

template <class DataType>
void expand_dims_op_cpu_impl(
    const at::Tensor& input_,
    const at::Tensor& output_,
    ArrayRef<int64_t> dims,
    Cache* cache) {
  Tensor input{C10Tensor(input_)};
  Tensor output{C10Tensor(output_)};

  if (!cache->initialized) {
    cache->dims = dims.vec();
    auto originalSize = cache->dims.size();
    CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");
    std::sort(cache->dims.begin(), cache->dims.end());
    cache->dims.erase(
        std::unique(cache->dims.begin(), cache->dims.end()), cache->dims.end());
    if (cache->dims.size() < originalSize) {
      LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
    }
    CAFFE_ENFORCE(
        cache->dims.front() >= 0, "Dimension ids must be non-negative.");
    cache->initialized = true;
  }

  output.CopyFrom(input);
  if (cache->dims.empty()) {
    return;
  }

  auto newDims = input.sizes().vec();
  CAFFE_ENFORCE_GE(
      input.sizes().size() + cache->dims.size(),
      cache->dims.back() + 1,
      "Input needs at least ",
      (1 + cache->dims.back() - cache->dims.size()),
      " dimensions given `dims`.");
  for (const auto dim : cache->dims) {
    newDims.insert(newDims.begin() + dim, 1);
  }
  output.Reshape(newDims);
}
} // namespace
} // namespace caffe2

namespace c10 {
C10_REGISTER_KERNEL(caffe2::ops::ExpandDims)
    .withCache<caffe2::Cache>()
    .kernel<decltype(caffe2::expand_dims_op_cpu_impl<float>), &caffe2::expand_dims_op_cpu_impl<float>>()
    .dispatchKey(CPUTensorId());
} // namespace c10
