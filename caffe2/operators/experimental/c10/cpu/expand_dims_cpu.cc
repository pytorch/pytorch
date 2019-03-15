#include <ATen/core/dispatch/OperatorRegistration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

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

static auto registry = c10::RegisterOperators().op(
    FunctionSchema(
        "_c10_experimental::ExpandDims",
        "",
        (std::vector<c10::Argument>{c10::Argument("input"),
                                    c10::Argument("output"),
                                    c10::Argument("dims", ListType::ofInts())}),
        (std::vector<c10::Argument>{})),
    c10::kernel<
        decltype(expand_dims_op_cpu_impl<float>),
        &expand_dims_op_cpu_impl<float>,
        Cache>(),
    c10::dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::ExpandDims",
    "",
    C10ExpandDims_DontUseThisOpYet)

} // namespace caffe2
