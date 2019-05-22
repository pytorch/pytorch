#include <ATen/core/op_registration/op_registration.h>
#include "caffe2/core/operator_c10wrapper.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

using caffe2::Tensor;

namespace caffe2 {
namespace {

template <class DataType>
class expand_dims_cpu final : public c10::OperatorKernel {
 public:
  void operator()(
      const at::Tensor& input_,
      const at::Tensor& output_,
      std::vector<int64_t> dims) {
    Tensor input(input_);
    Tensor output(output_);

    if (!initialized_) {
      dims_ = std::move(dims);
      auto originalSize = dims_.size();
      CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");
      std::sort(dims_.begin(), dims_.end());
      dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
      if (dims_.size() < originalSize) {
        LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
      }
      CAFFE_ENFORCE(dims_.front() >= 0, "Dimension ids must be non-negative.");
      initialized_ = true;
    }

    output.CopyFrom(input);
    if (dims_.empty()) {
      return;
    }

    auto newDims = input.sizes().vec();
    CAFFE_ENFORCE_GE(
        input.sizes().size() + dims_.size(),
        dims_.back() + 1,
        "Input needs at least ",
        (1 + dims_.back() - dims_.size()),
        " dimensions given `dims`.");
    for (const auto dim : dims_) {
      newDims.insert(newDims.begin() + dim, 1);
    }
    output.Reshape(newDims);
  }

 private:
  std::vector<int64_t> dims_;
  bool initialized_ = false;
};

static auto registry = c10::RegisterOperators().op(
    "_c10_experimental::ExpandDims",
    c10::RegisterOperators::options()
      .kernel<expand_dims_cpu<float>>()
      .dispatchKey(CPUTensorId()));

} // namespace

REGISTER_C10_OPERATOR_FOR_CAFFE2_DISPATCH_CPU(
    "_c10_experimental::ExpandDims",
    C10ExpandDims_DontUseThisOpYet)

} // namespace caffe2
