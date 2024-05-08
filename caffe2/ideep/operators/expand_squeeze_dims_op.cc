#include "caffe2/operators/expand_squeeze_dims_op.h"
#include <caffe2/ideep/ideep_utils.h>
#include <caffe2/ideep/operators/operator_fallback_ideep.h>

using namespace caffe2;

namespace {

class IDEEPExpandDimsOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();
  using FALLBACK_OP = IDEEPFallbackOp<ExpandDimsOp<CPUContext>, SkipIndices<0>>;

  IDEEPExpandDimsOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        fallback_(operator_def, ws) {
    dims_ = OperatorBase::GetRepeatedArgument<int>("dims");
    auto originalSize = dims_.size();
    CAFFE_ENFORCE_GT(originalSize, 0, "Parameter `dims` must be provided.");
    std::sort(dims_.begin(), dims_.end());
    dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
    if (dims_.size() < originalSize) {
      LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
    }
    CAFFE_ENFORCE_GE(dims_.front(), 0, "Dimension ids must be non-negative.");
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPExpandDimsOp() override {}

  bool RunOnDevice() override {
    if (!OperatorBase::InputBlob(INPUT).template IsType<itensor>()) {
      return fallback_.Run(0);
    }

    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);
    if (&X != Y) {
      // Copy if not inplace
      ideep::direct_copy::compute(X, *Y);
    }
    if (dims_.empty()) {
      return true;
    }

    auto newDims = X.get_dims();
    CAFFE_ENFORCE_GE(
        newDims.size() + dims_.size(),
        dims_.back() + 1,
        "Input needs at least ",
        (1 + dims_.back() - dims_.size()),
        " dimensions given `dims`.");

    for (const auto dim : dims_) {
      newDims.insert(newDims.begin() + dim, 1);
    }

    Y->reshape(newDims);
    return true;
  }

 private:
  std::vector<int> dims_;
  FALLBACK_OP fallback_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};


class IDEEPSqueezeOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();
  using FALLBACK_OP = IDEEPFallbackOp<SqueezeOp<CPUContext>, SkipIndices<0>>;

  IDEEPSqueezeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        fallback_(operator_def, ws) {
    dims_ = OperatorBase::GetRepeatedArgument<int>("dims");
    auto originalSize = dims_.size();
    CAFFE_ENFORCE_GT(originalSize, 0, "Parameter `dims` must be provided.");

    std::sort(dims_.begin(), dims_.end());
    dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
    if (dims_.size() < originalSize) {
      LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
    }
    CAFFE_ENFORCE_GE(dims_.front(), 0, "Dimension ids must be non-negative.");
  }
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ~IDEEPSqueezeOp() override {}

  bool RunOnDevice() override {
    if (!OperatorBase::InputBlob(INPUT).template IsType<itensor>()) {
      return fallback_.Run(0);
    }

    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    CAFFE_ENFORCE_GT(
        X.ndims(),
        dims_.back(),
        "Input needs at least ",
        (dims_.back() + 1),
        " dimensions.");
    const auto& ideep_dims = X.get_dims();
    std::vector<int64_t> dims(ideep_dims.begin(), ideep_dims.end());
    const auto new_dims = SqueezeOp<IDEEPContext>::ComputeDims(dims, dims_);
    itensor::dims new_dims_ideep(new_dims.begin(), new_dims.end());
    if (&X != Y) {
      // Copy if not inplace
      ideep::direct_copy::compute(X, *Y);
    }

    Y->reshape(new_dims_ideep);
    return true;
  }

 private:
  std::vector<int> dims_;
  FALLBACK_OP fallback_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};


REGISTER_IDEEP_OPERATOR(ExpandDims, IDEEPExpandDimsOp);
REGISTER_IDEEP_OPERATOR(Squeeze, IDEEPSqueezeOp);

} // namespace
