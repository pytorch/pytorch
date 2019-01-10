#include <caffe2/ideep/ideep_utils.h>
#include "caffe2/operators/expand_squeeze_dims_op.h"

namespace caffe2 {

class IDEEPSqueezeOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPSqueezeOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        dims_(OperatorBase::GetRepeatedArgument<int>("dims")) {
    auto originalSize = dims_.size();
    CAFFE_ENFORCE(originalSize > 0, "Parameter `dims` must be provided.");

    std::sort(dims_.begin(), dims_.end());
    dims_.erase(std::unique(dims_.begin(), dims_.end()), dims_.end());
    if (dims_.size() < originalSize) {
      LOG(WARNING) << "Parameter `dims` has repeated dimensions.";
    }
    CAFFE_ENFORCE(dims_.front() >= 0, "Dimension ids must be non-negative.");
  }

  virtual ~IDEEPSqueezeOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    CAFFE_ENFORCE_GT(
        X.ndims(),
        dims_.back(),
        "Input needs at least ",
        (dims_.back() + 1),
        " dimensions.");
    const auto& ideep_dims = X.get_dims();
    vector<int64_t> dims(ideep_dims.begin(), ideep_dims.end());
    const auto& new_dims = SqueezeOp<IDEEPContext>::ComputeDims(dims, dims_);
    itensor::dims new_dims_ideep(new_dims.begin(), new_dims.end());
    if (&X != Y) {
      // Copy if not inplace
      ideep::direct_copy::compute(X, *Y);
    }
    Y->reshape(new_dims_ideep);

    return true;
  }

 private:
  vector<int> dims_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(Squeeze, IDEEPSqueezeOp);

} // namespace caffe2
