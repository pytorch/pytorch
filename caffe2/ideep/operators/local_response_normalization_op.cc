#include <caffe2/ideep/ideep_utils.h>

using namespace caffe2;

namespace {

class IDEEPLRNOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPLRNOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        size_(OperatorBase::GetSingleArgument<int>("size", 0)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0)),
        bias_(OperatorBase::GetSingleArgument<float>("bias", 1)) {
    TORCH_DCHECK_GT(size_, 0);
    TORCH_DCHECK_EQ(size_ % 2, 1);
    TORCH_DCHECK_GT(alpha_, 0);
    TORCH_DCHECK_GT(beta_, 0);
  }
  ~IDEEPLRNOp() override = default;

  bool RunOnDevice() override {
    auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    ideep::lrn_forward::compute(X, *Y, size_, alpha_, beta_, bias_);

    return true;
  }

 private:
  const int size_;
  const float alpha_;
  const float beta_;
  const float bias_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPLRNGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPLRNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        size_(OperatorBase::GetSingleArgument<int>("size", 0)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0)),
        bias_(OperatorBase::GetSingleArgument<float>("bias", 1)) {
    TORCH_DCHECK_GT(size_, 0);
    TORCH_DCHECK_EQ(size_ % 2, 1);
    TORCH_DCHECK_GT(alpha_, 0);
    TORCH_DCHECK_GT(beta_, 0);
  }
  ~IDEEPLRNGradientOp() override = default;

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& Y = Input(FILTER);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dX = Output(INPUT_GRAD);

    ideep::lrn_backward::compute(X, dY, Y, *dX, size_, alpha_, beta_, bias_);

    return true;
  }

 private:
  const int size_;
  const float alpha_;
  const float beta_;
  const float bias_;

  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(INPUT_GRAD);
};


REGISTER_IDEEP_OPERATOR(LRN, IDEEPLRNOp);
REGISTER_IDEEP_OPERATOR(LRNGradient, IDEEPLRNGradientOp);

} // namespace
