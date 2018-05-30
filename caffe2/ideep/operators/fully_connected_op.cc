#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPFullyConnectedOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPFullyConnectedOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(OperatorBase::GetSingleArgument<int32_t>("axis_w", 1)),
        float16_compute_(
            OperatorBase::GetSingleArgument<bool>("float16_compute", false)) {}
  virtual ~IDEEPFullyConnectedOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    auto* Y = Output(OUTPUT);

    auto newDims = [&](itensor::dims adims, size_t axis) {
      if (adims.size() == 2 || axis == 1) return adims;
      auto dim0 = std::accumulate(
        adims.begin(), adims.begin() + axis, 1, std::multiplies<itensor::dim_t>());
      auto dim1 = std::accumulate(
        adims.begin() + axis, adims.end(), 1, std::multiplies<itensor::dim_t>());
      return itensor::dims({dim0, dim1});
    };

    itensor X_in = X;
    auto X_dims = newDims(X_in.get_dims(), axis_);
    if (X_in.get_dims() != X_dims) {
      X_in.reshape(X_dims);
    }

    itensor filter_in = filter;
    auto filter_dims = newDims(filter_in.get_dims(), axis_w_);
    if (filter_in.get_dims() != filter_dims) {
      filter_in.reshape(filter_dims);
    }

    if (InputSize() > BIAS) {
      ideep::inner_product_forward::compute(X_in, filter_in, Input(BIAS), *Y);
    } else {
      ideep::inner_product_forward::compute(X_in, filter_in, *Y);
    }

    return true;
  }

 private:
  size_t axis_{1};
  size_t axis_w_{1};
  bool float16_compute_;

  INPUT_TAGS(INPUT, FILTER, BIAS);
  OUTPUT_TAGS(OUTPUT);
};

class IDEEPFullyConnectedGradientOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPFullyConnectedGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int32_t>("axis", 1)),
        axis_w_(OperatorBase::GetSingleArgument<int32_t>("axis_w", 1)),
        float16_compute_(
            OperatorBase::GetSingleArgument<bool>("float16_compute", false)) {}
  virtual ~IDEEPFullyConnectedGradientOp() {}

  bool RunOnDevice() override {
    const auto& X = Input(INPUT);
    const auto& filter = Input(FILTER);
    const auto& dY = Input(OUTPUT_GRAD);
    auto* dfilter = Output(FILTER_GRAD);
    auto* dbias = Output(BIAS_GRAD);

    auto newDims = [&](itensor::dims adims, size_t axis) {
      if (adims.size() == 2 || axis == 1) return adims;
      auto dim0 = std::accumulate(
        adims.begin(), adims.begin() + axis, 1, std::multiplies<itensor::dim_t>());
      auto dim1 = std::accumulate(
        adims.begin() + axis, adims.end(), 1, std::multiplies<itensor::dim_t>());
      return itensor::dims({dim0, dim1});
    };

    itensor X_in = X;
    auto X_dims = newDims(X_in.get_dims(), axis_);
    if (X_in.get_dims() != X_dims) {
      X_in.reshape(X_dims);
    }

    itensor filter_in = filter;
    auto filter_dims = newDims(filter_in.get_dims(), axis_w_);
    if (filter_in.get_dims() != filter_dims) {
      filter_in.reshape(filter_dims);
    }

    ideep::inner_product_backward_weights::compute(X_in, dY, *dfilter, *dbias);

    if (OutputSize() > INPUT_GRAD) {
      ideep::inner_product_backward_data::compute(
          dY, filter_in, X_in.get_dims(), *Output(INPUT_GRAD));
    }

    return true;
  }

 private:
  size_t axis_{1};
  size_t axis_w_{1};
  bool float16_compute_;

  INPUT_TAGS(INPUT, FILTER, OUTPUT_GRAD);
  OUTPUT_TAGS(FILTER_GRAD, BIAS_GRAD, INPUT_GRAD);
};

REGISTER_IDEEP_OPERATOR(FC, IDEEPFullyConnectedOp);
REGISTER_IDEEP_OPERATOR(FCGradient, IDEEPFullyConnectedGradientOp);

} // namespace caffe2
