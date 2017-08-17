#ifndef CAFFE2_OPERATORS_LAYER_NORM_OP_H
#define CAFFE2_OPERATORS_LAYER_NORM_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace {
template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
} // namespace

template <class Context>
class LayerNormOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LayerNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 0.001)) {}
  ~LayerNormOp() {}

  template <typename T>
  bool DoRunWithType() {
    const auto& input = Input(0);
    auto* output = Output(0);
    auto* mean = Output(1);
    auto* stdev = Output(2);

    CAFFE_ENFORCE_GE(
        input.dims().size(), 2, "LayerNorm requires input dim >= 2");

    const auto canonical_axis = input.canonical_axis_index(axis_);
    const int left = input.size_to_dim(canonical_axis);
    const int right = input.size_from_dim(canonical_axis);

    output->ResizeLike(input);
    std::vector<TIndex> stats_dims(
        input.dims().begin(), input.dims().begin() + canonical_axis);
    stats_dims.push_back(1);
    mean->Resize(stats_dims);
    stdev->Resize(stats_dims);

    auto input_map =
        ConstEigenMatrixMapRowMajor<T>(input.template data<T>(), left, right);
    auto mean_map =
        EigenMatrixMapRowMajor<T>(mean->template mutable_data<T>(), left, 1);
    auto stdev_map =
        EigenMatrixMapRowMajor<T>(stdev->template mutable_data<T>(), left, 1);
    auto output_map = EigenMatrixMapRowMajor<T>(
        output->template mutable_data<T>(), left, right);

    auto sqr = [](float f) { return f * f; };
    auto add_ep = [this](float f) { return f + epsilon_; };
    auto fsqrt = [](float f) { return std::sqrt(f); };
    // Calculate row-wise statistics
    mean_map = input_map.rowwise().mean();
    stdev_map =
        (input_map.unaryExpr(sqr).rowwise().mean() - mean_map.unaryExpr(sqr))
            .unaryExpr(fsqrt);
    output_map =
        (input_map - mean_map.replicate(1, right))
            .cwiseQuotient(stdev_map.unaryExpr(add_ep).replicate(1, right));

    return true;
  }

  bool RunOnDevice() override {
    return DoRunWithType<float>();
  }

 protected:
  int axis_;
  float epsilon_;
};

} // namespace caffe2

#endif /* CAFFE2_OPERATORS_LAYER_NORM_OP_H */
