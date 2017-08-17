#include "caffe2/operators/layer_norm_op.h"

namespace caffe2 {

namespace {
template <typename T>
using EigenMatrixMapRowMajor = Eigen::Map<
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename T>
using ConstEigenMatrixMapRowMajor = Eigen::Map<
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
} // namespace

template <>
template <>
bool LayerNormOp<CPUContext>::DoRunWithType<float>() {
  const auto& input = Input(0);
  auto* output = Output(0);
  auto* mean = Output(1);
  auto* stdev = Output(2);

  CAFFE_ENFORCE_GE(input.dims().size(), 2, "LayerNorm requires input dim >= 2");

  const auto canonical_axis = input.canonical_axis_index(axis_);
  const int left = input.size_to_dim(canonical_axis);
  const int right = input.size_from_dim(canonical_axis);

  output->ResizeLike(input);
  std::vector<TIndex> stats_dims(
      input.dims().begin(), input.dims().begin() + canonical_axis);
  stats_dims.push_back(1);
  mean->Resize(stats_dims);
  stdev->Resize(stats_dims);

  auto input_map = ConstEigenMatrixMapRowMajor<float>(
      input.template data<float>(), left, right);
  auto mean_map = EigenMatrixMapRowMajor<float>(
      mean->template mutable_data<float>(), left, 1);
  auto stdev_map = EigenMatrixMapRowMajor<float>(
      stdev->template mutable_data<float>(), left, 1);
  auto output_map = EigenMatrixMapRowMajor<float>(
      output->template mutable_data<float>(), left, right);

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

REGISTER_CPU_OPERATOR(LayerNorm, LayerNormOp<CPUContext>);

OPERATOR_SCHEMA(LayerNorm)
    .NumInputs(1)
    .NumOutputs(3)
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(3);
      auto input_dims_long = GetDimsVector(in[0]);
      std::vector<int> input_dims(
          input_dims_long.begin(), input_dims_long.end());
      out[0] = CreateTensorShape(input_dims, TensorProto::FLOAT);

      ArgumentHelper helper(def);

      auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
      const auto canonical_axis =
          canonical_axis_index_(axis, in[0].dims().size());
      std::vector<int> stat_dims(
          input_dims.begin(), input_dims.begin() + canonical_axis);
      stat_dims.push_back(1);
      out[1] = CreateTensorShape(stat_dims, TensorProto::FLOAT);
      out[2] = CreateTensorShape(stat_dims, TensorProto::FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
Computes layer normalization as described in https://arxiv.org/pdf/1607.06450.pdf.
Given an input vector x \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}],
this op treats dimensions a_k thorugh a_{n-1} as feature vectors. For each
feature vector, the op contains the mean and standard deviation. Then,
it returns the normalized values (with respect to the feature vector).

Note that this op does not contain the scale an bias terms described in the
paper. Simply follow this op with an FC op to add those. Concretely, this op
implements:

h = \frac{1}{\sigma}(a - \mu)
where \mu = \frac{1}{H}\sum_{i=1}^{H} a_i
and \sigma = \sqrt{\frac{1}{H}\sum_{i=1}^{H}(a_i - \mu)^2}
where H is the number of hidden units (i.e. product of dimensions from 'axis'
to the end.)
)DOC")
    .Arg(
        "axis",
        "(int) default to 1; Describes axis of the inputs. Defaults to one "
        "because the 0th axis most likely describes the batch size")
    .Arg(
        "epsilon",
        "(float) default to 0.001. Small value to be added to the stdev when"
        " dividing out by that value. This prevents division by zero.")
    .Input(
        0,
        "input",
        "Input tensor which layer normalization will be applied to")
    .Output(0, "output", "Normalized values")
    .Output(1, "mean", "Mean values for each feature vector")
    .Output(2, "stddev", "Standard deviations for each feature vector");

} // namespace caffe2
