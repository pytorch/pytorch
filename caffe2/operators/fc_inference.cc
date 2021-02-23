#include "caffe2/operators/fc_inference.h"

namespace caffe2 {
std::vector<TensorShape> FCShapeInference(
    const OperatorDef& def,
    const vector<TensorShape>& in,
    bool pretransposed_weight) {
  vector<TensorShape> out(1);

  if (in[0].unknown_shape() || in[1].unknown_shape()) {
    out[0].set_unknown_shape(true);
    return out;
  }

  ArgumentHelper helper(def);

  auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
  const auto canonical_axis = canonical_axis_index_(axis, in[0].dims().size());
  auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
  const int canonical_axis_w =
      canonical_axis_index_(axis_w, in[1].dims().size());
  const int64_t N = pretransposed_weight
      ? size_from_dim_(canonical_axis_w, GetDimsVector(in[1]))
      : size_to_dim_(canonical_axis_w, GetDimsVector(in[1]));

  vector<int64_t> y_shape(in[0].dims().begin(), in[0].dims().end());
  CAFFE_ENFORCE_LE(canonical_axis + 1, y_shape.size());
  y_shape.resize(canonical_axis + 1);
  y_shape[canonical_axis] = N;

  out[0] = CreateTensorShape(y_shape, in[0].data_type());
  return out;
}

OpSchema::Cost CostInferenceForFC(
    const OperatorDef& def,
    const vector<TensorShape>& in,
    bool pretransposed_weight) {
  CAFFE_ENFORCE_GE(in.size(), 3, "FC requires at least three inputs");
  struct OpSchema::Cost c;
  ArgumentHelper helper(def);

  auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
  const auto canonical_axis = canonical_axis_index_(axis, in[0].dims().size());
  const uint64_t M = size_to_dim_(canonical_axis, GetDimsVector(in[0]));
  const uint64_t K = size_from_dim_(canonical_axis, GetDimsVector(in[0]));
  auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
  const int canonical_axis_w =
      canonical_axis_index_(axis_w, in[1].dims().size());
  const uint64_t N = pretransposed_weight
      ? size_from_dim_(canonical_axis_w, GetDimsVector(in[1]))
      : size_to_dim_(canonical_axis_w, GetDimsVector(in[1]));

  const auto& X = in[0];
  c.flops = M * N * (2 * K + 1);
  c.bytes_read = (K * (M + N) + N) * sizeof(X.data_type());
  c.bytes_written = M * N * sizeof(X.data_type());
  c.params_bytes = (K * N + N) * sizeof(X.data_type());
  return c;
}

std::vector<TensorShape> FCGradientShapeInference(
    const OperatorDef& def,
    const vector<TensorShape>& in,
    bool pretransposed_weight) {
  vector<TensorShape> out(2);
  ArgumentHelper helper(def);

  auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
  const int canonical_axis_w =
      canonical_axis_index_(axis_w, in[1].dims().size());
  const int N = pretransposed_weight
      ? size_from_dim_(canonical_axis_w, GetDimsVector(in[1]))
      : size_to_dim_(canonical_axis_w, GetDimsVector(in[1]));

  vector<int> dW_shape(in[1].dims().begin(), in[1].dims().end());
  out[0] = CreateTensorShape(dW_shape, in[1].data_type());
  out[1] = CreateTensorShape(vector<int>{N}, in[1].data_type()); // db
  if (def.output_size() == 3) {
    vector<int> dX_shape(in[0].dims().begin(), in[0].dims().end());
    out.push_back(CreateTensorShape(dX_shape, in[0].data_type()));
  }
  return out;
}

OpSchema::Cost CostInferenceForFCGradient(
    const OperatorDef& def,
    const vector<TensorShape>& in,
    bool pretransposed_weight) {
  struct OpSchema::Cost c;
  ArgumentHelper helper(def);
  std::vector<TensorShape> out =
      FCGradientShapeInference(def, in, pretransposed_weight);

  CAFFE_ENFORCE_LT(0, out.size());
  const TensorShape dW = out[0];
  const TensorShape db = out[1];

  auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
  const auto canonical_axis = canonical_axis_index_(axis, in[0].dims().size());
  const uint64_t M = size_to_dim_(canonical_axis, GetDimsVector(in[0]));
  const uint64_t K = size_from_dim_(canonical_axis, GetDimsVector(in[0]));
  auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
  const int canonical_axis_w =
      canonical_axis_index_(axis_w, in[1].dims().size());
  const uint64_t N = pretransposed_weight
      ? size_from_dim_(canonical_axis_w, GetDimsVector(in[1]))
      : size_to_dim_(canonical_axis_w, GetDimsVector(in[1]));

  uint64_t size_dW = nElemFromDim(dW);
  uint64_t size_db = nElemFromDim(db);

  c.flops = M * N * (2 * K + 1);
  c.bytes_written = (size_dW + size_db) * sizeof(float);
  c.params_bytes = (K * N + N) * sizeof(float);

  if (out.size() == 3) {
    const TensorShape dX = out[2];
    uint64_t size_dX = nElemFromDim(dX);

    c.flops += 2 * M * N * K;
    c.bytes_written += size_dX * sizeof(float);
  }
  return c;
}

} // namespace caffe2
