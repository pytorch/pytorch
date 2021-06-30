// Copyright 2004-present Facebook. All Rights Reserved.
#include "caffe2/quantization/server/compute_equalization_scale.h"
#include <functional>

namespace caffe2 {
using namespace std;

bool ComputeEqualizationScaleOp::RunOnDevice() {
  // Generate equalization scale based on the input data (last N samples of
  // the activations) and the weight
  const auto& X = Input(0);
  const auto& W = Input(1);
  CAFFE_ENFORCE_EQ(X.dim(), 2);
  CAFFE_ENFORCE_EQ(W.dim(), 2);

  const int64_t M = X.size_to_dim(1);
  const int64_t N = W.size_to_dim(1);
  const int64_t K = W.size_from_dim(1);
  auto* S = Output(0, K, at::dtype<float>());
  const float* X_data = X.template data<float>();
  const float* W_data = W.template data<float>();
  float* S_data = S->template mutable_data<float>();

  float WcolMax, XcolMax;
  for (int64_t j = 0; j < K; j++) {
    WcolMax = std::abs(W_data[j]);
    XcolMax = std::abs(X_data[j]);
    int64_t idx;
    for (int64_t i = 0; i < N; i++) {
      idx = i * K + j;
      WcolMax = std::max(WcolMax, std::abs(W_data[idx]));
    }
    for (int64_t i = 0; i < M; i++) {
      idx = i * K + j;
      XcolMax = std::max(XcolMax, std::abs(X_data[idx]));
    }
    if (WcolMax == 0 || XcolMax == 0) {
      S_data[j] = 1;
    } else {
      S_data[j] = std::sqrt(WcolMax / XcolMax);
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(ComputeEqualizationScale, ComputeEqualizationScaleOp);
OPERATOR_SCHEMA(ComputeEqualizationScale)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a weight matrix W and input matrix X, the output S is the equalization parameter
vector computed from W and X

S is computed by:
S[j] = max(abs(W[][j])) == 0 || max(abs(X[][j])) == 0 ? 1 :
  sqrt(max(abs(W[][j])) / max(abs(X[][j]))),

)DOC")
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      const int64_t K = size_from_dim_(1, GetDimsVector(in[1]));
      vector<int64_t> s_shape(2);
      s_shape[0] = 1;
      s_shape[1] = K;
      out[0] = CreateTensorShape(s_shape, TensorProto_DataType_FLOAT);
      return out;
    })
    .Input(
        0,
        "X",
        "The input data, or last N samples of the output activations.")
    .Input(1, "W", "The weight that we want to equalize with the input.")
    .Output(
        0,
        "S",
        "Scale computed that will be multiplied to the columns of input.")
    .SetDoc(
        R"DOC(Operator to compute equalization scale given the input data and weight)DOC");

} // namespace caffe2
