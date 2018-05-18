/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "caffe2/operators/relu_n_op.h"

#include "caffe2/utils/math.h"

namespace caffe2 {

template <>
bool ReluNOp<float, CPUContext>::RunOnDevice() {
  auto& X = Input(0);
  auto* Y = Output(0);
  Y->ResizeLike(X);

  EigenVectorMap<float>(Y->mutable_data<float>(), X.size()) =
      ConstEigenVectorMap<float>(X.data<float>(), X.size())
          .cwiseMax(0.f)
          .cwiseMin(n);
  return true;
}

// define a custom template unary functor
template <typename Scalar>
struct CwiseClampSignOp {
  CwiseClampSignOp(const Scalar& sup) : m_sup(sup) {}
  const Scalar operator()(const Scalar& x) const {
    return x < 0 ? 0 : (x >= m_sup ? 0 : 1);
  }
  Scalar m_sup;
};

template <>
bool ReluNGradientOp<float, CPUContext>::RunOnDevice() {
  auto& Y = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);
  CAFFE_ENFORCE_EQ(dY.size(), Y.size());
  dX->ResizeLike(Y);

  const float* Ydata = Y.data<float>();
  const float* dYdata = dY.data<float>();
  float* dXdata = dX->mutable_data<float>();
  // TODO: proper vectorization with Eigen
  EigenVectorArrayMap<float> dXvec(dXdata, dX->size());
  ConstEigenVectorArrayMap<float> Yvec(Ydata, Y.size());
  ConstEigenVectorArrayMap<float> dYvec(dYdata, dY.size());
  dXvec = dYvec * Yvec.unaryExpr(CwiseClampSignOp<float>(n));
  return true;
}

namespace {
OpSchema::Cost CostInferenceForReluN(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost cost = PointwiseCostInference<2>(def, in);
  cost.params_bytes = 0;
  return cost;
}
} // namespace

REGISTER_CPU_OPERATOR(ReluN, ReluNOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(ReluNGradient, ReluNGradientOp<float, CPUContext>);

// Input: X, output: Y
OPERATOR_SCHEMA(ReluN)
    .NumInputs(1)
    .NumOutputs(1)
    .Arg("n", "the cap of output")
    .AllowInplace({{0, 0}})
    .CostInferenceFunction(CostInferenceForReluN)
    .IdenticalTypeAndShape()
    .SetDoc(R"DOC(
Relu takes one input data (Tensor<T>) and produces one output data
(Tensor<T>) where the rectified linear function, y = min(max(0, x), n),
is applied to the tensor elementwise.
)DOC")
    .Input(0, "X", "1D input tensor")
    .Output(0, "Y", "1D input tensor");

// Input: Y, dY, output: dX
OPERATOR_SCHEMA(ReluNGradient)
    .NumInputs(2)
    .NumOutputs(1)
    .Arg("n", "the cap of forward op output")
    .AllowInplace({{1, 0}})
    .SetDoc(R"DOC(
ReluGradient takes both Y and dY and uses this to update dX according to the
chain rule and derivatives of the rectified linear function.
)DOC");

class GetReluNGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{O(0), GO(0)},
        vector<string>{GI(0)});
  }
};
REGISTER_GRADIENT(ReluN, GetReluNGradient);

} // namespace caffe2
