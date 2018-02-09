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

#include <functional>

#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(FC, FullyConnectedOp<CPUContext>);
REGISTER_CPU_OPERATOR(FCGradient, FullyConnectedGradientOp<CPUContext>);

REGISTER_CPU_OPERATOR(
    FCTransposed,
    FullyConnectedOp<
        CPUContext,
        DefaultEngine,
        false /* don't transpose weight */>);
REGISTER_CPU_OPERATOR(
    FCTransposedGradient,
    FullyConnectedGradientOp<
        CPUContext,
        DefaultEngine,
        false /* don't transpose weight */>);

namespace {
std::vector<TensorShape> FCShapeInference(
    const OperatorDef& def,
    const vector<TensorShape>& in,
    bool pretransposed_weight) {
  vector<TensorShape> out(1);
  ArgumentHelper helper(def);

  auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
  const auto canonical_axis = canonical_axis_index_(axis, in[0].dims().size());
  const int M = size_to_dim_(canonical_axis, GetDimsVector(in[0]));
  auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
  const int canonical_axis_w =
      canonical_axis_index_(axis_w, in[1].dims().size());
  const int N = pretransposed_weight
      ? size_from_dim_(canonical_axis_w, GetDimsVector(in[1]))
      : size_to_dim_(canonical_axis_w, GetDimsVector(in[1]));

  vector<int> y_shape(in[0].dims().begin(), in[0].dims().end());
  CAFFE_ENFORCE_LE(canonical_axis + 1, y_shape.size());
  y_shape.resize(canonical_axis + 1);
  y_shape[canonical_axis] = N;
  out[0] = CreateTensorShape(y_shape, in[0].data_type());
  return out;
}

OpSchema::Cost CostInferenceForFC(
    const OperatorDef& def,
    const vector<TensorShape>& in) {
  struct OpSchema::Cost c;
  ArgumentHelper helper(def);

  auto axis = helper.GetSingleArgument<int32_t>("axis", 1);
  const auto canonical_axis = canonical_axis_index_(axis, in[0].dims().size());
  const int M = size_to_dim_(canonical_axis, GetDimsVector(in[0]));
  const int K = size_from_dim_(canonical_axis, GetDimsVector(in[0]));
  auto axis_w = helper.GetSingleArgument<int32_t>("axis_w", 1);
  const int canonical_axis_w =
      canonical_axis_index_(axis_w, in[1].dims().size());
  const int N = size_to_dim_(canonical_axis_w, GetDimsVector(in[1]));
  c.flops = 2 * K * M * N + M * N;
  c.bytes_moved = M * N * sizeof(float);
  c.params_bytes = (K * N + N) * sizeof(float);
  return c;
}
} // namespace

using namespace std::placeholders;
OPERATOR_SCHEMA(FCTransposed)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, true))
    .SetDoc(R"DOC(
Same as FC, but weight matrix is supposed to be already pretransposed.
FCTransposed stands for calling blass with no noTrans, noTrans
)DOC");

OPERATOR_SCHEMA(FC)
    .NumInputs(3)
    .NumOutputs(1)
    .TensorInferenceFunction(std::bind(FCShapeInference, _1, _2, false))
    .CostInferenceFunction(
        OpSchema::CostInferenceFunctionType(CostInferenceForFC))
    .SetDoc(R"DOC(
Computes the result of passing an input vector X into a fully
connected layer with 2D weight matrix W and 1D bias vector b. That is,
the layer computes Y = X * W^T + b, where X has size (M x K),
W has size (N x K), b has size (N), and Y has size (M x N),
where M is often the batch size.


NOTE: X does not need to explicitly be a 2D vector; rather, it will be
coerced into one. For an arbitrary n-dimensional tensor
X \in [a_0, a_1, ...,a_{k-1}, a_k, ..., a_{n-1}] where a_i \in N+ and k is
the axis provided, then X will be coerced into a 2-dimensional tensor with
dimensions [a_0 * ... * a_{k-1}, a_k * ... * a_{n-1}]. For the default
case where axis=1, this means the X tensor will be coerced into a 2D tensor
of dimensions [a_0, a_1 * ... * a_{n-1}], where a_0 is often the batch size.
In this situation, we must have a_0 = M and a_1 * ... * a_{n-1} = K.
Lastly, even though b is a 1D vector of size N, it is copied/resized to
be size (M x N) implicitly and added to each vector in the batch.
Each of these dimensions must be matched correctly, or else the operator
will throw errors.
)DOC")
    .Arg(
        "axis",
        "(int32_t) default to 1; describes the axis of the inputs; "
        "defaults to one because the 0th axis most likely describes "
        "the batch_size")
    .Arg(
        "axis_w",
        "(int32_t) default to 1; describes the axis of the weight matrix W; "
        "defaults to one because the 0th axis most likely describes "
        "the batch_size")
    .Arg("float16_compute", "Whether to use float-16 compute kernel")
    .Input(
        0,
        "X",
        "input tensor that's coerced into a 2D matrix of size (MxK) "
        "as described above")
    .Input(
        1,
        "W",
        "A tensor that is coerced into a 2D blob of size (KxN) "
        "containing fully connected weight matrix")
    .Input(2, "b", "1D blob containing bias vector")
    .Output(0, "Y", "2D output tensor");

OPERATOR_SCHEMA(FCGradient).NumInputs(3).NumOutputs(2, 3);
OPERATOR_SCHEMA(FCTransposedGradient).NumInputs(3).NumOutputs(2, 3);

namespace {

class GetFCGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;

  std::vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 3);
    CAFFE_ENFORCE(def_.type() == "FC" || def_.type() == "FCTransposed");
    return SingleGradientDef(
        def_.type() + "Gradient",
        "",
        vector<string>{I(0), I(1), GO(0)},
        vector<string>{GI(1), GI(2), GI(0)});
  }
};

REGISTER_GRADIENT(FC, GetFCGradient);
REGISTER_GRADIENT(FCTransposed, GetFCGradient);

} // namespace

} // namespace caffe2
