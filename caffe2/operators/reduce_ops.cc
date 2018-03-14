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

#include "caffe2/operators/reduce_ops.h"

namespace caffe2 {

// For a Tensor X of n dimensions (dims[0], ..., dims[ndim-1]), given index
// is converted to corresponding n-dimensional index, e.g. for X.shape = (2,
// 3, 4) the linear index 12 maps to 3-dimensional index (1, 0, 0).
vector<TIndex> ConvertFromInputIndex(TIndex index, vector<TIndex>& dims) {
  TIndex ndim = dims.size();
  vector<TIndex> nd_idx(ndim);

  for (TIndex i = ndim - 1; i >= 0 && index > 0; i--) {
    nd_idx[i] = index % dims[i];
    index /= dims[i];
  }
  return nd_idx;
}

// For given n-dimensional index (nd_idx[0], ..., nd_idx[dims.size()-1]) and
// reduction axes, map the n-dimensional index to the corresponding linear
// index in the reduced tensor.
TIndex ConvertToOutputIndex(
    const vector<int>& axes,
    const vector<TIndex>& nd_idx,
    vector<TIndex>& dims) {
  TIndex index = 0;
  TIndex multiplier = 1;
  for (TIndex i = dims.size() - 1, j = axes.size() - 1; i >= 0; i--) {
    if (j >= 0 && axes[j] == i) {
      j--;
    } else {
      index += nd_idx[i] * multiplier;
      multiplier *= dims[i];
    }
  }
  return index;
}

template <typename T>
inline T Add(T x, T y) {
  return (x + y);
}

template <typename T, class Context>
void ComputeOp(
    const T* X_data,
    const TIndex X_size,
    vector<TIndex>& dims,
    T* Y_data,
    vector<int>& axes,
    int keepdims,
    T (*binary_op)(T, T)) {
  for (TIndex x_idx = 0; x_idx < X_size; x_idx++) {
    vector<TIndex> nd_idx = ConvertFromInputIndex(x_idx, dims);
    TIndex y_idx = ConvertToOutputIndex(axes, nd_idx, dims);
    Y_data[y_idx] = binary_op(Y_data[y_idx], X_data[x_idx]);
  }
}

template <typename T, class Context>
bool ReduceSumOp<T, Context>::Compute(
    const T* X_data,
    const TIndex X_size,
    vector<TIndex>& dims,
    T* Y_data,
    const TIndex Y_size,
    vector<int>& axes,
    int keepdims) {
  math::Set<T, Context>(Y_size, 0.f, Y_data, &context_);
  ComputeOp<T, Context>(X_data, X_size, dims, Y_data, axes, keepdims, Add);
  return true;
}

template <typename T, class Context>
bool ReduceMeanOp<T, Context>::Compute(
    const T* X_data,
    const TIndex X_size,
    vector<TIndex>& dims,
    T* Y_data,
    const TIndex Y_size,
    vector<int>& axes,
    int keepdims) {
  math::Set<T, Context>(Y_size, 0.f, Y_data, &context_);
  ComputeOp<T, Context>(X_data, X_size, dims, Y_data, axes, keepdims, Add);
  math::Scale(
      Y_size, static_cast<float>(Y_size) / X_size, Y_data, Y_data, &context_);

  return true;
}

REGISTER_CPU_OPERATOR(ReduceSum, ReduceSumOp<float, CPUContext>);

OPERATOR_SCHEMA(ReduceSum)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
  Computes the sum of the input tensor's element along the provided axes.
  The resulted tensor has the same rank as the input if keepdims equal 1.
  If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
)DOC")
    .Arg("axes", "A list of integers, along which to reduce.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension(s) or not, default 1 keeps the reduced dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(ReduceSum);

REGISTER_CPU_OPERATOR(ReduceMean, ReduceMeanOp<float, CPUContext>);

OPERATOR_SCHEMA(ReduceMean)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
      Computes the mean of the input tensor's element along the provided axes.
      The resulted tensor has the same rank as the input if keepdims equal 1.
      If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
    )DOC")
    .Arg("axes", "A list of integers, along which to reduce.")
    .Arg(
        "keepdims",
        "Keep the reduced dimension(s) or not, default 1 keeps the reduced dimension(s).")
    .Input(0, "data", "An input tensor.")
    .Output(0, "reduced", "Reduced output tensor.");

// TODO: Write gradient for this when needed
GRADIENT_NOT_IMPLEMENTED_YET(ReduceMean);

} // namespace caffe2
