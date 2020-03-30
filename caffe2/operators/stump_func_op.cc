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

#include "caffe2/operators/stump_func_op.h"

namespace caffe2 {

template <>
bool StumpFuncOp<float, float, CPUContext>::RunOnDevice() {
  auto& in = Input(0);
  const float* in_data = in.template data<float>();

  auto* out = Output(0, in.sizes(), at::dtype<float>());
  float* out_data = out->template mutable_data<float>();
  for (int i = 0; i < in.numel(); i++) {
    out_data[i] = (in_data[i] <= threshold_) ? low_value_ : high_value_;
  }
  return true;
}

template <>
bool StumpFuncIndexOp<float, int64_t, CPUContext>::RunOnDevice() {
  auto& in = Input(0);
  const float* in_data = in.template data<float>();

  int lo_cnt = 0;
  for (int i = 0; i < in.numel(); i++) {
    lo_cnt += (in_data[i] <= threshold_);
  }
  auto* out_lo = Output(0, {lo_cnt}, at::dtype<int64_t>());
  auto* out_hi = Output(1, {in.numel() - lo_cnt}, at::dtype<int64_t>());
  int64_t* lo_data = out_lo->template mutable_data<int64_t>();
  int64_t* hi_data = out_hi->template mutable_data<int64_t>();
  int lidx = 0;
  int hidx = 0;
  for (int i = 0; i < in.numel(); i++) {
    if (in_data[i] <= threshold_) {
      lo_data[lidx++] = i;
    } else {
      hi_data[hidx++] = i;
    }
  }
  return true;
}

REGISTER_CPU_OPERATOR(StumpFunc, StumpFuncOp<float, float, CPUContext>);

OPERATOR_SCHEMA(StumpFunc)
    .NumInputs(1)
    .NumOutputs(1)
    .Input(0, "X", "tensor of float")
    .Output(0, "Y", "tensor of float")
    .TensorInferenceFunction([](const OperatorDef&,
                                const vector<TensorShape>& input_types) {
      vector<TensorShape> out(1);
      out.at(0) = input_types.at(0);
      out.at(0).set_data_type(TensorProto_DataType::TensorProto_DataType_FLOAT);
      return out;
    })
    .SetDoc(R"DOC(
Converts each input element into either high_ or low_value
based on the given threshold.
)DOC");

NO_GRADIENT(StumpFunc);

REGISTER_CPU_OPERATOR(
    StumpFuncIndex,
    StumpFuncIndexOp<float, int64_t, CPUContext>);

OPERATOR_SCHEMA(StumpFuncIndex)
    .NumInputs(1)
    .NumOutputs(2)
    .Input(0, "X", "tensor of float")
    .Output(
        0,
        "Index_Low",
        "tensor of int64 indices for elements below/equal threshold")
    .Output(
        1,
        "Index_High",
        "tensor of int64 indices for elements above threshold")
    .SetDoc(R"DOC(
Split the elements and return the indices based on the given threshold.
)DOC");

NO_GRADIENT(StumpFuncIndex);

} // namespace caffe2
