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

#include "caffe2/quantization/server/fbgemm_fp16_pack_op.h"

#include <functional>

#include "caffe2/core/common.h"

namespace caffe2 {

// Expilictly register TypeMeta
CAFFE_KNOWN_TYPE(unique_ptr<fbgemm::PackedGemmMatrixFP16>);

REGISTER_CPU_OPERATOR(
    FbGemmPack,
    FbGemmPackOp<CPUContext, DefaultEngine, true, fbgemm::float16>);

REGISTER_CPU_OPERATOR(
    FbGemmPackTranspose,
    FbGemmPackOp<CPUContext, DefaultEngine, false, fbgemm::float16>);

using namespace std::placeholders;
OPERATOR_SCHEMA(FbGemmPack)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT16);
      return out;
    })
    .SetDoc(R"DOC(Prepack weight for fbgemm)DOC")
    .Input(0, "X", "row major format weight matrix")
    .Output(0, "Y", "Block row major packed format weight matrix");

OPERATOR_SCHEMA(FbGemmPackTranspose)
    .NumInputs(1)
    .NumOutputs(1)
    .AllowInplace({{0, 0}})
    .TensorInferenceFunction([](const OperatorDef& /* def */,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out;
      TensorShape X = in[0];

      X.set_dims(1, in[0].dims(0));
      X.set_dims(0, in[0].dims(1));
      out.push_back(std::move(X));
      out[0].set_data_type(TensorProto_DataType_FLOAT16);
      return out;
    })
    .SetDoc(R"DOC(Prepack weight for fbgemm)DOC")
    .Input(0, "X", "col major format weight matrix")
    .Output(0, "Y", "Block col major packed format weight matrix");

} // namespace caffe2
