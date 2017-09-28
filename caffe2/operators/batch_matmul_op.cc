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

#include "caffe2/operators/batch_matmul_op.h"
#include "caffe2/core/operator_schema.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(BatchMatMul, BatchMatMulOp<CPUContext>);

OPERATOR_SCHEMA(BatchMatMul)
    .NumInputs(2)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Batch Matrix multiplication Yi = Ai * Bi, where A has size (C x M x K), B has
size (C x K x N) where C is the batch size and i ranges from 0 to C-1.
)DOC")
    .Input(0, "A", "3D matrix of size (C x M x K)")
    .Input(1, "B", "3D matrix of size (C x K x N)")
    .Output(0, "Y", "3D matrix of size (C x M x N)")
    .Arg("trans_a", "Pass 1 to transpose A before multiplication")
    .Arg("trans_b", "Pass 1 to transpose B before multiplication")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      int a_dim0;
      int b_dim1;
      if (helper.GetSingleArgument<int>("trans_a", 0)) {
        a_dim0 = in[0].dims(2);
      } else {
        a_dim0 = in[0].dims(1);
      }

      if (helper.GetSingleArgument<int>("trans_b", 0)) {
        b_dim1 = in[1].dims(1);
      } else {
        b_dim1 = in[1].dims(2);
      }
      return vector<TensorShape> {
          CreateTensorShape(vector<TIndex> {
              in[0].dims(0), a_dim0, b_dim1},
              in[0].data_type())
      };
    });

class GetBatchMatMulGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    CAFFE_ENFORCE_EQ(def_.input_size(), 2);

    bool trans_a = 0;
    bool trans_b = 0;

    if (ArgumentHelper::HasArgument(Def(), "trans_a")) {
      trans_a = GetArgument(Def(), "trans_a").i();
    }
    if (ArgumentHelper::HasArgument(Def(), "trans_b")) {
      trans_b = GetArgument(Def(), "trans_b").i();
    }

    auto no_trans_arg = vector<Argument>();
    auto trans_a_arg = vector<Argument>{
        MakeArgument<int>("trans_a", 1)};
    auto trans_b_arg = vector<Argument>{
        MakeArgument<int>("trans_b", 1)};
    auto trans_both_arg = vector<Argument>{
        MakeArgument<int>("trans_a", 1),
        MakeArgument<int>("trans_b", 1)};

    if (ArgumentHelper::HasArgument(Def(), "use_scratch")) {
      no_trans_arg.push_back(MakeArgument<int>("use_scratch", 1));
      trans_a_arg.push_back(MakeArgument<int>("use_scratch", 1));
      trans_b_arg.push_back(MakeArgument<int>("use_scratch", 1));
      trans_both_arg.push_back(MakeArgument<int>("use_scratch", 1));
    }

    if (trans_a) {
      if (trans_b) {
        // A'B':
        // dA = B'G', dB = G'A'
        return vector<OperatorDef>{
            CreateOperatorDef(
                "BatchMatMul",
                "",
                vector<string>{I(1), GO(0)},
                vector<string>{GI(0)},
                trans_both_arg),
            CreateOperatorDef(
                "BatchMatMul",
                "",
                vector<string>{GO(0), I(0)},
                vector<string>{GI(1)},
                trans_both_arg)};
      } else {
        // A'B:
        // dA = BG', dB = AG
        return vector<OperatorDef>{
            CreateOperatorDef(
                "BatchMatMul",
                "",
                vector<string>{I(1), GO(0)},
                vector<string>{GI(0)},
                trans_b_arg),
            CreateOperatorDef(
                "BatchMatMul",
                "",
                vector<string>{I(0), GO(0)},
                vector<string>{GI(1)},
                no_trans_arg)};
      }
    } else {
      if (trans_b) {
        // AB':
        // dA = GB, dB = G'A
        return vector<OperatorDef>{
            CreateOperatorDef(
                "BatchMatMul",
                "",
                vector<string>{GO(0), I(1)},
                vector<string>{GI(0)},
                no_trans_arg),
            CreateOperatorDef(
                "BatchMatMul",
                "",
                vector<string>{GO(0), I(0)},
                vector<string>{GI(1)},
                trans_a_arg)};
      } else {
        // AB:
        // dA = GB', dB = A'G
        return vector<OperatorDef>{
            CreateOperatorDef(
                "BatchMatMul",
                "",
                vector<string>{GO(0), I(1)},
                vector<string>{GI(0)},
                trans_b_arg),
            CreateOperatorDef(
                "BatchMatMul",
                "",
                vector<string>{I(0), GO(0)},
                vector<string>{GI(1)},
                trans_a_arg)};
      }
    }
  }

  bool CopyArguments() const override {
    return false;
  }
};

REGISTER_GRADIENT(BatchMatMul, GetBatchMatMulGradient);

} // namespace caffe2
