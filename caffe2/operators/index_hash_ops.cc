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

#include "caffe2/operators/index_hash_ops.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(IndexHash, IndexHashOp<CPUContext>);

OPERATOR_SCHEMA(IndexHash)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This operator translates a list of indices into a list of hashed indices.
A seed can be fed as an argument to change the behavior of the hash function.
If a modulo is specified, all the hashed indices will be modulo the
specified number. All input and output indices are enforced to be positive.
)DOC")
    .Input(0, "Indices", "Input feature indices.")
    .Output(0, "HashedIndices", "Hashed feature indices.")
    .Arg("seed", "seed for the hash function")
    .Arg("modulo", "must be > 0, hashed ids will be modulo this number")
    .TensorInferenceFunction([](const OperatorDef& /* unused */,
                                const vector<TensorShape>& in) {
      std::vector<TensorShape> out(1);
      std::vector<TIndex> output_dims = GetDimsVector(in[0]);
      out[0] = CreateTensorShape(output_dims, in[0].data_type());
      return out;
    });

SHOULD_NOT_DO_GRADIENT(IndexHash);

} // namespace
} // namespace caffe2
