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

#include "caffe2/experiments/operators/funhash_op.h"

namespace caffe2 {
namespace {

REGISTER_CPU_OPERATOR(FunHash, FunHashOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(FunHashGradient, FunHashGradientOp<float, CPUContext>);

OPERATOR_SCHEMA(FunHash)
    .NumInputs(4, 5)
    .NumOutputs(1)
    .SetDoc(R"DOC(
This layer compresses a fully-connected layer for sparse inputs
via hashing.
It takes four required inputs and an optional fifth input.
The first three inputs `scalars`, `indices`, and `segment_ids` are
the sparse segmented representation of sparse data, which are the
same as the last three inputs of the `SparseSortedSegmentWeightedSum`
operator. If the argument `num_segments` is specified, it would be used
as the first dimension for the output; otherwise it would be derived
from the maximum segment ID.

The fourth input is a 1D weight vector. Each entry of the fully-connected
layer would be randomly mapped from one of the entries in this vector.

When the optional fifth input vector is present, each weight of the
fully-connected layer would be the linear combination of K entries
randomly mapped from the weight vector, provided the input
(length-K vector) serves as the coefficients.
)DOC")
    .Input(0, "scalars", "Values of the non-zero entries of the sparse data.")
    .Input(1, "indices", "Indices to the non-zero valued features.")
    .Input(2, "segment_ids",
        "Segment IDs corresponding to the non-zero entries.")
    .Input(3, "weight", "Weight vector")
    .Input(4, "alpha",
        "Optional coefficients for linear combination of hashed weights.")
    .Output(0, "output",
        "Output tensor with the first dimension equal to the number "
        "of segments.")
    .Arg("num_outputs", "Number of outputs")
    .Arg("num_segments", "Number of segments");

OPERATOR_SCHEMA(FunHashGradient)
    .NumInputs(5, 6)
    .NumOutputs(1, 2);

class GetFunHashGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    if (def_.input_size() == 4) {
      return SingleGradientDef(
          "FunHashGradient", "",
          vector<string>{GO(0), I(0), I(1), I(2), I(3)},
          vector<string>{GI(3)});
    }
    // def_.input_size() == 5
    return SingleGradientDef(
        "FunHashGradient", "",
        vector<string>{GO(0), I(0), I(1), I(2), I(3), I(4)},
        vector<string>{GI(3), GI(4)});
  }
};

REGISTER_GRADIENT(FunHash, GetFunHashGradient);

} // namespace
} // namespace caffe2
