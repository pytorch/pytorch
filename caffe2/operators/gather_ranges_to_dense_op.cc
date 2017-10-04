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

#include "caffe2/operators/gather_ranges_to_dense_op.h"

#include <cmath>

namespace caffe2 {
namespace {

OPERATOR_SCHEMA(GatherRangesToDense)
    .NumInputs(2)
    .NumOutputs(1, INT_MAX)
    .SetDoc(R"DOC(
Given DATA tensor of rank 1, and RANGES tensor of rank 3, gather values
corresponding to each range into a separate output tensor.

RANGES dimentions description:
1: represents list of examples within a batch
2: represents list features
3: two values which are start and length or a range (to be applied on DATA)

Each feature has fixed lengths which are passed as lengths argument and a
separate tensor will be produced for each feature.
i.e. DATA.dim(1) = len(lengths) = NumOuptuts.

Missing features (represented by empty ranges) filled with default_value.

Example:
  DATA  = [1, 2, 3, 4, 5, 6, 7, 8]
  RANGES = [
    [
      [2, 4],
      [0, 2],
    ],
    [
      [0, 0],
      [6, 2],
    ]
  ]
  lengths = [4, 2]
  OUTPUT[0] = [[3, 4, 5, 6], [0, 0, 0, 0]]
  OUTPUT[1] = [[1, 2], [7, 8]]
)DOC")
    .Input(0, "DATA", "Tensor of rank 1.")
    .Input(
        1,
        "RANGES",
        "Tensor of int32/int64 ranges, of dims (N, M, 2). "
        "Where N is number of examples and M is a size of each example. "
        "Last dimention represents a range in the format (start, lengths)")
    .Output(0, "OUTPUT", "1-D tensor of size sum of range lengths")
    .Arg("lengths", "Expected lengths for ranges")
    .TensorInferenceFunction([](const OperatorDef& def,
                                const vector<TensorShape>& in) {
      ArgumentHelper helper(def);
      auto lengths = helper.GetRepeatedArgument<int>("lengths");
      CAFFE_ENFORCE_EQ(in[0].dims_size(), 1, "DATA should be 1-D tensor.");
      CAFFE_ENFORCE_EQ(in[1].dims_size(), 3, "RANGES should be 3-D tensor.");
      CAFFE_ENFORCE_GT(lengths.size(), 0, "lengths should be non-empty.");
      std::vector<TensorShape> out(lengths.size());
      for (int i = 0; i < lengths.size(); ++i) {
        out[i].set_data_type(in[0].data_type());
        out[i].add_dims(in[1].dims(0));
        out[i].add_dims(lengths[i]);
      }
      return out;
    });

REGISTER_CPU_OPERATOR(GatherRangesToDense, GatherRangesToDenseOp<CPUContext>);
SHOULD_NOT_DO_GRADIENT(GatherRangesToDense);

} // namespace
} // namespace caffe2
