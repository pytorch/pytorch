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

#include "sparse_to_dense_op.h"

#include "caffe2/core/context.h"

namespace caffe2 {

REGISTER_CPU_OPERATOR(SparseToDense, SparseToDenseOp<CPUContext>);

OPERATOR_SCHEMA(SparseToDense)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Convert sparse representations to dense with given indices.

Transforms a sparse representation of map<id, value> represented as `indices`
vector and `values` tensor into a compacted tensor where the first dimension
is determined by the first dimension of the 3rd input if it is given or the
max index. Missing values are filled with zeros.

The op supports duplicated indices and performs summation over corresponding
values. This behavior is useful for converting GradientSlices into dense
representation.

After running this op:

```
output[indices[i], :] += values[i]  # sum over all indices[i] equal to the index
output[j, ...] = 0 if j not in indices
```
)DOC")
    .Input(0, "indices", "1-D int32/int64 tensor of concatenated ids of data")
    .Input(
        1,
        "values",
        "Data tensor, first dimension has to match `indices`, "
        "basic numeric types are supported")
    .Input(
        2,
        "data_to_infer_dim",
        "Optional: if provided, the first dimension of output is the first "
        "dimension of this tensor.")
    .Output(
        0,
        "output",
        "Output tensor of the same type as `values` of shape `[len(lengths), "
        "len(mask)] + shape(default_value)` (if `lengths` is not provided the "
        "first dimension is omitted)");


namespace {
class GetSparseToDenseGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "Gather", "", vector<string>{GO(0), I(0)}, vector<string>{GI(1)});
  }
};

REGISTER_GRADIENT(SparseToDense, GetSparseToDenseGradient);
}
} // namespace caffe2
