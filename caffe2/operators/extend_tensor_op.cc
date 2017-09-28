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

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

namespace {

template <class Context>
class ExtendTensorOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ExtendTensorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        growthPct_(OperatorBase::GetSingleArgument<int>("growthPct", 40)) {}

  bool RunOnDevice() override {
    auto& old_tensor = Input(0);
    auto& indices = Input(1);
    auto* new_tensor = Output(0);
    CAFFE_ENFORCE(indices.ndim() >= 1);
    CAFFE_ENFORCE(
        &old_tensor == new_tensor, "First argument must be in-place.");
    CAFFE_ENFORCE(new_tensor->ndim() == indices.ndim());
    CAFFE_ENFORCE(indices.ndim() == new_tensor->ndim());

    auto oldSize = new_tensor->size();
    auto maxElem = 1 +
        *(std::max_element(
            indices.template data<int>(),
            indices.template data<int>() + indices.size()));

    auto extendSize = (TIndex)maxElem - oldSize;
    if (extendSize > 0) {
      new_tensor->Extend(extendSize, growthPct_, &context_);
      if (!new_tensor->meta().ctor()) {
        auto oldSizeBytes = oldSize * new_tensor->meta().itemsize();
        auto* dst = (char*)new_tensor->raw_mutable_data() + oldSizeBytes;
        math::Set<char, Context>(
            new_tensor->nbytes() - oldSizeBytes, 0, dst, &context_);
      }
    }
    return true;
  }

  int growthPct_;
};

REGISTER_CPU_OPERATOR(ExtendTensor, ExtendTensorOp<CPUContext>);

OPERATOR_SCHEMA(ExtendTensor)
    .NumInputs(2)
    .NumOutputs(1)
    .EnforceInplace({{0, 0}})
    .SetDoc(R"DOC(
Extend input 0 if necessary based on max element in input 1.
Input 0 must be the same as output, that is, it is required to be in-place.
Input 0 may have to be re-allocated in order for accommodate to the new size.
Currently, an exponential growth ratio is used in order to ensure amortized
constant time complexity.
All except the outer-most dimension must be the same between input 0 and 1.
)DOC")
    .Input(0, "tensor", "The tensor to be extended.")
    .Input(
        1,
        "new_indices",
        "The size of tensor will be extended based on max element in "
        "new_indices.")
    .Output(
        0,
        "extended_tensor",
        "Same as input 0, representing the mutated tensor.");
}
} // namespace
