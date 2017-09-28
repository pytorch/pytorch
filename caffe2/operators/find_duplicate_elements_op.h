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

#ifndef CAFFE2_OPERATORS_FIND_DUPLICATE_ELEMENTS_OP_H
#define CAFFE2_OPERATORS_FIND_DUPLICATE_ELEMENTS_OP_H

#include <unordered_map>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <class Context>
class FindDuplicateElementsOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FindDuplicateElementsOp);
  USE_DISPATCH_HELPER;

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double, int, long, std::string>>::
        call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    const auto& data = Input(0);
    CAFFE_ENFORCE(data.ndim() == 1, "data should be 1-D.");

    const auto* data_ptr = data.template data<T>();
    std::unordered_map<T, int64_t> dict;
    std::vector<int64_t> dupIndices;
    // i is the index of unique elements, j is the index of all elements
    for (int64_t i = 0, j = 0; j < data.dims()[0]; ++i, ++j) {
      bool retVal = dict.insert({data_ptr[j], i}).second;
      if (!retVal) {
        --i;
        dupIndices.push_back(j);
      }
    }

    const auto dupSize = dupIndices.size();
    auto* output = Output(0);
    output->Resize(dupSize);
    auto* out_ptr = output->template mutable_data<int64_t>();
    for (int64_t i = 0; i < dupSize; ++i) {
      out_ptr[i] = dupIndices[i];
    }

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FIND_DUPLICATE_ELEMENTS_OP_H
