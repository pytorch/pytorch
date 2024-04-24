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

#include "caffe2/operators/utility_ops.h"

namespace caffe2 {

template <class Context>
class SumReluOp : public SumOp<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SumReluOp(const OperatorDef& operator_def, Workspace* ws)
      : SumOp<Context>(operator_def, ws) {}

  template <typename T, typename M>
  bool DoRunWithType() {
    if (!SumOp<Context>::template DoRunWithType<T>()) {
      return false;
    }

    auto* output = Output(0);
    T* output_data = output->template mutable_data<T>();
    for (int i = 0; i < output->size(); ++i) {
      output_data[i] = std::max(static_cast<T>(0), output_data[i]);
    }
    return true;
  }

  bool RunOnDevice() override {
    if (Input(0).template IsType<float>()) {
      return DoRunWithType<float, float>();
    } else if (Input(0).template IsType<double>()) {
      return DoRunWithType<double, double>();
    } else if (Input(0).template IsType<int>()) {
      return DoRunWithType<int, int>();
    } else {
      CAFFE_THROW(
          "Sum operator only supports 32-bit float, 64-bit double and ints, but",
          " input was of type ",
          Input(0).dtype().name());
    }
  }
};

REGISTER_CPU_OPERATOR(SumRelu, SumReluOp<CPUContext>);

} // namespace caffe2
