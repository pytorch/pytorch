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

#ifndef CAFFE_OPERATORS_REPLACE_NAN_OP_H_
#define CAFFE_OPERATORS_REPLACE_NAN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class ReplaceNaNOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReplaceNaNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<float, double>>::call(this, Input(0));
  }

  template <typename T>
  bool DoRunWithType() {
    T value = OperatorBase::GetSingleArgument<T>("value", 0);

    auto& input = Input(0);
    auto* output = Output(0);
    output->ResizeLike(input);

    const T* input_data = input.template data<T>();
    T* output_data = output->template mutable_data<T>();
    for (TIndex i = 0; i < input.size(); i++) {
      if (std::isnan(input_data[i])) {
        output_data[i] = value;
      } else {
        output_data[i] = input_data[i];
      }
    }

    return true;
  }
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_REPLACE_NAN_OP_H_
