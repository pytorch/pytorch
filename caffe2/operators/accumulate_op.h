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

#ifndef CAFFE2_OPERATORS_ACCUMULATE_OP_H_
#define CAFFE2_OPERATORS_ACCUMULATE_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class AccumulateOp final : public Operator<Context> {
 public:
  AccumulateOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        gamma_(static_cast<T>(
            OperatorBase::template GetSingleArgument<float>("gamma", 1.0))) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    if (output->dims() != input.dims()) {
      LOG(INFO) << "Reshaping and initializing output.";
      output->ResizeLike(input);
      math::Set<T, Context>(
          output->size(), 0, output->template mutable_data<T>(), &context_);
    }
    math::Axpby<T, Context>(
        input.size(),
        static_cast<T>(1),
        input.template data<T>(),
        gamma_,
        output->template mutable_data<T>(),
        &context_);
    return true;
  }

 protected:
  T gamma_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_ACCUMULATE_OP_H_
