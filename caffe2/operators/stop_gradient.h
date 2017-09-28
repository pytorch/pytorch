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

#ifndef CAFFE2_OPERATORS_STOP_GRADIENT_H_
#define CAFFE2_OPERATORS_STOP_GRADIENT_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class StopGradientOp : public Operator<Context> {
 public:
  USE_SIMPLE_CTOR_DTOR(StopGradientOp)
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  bool RunOnDevice() override {
    const auto& in = Input(0);
    auto* out = Output(0);
    if (out != &in) {
      out->CopyFrom(in, &context_);
    }
    return true;
  }
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_STOP_GRADIENT_H_
