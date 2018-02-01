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

#ifndef CAFFE_OPERATORS_MOD_OP_H_
#define CAFFE_OPERATORS_MOD_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class ModOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ModOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    divisor_ = OperatorBase::GetSingleArgument<int64_t>("divisor", 0);
    CAFFE_ENFORCE_NE(divisor_, 0, "divisor must not be 0");
    sign_follow_divisor_ =
        OperatorBase::GetSingleArgument<bool>("sign_follow_divisor", false);
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int, int64_t>>::call(this, Input(DATA));
  }

  template <typename T>
  bool DoRunWithType();

 protected:
  INPUT_TAGS(DATA);

 private:
  int64_t divisor_;
  bool sign_follow_divisor_;
};

} // namespace caffe2

#endif // CAFFE_OPERATORS_MOD_OP_H_
