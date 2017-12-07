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
#ifndef CAFFE2_OPERATORS_ASSERT_OP_H_
#define CAFFE2_OPERATORS_ASSERT_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class AssertOp final : public Operator<Context> {
 public:
  AssertOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        error_msg_(
            OperatorBase::GetSingleArgument<std::string>("error_msg", "")) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <typename T>
  bool DoRunWithType() {
    // Copy into CPU context for comparison
    cmp_tensor_.CopyFrom(Input(0));
    auto* cmp_data = cmp_tensor_.template data<T>();

    for (TIndex i = 0; i < cmp_tensor_.size(); ++i) {
      CAFFE_ENFORCE((bool)cmp_data[i], [&]() {
        std::stringstream ss;
        ss << "Assert failed for element " << i
           << " in tensor, value: " << cmp_data[i] << "\n";
        if (!error_msg_.empty()) {
          ss << "Error message: " << error_msg_;
        }
        return ss.str();
      }());
    }
    return true;
  }

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<long, int, bool>>::call(this, Input(0));
  }

 private:
  TensorCPU cmp_tensor_;
  std::string error_msg_;
};

} // namespace caffe2

#endif /* CAFFE2_OPERATORS_ASSERT_OP_H_ */
