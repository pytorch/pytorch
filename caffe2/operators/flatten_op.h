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

#ifndef CAFFE2_OPERATORS_FLATTEN_OP_H_
#define CAFFE2_OPERATORS_FLATTEN_OP_H_

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class FlattenOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  FlattenOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)) {}

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);
    CAFFE_ENFORCE_GE(
        input.dims().size(), axis_, "The rank of the tensor must be >= axis.");
    output->Resize(input.size_to_dim(axis_), input.size_from_dim(axis_));
    context_.template CopyItems<Context, Context>(
        input.meta(),
        input.size(),
        input.raw_data(),
        output->raw_mutable_data(input.meta()));
    return true;
  }

 private:
  int axis_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FLATTEN_OP_H_
