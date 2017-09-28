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


#ifndef CAFFE2_OPERATORS_PREPEND_DIM_OP_H_
#define CAFFE2_OPERATORS_PREPEND_DIM_OP_H_

#include "caffe2/core/common_omp.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class PrependDimOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  PrependDimOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        dim_size_(OperatorBase::GetSingleArgument<int64_t>("dim_size", 0)) {
    CAFFE_ENFORCE_GT(
        dim_size_, 0, "Argument dim_size must be greater than zero.");
  }

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);

    CAFFE_ENFORCE(input.ndim() > 0, "Input must be at least 1D.");
    CAFFE_ENFORCE(
        input.dim(0) % dim_size_ == 0,
        "First dimension must be multiple of prepend_dim.");

    vector<int64_t> actual_new_shape(input.ndim() + 1);
    actual_new_shape[0] = dim_size_;
    actual_new_shape[1] = input.dim(0) / dim_size_;
    for (int i = 1; i < input.dims().size(); ++i) {
      actual_new_shape[i + 1] = input.dim(i);
    }
    output->Resize(actual_new_shape);

    if (output != &input) {
      // If we are not doing in-place computation, a copy is needed.
      context_.template CopyBytes<Context, Context>(
          input.nbytes(),
          input.raw_data(),
          output->raw_mutable_data(input.meta()));
    }
    return true;
  }

 private:
  int64_t dim_size_;
};

template <class Context>
class MergeDimOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  MergeDimOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}

  bool RunOnDevice() override {
    auto& input = Input(0);
    auto* output = Output(0);

    CAFFE_ENFORCE(input.ndim() > 1, "Input must be at least 2D.");

    vector<int64_t> actual_new_shape(input.ndim() - 1);
    actual_new_shape[0] = input.dim(0) * input.dim(1);
    for (int i = 1; i < input.dims().size() - 1; ++i) {
      actual_new_shape[i] = input.dim(i + 1);
    }
    output->Resize(actual_new_shape);

    if (output != &input) {
      // If we are not doing in-place computation, a copy is needed.
      context_.template CopyBytes<Context, Context>(
          input.nbytes(),
          input.raw_data(),
          output->raw_mutable_data(input.meta()));
    }
    return true;
  }

 private:
  int64_t dim_size_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_PREPEND_DIM_OP_H_
