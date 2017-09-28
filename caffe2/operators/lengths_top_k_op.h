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


#ifndef CAFFE2_OPERATORS_LENGTHS_TOP_K_OP_H_
#define CAFFE2_OPERATORS_LENGTHS_TOP_K_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/conv_pool_op_base.h"
#include "caffe2/utils/math.h"

namespace caffe2 {
template <typename T, class Context>
class LengthsTopKOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  LengthsTopKOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws), OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE_GE(k_, 1, "k argument must be >= 1");
  }

  bool RunOnDevice() override;

 protected:
  int k_;
  INPUT_TAGS(X_IN, Y_IN);
  OUTPUT_TAGS(TOPK_VALUES_OUT, TOPK_INDICES_OUT);
};

template <typename T, class Context>
class LengthsTopKGradientOp : public Operator<Context> {
 public:
  LengthsTopKGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws), OP_SINGLE_ARG(int, "k", k_, -1) {
    CAFFE_ENFORCE_GE(k_, 1, "k argument must be >= 1");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int k_;
  INPUT_TAGS(LENGTH_IN, INDICES_IN, DER_TOPK_IN);
  OUTPUT_TAGS(DER_X_OUT);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_TOP_K_OP_H_
