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

#ifndef CAFFE2_OPERATORS_LPNORM_OP_H_
#define CAFFE2_OPERATORS_LPNORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LpNormOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LpNormOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OperatorBase::GetSingleArgument<int>("p", 2)),
        average_(OperatorBase::GetSingleArgument<bool>("average", false)) {
    CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
  }

  bool RunOnDevice() override;

 protected:
  int p_;
  bool average_;
  INPUT_TAGS(X_IN);
  OUTPUT_TAGS(OUT);
  // Input: X; Output: Norm
};

template <typename T, class Context>
class LpNormGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LpNormGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        p_(OperatorBase::GetSingleArgument<int>("p", 2)),
        average_(OperatorBase::GetSingleArgument<bool>("average", false)) {
    CAFFE_ENFORCE(p_ == 1 || p_ == 2, "p should be either 1 or 2.");
  }

  bool RunOnDevice() override;

 protected:
  int p_;
  bool average_;
  INPUT_TAGS(X_IN, DER_NORM_IN);
  OUTPUT_TAGS(DER_X_OUT);
  // Input: X, dNorm; Output: dX
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LPNORM_OP_H_
