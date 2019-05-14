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


#ifndef CAFFE2_FB_OPERATORS_UTILITY_OPS_H_
#define CAFFE2_FB_OPERATORS_UTILITY_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// Converts each input element into either high_ or low_value
// based on the given threshold.
//
// out[i] = low_value if in[i] <= threshold else high_value
template <typename TIN, typename TOUT, class Context>
class StumpFuncOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit StumpFuncOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        threshold_(this->template GetSingleArgument<TIN>("threshold", 0)),
        low_value_(this->template GetSingleArgument<TOUT>("low_value", 0)),
        high_value_(this->template GetSingleArgument<TOUT>("high_value", 0)) {}

  bool RunOnDevice() override;

 protected:
  TIN threshold_;
  TOUT low_value_;
  TOUT high_value_;

  // Input: label, output: weight
};

template <typename TIN, typename TOUT, class Context>
class StumpFuncIndexOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  template <class... Args>
  explicit StumpFuncIndexOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        threshold_(this->template GetSingleArgument<TIN>("threshold", 0)) {}

  bool RunOnDevice() override;

 protected:
  TIN threshold_;
  // Input: label, output: indices
};

} // caffe2

#endif // CAFFE2_FB_OPERATORS_UTILITY_OPS_H_
