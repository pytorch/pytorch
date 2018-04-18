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

#include <caffe2/ideep/ideep_utils.h>

namespace caffe2 {

class IDEEPLRNOp final : public IDEEPOperator {
 public:
  USE_IDEEP_DEF_ALIASES();
  USE_IDEEP_OPERATOR_FUNCTIONS();

  IDEEPLRNOp(const OperatorDef& operator_def, Workspace* ws)
      : IDEEPOperator(operator_def, ws),
        size_(OperatorBase::GetSingleArgument<int>("size", 0)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0)),
        bias_(OperatorBase::GetSingleArgument<float>("bias", 1)),
        pre_pad_((size_ - 1) / 2) {
    DCHECK_GT(size_, 0);
    DCHECK_EQ(size_ % 2, 1);
    DCHECK_GT(alpha_, 0);
    DCHECK_GT(beta_, 0);
  }
  virtual ~IDEEPLRNOp() {}

  bool RunOnDevice() override {
    auto& X = Input(INPUT);
    auto* Y = Output(OUTPUT);

    ideep::lrn_forward::compute(X, *Y, size_, alpha_, beta_, bias_);

    return true;
  }

 private:
  const int size_;
  const float alpha_;
  const float beta_;
  const float bias_;
  const int pre_pad_;

  INPUT_TAGS(INPUT);
  OUTPUT_TAGS(OUTPUT);
};

REGISTER_IDEEP_OPERATOR(LRN, IDEEPLRNOp);

} // namespace caffe2
