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

#pragma once

#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class SparseNormalizeOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SparseNormalizeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        use_max_norm_(
            OperatorBase::GetSingleArgument<bool>("use_max_norm", true)),
        norm_(OperatorBase::GetSingleArgument<float>("norm", 1.0)) {
    CAFFE_ENFORCE_GE(norm_, 0, "norm should be bigger than 0");
  }

  bool RunOnDevice() override {
    CAFFE_ENFORCE_EQ(
        Input(PARAM).size_from_dim(1),
        Input(GRAD).size_from_dim(Input(INDICES).ndim()));

    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename SIndex>
  bool DoRunWithType();

 protected:
  bool use_max_norm_;
  float norm_;
  INPUT_TAGS(PARAM, INDICES, GRAD);
  OUTPUT_TAGS(OUTPUT_PARAM);
};

} // namespace caffe2
