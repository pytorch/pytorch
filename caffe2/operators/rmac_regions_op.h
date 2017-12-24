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


#ifndef CAFFE2_OPERATORS_RMAC_REGIONS_OP_H
#define CAFFE2_OPERATORS_RMAC_REGIONS_OP_H

#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class RMACRegionsOp final : public Operator<Context> {
 public:
  RMACRegionsOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        scales_(OperatorBase::GetSingleArgument<int>("scales", 3)),
        overlap_(OperatorBase::GetSingleArgument<float>("overlap", 0.4f)) {}

  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  int scales_;
  float overlap_;
  Tensor<Context> num_rois_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_RMAC_REGIONS_OP_H
