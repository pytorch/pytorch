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

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/transforms/single_op_transform.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

class ConvToNNPackTransform : public SingleOpTransform {
 protected:
  // Specify what the op needs to be to match the pattern.
  bool MatchOperator(const OperatorDef& op) override {
    return (
        op.type() == "Conv" && op.device_option().device_type() == CPU &&
        op.engine() != "NNPACK");
  }

  // Specify how the operator should be replaced.
  void ReplaceOperator(OperatorDef* op) override {
    op->set_engine("NNPACK");
  }
};

} // namespace caffe2
