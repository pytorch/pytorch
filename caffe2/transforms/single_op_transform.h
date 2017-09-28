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
#include "caffe2/core/transform.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

/**
 * Single Op Transform Base class
 *
 * A transform which is applied to a single node, in place.
 *
 * Transforms which derive from SingleOpTransform need to override:
 * ReplaceOperator and MatchOperator.
 */
class SingleOpTransform : public Transform {
 protected:
  bool PatternRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph,
      int idx) override;
  bool ValidatorRule(
      const transform::Graph& g,
      const std::vector<int>& subgraph) override;
  bool ReplaceRule(const std::vector<int>& subgraph, transform::Graph* g_ptr)
      override;

  // Specify what the op needs to be to match the pattern.
  virtual bool MatchOperator(const OperatorDef& op) = 0;

  // Specify how the operator should be replaced.
  virtual void ReplaceOperator(OperatorDef* op) = 0;
};

} // namespace caffe2
