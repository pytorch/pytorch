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
 * Common Subexpression Elimination
 *
 * This transforms looks for specific operators (denoted by whitelisted_ops_),
 * and removes unnecessary repetition of that operator.
 *
 * Consider some operator of X, that reads from blob b_ written to by W.
 * X_a and X_b read the output of X. However, another operator Y, is the same
 * type as X, has the same arguments as X, and reads from the same input b_,
 * written to by W. It's output is the same as X. Y_a, Y_b, and Y_c read from Y.
 *
 * Then, we can eliminate the common subexpressions X and Y, and merge them to
 * Z, where X_a, X_b, Y_a, Y_b, and Y_c all read from Z.
 *
 *
 * TODO(benz): Fix the error to not match nodes that write to external output.
 */
class CommonSubexpressionEliminationTransform : public Transform {
 public:
  CommonSubexpressionEliminationTransform() {
    SetPatternMatchType(SORTED_WRT_EXECUTION_ORDER);
  }

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

 private:
  bool IsWhitelisted(string op_type) {
    return whitelisted_ops_.count(op_type);
  }
  std::set<string> whitelisted_ops_ = {"LearningRate", "FC"};
};

} // namespace caffe2
