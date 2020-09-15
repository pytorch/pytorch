
#pragma once

#include "caffe2/core/common.h"
#include "caffe2/core/transform.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

/**
 * Common Subexpression Elimination
 *
 * This transforms looks for specific operators (denoted by allowed_ops_),
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
class CAFFE2_API CommonSubexpressionEliminationTransform : public Transform {
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
  bool IsAllowed(string op_type) {
    return allowed_ops_.count(op_type);
  }
  std::set<string> allowed_ops_ = {"LearningRate", "FC"};
};

} // namespace caffe2
