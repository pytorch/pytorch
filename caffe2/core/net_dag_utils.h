#ifndef CAFFE2_CORE_NET_DAG_UTILS_H_
#define CAFFE2_CORE_NET_DAG_UTILS_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread> // NOLINT
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "c10/util/Registry.h"
#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/simple_queue.h"

namespace caffe2 {
namespace dag_utils {

struct OperatorNode {
  unique_ptr<OperatorBase> operator_;
  vector<int> children_;
  vector<int> parents_;
  std::atomic<int> runtime_parent_count_;
  bool is_chain_start_ = false;
  std::atomic_flag scheduled_ = ATOMIC_FLAG_INIT;
};

struct OpGraphNode {
  vector<int> children_;
  vector<int> parents_;
  int visited_inputs = 0;
  int num_orig_parents;
};

using ExecutionChains = std::unordered_map<int, std::vector<int>>;

C10_EXPORT ExecutionChains computeChains(std::vector<OperatorNode>& orig_nodes);

// Instead of breaking down the DAG into chains, we partition it into clusters
// of sync ops and individual async op. This is useful for disturbuted inference
// case where we have sync and async cpu ops. Note that we have go sync each
// aysnc op instead of put them into the chain and sync its tail like GPU op,
// because CPU async ops are typically rpc calls and are not guaranteed to be
// linearized at remote site.
C10_EXPORT ExecutionChains computeGroups(std::vector<OperatorNode>& orig_nodes);

C10_EXPORT ExecutionChains singleChains(std::vector<OperatorNode>& nodes);

C10_EXPORT std::vector<OperatorNode> prepareOperatorNodes(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws);

std::vector<OpGraphNode> prepareChainGraphNodes(
    const std::vector<dag_utils::OperatorNode>& operator_nodes,
    const std::vector<std::vector<int>>& execution_chains);

} // namespace dag_utils
} // namespace caffe2

#endif // CAFFE2_CORE_NET_DAG_UTILS_H_
