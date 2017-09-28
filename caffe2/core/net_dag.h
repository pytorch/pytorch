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

#ifndef CAFFE2_CORE_NET_DAG_H_
#define CAFFE2_CORE_NET_DAG_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread> // NOLINT
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/simple_queue.h"

namespace caffe2 {

namespace internal {
struct OperatorNode {
  unique_ptr<OperatorBase> operator_;
  vector<int> children_;
  vector<int> parents_;
  std::atomic<int> runtime_parent_count_;
  bool is_chain_start_ = false;
};

struct OpGraphNode {
  vector<int> children_;
  vector<int> parents_;
  int visited_inputs = 0;
  int num_orig_parents;
};
}

class DAGNetBase : public NetBase {
 public:
  using ExecutionChains = std::unordered_map<int, std::vector<int>>;
  DAGNetBase(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~DAGNetBase() override;
  bool RunAsync() override;
  // WorkerFunction() is a function wrapper to allow us to run worker threads.
  // It checks out one ready-to-run operator from the job queue, runs it,
  // notifies all its children, and for any children that is ready, enqueues
  // it to the job queue.
  void WorkerFunction();
  vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) override;

  const ExecutionChains& TEST_execution_chains() const {
    return execution_chains_;
  }

  vector<OperatorBase*> GetOperators() const override {
    vector<OperatorBase*> op_list;
    for (auto& op_node : operator_nodes_) {
      op_list.push_back(op_node.operator_.get());
    }
    return op_list;
  }

 protected:
  virtual bool RunAt(const std::vector<int>& chain) = 0;

  vector<internal::OperatorNode> operator_nodes_;
  ExecutionChains execution_chains_;
  vector<int> initial_frontier_;
  std::unique_ptr<SimpleQueue<int>> job_queue_;
  std::vector<std::thread> workers_;
  int num_workers_;
  int num_workers_first_iteration_;
  int remaining_ops_;

  bool success_;
  int iter_;
  std::mutex remaining_ops_mutex_;
  std::condition_variable cv_;
  std::mutex run_in_progress_;

  DISABLE_COPY_AND_ASSIGN(DAGNetBase);
};

class DAGNet : public DAGNetBase {
 public:
  using DAGNetBase::DAGNetBase;

 protected:
  bool RunAt(const std::vector<int>& chain) override;
  bool SupportsAsync() override {
    return false;
  }
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_DAG_H_
