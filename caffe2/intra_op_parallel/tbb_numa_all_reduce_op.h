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

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

#include "caffe2/intra_op_parallel/intra_op_parallel.h"

#ifdef INTRA_OP_PARALLEL_CAN_USE_TBB

#include <tbb/flow_graph.h>
#include <tbb/task_arena.h>
#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include <tbb/task_scheduler_observer.h>

namespace caffe2 {

namespace tbb {

class NUMAAllReduceOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  NUMAAllReduceOp(const OperatorDef& operator_def, Workspace* ws);
  ~NUMAAllReduceOp() override;

 private:
  bool RunOnDevice() override;

  typedef ::tbb::flow::continue_node<::tbb::flow::continue_msg> cn_type;

  std::vector<std::unique_ptr<::tbb::flow::graph>> dags_;
  // TODO: reuse arena in net_tbb_task_graph
  std::vector<std::unique_ptr<::tbb::task_arena>> arena_;
  std::unique_ptr<cn_type> dag_root_, dag_exit_;
  std::vector<std::unique_ptr<::tbb::task_scheduler_observer>> observers_;

  std::vector<std::unique_ptr<cn_type>> flow_nodes_;
  std::vector<std::unique_ptr<::tbb::flow::graph_node>> cross_graph_edges_;

  int max_num_tasks_;
  std::vector<float*> push_bufs_;
};

} // namespace tbb

} // namespace caffe2

#endif // INTRA_OP_PARALLEL_CAN_USE_TBB
