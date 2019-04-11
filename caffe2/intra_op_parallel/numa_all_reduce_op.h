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

#include <array>
#include <vector>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

namespace intra_op_parallel {

/**
 * This function chooses a ring based on task id, and sets the index of
 * numa_node_id in the ring (idx_in_ring), the numa node precedes numa_node_id
 * in the ring (prev_numa_node_id), and the numa node succeeds numa_node_id in
 * the ring (next_numa_node_id) .
 */
void get_my_ring_info(
    int numa_node_id,
    int task,
    int num_numa_nodes,
    int* idx_in_ring,
    int* prev_numa_node_id,
    int* next_numa_node_id);

class NUMAAllReduceOp final : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  NUMAAllReduceOp(const OperatorDef& operator_def, Workspace* ws);
  ~NUMAAllReduceOp() override;

 private:
  bool RunOnDevice() override;
  bool HasAsyncPart() const override {
    return true;
  }

  bool RunOnDeviceParallel_(int numa_node_id, int worker_id, int num_workers);

  int max_num_tasks_;
  std::atomic<int> count_;
  std::vector<float*> push_bufs_;

  // Per-worker synchronization variable
  std::vector<std::condition_variable> cv_for_peer_sync_;
  std::vector<std::mutex> mutex_for_peer_sync_;
  std::vector<std::unique_ptr<std::atomic<int>>> generations_;
};

} // namespace intra_op_parallel

} // namespace caffe2
