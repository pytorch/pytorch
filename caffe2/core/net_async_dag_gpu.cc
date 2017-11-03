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

#include "caffe2/core/net_async_dag_gpu.h"

#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>

#include "caffe2/core/operator.h"
#include "caffe2/core/static_tracepoint.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/proto_utils.h"

#ifdef CAFFE2_USE_NVTX
#include <nvToolsExt.h>
#endif

CAFFE2_DEFINE_bool(caffe2_use_nvtx, false, "Use NVTX ranges for profiling");

CAFFE2_DECLARE_bool(caffe2_dag_net_collect_stats);

namespace caffe2 {

namespace {

using Color = int32_t;
constexpr Color kRunColor = 0x0000CCFF; // blue
constexpr Color kRecordColor = 0x00FF3300; // red
constexpr Color kWaitColor = 0x0066FF33; // green

#ifdef CAFFE2_USE_NVTX

class ProfiledRange {
 public:
  ProfiledRange(const OperatorDef& def, Color color) {
    if (!FLAGS_caffe2_use_nvtx) {
      return;
    }
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = color;
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = def.type().c_str();
    range_ = nvtxRangeStartEx(&eventAttrib);
    CAFFE_ENFORCE(range_, "Start range is invalid.");
  }

  ~ProfiledRange() {
    if (!FLAGS_caffe2_use_nvtx) {
      return;
    }
    nvtxRangeEnd(range_);
  }

 private:
  nvtxRangeId_t range_ = 0;
  DISABLE_COPY_AND_ASSIGN(ProfiledRange);
};

#else

class ProfiledRange {
 public:
  ProfiledRange(const OperatorDef& def, Color color) {}

 private:
  DISABLE_COPY_AND_ASSIGN(ProfiledRange);
};

#endif // ifdef CAFFE2_USE_NVTX

} // namespace

AsyncDAGNet::AsyncDAGNet(
    const std::shared_ptr<const NetDef>& net_def,
    Workspace* ws)
    : DAGNetBase(net_def, ws) {
  VLOG(1) << "Constructing Async DAG Net " << net_def->name();
  eventRecorded_.resize(net_def->op_size());

  // For all chains, their tail should consist the list of events that we are
  // needing for synchronization in the Run() inteface, unless there are other
  // chains depending on it.
  events_.reserve(execution_chains_.size());
  for (const auto& chain : execution_chains_) {
    const int tail_op_idx = chain.second.back();
    if (operator_nodes_[tail_op_idx].children_.empty()) {
      events_.push_back(&operator_nodes_[tail_op_idx].operator_->event());
    }
  }
  VLOG(1) << "Total " << execution_chains_.size()
          << " chains, final waiting on " << events_.size() << " events";
}

bool AsyncDAGNet::RunAt(int chain_id, const std::vector<int>& chain) {
  CAFFE_ENFORCE(!chain.empty(), "Chain should not be empty.");
  const auto source_idx = chain.front();
  const auto& parents = operator_nodes_[source_idx].parents_;
  // Help ensure that our chaining is correct by verifying at least
  // one parent recorded an event.
  CAFFE_ENFORCE(
      parents.empty() ||
          std::any_of(
              parents.begin(),
              parents.end(),
              [this](int p) { return eventRecorded_[p]; }),
      "None of the parent is recorded for an event.");

  for (auto source_parent_idx : operator_nodes_[source_idx].parents_) {
    ProfiledRange r(
        operator_nodes_[source_parent_idx].operator_->debug_def(), kWaitColor);
    operator_nodes_[source_idx].operator_->Wait(
        *operator_nodes_[source_parent_idx].operator_);
  }

  if (FLAGS_caffe2_dag_net_collect_stats) {
    const auto& device_option =
        operator_nodes_[chain_id].operator_->event().GetDeviceOption();
    CAFFE_EVENT(
        stats_[device_option.device_type()],
        task_wait_time_us,
        task_timers_[chain_id]->MicroSeconds());
  }

  // We've waited on all our parent indices.
  bool success = true;
  for (auto idx : chain) {
    ProfiledRange r(operator_nodes_[idx].operator_->debug_def(), kRunColor);
    success &= operator_nodes_[idx].operator_->RunAsync();
  }

  const auto& sink_idx = chain.back();
  CAFFE_ENFORCE(
      !eventRecorded_[sink_idx],
      "An event for ",
      sink_idx,
      " should not be recorded.");
  eventRecorded_[sink_idx] = 1;

  if (FLAGS_caffe2_dag_net_collect_stats) {
    const auto& device_option =
        operator_nodes_[chain_id].operator_->event().GetDeviceOption();
    CAFFE_EVENT(
        stats_[device_option.device_type()],
        task_time_to_scheduled_us,
        task_timers_[chain_id]->MicroSeconds());
  }
  return success;
}

bool AsyncDAGNet::DoRunAsync() {
  // Reset the event tracking at each iteration
  eventRecorded_.assign(eventRecorded_.size(), 0);

  const auto result = DAGNetBase::DoRunAsync();
  return result;
}

REGISTER_NET(async_dag, AsyncDAGNet);

} // namespace caffe2
