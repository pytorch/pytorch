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
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

#include "caffe2/core/hip/context_hip.h"

C10_DEFINE_bool(caffe2_use_nvtx, false, "Use NVTX ranges for profiling");

C10_DEFINE_bool(
    caffe2_async_dag_use_multiple_streams,
    false,
    "Use multiple streams per thread");

C10_DECLARE_bool(caffe2_dag_net_collect_stats);

C10_DECLARE_bool(caffe2_net_async_finish_chain);

C10_DECLARE_int(caffe2_streams_per_gpu);

C10_DECLARE_bool(caffe2_net_async_check_stream_status);

namespace caffe2 {

thread_local std::vector<int> AsyncDAGNet::stream_counters_;

namespace {

using Color                  = int32_t;
constexpr Color kRunColor    = 0x0000CCFF; // blue
constexpr Color kRecordColor = 0x00FF3300; // red
constexpr Color kWaitColor   = 0x0066FF33; // green

class ProfiledRange
{
    public:
    ProfiledRange(const OperatorDef& def, Color color) {}

    private:
     C10_DISABLE_COPY_AND_ASSIGN(ProfiledRange);
};

} // namespace

AsyncDAGNet::AsyncDAGNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws)
    : DAGNetBase(net_def, ws)
{
    VLOG(1) << "Constructing Async DAG Net " << net_def->name();
    eventRecorded_.resize(net_def->op_size());

    // For all chains, their tail should consist the list of events that we are
    // needing for synchronization in the Run() inteface, unless there are other
    // chains depending on it.
    events_.reserve(execution_chains_.size());
    for(const auto& chain : execution_chains_)
    {
        const int tail_op_idx = chain.second.back();
        if(operator_nodes_[tail_op_idx].children_.empty())
        {
            events_.push_back(&operator_nodes_[tail_op_idx].operator_->event());
        }
    }
    VLOG(1) << "Total " << execution_chains_.size() << " chains, final waiting on "
            << events_.size() << " events";
}

int AsyncDAGNet::stream(const DeviceOption& device_option)
{
    int stream_id = 0;
    if (device_option.device_type() == PROTO_HIP) {
      int gpu_id = device_option.hip_gpu_id();
      CAFFE_ENFORCE_GE(
          gpu_id, 0, "Invalid gpu id: " + caffe2::to_string(gpu_id));
      if ((unsigned)gpu_id >= stream_counters_.size()) {
        stream_counters_.resize(gpu_id + 1, 0);
      }
      do {
        stream_id = stream_counters_[gpu_id]++;
        stream_counters_[gpu_id] %= c10::FLAGS_caffe2_streams_per_gpu;
      } while (c10::FLAGS_caffe2_net_async_check_stream_status &&
               !HIPContext::IsStreamFree(device_option, stream_id));
    }
    return stream_id;
}

bool AsyncDAGNet::RunAt(int chain_id, const std::vector<int>& chain)
{
    CAFFE_ENFORCE(!chain.empty(), "Chain should not be empty.");
    const auto source_idx = chain.front();
    const auto& parents   = operator_nodes_[source_idx].parents_;
    // Help ensure that our chaining is correct by verifying at least
    // one parent recorded an event.
    CAFFE_ENFORCE(parents.empty() || std::any_of(parents.begin(),
                                                 parents.end(),
                                                 [this](int p) { return eventRecorded_[p]; }),
                  "None of the parent is recorded for an event.");

    int stream_id = 0;
    if (c10::FLAGS_caffe2_async_dag_use_multiple_streams) {
      stream_id = stream(
          operator_nodes_[source_idx].operator_->event().GetDeviceOption());
    }

    std::vector<const Event*> parent_events;
    parent_events.reserve(operator_nodes_[source_idx].parents_.size());
    for(auto source_parent_idx : operator_nodes_[source_idx].parents_)
    {
        parent_events.push_back(&operator_nodes_[source_parent_idx].operator_->event());
    }
    {
        ProfiledRange r(operator_nodes_[source_idx].operator_->debug_def(), kWaitColor);
        operator_nodes_[source_idx].operator_->WaitEvents(parent_events, stream_id);
    }

    if (c10::FLAGS_caffe2_dag_net_collect_stats) {
      const auto& device_option =
          operator_nodes_[source_idx].operator_->event().GetDeviceOption();
      CAFFE_EVENT(
          stats_[device_option.device_type()],
          task_wait_time_us,
          task_timers_[chain_id]->MicroSeconds());
    }

    // We've waited on all our parent indices.
    bool success = true;
    for(auto idx : chain)
    {
        ProfiledRange r(operator_nodes_[idx].operator_->debug_def(), kRunColor);
        {
          TRACE_EVENT(
              tracing::TRACE_OP,
              idx,
              tracing::TRACE_TASK,
              chain_id,
              tracing::TRACE_STREAM,
              stream_id);
          success &= operator_nodes_[idx].operator_->RunAsync(stream_id);
        }
    }

    const auto& sink_idx = chain.back();
    if (success && c10::FLAGS_caffe2_net_async_finish_chain) {
      operator_nodes_[sink_idx].operator_->event().Finish();
    }
    CAFFE_ENFORCE(!eventRecorded_[sink_idx], "An event for ", sink_idx, " should not be recorded.");
    eventRecorded_[sink_idx] = 1;

    if (c10::FLAGS_caffe2_dag_net_collect_stats) {
      const auto& device_option =
          operator_nodes_[source_idx].operator_->event().GetDeviceOption();
      CAFFE_EVENT(
          stats_[device_option.device_type()],
          task_time_to_scheduled_us,
          task_timers_[chain_id]->MicroSeconds());
    }
    return success;
}

bool AsyncDAGNet::DoRunAsync()
{
    // Reset the event tracking at each iteration
    eventRecorded_.assign(eventRecorded_.size(), 0);

    const auto result = DAGNetBase::DoRunAsync();
    return result;
}

REGISTER_NET(async_dag, AsyncDAGNet);

} // namespace caffe2
