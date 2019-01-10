#include "caffe2/operators/rnn/recurrent_network_executor_gpu.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

template <>
std::unique_ptr<RecurrentNetworkExecutorBase> createRNNExecutor<CUDAContext>(
    const NetDef& step_net_def,
    std::map<string, string>& recurrent_input_map,
    std::string timestep_blob,
    ArgumentHelper arg_helper) {
  auto* exec = new CUDARecurrentNetworkExecutor(
      step_net_def, recurrent_input_map, timestep_blob);
  int max_streams = arg_helper.GetSingleArgument<int>("rnn_executor.max_cuda_streams", 0);
  if (max_streams > 0) {
    exec->setMaxStreams(max_streams);
    LOG(INFO) << "Set max streams:" << max_streams;
  }
  std::unique_ptr<RecurrentNetworkExecutorBase> ptr(exec);
  return ptr;
}

CUDARecurrentNetworkExecutor::~CUDARecurrentNetworkExecutor() {
  for (cudaEvent_t ev : events_) {
    if (ev != nullptr) {
      CUDA_CHECK(cudaEventDestroy(ev));
    }
  }
}

/**
 * Special execution for CUDA. It tries to run ops with as little overhead as
 * possible, but to identify opportunities to run ops with "frontier execution"
 * parallelism, i.e by starting kernel from next timestep in parallel with
 * the current timestep. This is done by assigning streams.
 */
void CUDARecurrentNetworkExecutor::_ExecRange(int from, int to) {
  int direction = to > from ? 1 : -1;

  int max_streams = max_parallel_timesteps_ > 0 ?
                    std::min(max_parallel_timesteps_, max_cuda_streams_)
                    : max_cuda_streams_;
  int stream_seq = 0;
  int num_ops = timestep_ops_[0].size();

  events_.resize(num_ops * timestep_ops_.size(), nullptr);

  int gpu_id = -1;

  // Loop over timesteps
  for (int t = from; t != to; t += direction) {
    bool first_timestep = t == from;
    bool last_timestep =
        (direction == -1 && t == 0) || (direction == 1 && t == to - 1);
    auto& ops = timestep_ops_[t];
    int stream_id = stream_seq % max_streams;

    for (int i = 0; i < ops.size(); i++) {
      auto& rnn_op = ops[i];

      // Special handling for link ops -- we just run them directly
      // they do not execute any kernels.
      if (rnn_op.link_op) {
        rnn_op.op->RunAsync(stream_id);
        CAFFE_ENFORCE(
            rnn_op.dependencies.empty(),
            "GPU executor ignores link dependencies");
        continue;
      }

      if (gpu_id == -1 && rnn_op.op->device_option().device_type() == 1) {
        gpu_id = rnn_op.op->device_option().cuda_gpu_id();
      } else {
        CAFFE_ENFORCE(
            rnn_op.op->device_option().device_type() == 0 ||
                rnn_op.op->device_option().cuda_gpu_id() == gpu_id,
            "RNN Executor only supports ops on one GPU");
      }

      // If have recurrent parents, add for event waits so that those
      // parents complete their work.
      if (has_timestep_parallelism_ && !first_timestep) {
        for (int parent : rnn_op.parents) {
          if (parent > i) {
            int parent_ev_idx = (t - direction) * num_ops + parent;
            CHECK(events_.size() > parent_ev_idx);
            CAFFE_ENFORCE(events_[parent_ev_idx] != nullptr);
            CUDA_CHECK(cudaStreamWaitEvent(
                CUDAContext::cuda_stream(gpu_id, stream_id),
                events_[parent_ev_idx],
                0));
        }
        }
      }

      // Run the op in the given stream
      rnn_op.op->RunAsync(stream_id);

      // Create and record event for this op, if it has at least one
      // recurrent dependency.
      if (has_timestep_parallelism_ && !last_timestep) {
        for (int dep : rnn_op.dependencies) {
          if (dep < i) {
            int event_idx = t * num_ops + i;
            // Create event for recurrent connections
            if (events_[event_idx] == nullptr) {
              CUDA_CHECK(cudaEventCreate(&events_[event_idx]));
            }
            CUDA_CHECK(cudaEventRecord(
                events_[event_idx],
                CUDAContext::cuda_stream(gpu_id, stream_id)));
            break;
          }
        }
      }
    } // for over ops

    // Next timestep will run on different stream
    if (has_timestep_parallelism_) {
      stream_seq++;
    }
  } // for over timesteps

  /**
   * Wait for all the started streams to complete.
   */
  for (int stream_id = 0; stream_id <= std::min(stream_seq, max_streams - 1);
       stream_id++) {
    VLOG(1) << "Wait for stream:" << stream_id;
    CUDA_CHECK(
        cudaStreamSynchronize(CUDAContext::cuda_stream(gpu_id, stream_id)));
  }
}

bool CUDARecurrentNetworkExecutor::Run(int T) {
  _ExecRange(0, T);
  return true;
}

bool CUDARecurrentNetworkExecutor::RunBackwards(int T) {
  _ExecRange(T - 1, -1);
  return true;
}
}
