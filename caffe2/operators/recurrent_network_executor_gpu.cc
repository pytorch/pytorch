#include "caffe2/operators/recurrent_network_executor_gpu.h"

#include "caffe2/core/context_gpu.h"

namespace caffe2 {

template <>
std::unique_ptr<RecurrentNetworkExecutorBase> createRNNExecutor<CUDAContext>(
    const NetDef& step_net_def,
    std::map<string, string>& recurrent_input_map) {
  std::unique_ptr<RecurrentNetworkExecutorBase> ptr(
      new CUDARecurrentNetworkExecutor(step_net_def, recurrent_input_map));
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

  int max_streams = 8; // TODO
  int stream_seq = 0;
  std::vector<int> run_streams;

  int num_ops = step_net_def_.op_size();

  run_streams.resize(num_ops, -1);
  events_.resize(num_ops);

  int gpu_id = -1;

  // Loop over timesteps
  for (int t = from; t != to; t += direction) {
    bool first_timestep = t == from;
    bool last_timestep =
        (direction == -1 && t == 0) || (direction == 1 && t == to - 1);
    auto& ops = timestep_ops_[t];

    int num_early_starts = 0;
    for (int i = 0; i < ops.size(); i++) {
      auto& rnn_op = ops[i];
      int stream_id = stream_seq % max_streams;

      // Special handling for link ops -- we just run them directly
      // they do not execute any kernels.
      if (rnn_op.link_op) {
        rnn_op.op->RunAsync(stream_id);
        CAFFE_ENFORCE(
            rnn_op.dependencies.empty(),
            "GPU executor ignores link dependencies");
        continue;
      }

      if (gpu_id == -1) {
        gpu_id = rnn_op.op->device_option().cuda_gpu_id();
      } else {
        CAFFE_ENFORCE(
            rnn_op.op->device_option().device_type() == 0 ||
                rnn_op.op->device_option().cuda_gpu_id() == gpu_id,
            "RNN Executor only supports ops on one GPU");
      }

      // Check which stream I have
      CHECK(run_streams.size() > i);
      run_streams[i] = stream_id;

      // If have recurrent parents, add for event waits so that those
      // parents complete their work.
      for (int parent : rnn_op.parents) {
        if (!first_timestep && parent > i) {
          CHECK(events_.size() > parent);
          if (events_[parent] != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(
                CUDAContext::cuda_stream(gpu_id, stream_id),
                events_[parent],
                0));
          }
        }
      }

      // Run the op in the given stream
      rnn_op.op->RunAsync(stream_id);

      // Pass my stream to dependents
      for (int dep : rnn_op.dependencies) {
        if (!last_timestep && dep < i) {
          // Create event for recurrent connections
          if (events_[i] == nullptr) {
            CUDA_CHECK(cudaEventCreate(&events_[i]));
          }
          CUDA_CHECK(cudaEventRecord(
              events_[i], CUDAContext::cuda_stream(gpu_id, stream_id)));
          break;
        }
      }

      // Count early starts: these are operators that can be started
      // without waiting for multiple dependencies from previous timestep.
      // Thus, they can be chained.
      if (i < ops.size() - 1) {
        for (int dep : rnn_op.dependencies) {
          if (dep < i && ops[dep].parents.size() == 1) {
            num_early_starts++;
          }
        }
      }
    }

    // If this op had dependenceis that can be started early, we let them
    // continue on this stream and instead next op will start in another
    // stream.
    if (num_early_starts > 0) {
      stream_seq++;
    }
  }

  /**
   * Wait for all the started streams to complete.
   */
  for (int stream_id = 0; stream_id < std::min(stream_seq, max_streams);
       stream_id++) {
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
