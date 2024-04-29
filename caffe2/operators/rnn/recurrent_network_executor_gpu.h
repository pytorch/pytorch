#ifndef CAFFE2_OPERATORS_RECURRENT_NETWORK_GPU_EXECUTOR_H_
#define CAFFE2_OPERATORS_RECURRENT_NETWORK_GPU_EXECUTOR_H_

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/rnn/recurrent_network_executor.h"


#include <map>

namespace caffe2 {

class CUDARecurrentNetworkExecutor : public RecurrentNetworkExecutorBase {
 public:
  CUDARecurrentNetworkExecutor(
      const NetDef& step_net_def,
      std::map<string, string>& recurrent_input_map,
      std::string timestep_blob)
  : RecurrentNetworkExecutorBase(step_net_def, recurrent_input_map, timestep_blob) {}

  ~CUDARecurrentNetworkExecutor();

 protected:
  bool Run(int T) override;

  bool RunBackwards(int T) override;

  bool ignoreLinkDependencies() override {
    return true;
  }

  void AnalyzeOps() override {
    /**
      * Check if there is an op that only depends on ops from previous
      * timestep, and that ops is not the last op. Then we can start computation
      * in subsequent timesteps before the whole previous timestep has finished.
      * If there is no parallelism, we can avoid overhead of event-based
      * dependency management.
      */
    has_timestep_parallelism_ = false;
    for (auto& rnn_op : timestep_ops_template_) {
      int i = rnn_op.order;
      if (rnn_op.parents.size() >= 1 && i < timestep_ops_template_.size() - 1) {
        bool only_recurrent_deps = std::all_of(
                  rnn_op.parents.begin(),
                  rnn_op.parents.end(), [&](const int &parent) {
                    return parent > i;
                  }
        );
        if (only_recurrent_deps) {
          VLOG(1) << "Timestep parallel op: " << ProtoDebugString(step_net_def_.op(i));
          has_timestep_parallelism_ = true;

          for (int dep : rnn_op.parents) {
            if (dep == timestep_ops_template_.size() - 1) {
              // This op depends on the last op of the previous iteration,
              // so it will block any parallelism
              has_timestep_parallelism_ = false;
              break;
            }
          }
          break;
        }
      }
    }
    LOG(INFO) << "Analyzed ops for timestep parallelism: " << has_timestep_parallelism_;
 }

 public:

   void setMaxStreams(int n) {
     max_cuda_streams_ = n;
   }

 private:
  void _ExecRange(int from, int to);

  std::vector<cudaEvent_t> events_;
  bool has_timestep_parallelism_ = false;
  int max_cuda_streams_ = 2;
};
}
#endif
