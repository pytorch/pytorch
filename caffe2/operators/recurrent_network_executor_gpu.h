#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/recurrent_network_executor.h"

#include <map>

namespace caffe2 {

class CUDARecurrentNetworkExecutor : public RecurrentNetworkExecutorBase {
 public:
  CUDARecurrentNetworkExecutor(
      const NetDef& step_net_def,
      std::map<string, string>& recurrent_input_map)
      : RecurrentNetworkExecutorBase(step_net_def, recurrent_input_map) {}

  ~CUDARecurrentNetworkExecutor();

 protected:
  bool Run(int T) override;

  bool RunBackwards(int T) override;

  bool ignoreLinkDependencies() override {
    return true;
  }

 private:
  void _ExecRange(int from, int to);

  std::vector<cudaEvent_t> events_;
};
}
