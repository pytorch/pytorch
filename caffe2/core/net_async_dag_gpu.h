#ifndef CAFFE2_CORE_NET_ASYNC_DAG_GPU_H_
#define CAFFE2_CORE_NET_ASYNC_DAG_GPU_H_

#include "caffe2/core/common.h"
#include "caffe2/core/net_dag.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {

// Run an event-driven graph - before each operator chain, wait on each parent
// operator for the chain source, then execute each operator. Due to the chain
// construction mechanism, operators in the same chain implicitly runs on the
// same stream.
// AsyncDAGNet is only registered in gpu mode, because CPU code is always sync
// and a CPU only AsyncDAG net is essentially a DAG net.
class AsyncDAGNet : public DAGNetBase {
 public:
  AsyncDAGNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  bool SupportsAsync() override {
    return true;
  }
  bool RunAt(int chain_id, const std::vector<int>& chain) override;

 protected:
  bool DoRunAsync() override;

  // Tracks whether a given op has had an event recorded in each
  // RunAt() iteration.
  std::vector<int32_t> eventRecorded_;

  int stream(const DeviceOption& device_option);
  static thread_local std::vector<int> stream_counters_;

  DISABLE_COPY_AND_ASSIGN(AsyncDAGNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_ASYNC_DAG_GPU_H_
