#ifndef CAFFE2_CORE_NET_H_
#define CAFFE2_CORE_NET_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread>  // NOLINT
#include <typeinfo>
#include <vector>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/simple_queue.h"

namespace caffe2 {

class OperatorBase;

// Net is a thin struct that owns all the operators together with the operator
// contexts.
class NetBase {
 public:
  NetBase(const NetDef& net_def, Workspace* ws) {}
  virtual ~NetBase() {}
  virtual bool Verify() = 0;
  virtual bool Run() = 0;
  virtual void TEST_Benchmark(const int warmup_runs, const int main_runs,
                              const bool run_individual) {
    CAFFE_LOG_ERROR << "Benchmark not implemented for this net type.";
  }

  DISABLE_COPY_AND_ASSIGN(NetBase);
};


DECLARE_REGISTRY(NetRegistry, NetBase, const NetDef&, Workspace*);
#define REGISTER_NET_CREATOR(key, ...) \
  REGISTER_CREATOR(NetRegistry, key, __VA_ARGS__)
#define REGISTER_NET(name, ...) \
  REGISTER_CLASS(NetRegistry, name, __VA_ARGS__)
NetBase* CreateNet(const NetDef& net_def, Workspace* ws);

// This is the very basic structure you need to run a network - all it
// does is simply to run everything in sequence. If you want more fancy control
// such as a DAG-like execution, check out other better net implementations.
class SimpleNet final : public NetBase {
 public:
  SimpleNet(const NetDef& net_def, Workspace* ws);
  bool Verify() override;
  bool Run() override;
  void TEST_Benchmark(const int warmup_runs, const int main_runs,
                      const bool run_individual) override;

 protected:
  vector<unique_ptr<OperatorBase> > operators_;

  DISABLE_COPY_AND_ASSIGN(SimpleNet);
};

namespace internal {
struct OperatorNode {
  unique_ptr<OperatorBase> operator_;
  vector<int> children_;
  vector<int> parents_;
  std::atomic<int> runtime_parent_count_;
};
}

class DAGNet final : public NetBase {
 public:
  DAGNet(const NetDef& net_def, Workspace* ws);
  ~DAGNet();
  bool Verify() override;
  bool Run() override;
  // WorkerFunction() is a function wrapper to allow us to run worker threads.
  // It checks out one ready-to-run operator from the job queue, runs it,
  // notifies all its children, and for any children that is ready, enqueues
  // it to the job queue.
  void WorkerFunction();
  void TEST_Benchmark(const int warmup_runs, const int main_runs,
                      const bool run_individual) override;

 protected:
  vector<internal::OperatorNode> operator_nodes_;
  vector<int> initial_frontier_;
  SimpleQueue<int> job_queue_;
  std::vector<std::thread> workers_;
  int remaining_ops_;
  bool success_;
  std::mutex remaining_ops_mutex_;
  std::condition_variable cv_;
  std::mutex run_in_progress_;

  DISABLE_COPY_AND_ASSIGN(DAGNet);
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_NET_H_
