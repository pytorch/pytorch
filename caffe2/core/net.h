#ifndef CAFFE2_CORE_NET_H_
#define CAFFE2_CORE_NET_H_

#include <atomic>
#include <climits>
#include <cstddef>
#include <thread>  // NOLINT
#include <typeinfo>
#include <vector>
#include <unordered_map>

#include "caffe2/core/blob.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/observer.h"
#include "caffe2/core/operator_schema.h"
#include "caffe2/core/registry.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2.pb.h"
#include "caffe2/utils/simple_queue.h"

namespace caffe2 {

class OperatorBase;
class Workspace;
// Net is a thin struct that owns all the operators together with the operator
// contexts.
class NetBase {
 public:
  NetBase(const NetDef& net_def, Workspace* ws);
  virtual ~NetBase() noexcept {}
  virtual bool Run() = 0;

  // RunAsync runs the net on the current stream, but potentially does
  // not synchronize with respect to the host, and thus may require
  // external synchronization (with respect to the current stream)
  // after execution.
  virtual bool RunAsync() { return Run(); }
  /**
   * Benchmarks a network.
   *
   * This function returns a vector of float recording the number of milli-
   * seconds spent during the benchmark. The 0-th item is the time spent per
   * each network run, and if a net instantiation supports run_individual,
   * the remainder of the vector returns the number of milliseconds spent per
   * opeartor.
   */
  virtual vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) {
    LOG(ERROR) << "Benchmark not implemented for this net type.";
    return vector<float>();
  }

  inline const vector<string>& external_output() const {
    return external_output_;
  }

  inline const vector<string>& external_input() const {
    return external_input_;
  }

  /* Used to attach Observers to operators of a Net
   *
   * Returns pointers to objects owned with unique_ptrs.
   * Use with caution.
   */
  virtual vector<OperatorBase*> getOperators() const = 0;

  void SetObserver(ObserverBase<NetBase>* observer) {
    observer_ = observer;
  }

  void RemoveObserver() {
    observer_ = nullptr;
  }

 protected:
  vector<string> external_input_;
  vector<string> external_output_;
  string name_;
  ObserverBase<NetBase>* observer_ = nullptr;
  DISABLE_COPY_AND_ASSIGN(NetBase);
};

CAFFE_DECLARE_REGISTRY(NetRegistry, NetBase, const NetDef&, Workspace*);
#define REGISTER_NET_CREATOR(key, ...) \
  CAFFE_REGISTER_CREATOR(NetRegistry, key, __VA_ARGS__)
#define REGISTER_NET(name, ...) \
  CAFFE_REGISTER_CLASS(NetRegistry, name, __VA_ARGS__)

/**
 * @brief Creates a network, accessing / creating blobs in the given workspace.
 *
 * Note that this is different from Workspace::CreateNet. The latter adds the
 * created net object to the workspace's net map, while this function returns
 * a standalone net object.
 */
unique_ptr<NetBase> CreateNet(const NetDef& net_def, Workspace* ws);

// This is the very basic structure you need to run a network - all it
// does is simply to run everything in sequence. If you want more fancy control
// such as a DAG-like execution, check out other better net implementations.
class SimpleNet : public NetBase {
 public:
  SimpleNet(const NetDef& net_def, Workspace* ws);
  bool Run() override;
  bool RunAsync() override;
  vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) override;

  /*
   * This returns a list of pointers to objects stored in unique_ptrs.
   * Used by Observers.
   *
   * Think carefully before using.
   */
  vector<OperatorBase*> getOperators() const override {
    vector<OperatorBase*> op_list;
    for (auto& op : operators_) {
      op_list.push_back(op.get());
    }
    return op_list;
  }

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
  bool is_chain_start_ = false;
};

struct OpGraphNode {
  vector<int> children_;
  vector<int> parents_;
  int visited_inputs = 0;
  int num_orig_parents;
};
}

class DAGNetBase : public NetBase {
 public:
  using ExecutionChains = std::unordered_map<int, std::vector<int>>;
  DAGNetBase(const NetDef& net_def, Workspace* ws);
  ~DAGNetBase() override;
  bool Run() override;
  // WorkerFunction() is a function wrapper to allow us to run worker threads.
  // It checks out one ready-to-run operator from the job queue, runs it,
  // notifies all its children, and for any children that is ready, enqueues
  // it to the job queue.
  void WorkerFunction();
  vector<float> TEST_Benchmark(
      const int warmup_runs,
      const int main_runs,
      const bool run_individual) override;

  const ExecutionChains& TEST_execution_chains() const {
    return execution_chains_;
  }

  vector<OperatorBase*> getOperators() const override {
    vector<OperatorBase*> op_list;
    for (auto& op_node : operator_nodes_) {
      op_list.push_back(op_node.operator_.get());
    }
    return op_list;
  }

 protected:
  virtual bool RunAt(const std::vector<int>& chain) = 0;

  vector<internal::OperatorNode> operator_nodes_;
  ExecutionChains execution_chains_;
  vector<int> initial_frontier_;
  std::unique_ptr<SimpleQueue<int>> job_queue_;
  std::vector<std::thread> workers_;
  int num_workers_;
  int num_workers_first_iteration_;
  int remaining_ops_;

  bool success_;
  int iter_;
  std::mutex remaining_ops_mutex_;
  std::condition_variable cv_;
  std::mutex run_in_progress_;

  DISABLE_COPY_AND_ASSIGN(DAGNetBase);
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_NET_H_
