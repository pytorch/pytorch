#ifndef CAFFE2_CORE_NET_SIMPLE_H_
#define CAFFE2_CORE_NET_SIMPLE_H_

#include <vector>

#include "c10/util/Registry.h"
#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/net.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/workspace.h"
#include "caffe2/proto/caffe2_pb.h"

namespace caffe2 {

struct IndividualMetrics {
 public:
  explicit IndividualMetrics(const std::vector<OperatorBase*>& operators)
      : main_runs_(0), operators_(operators) {
    const auto num_ops = operators_.size();
    time_per_op.resize(num_ops, 0.0);
  }
  // run ops while collecting profiling results
  void RunOpsWithProfiling();

  // print out profiling results
  void PrintOperatorProfilingResults();

  const vector<float>& GetTimePerOp() {
    return time_per_op;
  }

  float setup_time{0.0};
  float memory_alloc_time{0.0};
  float memory_dealloc_time{0.0};
  float output_dealloc_time{0.0};

 private:
  int main_runs_;
  const std::vector<OperatorBase*>& operators_;

  vector<float> time_per_op;
  vector<uint64_t> flops_per_op;
  vector<uint64_t> memory_bytes_read_per_op;
  vector<uint64_t> memory_bytes_written_per_op;
  vector<uint64_t> param_bytes_per_op;

  CaffeMap<string, int> num_ops_per_op_type_;
  CaffeMap<string, float> time_per_op_type;
  CaffeMap<string, float> flops_per_op_type;
  CaffeMap<string, float> memory_bytes_read_per_op_type;
  CaffeMap<string, float> memory_bytes_written_per_op_type;
  CaffeMap<string, float> param_bytes_per_op_type;
};

// This is the very basic structure you need to run a network - all it
// does is simply to run everything in sequence. If you want more fancy control
// such as a DAG-like execution, check out other better net implementations.
class TORCH_API SimpleNet : public NetBase {
 public:
  SimpleNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  bool SupportsAsync() override {
    return false;
  }

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
  vector<OperatorBase*> GetOperators() const override {
    vector<OperatorBase*> op_list;
    for (auto& op : operators_) {
      op_list.push_back(op.get());
    }
    return op_list;
  }

 protected:
  bool Run() override;
  bool RunAsync() override;

  vector<unique_ptr<OperatorBase>> operators_;

  C10_DISABLE_COPY_AND_ASSIGN(SimpleNet);
};

} // namespace caffe2

#endif // CAFFE2_CORE_NET_SIMPLE_H_
