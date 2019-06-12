#pragma once

#include "caffe2/core/net_dag.h"
#include "caffe2/proto/prof_dag.pb.h"

namespace caffe2 {

struct Stats {
  float sum;
  float sqrsum;
  size_t cnt;
};

/**
 * This net type is identical to DAGNet, except that it
 * measures the time taken for each and every operator.
 *
 * To collect statistics from stable runs, this net ignores the first run.
 * Thus, at least two runs are required for this net to print operator metrics.
 */
class ProfDAGNet : public DAGNetBase {
 public:
  ProfDAGNet(const std::shared_ptr<const NetDef>& net_def, Workspace* ws);
  ~ProfDAGNet();
  bool SupportsAsync() override {
    return false;
  }
  ProfDAGProtos GetOperatorStats();

  // GetPerOperatorCost collects the execution time of each operator, the
  // output is formatted as a map: (netName__opIndex__opType, cost)
  ProfDAGProtos GetPerOperatorCost();

 protected:
  bool DoRunAsync() override;
  bool RunAt(int chain_id, const std::vector<int>& chain) override;
  void PrintStats();
  void ValidateOpTensorDevices();
  ProfDAGProto ProtoMsg(std::pair<std::string, Stats> op_stat) const;
  // Cumulative sum and sum squared time spent per operator instance in net.
  std::vector<Stats> time_per_op_total_;
  // Cumulative sum and sum squared time spent per unique operator type.
  CaffeMap<std::string, Stats> time_per_op_type_total_;
  int runs_ = 0;
};

} // namespace caffe2
