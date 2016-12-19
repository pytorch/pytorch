#include <cmath>

#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/timer.h"

namespace caffe2 {
namespace {

struct Stats {
  float sum;
  float sqrsum;
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
  ProfDAGNet(const NetDef& net_def, Workspace* ws)
      : DAGNetBase(net_def, ws), time_per_op_(operator_nodes_.size()) {
    VLOG(1) << "Constructing ProfDAGNet " << name_;
  }

  bool Run() override {
    runs_++;

    // don't collect metrics from first run
    if (runs_ <= 1) {
      return DAGNetBase::Run();
    }

    CAFFE_ENFORCE(
        time_per_op_.size() == operator_nodes_.size(),
        "Data collected for ",
        time_per_op_.size(),
        " ops, expected ",
        operator_nodes_.size(),
        " ops.");

    // create a copy and later collect the differences
    vector<Stats> time_per_op_run(time_per_op_);
    bool success = DAGNetBase::Run();

    // aggregate this run's stats per operator type
    CaffeMap<string, float> time_per_op_type_run;
    for (int idx = 0; idx < operator_nodes_.size(); idx++) {
      const auto& node = operator_nodes_[idx];
      const string& op_type = node.operator_->def().type();
      time_per_op_type_run[op_type] +=
          time_per_op_[idx].sum - time_per_op_run[idx].sum;
    }

    for (const auto& item : time_per_op_type_run) {
      time_per_op_type_[item.first].sum += item.second;
      time_per_op_type_[item.first].sqrsum += item.second * item.second;
    }

    return success;
  }

  ~ProfDAGNet() {
    VLOG(1) << "Closing ProfDAGNet " << name_;
    if (runs_ <= 1) {
      LOG(INFO) << "Insufficient runs to produce meaningful data.";
      return;
    }
    PrintStats();
  }

 protected:
  bool RunAt(const std::vector<int>& chain) override {
    bool success = true;
    Timer timer;
    for (const auto idx : chain) {
      // don't collect metrics from first run
      if (runs_ <= 1) {
        success &= operator_nodes_[idx].operator_->Run();

      } else {
        timer.Start();
        success &= operator_nodes_[idx].operator_->Run();
        float spent = timer.MilliSeconds();

        CAFFE_ENFORCE(
            time_per_op_.size() > idx,
            "Expecting ",
            time_per_op_.size(),
            " ops, but op #",
            idx,
            " was given.");
        time_per_op_[idx].sum += spent;
        time_per_op_[idx].sqrsum += spent * spent;
      }
    }
    return success;
  }

  void PrintStats() {
    CAFFE_ENFORCE(
        time_per_op_.size() == operator_nodes_.size(),
        "Data collected for ",
        time_per_op_.size(),
        " ops, expected ",
        operator_nodes_.size(),
        " ops.");

    CAFFE_ENFORCE(runs_ > 1, "# of runs: ", runs_, ", expected > 1.");
    int measured_runs = runs_ - 1;

    for (int idx = 0; idx < operator_nodes_.size(); idx++) {
      auto& node = operator_nodes_[idx];
      const string& op_type = node.operator_->def().type();
      const string& print_name = node.operator_->def().name().size()
          ? node.operator_->def().name()
          : (node.operator_->def().output_size()
                 ? node.operator_->def().output(0)
                 : "NO_OUTPUT");

      float mean = time_per_op_[idx].sum / measured_runs;
      float stddev =
          std::sqrt(time_per_op_[idx].sqrsum / measured_runs - mean * mean);
      LOG(INFO) << "Op #" << idx << " (" << print_name << ", " << op_type
                << ") " << mean << " ms/iter (" << stddev << " ms/iter)";
    }

    LOG(INFO) << "Time per operator type:";
    for (const auto& item : time_per_op_type_) {
      float mean = item.second.sum / measured_runs;
      float stddev =
          std::sqrt(item.second.sqrsum / measured_runs - mean * mean);
      LOG(INFO) << std::setw(10) << std::setfill(' ') << mean << " ms/iter ("
                << std::setw(10) << std::setfill(' ') << stddev << " ms/iter) "
                << item.first;
    }
  }

  vector<Stats> time_per_op_;
  CaffeMap<string, Stats> time_per_op_type_;
  int runs_ = 0;
};

REGISTER_NET(prof_dag, ProfDAGNet);

} // namespace
} // namespace caffe2
