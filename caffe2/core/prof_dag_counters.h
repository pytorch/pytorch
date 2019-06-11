#ifndef PROF_DAG_COUNTERS_H
#define PROF_DAG_COUNTERS_H

#include "caffe2/core/common.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/timer.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/proto/prof_dag.pb.h"

#include <unordered_map>

namespace caffe2 {

class ProfDAGStats {
 public:
  ProfDAGStats() : sum_(0.0), sqrsum_(0.0), cnt_(0) {}
  explicit ProfDAGStats(float time_ms)
      : sum_(time_ms), sqrsum_(time_ms * time_ms), cnt_(1) {}

  ProfDAGStats& operator+=(const ProfDAGStats& rhs) {
    sum_ += rhs.sum_;
    sqrsum_ += rhs.sqrsum_;
    cnt_ += rhs.cnt_;
    return *this;
  }

  std::pair<float, float> computeMoments() const {
    CAFFE_ENFORCE_GT(cnt_, 0);
    float mean = sum_ / cnt_;
    float stddev = std::sqrt(std::abs(sqrsum_ / cnt_ - mean * mean));
    return {mean, stddev};
  }

  float sum() const {
    return sum_;
  }

  float sqrsum() const {
    return sqrsum_;
  }

  size_t cnt() const {
    return cnt_;
  }

 private:
  float sum_;
  float sqrsum_;
  size_t cnt_;
};

class ProfDAGReport {
 public:
  friend class ProfDAGCounters;
  // Collects the execution time per each operator type
  ProfDAGProtos GetOperatorStats() const;

  // Collects the execution time of each operator, the output is
  // formatted as a map: (netName__opIndex__opType, cost)
  ProfDAGProtos GetPerOperatorCost() const;

  ProfDAGReport& operator+=(const ProfDAGReport& rhs);

  void PrintStats();

 private:
  ProfDAGProto statsProto(
      const std::string& name,
      const ProfDAGStats& stats,
      const std::vector<std::string>& op_extra_info) const;

  bool hasStats() const;

  std::vector<std::string> op_types_;
  std::vector<std::vector<std::string>> op_extra_info_;

  std::string net_name_;

  int num_runs_;
  // Cumulative stats per operator instance of the net
  std::vector<ProfDAGStats> time_per_op_total_;

  // Cumulative stats per unique operator type
  CaffeMap<std::string, ProfDAGStats> time_per_op_type_total_;

  CaffeMap<std::string, ProfDAGStats> times_per_run_per_type_total_;

  ProfDAGStats runtime_stats_;
};

/**
 * A simple wrapper around prof_dag's counters
 */
class ProfDAGCounters {
 public:
  explicit ProfDAGCounters(const std::shared_ptr<const NetDef>& net_def);

  // ReportRunStart/End are called at the beginning and at the end of
  // each net's run
  void ReportRunStart();
  void ReportRunEnd();

  void AddPerOpStartTime(size_t op_id);
  void AddPerOpEndTime(size_t op_id);
  void AddPerOpAsyncEndTime(size_t op_id);
  ProfDAGReport GetReport() const;

 private:
  Timer timer_;

  std::vector<float> op_start_times_run_;
  std::vector<float> op_end_times_run_;
  std::vector<float> op_async_end_times_run_;
  ProfDAGReport report_;
};

} // namespace caffe2

#endif
