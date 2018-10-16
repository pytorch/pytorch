#pragma once

#include "caffe2/core/observer.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

class CostObserver;

class CostOpObserver : public ObserverBase<OperatorBase> {
 public:
  struct DetailedStat {
    int64_t invocations = 0;
    double timeSpentSum = 0;
    string opType;
    string displayName;
    struct OpSchema::Cost c;
  };

  CostOpObserver(OperatorBase* op, DetailedStat* stat);

 private:
  void Start() override;
  void Stop() override;

 private:
  DetailedStat* stat_;
};

class CostObserver : public ObserverBase<NetBase> {
 public:
  explicit CostObserver(NetBase* subject_);
  ~CostObserver();
  CaffeMap<string, OpSchema::Cost> getCostPerOpType() const {
    CaffeMap<string, OpSchema::Cost> cost_per_op_type;
    for (int idx = 0; idx < detailedOpStats_.size(); ++idx) {
      const auto& stat = detailedOpStats_[idx];
      uint64_t flops = stat.c.flops;
      uint64_t bytes_read = stat.c.bytes_read;
      uint64_t bytes_written = stat.c.bytes_written;

      cost_per_op_type[stat.opType].flops += flops;
      cost_per_op_type[stat.opType].bytes_read += bytes_read;
      cost_per_op_type[stat.opType].bytes_written += bytes_written;
    }
    return cost_per_op_type;
  }

 private:
  void Start() override;
  void Stop() override;

  std::vector<CostOpObserver::DetailedStat> detailedOpStats_;
  std::string net_name_;
};

} // namespace caffe2
