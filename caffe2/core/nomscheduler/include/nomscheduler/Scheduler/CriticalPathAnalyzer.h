//===----------------------------------------------------------------------===//
//
// Tool to analyze (theoretical) critical path of a task scheduling problems.
//
//===----------------------------------------------------------------------===//

#ifndef NOM_SCHEDULER_CRITICAL_PATH_ANALYZER_H
#define NOM_SCHEDULER_CRITICAL_PATH_ANALYZER_H

#include <vector>

#include "Scheduler.h"

namespace nomscheduler {

class CriticalPathOutput {
 public:
  float getTotalCost() const {
    return totalCost_;
  }

  std::vector<int> getTaskIds() const {
    return taskIds_;
  }

  std::vector<int> getDeviceIds() const {
    return deviceIds_;
  }

  void setOutput(
      float totalCost,
      const std::vector<int>& taskIds,
      const std::vector<int>& deviceIds) {
    totalCost_ = totalCost;
    taskIds_ = taskIds;
    deviceIds_ = deviceIds;
  }

 private:
  float totalCost_;

  // Task along the critical path.
  std::vector<int> taskIds_;
  // Device assignment of the tasks along the critical path.
  std::vector<int> deviceIds_;
};

class CriticalPathAnalyzer {
 public:
  // Analyze the theoretical critical path(s) of an input scheduling problem.
  // Communication cost should be taken into account.
  // Formal definition
  // The best-scenario computation cost of a path
  // Task1 -> ... -> TaskK
  // in the task dependency graph
  // is defined by chossing a device assignment
  // (Device1, ... , DeviceK)
  // that minimizes the total computation cost (communication cost is taken
  // into account).
  //
  // The critical path is defined as the path that has the maximum
  // best-scenario computation cost.
  CriticalPathOutput analyze(const SchedulerInput& input);
};

} // namespace nomscheduler

#endif // NOM_SCHEDULER_CRITICAL_PATH_ANALYZER_H
