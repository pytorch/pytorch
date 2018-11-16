#ifndef NOM_SCHEDULER_HEFT_SCHEDULER_INTERNAL_H
#define NOM_SCHEDULER_HEFT_SCHEDULER_INTERNAL_H

#include "Scheduler.h"

namespace heftscheduler {

// Internal state associated with a task while the algorithm is running.
struct TaskState {
  // Average computation cost across all devices.
  float avgComputationCost;

  // The upward rank of a task T is defined recursively as:
  // upwardRank(T) =
  // avgComputationCost(T) + max(avgCommCost(T, T') + upwardRank(T'))
  // Basically, upwardRank(T) is the length of the critical path from
  // task T to the exit task, including the computation cost of task T.
  float upwardRank;
  bool upwardRankComputed = false;
};

// Represents a slot of time to schedule tasks on a device.
// Additionally store the number of used cores, for intra-op parallelism.
struct CoreSlot {
  float startTime, endTime;
  int usedCores;
};

// Internal state associated with a device while the algorithm is running.
struct DeviceState {
  // Maintain a list of slots per device.
  // The slots are guaranteed to be continuous.
  std::vector<CoreSlot> slots;
};

// Internal state of the HEFT scheduling algorithm.
struct AlgorithmState {
  explicit AlgorithmState(const nomscheduler::SchedulerInput& input) {
    int nTasks = input.getNumberOfTasks();

    tasksState.resize(nTasks);

    // Initial, unsorted values.
    taskIdsByUpwardRank.resize(nTasks);
    for (int taskId = 0; taskId < nTasks; taskId++) {
      taskIdsByUpwardRank[taskId] = taskId;
    }

    int nDevices = input.getNumberOfDevices();
    devicesState.resize(nDevices);
    for (int deviceId = 0; deviceId < nDevices; deviceId++) {
      CoreSlot all;
      all.startTime = 0;
      all.endTime = std::numeric_limits<float>::infinity();
      all.usedCores = 0;
      // Initially, there is no task scheduled on each device, so we just make
      // one slot that covers the entire time horizon.
      devicesState.at(deviceId).slots.emplace_back(all);
    }
  }

  std::vector<TaskState> tasksState;

  std::vector<DeviceState> devicesState;

  // Task ids sorted by upward ranks in decreasing order.
  // It can be shown that this is also a topological order.
  std::vector<int> taskIdsByUpwardRank;

  // Average data transfer rate between two devices.
  float avgDataTransferRate;
};

} // namespace heftscheduler

#endif // NOM_SCHEDULER_HEFT_SCHEDULER_INTERNAL_H
