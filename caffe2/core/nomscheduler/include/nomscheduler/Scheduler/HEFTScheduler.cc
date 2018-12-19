#include "HEFTScheduler.h"
#include <vector>

namespace {

void computeAverageComputationCost(
    const nomscheduler::SchedulerInput& input,
    heftscheduler::AlgorithmState& state,
    int taskId) {
  float sum = 0;
  for (int deviceId = 0; deviceId < input.getNumberOfDevices(); deviceId++) {
    sum += input.getTaskDeviceCostModel(taskId, deviceId).getComputationCost();
  }
  state.tasksState.at(taskId).avgComputationCost =
      sum / input.getNumberOfDevices();
}

void computeAverageComputationCost(
    const nomscheduler::SchedulerInput& input,
    heftscheduler::AlgorithmState& state) {
  for (int taskId = 0; taskId < input.getNumberOfTasks(); taskId++) {
    computeAverageComputationCost(input, state, taskId);
  }
}

void computeAverageDataTransferRate(
    const nomscheduler::SchedulerInput& input,
    heftscheduler::AlgorithmState& state) {
  state.avgDataTransferRate = 0;

  int nDevices = input.getNumberOfDevices();
  for (int deviceId1 = 0; deviceId1 < nDevices; deviceId1++) {
    for (int deviceId2 = 0; deviceId2 < nDevices; deviceId2++) {
      if (deviceId1 != deviceId2) {
        state.avgDataTransferRate +=
            input.getDeviceEdge(deviceId1, deviceId2).getDataTransferRate();
      }
    }
  }

  if (nDevices > 1) {
    state.avgDataTransferRate /= nDevices * (nDevices - 1);
  }
}

void computeUpwardRank(
    const nomscheduler::SchedulerInput& input,
    heftscheduler::AlgorithmState& state,
    int taskId) {
  auto& taskState = state.tasksState.at(taskId);
  if (taskState.upwardRankComputed) {
    return;
  }

  float maxDependentCost = 0;
  for (auto outEdge : input.getTaskNode(taskId)->getOutEdges()) {
    auto dependentTask = outEdge->head();
    auto dependentTaskId = dependentTask->data().getId();
    computeUpwardRank(input, state, dependentTaskId);

    float avgCommCost =
        input.getTaskEdge(taskId, dependentTaskId).getDataSize() *
        state.avgDataTransferRate;

    maxDependentCost = std::max(
        maxDependentCost,
        avgCommCost + state.tasksState.at(dependentTaskId).upwardRank);
  }

  taskState.upwardRankComputed = true;
  taskState.upwardRank = taskState.avgComputationCost + maxDependentCost;
}

void computeUpwardRank(
    const nomscheduler::SchedulerInput& input,
    heftscheduler::AlgorithmState& state) {
  for (int taskId = 0; taskId < input.getNumberOfTasks(); taskId++) {
    computeUpwardRank(input, state, taskId);
  }
}

void sortTasksByUpwardRank(heftscheduler::AlgorithmState& state) {
  std::sort(
      state.taskIdsByUpwardRank.begin(),
      state.taskIdsByUpwardRank.end(),
      [&state](int taskId1, int taskId2) -> bool {
        return state.tasksState[taskId1].upwardRank >
            state.tasksState[taskId2].upwardRank;
      });
}

// Task assignment information on a specific device.
struct TaskAssignment {
  bool possible;
  float start, end;
};

// Compute the earliest possible start time of a task on a device based
// on dependency graph information and current schedule.
float computeEarliestPossibleStartTimeFromDAG(
    const nomscheduler::SchedulerInput& input,
    const heftscheduler::AlgorithmState& state,
    const nomscheduler::SchedulerOutput& output,
    int taskId,
    int deviceId) {
  float result = 0.0f;
  for (auto& inEdge : input.getTaskNode(taskId)->getInEdges()) {
    auto prereqTaskId = inEdge->tail()->data().getId();
    auto& prereqScheduleItem = output.getTaskScheduleItem(prereqTaskId);
    // Since the algorithm schedule tasks in topological order, at this point
    // all the prerequisites should have been scheduled.
    assert(prereqScheduleItem.isAssigned());

    // Communication time to send output from the prerequisite task to the
    // current task.
    float commTime = inEdge->data().getDataSize() *
        input.getDeviceEdge(prereqScheduleItem.getAssignedDeviceId(), deviceId)
            .getDataTransferRate();
    result = std::max(result, prereqScheduleItem.getEndTime() + commTime);
  }
  return result;
}

void computeEarliestTaskAssignment(
    const nomscheduler::SchedulerInput& input,
    const heftscheduler::AlgorithmState& state,
    const nomscheduler::SchedulerOutput& output,
    int taskId,
    int deviceId,
    TaskAssignment& assignment) {
  assignment.possible = false;
  auto& costModel = input.getTaskDeviceCostModel(taskId, deviceId);
  if (!costModel.isPossible()) {
    // If the task cannot be scheduled on the device.
    return;
  }
  float earliestPossibleStartTime = computeEarliestPossibleStartTimeFromDAG(
      input, state, output, taskId, deviceId);

  int coresUsedByTask = input.getTask(taskId).getIntraDeviceParallelism();
  int coresInDevice = input.getDevice(deviceId).getNumberOfCores();

  auto& slots = state.devicesState.at(deviceId).slots;
  for (int slotId = 0; slotId < slots.size(); slotId++) {
    auto& slot = slots.at(slotId);
    if (earliestPossibleStartTime > slot.endTime) {
      // Ignore slots that end before the earliest possible start time.
      continue;
    }
    float requiredStart = std::max(earliestPossibleStartTime, slot.startTime);
    float requiredEnd = requiredStart + costModel.getComputationCost();

    // Find a range of slots that can accommodate the task.
    bool found = false;
    for (int endSlotId = slotId; endSlotId < slots.size(); endSlotId++) {
      auto& endSlot = slots[endSlotId];
      if (endSlot.usedCores + coresUsedByTask > coresInDevice) {
        // Not enough cores to execute the task.
        break;
      }
      if (requiredEnd <= endSlot.endTime) {
        // We found a range of slots that covers the time window to execute
        // the task.
        found = true;
        break;
      }
    }
    if (found) {
      assignment.possible = true;
      assignment.start = requiredStart;
      assignment.end = requiredEnd;
      break;
    }
  }
}

// Update the list of slots for a device, given the schedule of a new task
// on that device.
void updateSlots(
    const nomscheduler::SchedulerInput& input,
    heftscheduler::AlgorithmState& state,
    int taskId,
    int deviceId,
    const TaskAssignment& assignment) {
  auto& slots = state.devicesState.at(deviceId).slots;

  // Find start slot of the slot ranges that cover the task assignment time
  // window.
  int startSlotId = 0;
  while (startSlotId < slots.size() &&
         assignment.start >= slots.at(startSlotId).endTime) {
    startSlotId++;
  }
  // Start slot must exist.
  assert(
      startSlotId < slots.size() &&
      assignment.start < slots.at(startSlotId).endTime);

  if (assignment.start > slots.at(startSlotId).startTime) {
    auto& startSlot = slots.at(startSlotId);
    // Split the start slot into two.
    heftscheduler::CoreSlot newSlot;
    newSlot.startTime = startSlot.startTime;
    newSlot.endTime = assignment.start;
    newSlot.usedCores = startSlot.usedCores;

    startSlot.startTime = assignment.start;
    slots.insert(slots.begin() + startSlotId, newSlot);
    startSlotId++;
  }

  // Find end slot of the slot ranges that cover the task assignment time
  // window.
  int endSlotId = startSlotId;
  while (endSlotId < slots.size() &&
         assignment.end > slots.at(endSlotId).endTime) {
    endSlotId++;
  }
  // End slot must exist.
  assert(
      endSlotId < slots.size() &&
      assignment.end <= slots.at(endSlotId).endTime);

  if (assignment.end < slots.at(endSlotId).endTime) {
    // Split the end slot into two.
    auto& endSlot = slots.at(endSlotId);
    heftscheduler::CoreSlot newSlot;
    newSlot.startTime = endSlot.startTime;
    newSlot.endTime = assignment.end;
    newSlot.usedCores = endSlot.usedCores;

    endSlot.startTime = assignment.end;
    slots.insert(slots.begin() + endSlotId, newSlot);
  }

  // Now we just update the usedCores count of the slots.
  int coresUsedByTask = input.getTask(taskId).getIntraDeviceParallelism();
  for (int slotId = startSlotId; slotId <= endSlotId; slotId++) {
    slots.at(slotId).usedCores += coresUsedByTask;
  }
}

void scheduleTask(
    const nomscheduler::SchedulerInput& input,
    heftscheduler::AlgorithmState& state,
    nomscheduler::SchedulerOutput& output,
    int taskId) {
  // For each device, calculate the earliest possible assignment of the task
  // to the device (or if the task can even be assigned to the device at all).
  std::vector<TaskAssignment> assignments(input.getNumberOfDevices());
  for (int deviceId = 0; deviceId < input.getNumberOfDevices(); deviceId++) {
    computeEarliestTaskAssignment(
        input, state, output, taskId, deviceId, assignments.at(deviceId));
  }

  // Select the device that minimize the earlist finish time of the task.
  int scheduledDeviceId = -1;
  for (int deviceId = 0; deviceId < input.getNumberOfDevices(); deviceId++) {
    auto& assignment = assignments.at(deviceId);
    if (assignment.possible &&
        ((scheduledDeviceId == -1 ||
          assignment.end < assignments.at(scheduledDeviceId).end))) {
      scheduledDeviceId = deviceId;
    }
  }

  if (scheduledDeviceId == -1) {
    // No device can execute the task.
    output.setFailure(true);
  } else {
    auto& assignment = assignments.at(scheduledDeviceId);
    auto& taskScheduleItem = output.getMutableTaskScheduleItem(taskId);
    taskScheduleItem.setAssignedDeviceId(scheduledDeviceId);
    taskScheduleItem.setStartTime(assignment.start);
    taskScheduleItem.setEndTime(assignment.end);
    updateSlots(input, state, taskId, scheduledDeviceId, assignment);
  }
}

void scheduleTasks(
    const nomscheduler::SchedulerInput& input,
    heftscheduler::AlgorithmState& state,
    nomscheduler::SchedulerOutput& output) {
  // Loop over tasks in decreasing order of upward ranks (which is also
  // a topological order) and schedule each one.
  for (int taskId : state.taskIdsByUpwardRank) {
    scheduleTask(input, state, output, taskId);
    if (output.isFailure()) {
      break;
    }
  }
}

} // namespace

namespace nomscheduler {

std::pair<SchedulerOutput, heftscheduler::AlgorithmState>
HEFTScheduler::scheduleInternal(const SchedulerInput& input) {
  heftscheduler::AlgorithmState state(input);

  computeAverageComputationCost(input, state);
  computeAverageDataTransferRate(input, state);
  computeUpwardRank(input, state);
  sortTasksByUpwardRank(state);

  SchedulerOutput output(input.getNumberOfTasks());
  output.setFailure(false);
  scheduleTasks(input, state, output);
  return std::make_pair(output, state);
}

SchedulerOutput HEFTScheduler::schedule(const SchedulerInput& input) {
  return scheduleInternal(input).first;
}

} // namespace nomscheduler
