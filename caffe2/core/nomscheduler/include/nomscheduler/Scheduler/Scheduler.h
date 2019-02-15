//===----------------------------------------------------------------------===//
//
// nomnigraph supports for task scheduling problems.
//
//===----------------------------------------------------------------------===//

#ifndef NOM_SCHEDULER_SCHEDULER_H
#define NOM_SCHEDULER_SCHEDULER_H

#include "caffe2/core/common.h"
#include "nomnigraph/Graph/Graph.h"

#include <algorithm>
#include <vector>

namespace nomscheduler {

// Models a processing unit (such as CPU/GPU/accelerator/...) that can execute a
// task.
class Device {
 public:
  Device() {}

  int getNumberOfCores() const {
    return numberOfCores_;
  }

  void setNumberOfCores(int numberOfCores) {
    numberOfCores_ = numberOfCores;
  }

  // unit: GB
  float getMaxMemory() const {
    return maxMemory_;
  }

  void setMaxMemory(float maxMemory) {
    maxMemory_ = maxMemory;
  }

 private:
  int numberOfCores_;
  float maxMemory_;
};

// Models a link between two devices.
class DeviceEdge {
 public:
  DeviceEdge() {}

  // data transfer rate between two devices (unit: s / bytes)
  float getDataTransferRate() const {
    return dataTransferRate_;
  }

  void setDataTransferRate(float dataTransferRate) {
    dataTransferRate_ = dataTransferRate;
  }

 private:
  float dataTransferRate_;
};

// Models a unit of work that can be scheduled to run on a device.
class Task {
 public:
  Task(int taskId) : taskId_(taskId) {}

  // number of cores that will be used by the task
  int getIntraDeviceParallelism() const {
    return intraDeviceParallelism_;
  }

  void setIntraDeviceParallelism(int intraDeviceParallelism) {
    intraDeviceParallelism_ = intraDeviceParallelism;
  }

  // static memory consumed by the task, unit: GB
  float getStaticMemoryConsumed() const {
    return staticMemoryConsumed_;
  }

  void setStaticMemoryConsumed(float staticMemoryConsumed) {
    staticMemoryConsumed_ = staticMemoryConsumed;
  }

  int getId() const {
    return taskId_;
  }

 private:
  int intraDeviceParallelism_;
  float staticMemoryConsumed_;
  int taskId_;
};

// Model a dependency between two tasks. An edge between Task A -> Task B
// means that Task B depends on the output of Task A, and so Task B must start
// after task A finishes.
// The edge A->B also holds the size of the data that needs to be transferred
// from task A to task B, i.e. the total size of the blobs produced by A
// and consumed by B.
class TaskEdge {
 public:
  // size of data transfered between two tasks (unit : bytes)
  float getDataSize() const {
    return dataSize_;
  }

  void setDataSize(float dataSize) {
    dataSize_ = dataSize;
  }

 private:
  float dataSize_;
};

// Represents the cost model of a task executed on a specific device.
class TaskDeviceEdge {
 public:
  // estimated computation cost for a task executed by a device
  // (runtime, unit: ms)
  float getComputationCost() const {
    return computationCost_;
  }

  void setComputationCost(float computationCost) {
    computationCost_ = computationCost;
  }

  // Return true if the task can be executed by the device.
  bool isPossible() const {
    return possible_;
  }

  void setPossible(bool possible) {
    possible_ = possible;
  }

 private:
  float computationCost_;
  bool possible_ = true;
};

// dependency DAG between tasks
using TaskGraph = nom::Graph<Task, TaskEdge>;

// (undirected) graph between devices, to represent communication links
// between devices
using DeviceGraph = nom::Graph<Device, DeviceEdge>;

// (bipartite) task - device graph, represents estimated cost model for
// task execution on each device
// We don't currently store data on this graph's node, so int type is just a
// placeholder.
using TaskDeviceCostModelGraph = nom::Graph<int /*unused*/, TaskDeviceEdge>;

// Input to the scheduler. Underneath, the input is represented by one
// TaskGraph, one DeviceGraph and one TaskDeviceCostModelGraph.
class SchedulerInput {
 public:
  SchedulerInput(int numTasks, int numDevices) {
    for (int taskId = 0; taskId < numTasks; taskId++) {
      tasks_.emplace_back(taskGraph_.createNode(taskId));
      taskNodes_.emplace_back(costModelGraph_.createNode());
    }

    for (int deviceId = 0; deviceId < numDevices; deviceId++) {
      devices_.emplace_back(deviceGraph_.createNode());
      deviceNodes_.emplace_back(costModelGraph_.createNode());
    }

    for (int taskId = 0; taskId < numTasks; taskId++) {
      for (int deviceId = 0; deviceId < numDevices; deviceId++) {
        costModelGraph_.createEdge(
            taskNodes_[taskId], deviceNodes_[deviceId], TaskDeviceEdge());
      }
    }

    for (int deviceId1 = 0; deviceId1 < numDevices; deviceId1++) {
      for (int deviceId2 = 0; deviceId2 < numDevices; deviceId2++) {
        deviceGraph_.createEdge(
            devices_[deviceId1], devices_[deviceId2], DeviceEdge());
      }
    }
  }

  int getNumberOfDevices() const {
    return deviceGraph_.getNodesCount();
  }

  int getNumberOfTasks() const {
    return taskGraph_.getNodesCount();
  }

  Device* getMutableDevice(int deviceId) {
    return devices_[deviceId]->mutableData();
  }

  Task* getMutableTask(int taskId) {
    return tasks_[taskId]->mutableData();
  }

  const Device& getDevice(int deviceId) const {
    return devices_[deviceId]->data();
  }

  const Task& getTask(int taskId) const {
    return tasks_[taskId]->data();
  }

  void createTaskDependency(int taskId1, int taskId2) {
    taskGraph_.createEdge(tasks_[taskId1], tasks_[taskId2], TaskEdge());
  }

  TaskDeviceEdge* getMutableTaskDeviceCostModel(int taskId, int deviceId) {
    return costModelGraph_.getEdge(taskNodes_[taskId], deviceNodes_[deviceId])
        ->mutableData();
  }

  const TaskDeviceEdge& getTaskDeviceCostModel(int taskId, int deviceId) const {
    return costModelGraph_.getEdge(taskNodes_[taskId], deviceNodes_[deviceId])
        ->data();
  }

  DeviceEdge* getMutableDeviceEdge(int deviceId1, int deviceId2) {
    return deviceGraph_.getEdge(devices_[deviceId1], devices_[deviceId2])
        ->mutableData();
  }

  const DeviceEdge& getDeviceEdge(int deviceId1, int deviceId2) const {
    return deviceGraph_.getEdge(devices_[deviceId1], devices_[deviceId2])
        ->data();
  }

  TaskEdge* getMutableTaskEdge(int taskId1, int taskId2) {
    return taskGraph_.getEdge(tasks_[taskId1], tasks_[taskId2])->mutableData();
  }

  const TaskEdge& getTaskEdge(int taskId1, int taskId2) const {
    return taskGraph_.getEdge(tasks_[taskId1], tasks_[taskId2])->data();
  }

  TaskGraph::NodeRef getTaskNode(int taskId) const {
    return tasks_[taskId];
  }

 private:
  TaskGraph taskGraph_;
  DeviceGraph deviceGraph_;
  TaskDeviceCostModelGraph costModelGraph_;

  std::vector<TaskGraph::NodeRef> tasks_;
  std::vector<DeviceGraph::NodeRef> devices_;

  std::vector<TaskDeviceCostModelGraph::NodeRef> taskNodes_;
  std::vector<TaskDeviceCostModelGraph::NodeRef> deviceNodes_;
};

// Represents a schedule item for a task. Consists of the device that the task
// should be assigned to, and the (estimated) start and end time of the task
// execution based on the cost model given to the scheduler.
class TaskScheduleItem {
 public:
  int getAssignedDeviceId() const {
    return assignedDeviceId_;
  }

  bool isAssigned() const {
    return assignedDeviceId_ != -1;
  }

  void setAssignedDeviceId(int assignedDeviceId) {
    assignedDeviceId_ = assignedDeviceId;
  }

  float getStartTime() const {
    return startTime_;
  }

  void setStartTime(float startTime) {
    startTime_ = startTime;
  }

  float getEndTime() const {
    return endTime_;
  }

  void setEndTime(float endTime) {
    endTime_ = endTime;
  }

 private:
  int assignedDeviceId_ = -1;
  float startTime_;
  float endTime_;
};

// Represents an output of the static scheduler - a map from each task
// to a TaskScheduleItem for that task.
class SchedulerOutput {
 public:
  SchedulerOutput(int numTasks) {
    for (int i = 0; i < numTasks; i++) {
      taskScheduleItems_.emplace_back();
    }
  }

  TaskScheduleItem& getMutableTaskScheduleItem(int taskId) {
    return taskScheduleItems_[taskId];
  }

  const TaskScheduleItem& getTaskScheduleItem(int taskId) const {
    return taskScheduleItems_[taskId];
  }

  // The finish time of the schedule, which is just the maximum end time
  // of all the schedule items.
  float getFinishTime() const {
    float result = 0;
    for (auto& scheduleItem : taskScheduleItems_) {
      result = std::max(result, scheduleItem.getEndTime());
    }
    return result;
  }

  // Fails to compute a schedule.
  bool isFailure() const {
    return failure_;
  }

  void setFailure(bool failure) {
    failure_ = failure;
  }

 private:
  std::vector<TaskScheduleItem> taskScheduleItems_;
  bool failure_;
};

// Interface for static schedulers.
class Scheduler {
 public:
  virtual ~Scheduler() {}
  virtual SchedulerOutput schedule(const SchedulerInput&) = 0;
};

} // namespace nomscheduler

#endif // NOM_SCHEDULER_SCHEDULER_H
