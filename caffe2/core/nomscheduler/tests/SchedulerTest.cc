#include <algorithm>
#include <cstdio>
#include <limits>
#include <sstream>
#include <unordered_set>

#include "nomscheduler/Scheduler/CriticalPathAnalyzer.h"
#include "nomscheduler/Scheduler/HEFTScheduler.h"
#include "nomscheduler/Scheduler/Scheduler.h"

#include <gtest/gtest.h>

namespace nomscheduler {

SchedulerInput loadSchedulerInputFromString(const std::string& fileInput) {
  std::stringstream ss;
  ss << fileInput;

  int numTasks, numDevices;
  ss >> numTasks >> numDevices;

  SchedulerInput result(numTasks, numDevices);

  // Cores per devices
  for (int id = 0; id < numDevices; id++) {
    int numCores;
    ss >> numCores;
    result.getMutableDevice(id)->setNumberOfCores(numCores);
  }

  // Parallelism per task
  for (int id = 0; id < numTasks; id++) {
    int parallelismLevel;
    ss >> parallelismLevel;
    result.getMutableTask(id)->setIntraDeviceParallelism(parallelismLevel);
  }

  // The computation costs of each task
  for (int taskId = 0; taskId < numTasks; taskId++) {
    for (int deviceId = 0; deviceId < numDevices; deviceId++) {
      float cost;
      ss >> cost;
      if (cost < 0) {
        result.getMutableTaskDeviceCostModel(taskId, deviceId)
            ->setPossible(false);
      } else {
        result.getMutableTaskDeviceCostModel(taskId, deviceId)
            ->setComputationCost(cost);
      }
    }
  }

  for (int deviceId1 = 0; deviceId1 < numDevices; deviceId1++) {
    for (int deviceId2 = 0; deviceId2 < numDevices; deviceId2++) {
      float rate;
      ss >> rate;
      result.getMutableDeviceEdge(deviceId1, deviceId2)
          ->setDataTransferRate(rate);
    }
  }

  for (int taskId1 = 0; taskId1 < numTasks; taskId1++) {
    for (int taskId2 = 0; taskId2 < numTasks; taskId2++) {
      float dataSize;
      ss >> dataSize;
      if (dataSize > 0) {
        result.createTaskDependency(taskId1, taskId2);
        result.getMutableTaskEdge(taskId1, taskId2)->setDataSize(dataSize);
      }
    }
  }

  for (int deviceId = 0; deviceId < numDevices; deviceId++) {
    float maxMemory;
    ss >> maxMemory;
    result.getMutableDevice(deviceId)->setMaxMemory(maxMemory);
  }

  for (int id = 0; id < numTasks; id++) {
    float staticMemoryConsumed;
    ss >> staticMemoryConsumed;
    result.getMutableTask(id)->setStaticMemoryConsumed(staticMemoryConsumed);
  }

  return result;
}

// A simple scheduling algorithm, just for testing and comparison purpose.
// For each iteration:
// - Pick any task that is ready to schedule (no dependency)
// - Then pick a device that has the earliest next available time to
// schedule that task.
// For simplicity, this algorithm does not take into account any resource
// constraints.
class SimpleScheduler : Scheduler {
 public:
  SchedulerOutput schedule(const SchedulerInput& input) override {
    int numTasks = input.getNumberOfTasks();
    SchedulerOutput result(numTasks);

    std::unordered_set<TaskGraph::NodeRef> scheduledTasks;

    // Next available time per device.
    std::vector<float> nextFreeTime;
    for (int i = 0; i < input.getNumberOfDevices(); i++) {
      nextFreeTime.emplace_back(0);
    }

    while (scheduledTasks.size() < numTasks) {
      for (int taskId = 0; taskId < numTasks; taskId++) {
        auto taskNode = input.getTaskNode(taskId);
        if (scheduledTasks.count(taskNode)) {
          continue;
        }

        bool hasDependency = false;
        for (auto& inEdge : taskNode->getInEdges()) {
          auto tail = inEdge->tail();
          if (!scheduledTasks.count(tail)) {
            hasDependency = true;
            break;
          }
        }

        if (!hasDependency) {
          scheduledTasks.insert(taskNode);

          // Find the device with earliest next available time.
          int earliestDeviceId = 0;
          for (int deviceId = 1; deviceId < input.getNumberOfDevices();
               deviceId++) {
            if (nextFreeTime[deviceId] < nextFreeTime[earliestDeviceId]) {
              earliestDeviceId = deviceId;
            }
          }

          // Schedule the task on the device.
          auto& taskScheduleItem = result.getMutableTaskScheduleItem(taskId);
          taskScheduleItem.setAssignedDeviceId(earliestDeviceId);
          taskScheduleItem.setStartTime(nextFreeTime[earliestDeviceId]);
          auto computationCost =
              input.getTaskDeviceCostModel(taskId, earliestDeviceId)
                  .getComputationCost();
          taskScheduleItem.setEndTime(
              taskScheduleItem.getStartTime() + computationCost);

          // Update next available time for the device.
          nextFreeTime[earliestDeviceId] = taskScheduleItem.getEndTime();
          break;
        }
      }
    }

    return result;
  }
};

} // namespace nomscheduler

nomscheduler::SchedulerInput getTestInput() {
  return nomscheduler::loadSchedulerInputFromString(R"(
  10 3

  1 1 1
  1 1 1 1 1 1 1 1 1 1

  14 16 9
  13 19 18
  11 13 19
  13 8 17
  12 13 10
  13 16 9
  7 15 11
  5 11 14
  18 12 20
  21 7 16

  0 1 1
  1 0 1
  1 1 0

  -1 18 12 9 11 14 -1 -1 -1 -1
  -1 -1 -1 -1 -1 -1 -1 19 16 -1
  -1 -1 -1 -1 -1 -1 23 -1 -1 -1
  -1 -1 -1 -1 -1 -1 -1 27 23 -1
  -1 -1 -1 -1 -1 -1 -1 -1 13 -1
  -1 -1 -1 -1 -1 -1 -1 15 -1 -1
  -1 -1 -1 -1 -1 -1 -1 -1 -1 17
  -1 -1 -1 -1 -1 -1 -1 -1 -1 11
  -1 -1 -1 -1 -1 -1 -1 -1 -1 13
  -1 -1 -1 -1 -1 -1 -1 -1 -1 -1

  256 256 256

  12 1 1 18 12 1 12 1 10 6
)");
}

TEST(Scheduler, SchedulerTest) {
  auto input = getTestInput();

  EXPECT_EQ(input.getNumberOfTasks(), 10);
  EXPECT_EQ(input.getNumberOfDevices(), 3);

  EXPECT_EQ(input.getDevice(1).getNumberOfCores(), 1);
  EXPECT_EQ(input.getTask(6).getIntraDeviceParallelism(), 1);

  EXPECT_EQ(input.getTaskDeviceCostModel(0, 0).getComputationCost(), 14);
  EXPECT_EQ(input.getTaskDeviceCostModel(8, 1).getComputationCost(), 12);
  EXPECT_EQ(input.getDeviceEdge(0, 2).getDataTransferRate(), 1.0f);

  EXPECT_EQ(input.getTaskEdge(0, 3).getDataSize(), 9);

  EXPECT_EQ(input.getDevice(2).getMaxMemory(), 256);
  EXPECT_EQ(input.getTask(3).getStaticMemoryConsumed(), 18);

  auto scheduler = nomscheduler::SimpleScheduler();
  auto output = scheduler.schedule(input);
  EXPECT_EQ(output.getFinishTime(), 55);
}

TEST(Scheduler, HEFTSchedulerTest) {
  auto error = 1E-3;

  auto input = getTestInput();
  auto scheduler = nomscheduler::HEFTScheduler();
  auto outputAndState = scheduler.scheduleInternal(input);
  auto state = outputAndState.second;
  EXPECT_NEAR(state.avgDataTransferRate, 1.0f, error);

  auto task9 = state.tasksState.at(9);
  EXPECT_NEAR(task9.avgComputationCost, 14.6666f, error);
  // This task has no dependency
  EXPECT_NEAR(task9.upwardRank, task9.avgComputationCost, error);

  auto task8 = state.tasksState.at(8);
  EXPECT_NEAR(task8.avgComputationCost, 16.6666f, error);
  EXPECT_NEAR(
      task8.upwardRank,
      task8.avgComputationCost +
          input.getTaskEdge(8, 9).getDataSize() / state.avgDataTransferRate +
          state.tasksState.at(9).upwardRank,
      error);
  EXPECT_NEAR(task8.upwardRank, 44.333f, error);

  auto task0 = state.tasksState.at(0);
  EXPECT_NEAR(task0.avgComputationCost, 13.0f, error);
  EXPECT_NEAR(
      task0.upwardRank,
      task0.avgComputationCost +
          input.getTaskEdge(0, 1).getDataSize() / state.avgDataTransferRate +
          state.tasksState.at(1).upwardRank,
      error);
  EXPECT_NEAR(task0.upwardRank, 108.0f, error);

  auto sortedTaskIds = std::vector<int>{0, 2, 3, 1, 4, 5, 8, 6, 7, 9};
  EXPECT_EQ(state.taskIdsByUpwardRank, sortedTaskIds);

  // Verify the output of the HEFT scheduler.
  // The input and output in this unit test matches the example in the
  // original HEFT paper.
  auto output = outputAndState.first;
  EXPECT_FALSE(output.isFailure());
  EXPECT_NEAR(output.getFinishTime(), 80, error);

  auto expectedAssignedDeviceId =
      std::vector<int>{2, 0, 2, 1, 2, 1, 2, 0, 1, 1};
  auto expectedStartTime =
      std::vector<float>{0, 27, 9, 18, 28, 26, 38, 57, 56, 73};
  auto assignedDeviceId = std::vector<int>();
  auto scheduledStartTime = std::vector<float>();
  for (int taskId = 0; taskId < input.getNumberOfTasks(); taskId++) {
    auto& taskScheduleItem = output.getTaskScheduleItem(taskId);
    assignedDeviceId.emplace_back(taskScheduleItem.getAssignedDeviceId());
    scheduledStartTime.emplace_back(taskScheduleItem.getStartTime());
  }
  EXPECT_EQ(assignedDeviceId, expectedAssignedDeviceId);
  EXPECT_EQ(scheduledStartTime, expectedStartTime);
}

TEST(Scheduler, CriticalPathAnalyzer) {
  auto input = getTestInput();
  auto analyzer = nomscheduler::CriticalPathAnalyzer();
  auto output = analyzer.analyze(input);
  EXPECT_EQ(output.getTotalCost(), 54.0f);
  auto expectedTaskIds = std::vector<int>{0, 1, 8, 9};
  auto expectedDeviceIds = std::vector<int>{1, 1, 1, 1};
  EXPECT_EQ(output.getTaskIds(), expectedTaskIds);
  EXPECT_EQ(output.getDeviceIds(), expectedDeviceIds);
}
