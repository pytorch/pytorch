#include "CriticalPathAnalyzer.h"
#include <vector>

namespace {

// Data structure used by the dynamic programming algorithm.
struct PathTrace {
  float cost;
  bool computed = false;
  // Successive (taskId, deviceId) on the critical path.
  int depTaskId = -1;
  int depDeviceId = -1;
};

// List of tasks and device assignment along a path.
struct FullPathTrace {
  std::vector<int> taskIds;
  std::vector<int> deviceIds;
};

struct State {
  explicit State(const nomscheduler::SchedulerInput& input) {
    int nTasks = input.getNumberOfTasks();
    int nDevices = input.getNumberOfDevices();
    pathTrace.resize(nTasks);
    for (int taskId = 0; taskId < nTasks; taskId++) {
      pathTrace.at(taskId).resize(nDevices);
    }
  }

  // Use dynamic programming to compute the theorical critical path.
  // pathTrace[T][D] = theoretical critical path starting from task T,
  // given that task T is executed on device D.
  // Dynamic programming can be used because the critical path has optimize
  // substructure (on a DAG), and can be expressed recursively as follows.
  // pathTrace[T][D] =
  // argmax(T', compCost(T, D) +
  // argmin(D', commCost(T, D, T', D') + pathTrace[T'][D'])
  std::vector<std::vector<PathTrace>> pathTrace;
};

void computePathTrace(
    const nomscheduler::SchedulerInput& input,
    State& state,
    int taskId,
    int deviceId) {
  auto& trace = state.pathTrace.at(taskId).at(deviceId);
  if (trace.computed) {
    return;
  }

  int nDevices = input.getNumberOfDevices();
  float computationCost =
      input.getTaskDeviceCostModel(taskId, deviceId).getComputationCost();
  float maxCost = computationCost;

  // Recursively compute (with memoization) critical path trace on all possible
  // (depTaskId, depDeviceId) combinations where depTaskId is a dependent of
  // taskId.
  for (auto outEdge : input.getTaskNode(taskId)->getOutEdges()) {
    auto depTaskId = outEdge->head()->data().getId();

    float dataSize = outEdge->data().getDataSize();

    int bestCaseDeviceId = -1;
    float bestCaseCost = 0;

    for (int depDeviceId = 0; depDeviceId < nDevices; depDeviceId++) {
      float commCost = dataSize *
          input.getDeviceEdge(deviceId, depDeviceId).getDataTransferRate();

      // Make sure that path trace is computed on (depTaskId, depDeviceId)
      computePathTrace(input, state, depTaskId, depDeviceId);

      auto& depTrace = state.pathTrace.at(depTaskId).at(depDeviceId);

      float totalCost = computationCost + commCost + depTrace.cost;

      if (bestCaseDeviceId == -1 || totalCost < bestCaseCost) {
        // Given depTaskId, choose the device assignment that minimizes the
        // total computation cost.
        bestCaseDeviceId = depDeviceId;
        bestCaseCost = totalCost;
      }
    }

    if (bestCaseCost > maxCost) {
      maxCost = bestCaseCost;
      trace.depTaskId = depTaskId;
      trace.depDeviceId = bestCaseDeviceId;
    }
  }

  trace.computed = true;
  trace.cost = maxCost;
}

FullPathTrace
constructFullPathTrace(const State& state, int taskId, int deviceId) {
  FullPathTrace output;
  int t = taskId;
  int d = deviceId;
  while (t != -1) {
    output.taskIds.emplace_back(t);
    output.deviceIds.emplace_back(d);
    auto& trace = state.pathTrace.at(t).at(d);
    t = trace.depTaskId;
    d = trace.depDeviceId;
  }
  return output;
}

} // namespace

namespace nomscheduler {

CriticalPathOutput CriticalPathAnalyzer::analyze(const SchedulerInput& input) {
  CriticalPathOutput output;

  auto state = State(input);

  float maxCost = 0;
  int nTasks = input.getNumberOfTasks();
  int nDevices = input.getNumberOfDevices();

  int criticalPathTaskId = -1;
  int criticalPathDeviceId = -1;

  for (int taskId = 0; taskId < nTasks; taskId++) {
    float bestCaseDeviceId = -1;
    float bestCaseCost = 0;
    for (int deviceId = 0; deviceId < nDevices; deviceId++) {
      computePathTrace(input, state, taskId, deviceId);
      auto& trace = state.pathTrace.at(taskId).at(deviceId);
      if (bestCaseDeviceId == -1 || trace.cost < bestCaseCost) {
        bestCaseCost = trace.cost;
        bestCaseDeviceId = deviceId;
      }
    }
    if (bestCaseCost > maxCost) {
      maxCost = bestCaseCost;
      criticalPathTaskId = taskId;
      criticalPathDeviceId = bestCaseDeviceId;
    }
  }

  auto fullPathTrace =
      constructFullPathTrace(state, criticalPathTaskId, criticalPathDeviceId);
  output.setOutput(maxCost, fullPathTrace.taskIds, fullPathTrace.deviceIds);

  return output;
}

} // namespace nomscheduler
